import copy
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from math import sqrt
from typing import Any, Awaitable, Callable, Optional

from embedding_service import InputValidationError, create_embeddings

PROJECTOR_CACHE_TTL_SECONDS = int(os.getenv("PROJECTOR_CACHE_TTL_SECONDS", "300"))
PROJECTOR_CACHE_MAX_ITEMS = int(os.getenv("PROJECTOR_CACHE_MAX_ITEMS", "32"))

_projector_cache_lock = threading.RLock()
_projector_cache: dict[str, tuple[float, dict[str, Any]]] = {}


class ProjectorDependencyError(RuntimeError):
    pass


@dataclass
class ProjectorConfig:
    projection_method: str
    metric: str
    neighbors_k: int
    point_size: float


def _hash_cache_key(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> Optional[dict[str, Any]]:
    now = time.time()
    with _projector_cache_lock:
        item = _projector_cache.get(key)
        if not item:
            return None
        expires_at, payload = item
        if expires_at < now:
            _projector_cache.pop(key, None)
            return None
        return copy.deepcopy(payload)


def _cache_set(key: str, payload: dict[str, Any]) -> None:
    expires_at = time.time() + max(1, PROJECTOR_CACHE_TTL_SECONDS)
    with _projector_cache_lock:
        _projector_cache[key] = (expires_at, copy.deepcopy(payload))
        if len(_projector_cache) <= max(1, PROJECTOR_CACHE_MAX_ITEMS):
            return
        # Remove the oldest entry first.
        oldest_key = min(_projector_cache.items(), key=lambda item: item[1][0])[0]
        _projector_cache.pop(oldest_key, None)


def _normalize_inputs(raw_inputs: Any) -> list[str]:
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise InputValidationError("`inputs` must be a non-empty array of strings.")
    normalized: list[str] = []
    for index, value in enumerate(raw_inputs):
        if not isinstance(value, str) or not value.strip():
            raise InputValidationError(f"`inputs[{index}]` must be a non-empty string.")
        normalized.append(value)
    return normalized


def _normalize_labels(raw_labels: Any, size: int) -> list[str]:
    if raw_labels is None:
        return ["" for _ in range(size)]
    if not isinstance(raw_labels, list):
        raise InputValidationError("`labels` must be an array of strings.")
    if len(raw_labels) != size:
        raise InputValidationError("`labels` must have the same length as `inputs`.")

    normalized: list[str] = []
    for index, value in enumerate(raw_labels):
        if value is None:
            normalized.append("")
            continue
        if not isinstance(value, str):
            raise InputValidationError(f"`labels[{index}]` must be a string.")
        normalized.append(value)
    return normalized


def _normalize_projector_config(payload: dict[str, Any]) -> ProjectorConfig:
    projection_method = str(payload.get("projection_method") or "umap").strip().lower()
    metric = str(payload.get("metric") or "cosine").strip().lower()
    neighbors_k = int(payload.get("neighbors_k") or 10)
    point_size = float(payload.get("point_size") or 3.5)

    if projection_method not in ("umap", "tsne", "pca"):
        raise InputValidationError("`projection_method` must be `umap`, `tsne`, or `pca`.")
    if metric not in ("cosine", "euclidean"):
        raise InputValidationError("`metric` must be `cosine` or `euclidean`.")
    if neighbors_k < 1 or neighbors_k > 256:
        raise InputValidationError("`neighbors_k` must be between 1 and 256.")
    if point_size <= 0 or point_size > 64:
        raise InputValidationError("`point_size` must be in (0, 64].")

    return ProjectorConfig(
        projection_method=projection_method,
        metric=metric,
        neighbors_k=neighbors_k,
        point_size=point_size,
    )


def _extract_vectors(embedding_payload: dict[str, Any], expected_size: int) -> list[list[float]]:
    rows = embedding_payload.get("data")
    if not isinstance(rows, list):
        raise ProjectorDependencyError("Embedding payload has no valid `data` array.")
    if len(rows) != expected_size:
        raise ProjectorDependencyError(
            f"Embedding payload row count mismatch: expected {expected_size}, got {len(rows)}."
        )

    vectors: list[list[float]] = []
    for index, row in enumerate(rows):
        embedding = row.get("embedding") if isinstance(row, dict) else None
        if not isinstance(embedding, list) or not embedding:
            raise ProjectorDependencyError(f"Embedding row {index} is missing a non-empty `embedding` array.")
        values: list[float] = []
        for value in embedding:
            try:
                values.append(float(value))
            except (TypeError, ValueError) as exc:
                raise ProjectorDependencyError(f"Embedding row {index} contains non-numeric values.") from exc
        vectors.append(values)
    return vectors


def _project_by_pca(vectors: list[list[float]]) -> list[tuple[float, float]]:
    if len(vectors) <= 1:
        return [(0.0, 0.0) for _ in vectors]

    try:
        import numpy as np
    except ImportError:
        # Fallback when numpy is absent in local dev: project by first two dimensions.
        projected: list[tuple[float, float]] = []
        for row in vectors:
            projected.append((float(row[0]), float(row[1] if len(row) > 1 else 0.0)))
        return projected

    matrix = np.asarray(vectors, dtype=float)
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    if matrix.shape[1] == 1:
        column = matrix[:, 0]
        return [(float(value), 0.0) for value in column]

    _, _, v_t = np.linalg.svd(matrix, full_matrices=False)
    basis = v_t[:2].T
    projected = matrix @ basis
    return [(float(row[0]), float(row[1])) for row in projected]


def _project_by_tsne(vectors: list[list[float]], metric: str) -> list[tuple[float, float]]:
    if len(vectors) <= 2:
        return _project_by_pca(vectors)

    try:
        import numpy as np
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ProjectorDependencyError(
            "t-SNE projection requires `numpy` and `scikit-learn`. "
            "Please install dependencies from requirements.txt."
        ) from exc

    matrix = np.asarray(vectors, dtype=float)
    perplexity = max(2.0, min(30.0, float(len(vectors) - 1)))
    reducer = TSNE(
        n_components=2,
        metric=metric,
        perplexity=perplexity,
        learning_rate="auto",
        init="random",
        random_state=42,
    )
    projected = reducer.fit_transform(matrix)
    return [(float(row[0]), float(row[1])) for row in projected]


def _project_by_umap(vectors: list[list[float]], metric: str, neighbors_k: int) -> list[tuple[float, float]]:
    if len(vectors) <= 2:
        return _project_by_pca(vectors)

    try:
        import numpy as np
        import umap
    except ImportError as exc:
        raise ProjectorDependencyError(
            "UMAP projection requires `numpy` and `umap-learn`. "
            "Please install dependencies from requirements.txt."
        ) from exc

    matrix = np.asarray(vectors, dtype=float)
    n_neighbors = min(max(2, neighbors_k), max(2, len(vectors) - 1))
    reducer = umap.UMAP(n_components=2, metric=metric, n_neighbors=n_neighbors, random_state=42)
    projected = reducer.fit_transform(matrix)
    return [(float(row[0]), float(row[1])) for row in projected]


def _project_vectors(vectors: list[list[float]], config: ProjectorConfig) -> list[tuple[float, float]]:
    if config.projection_method == "pca":
        return _project_by_pca(vectors)
    if config.projection_method == "tsne":
        return _project_by_tsne(vectors, config.metric)
    return _project_by_umap(vectors, config.metric, config.neighbors_k)


def _normalize_coordinates(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return []

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)

    normalized: list[tuple[float, float]] = []
    for x, y in points:
        nx = ((x - min_x) / span_x) * 2.0 - 1.0
        ny = ((y - min_y) / span_y) * 2.0 - 1.0
        normalized.append((nx, ny))
    return normalized


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for va, vb in zip(a, b):
        dot += va * vb
        norm_a += va * va
        norm_b += vb * vb
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 1.0
    return 1.0 - dot / (sqrt(norm_a) * sqrt(norm_b))


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    total = 0.0
    for va, vb in zip(a, b):
        diff = va - vb
        total += diff * diff
    return sqrt(total)


def _compute_neighbors(vectors: list[list[float]], metric: str, neighbors_k: int) -> dict[str, list[dict[str, float | int]]]:
    count = len(vectors)
    if count <= 1:
        return {"0": []} if count == 1 else {}

    k = min(neighbors_k, count - 1)
    neighbors: dict[str, list[dict[str, float | int]]] = {}
    distance_fn = _cosine_distance if metric == "cosine" else _euclidean_distance

    for i in range(count):
        distances: list[tuple[int, float]] = []
        for j in range(count):
            if i == j:
                continue
            distances.append((j, distance_fn(vectors[i], vectors[j])))
        distances.sort(key=lambda item: item[1])
        selected = distances[:k]
        payload: list[dict[str, float | int]] = []
        for index, distance in selected:
            similarity = max(-1.0, min(1.0, 1.0 - distance)) if metric == "cosine" else 1.0 / (1.0 + distance)
            payload.append(
                {
                    "index": index,
                    "distance": round(float(distance), 8),
                    "similarity": round(float(similarity), 8),
                }
            )
        neighbors[str(i)] = payload
    return neighbors


def _build_embedding_payload(request_payload: dict[str, Any], inputs: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input": inputs if len(inputs) > 1 else inputs[0],
    }
    if request_payload.get("model"):
        payload["model"] = request_payload["model"]
    if request_payload.get("input_type"):
        payload["input_type"] = request_payload["input_type"]
    if request_payload.get("instruction"):
        payload["instruction"] = request_payload["instruction"]
    return payload


async def create_projector_payload(
    request_payload: dict[str, Any],
    embedder: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] = create_embeddings,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    inputs = _normalize_inputs(request_payload.get("inputs"))
    labels = _normalize_labels(request_payload.get("labels"), len(inputs))
    config = _normalize_projector_config(request_payload)

    cache_material = {
        "inputs": inputs,
        "labels": labels,
        "model": request_payload.get("model"),
        "input_type": request_payload.get("input_type"),
        "instruction": request_payload.get("instruction"),
        "projection_method": config.projection_method,
        "metric": config.metric,
        "neighbors_k": config.neighbors_k,
        "point_size": config.point_size,
    }
    cache_key = _hash_cache_key(cache_material)
    cached = _cache_get(cache_key)
    if cached is not None:
        cached["projection_meta"]["cache_hit"] = True
        return cached

    embedding_payload = _build_embedding_payload(request_payload, inputs)
    embedding_response = await embedder(embedding_payload)
    vectors = _extract_vectors(embedding_response, len(inputs))
    coordinates = _project_vectors(vectors, config)
    normalized_coordinates = _normalize_coordinates(coordinates)
    neighbors = _compute_neighbors(vectors, config.metric, config.neighbors_k)

    points: list[dict[str, Any]] = []
    for index, (raw_xy, normalized_xy) in enumerate(zip(coordinates, normalized_coordinates)):
        points.append(
            {
                "id": str(index),
                "index": index,
                "text": inputs[index],
                "label": labels[index],
                "x": round(float(raw_xy[0]), 8),
                "y": round(float(raw_xy[1]), 8),
                "normalized_x": round(float(normalized_xy[0]), 8),
                "normalized_y": round(float(normalized_xy[1]), 8),
            }
        )

    duration_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
    dimensions = len(vectors[0]) if vectors else 0
    response = {
        "object": "projector",
        "model": embedding_response.get("model") or request_payload.get("model"),
        "usage": embedding_response.get("usage"),
        "points": points,
        "neighbors": neighbors,
        "projection_meta": {
            "projection_method": config.projection_method,
            "metric": config.metric,
            "neighbors_k": min(config.neighbors_k, max(0, len(points) - 1)),
            "point_size": config.point_size,
            "count": len(points),
            "dimensions": dimensions,
            "cache_hit": False,
            "duration_ms": duration_ms,
        },
    }
    _cache_set(cache_key, response)
    return response
