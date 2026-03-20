from fastapi.testclient import TestClient

import app
import embedding_service


def test_embeddings_route_returns_openai_shape(monkeypatch):
    async def fake_create_embeddings(payload):
        assert payload["input_type"] == "query"
        return {
            "object": "list",
            "model": "Qwen/Qwen3-Embedding-8B",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
            ],
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }

    monkeypatch.setattr(app, "create_embeddings", fake_create_embeddings)

    with TestClient(app.create_application()) as client:
        response = client.post(
            "/v1/embeddings",
            json={
                "input": "What is the capital of China?",
                "input_type": "query",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"][0]["object"] == "embedding"
    assert payload["data"][0]["embedding"] == [0.1, 0.2, 0.3]


def test_health_route_reflects_backend_status(monkeypatch):
    async def fake_health():
        return {
            "status": "ok",
            "backend_ready": True,
            "model_id": "Qwen/Qwen3-Embedding-8B",
        }

    monkeypatch.setattr(app, "get_health_payload", fake_health)

    with TestClient(app.create_application()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["backend_ready"] is True


def test_admin_reload_requires_token(monkeypatch):
    monkeypatch.setattr(app, "ADMIN_TOKEN", "secret")

    with TestClient(app.create_application()) as client:
        response = client.post("/admin/reload", json={})

    assert response.status_code == 401


def test_admin_reload_success(monkeypatch):
    async def fake_reload(payload):
        assert payload["model_id"] == "Qwen/Qwen3-Embedding-8B"
        return {"status": "ok", "backend_ready": True}

    monkeypatch.setattr(app, "ADMIN_TOKEN", "secret")
    monkeypatch.setattr(app, "reload_backend", fake_reload)

    with TestClient(app.create_application()) as client:
        response = client.post(
            "/admin/reload",
            headers={"x-admin-token": "secret"},
            json={"model_id": "Qwen/Qwen3-Embedding-8B"},
        )

    assert response.status_code == 200
    assert response.json()["backend_ready"] is True


def test_backend_proxy_error_is_passthrough(monkeypatch):
    async def fake_create_embeddings(payload):
        raise embedding_service.BackendProxyError(
            "bad request",
            status_code=400,
            payload={
                "error": {
                    "message": "dimensions not supported",
                    "type": "BadRequestError",
                    "param": None,
                    "code": 400,
                }
            },
        )

    monkeypatch.setattr(app, "create_embeddings", fake_create_embeddings)

    with TestClient(app.create_application()) as client:
        response = client.post("/v1/embeddings", json={"input": "hello"})

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "dimensions not supported"
