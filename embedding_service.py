import asyncio
import os
import shlex
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Optional

import httpx

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-Embedding-8B")
MODEL_REVISION = os.getenv("MODEL_REVISION")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "12302"))
VLLM_HOST = os.getenv("VLLM_HOST", "127.0.0.1")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8001"))
HF_HOME = os.getenv("HF_HOME", "/models")
DTYPE = os.getenv("DTYPE", "float16")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "32768"))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.92"))
DEFAULT_QUERY_INSTRUCTION = os.getenv(
    "DEFAULT_QUERY_INSTRUCTION",
    "Given a web search query, retrieve relevant passages that answer the query",
)
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
BACKEND_START_TIMEOUT = int(os.getenv("BACKEND_START_TIMEOUT", "600"))
BACKEND_HTTP_TIMEOUT = float(os.getenv("BACKEND_HTTP_TIMEOUT", "120"))
BACKEND_POLL_INTERVAL = float(os.getenv("BACKEND_POLL_INTERVAL", "1.0"))
REQUEST_READY_TIMEOUT = float(os.getenv("REQUEST_READY_TIMEOUT", "3.0"))
BACKEND_PORT_SCAN_WINDOW = int(os.getenv("BACKEND_PORT_SCAN_WINDOW", "4"))
VLLM_EXTRA_ARGS = os.getenv("VLLM_EXTRA_ARGS", "")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "").strip()
MANAGE_BACKEND_PROCESS = (
    os.getenv("MANAGE_BACKEND_PROCESS", "0" if VLLM_BASE_URL else "1").strip().lower()
    not in ("0", "false", "no", "off")
)
PRELOAD_MODEL = os.getenv("PRELOAD_MODEL", "1").strip().lower() not in ("0", "false", "no", "off")

_DEFAULT_BACKEND_PATH = f"http://{VLLM_HOST}:{VLLM_PORT}"
_backend_lock = threading.RLock()
_backend_process: Optional[subprocess.Popen[Any]] = None
_backend_started_at: Optional[float] = None
_backend_ready = False
_backend_last_error = ""


class InputValidationError(ValueError):
    pass


class BackendUnavailableError(RuntimeError):
    pass


class BackendProxyError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502, payload: Optional[dict[str, Any]] = None):
        self.status_code = status_code
        self.payload = payload
        super().__init__(message)


@dataclass
class BackendSettings:
    model_id: str = MODEL_ID
    model_revision: Optional[str] = MODEL_REVISION
    backend_host: str = VLLM_HOST
    backend_port: int = VLLM_PORT
    dtype: str = DTYPE
    max_model_len: int = MAX_MODEL_LEN
    gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION
    default_query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    hf_home: str = HF_HOME
    manage_backend_process: bool = MANAGE_BACKEND_PROCESS
    preload_model: bool = PRELOAD_MODEL
    extra_args: str = VLLM_EXTRA_ARGS
    backend_base_url: str = VLLM_BASE_URL or _DEFAULT_BACKEND_PATH


_settings = BackendSettings()


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off")


def _apply_proxy_env(target_env: Optional[dict[str, str]] = None) -> dict[str, str]:
    env = target_env if target_env is not None else os.environ
    http_proxy = (env.get("HTTP_PROXY") or os.getenv("HTTP_PROXY") or "").strip()
    https_proxy = (env.get("HTTPS_PROXY") or os.getenv("HTTPS_PROXY") or "").strip()
    no_proxy = (env.get("NO_PROXY") or os.getenv("NO_PROXY") or "").strip()

    if http_proxy:
        env["HTTP_PROXY"] = http_proxy
        env["http_proxy"] = http_proxy
        if not https_proxy:
            https_proxy = http_proxy
        env["HTTPS_PROXY"] = https_proxy
        env["https_proxy"] = https_proxy

    env["NO_PROXY"] = no_proxy or "localhost,127.0.0.1"
    env["no_proxy"] = env["NO_PROXY"]
    return env


def _build_backend_env() -> dict[str, str]:
    env = _apply_proxy_env(dict(os.environ))
    env["HF_HOME"] = _settings.hf_home
    env.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    return env


def _build_vllm_command() -> list[str]:
    command = [
        "vllm",
        "serve",
        _settings.model_id,
        "--host",
        _settings.backend_host,
        "--port",
        str(_settings.backend_port),
        "--task",
        "embed",
        "--dtype",
        _settings.dtype,
        "--max-model-len",
        str(_settings.max_model_len),
        "--gpu-memory-utilization",
        str(_settings.gpu_memory_utilization),
        "--served-model-name",
        _settings.model_id,
    ]
    if _settings.model_revision:
        command.extend(["--revision", _settings.model_revision])
    if _env_flag("TRUST_REMOTE_CODE", "0"):
        command.append("--trust-remote-code")
    if _settings.extra_args.strip():
        command.extend(shlex.split(_settings.extra_args))
    return command


def _backend_alive() -> bool:
    return _backend_process is not None and _backend_process.poll() is None


def _mark_backend_not_ready(message: str = "") -> None:
    global _backend_ready, _backend_last_error
    _backend_ready = False
    if message:
        _backend_last_error = message


def _backend_state(healthy: bool, process_alive: bool, exit_code: Optional[int]) -> str:
    if healthy:
        return "ready"
    if process_alive:
        return "starting"
    if exit_code is not None:
        return "exited"
    return "stopped"


def _backend_message(
    healthy: bool,
    process_alive: bool,
    exit_code: Optional[int],
    probe_message: str,
) -> str:
    if healthy:
        return ""
    if process_alive:
        return (
            "vLLM 进程已启动，但健康检查尚未就绪。"
            "这通常表示模型仍在加载权重到 GPU，或内部 engine 仍在初始化。"
        )
    if exit_code is not None:
        return _backend_last_error or f"vLLM exited with code {exit_code}."
    return _backend_last_error or probe_message or "Backend is not reachable."


def _start_backend_process_locked() -> None:
    global _backend_process, _backend_started_at, _backend_last_error, _backend_ready

    if not _settings.manage_backend_process:
        return

    if _backend_alive():
        return

    command = _build_vllm_command()
    try:
        _backend_process = subprocess.Popen(
            command,
            env=_build_backend_env(),
            stdin=subprocess.DEVNULL,
            stdout=None,
            stderr=None,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        _backend_process = None
        _backend_ready = False
        _backend_last_error = "Failed to start vLLM: `vllm` command not found."
        raise BackendUnavailableError(_backend_last_error) from exc

    _backend_started_at = time.time()
    _backend_ready = False
    _backend_last_error = ""


def _stop_backend_process_locked() -> None:
    global _backend_process, _backend_ready

    proc = _backend_process
    _backend_process = None
    _backend_ready = False

    if proc is None:
        return

    if proc.poll() is not None:
        return

    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


async def _probe_backend_health() -> tuple[bool, Optional[dict[str, Any]], str]:
    timeout = httpx.Timeout(5.0, connect=2.0)
    ports_to_try = [_settings.backend_port]
    for offset in range(1, max(1, BACKEND_PORT_SCAN_WINDOW) + 1):
        ports_to_try.append(_settings.backend_port + offset)

    last_message = ""
    async with httpx.AsyncClient(timeout=timeout) as client:
        for port in ports_to_try:
            base_url = f"http://{_settings.backend_host}:{port}"
            try:
                response = await client.get(f"{base_url}/health")
            except httpx.HTTPError as exc:
                last_message = str(exc)
                continue

            if response.status_code >= 400:
                last_message = f"vLLM health returned HTTP {response.status_code}"
                continue

            payload: Optional[dict[str, Any]]
            try:
                payload = response.json()
            except ValueError:
                payload = None

            _settings.backend_base_url = base_url
            return True, payload, ""

    return False, None, last_message or "All connection attempts failed"


async def wait_for_backend_ready(timeout_s: Optional[float] = None) -> None:
    global _backend_ready, _backend_last_error

    if not _settings.manage_backend_process:
        healthy, _, message = await _probe_backend_health()
        _backend_ready = healthy
        if not healthy:
            _backend_last_error = message or "Backend is not reachable."
            raise BackendUnavailableError(_backend_last_error)
        return

    deadline = time.monotonic() + float(timeout_s or BACKEND_START_TIMEOUT)
    while time.monotonic() < deadline:
        with _backend_lock:
            proc = _backend_process
            if proc is None:
                _mark_backend_not_ready("Backend process is not running.")
                raise BackendUnavailableError(_backend_last_error)
            if proc.poll() is not None:
                _mark_backend_not_ready(f"vLLM exited with code {proc.returncode}.")
                raise BackendUnavailableError(_backend_last_error)

        healthy, _, message = await _probe_backend_health()
        if healthy:
            _backend_ready = True
            _backend_last_error = ""
            return

        _backend_ready = False
        if message:
            _backend_last_error = message
        await asyncio.sleep(BACKEND_POLL_INTERVAL)

    _backend_ready = False
    _backend_last_error = f"Timed out after {int(timeout_s or BACKEND_START_TIMEOUT)}s waiting for vLLM."
    raise BackendUnavailableError(_backend_last_error)


async def ensure_backend_started(wait_ready: bool = True, timeout_s: Optional[float] = None) -> None:
    with _backend_lock:
        _start_backend_process_locked()
    if wait_ready:
        await wait_for_backend_ready(timeout_s=timeout_s)


async def maybe_preload_backend() -> None:
    if not _settings.preload_model:
        return
    try:
        await ensure_backend_started(wait_ready=True)
    except Exception as exc:
        _mark_backend_not_ready(str(exc))


async def shutdown_backend() -> None:
    with _backend_lock:
        _stop_backend_process_locked()


def get_current_model_id() -> str:
    return _settings.model_id


def get_current_settings() -> dict[str, Any]:
    snapshot = asdict(_settings)
    snapshot["public_host"] = HOST
    snapshot["public_port"] = PORT
    return snapshot


def _normalize_text_list(raw_input: Any) -> tuple[list[str], bool]:
    if isinstance(raw_input, str):
        if not raw_input.strip():
            raise InputValidationError("`input` must not be empty.")
        return [raw_input], False

    if isinstance(raw_input, list):
        if not raw_input:
            raise InputValidationError("`input` array must not be empty.")
        normalized: list[str] = []
        for index, item in enumerate(raw_input):
            if not isinstance(item, str):
                raise InputValidationError(f"`input[{index}]` must be a string.")
            if not item.strip():
                raise InputValidationError(f"`input[{index}]` must not be empty.")
            normalized.append(item)
        return normalized, True

    raise InputValidationError("`input` must be a string or an array of strings.")


def _validate_dimensions(dimensions: Optional[int]) -> Optional[int]:
    if dimensions is None:
        return None
    if not isinstance(dimensions, int):
        raise InputValidationError("`dimensions` must be an integer.")
    if dimensions < 32 or dimensions > 4096:
        raise InputValidationError("`dimensions` must be between 32 and 4096.")
    return dimensions


def _validate_input_type(input_type: Optional[str]) -> Optional[str]:
    if input_type is None:
        return None
    if input_type not in ("query", "document"):
        raise InputValidationError("`input_type` must be `query` or `document`.")
    return input_type


def format_query_text(text: str, instruction: str) -> str:
    cleaned_instruction = instruction.strip()
    return f"Instruct: {cleaned_instruction}\nQuery:{text}"


def prepare_backend_payload(request_payload: dict[str, Any]) -> dict[str, Any]:
    texts, input_was_list = _normalize_text_list(request_payload.get("input"))
    input_type = _validate_input_type(request_payload.get("input_type"))
    dimensions = _validate_dimensions(request_payload.get("dimensions"))
    instruction = (request_payload.get("instruction") or "").strip() or _settings.default_query_instruction

    prepared_texts = texts
    if input_type == "query":
        prepared_texts = [format_query_text(text, instruction) for text in texts]

    backend_input: str | list[str]
    if input_was_list:
        backend_input = prepared_texts
    else:
        backend_input = prepared_texts[0]

    payload: dict[str, Any] = {
        "input": backend_input,
        "model": request_payload.get("model") or _settings.model_id,
    }
    if dimensions is not None:
        payload["dimensions"] = dimensions
    if request_payload.get("encoding_format") is not None:
        payload["encoding_format"] = request_payload["encoding_format"]
    if request_payload.get("user") is not None:
        payload["user"] = request_payload["user"]
    return payload


async def _post_embeddings(payload: dict[str, Any]) -> dict[str, Any]:
    timeout = httpx.Timeout(BACKEND_HTTP_TIMEOUT, connect=10.0)
    base_url = _settings.backend_base_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{base_url}/v1/embeddings", json=payload)
    except httpx.HTTPError as exc:
        _mark_backend_not_ready(str(exc))
        raise BackendUnavailableError(f"Failed to reach vLLM backend: {exc}") from exc

    if response.status_code >= 400:
        try:
            payload = response.json()
        except ValueError:
            payload = {
                "error": {
                    "message": response.text or f"Backend returned HTTP {response.status_code}",
                    "type": "backend_error",
                    "param": None,
                    "code": response.status_code,
                }
            }
        raise BackendProxyError(
            f"Backend returned HTTP {response.status_code}",
            status_code=response.status_code,
            payload=payload,
        )

    try:
        return response.json()
    except ValueError as exc:
        raise BackendProxyError("Backend returned non-JSON response.", status_code=502) from exc


async def create_embeddings(request_payload: dict[str, Any]) -> dict[str, Any]:
    global _backend_ready
    backend_payload = prepare_backend_payload(request_payload)
    try:
        await ensure_backend_started(wait_ready=True, timeout_s=REQUEST_READY_TIMEOUT)
    except BackendUnavailableError as exc:
        if _backend_alive():
            raise BackendUnavailableError(
                "Backend is still starting and may still be loading model weights on GPU. "
                "Please wait a bit and refresh /health."
            ) from exc
        raise
    response_payload = await _post_embeddings(backend_payload)
    _backend_ready = True
    return response_payload


async def embed_texts(
    texts: str | list[str],
    input_type: Optional[str] = None,
    instruction: Optional[str] = None,
    dimensions: Optional[int] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"input": texts}
    if input_type is not None:
        payload["input_type"] = input_type
    if instruction is not None:
        payload["instruction"] = instruction
    if dimensions is not None:
        payload["dimensions"] = dimensions
    return await create_embeddings(payload)


async def get_health_payload() -> dict[str, Any]:
    healthy, backend_health, message = await _probe_backend_health()
    if healthy:
        global _backend_ready, _backend_last_error
        _backend_ready = True
        _backend_last_error = ""
    elif not _backend_alive() and _settings.manage_backend_process:
        _backend_ready = False
        if message:
            _backend_last_error = message

    backend_pid: Optional[int] = None
    backend_exit_code: Optional[int] = None
    backend_process_alive = False
    with _backend_lock:
        if _backend_process is not None:
            backend_pid = _backend_process.pid
            backend_exit_code = _backend_process.poll()
            backend_process_alive = backend_exit_code is None

    backend_state = _backend_state(healthy, backend_process_alive, backend_exit_code)
    backend_message = _backend_message(healthy, backend_process_alive, backend_exit_code, message)

    return {
        "status": "ok" if healthy else "degraded",
        "backend": "vllm",
        "backend_ready": healthy,
        "backend_state": backend_state,
        "backend_process_alive": backend_process_alive,
        "backend_url": _settings.backend_base_url,
        "backend_pid": backend_pid,
        "backend_exit_code": backend_exit_code,
        "backend_last_error": backend_message,
        "backend_health": backend_health,
        "model_id": _settings.model_id,
        "model_revision": _settings.model_revision,
        "port": PORT,
        "backend_port": _settings.backend_port,
        "dtype": _settings.dtype,
        "backend_target_device": "cuda",
        "cpu_fallback": False,
        "max_model_len": _settings.max_model_len,
        "gpu_memory_utilization": _settings.gpu_memory_utilization,
        "default_query_instruction": _settings.default_query_instruction,
        "manage_backend_process": _settings.manage_backend_process,
        "preload_model": _settings.preload_model,
        "started_at": _backend_started_at,
        "server_time": datetime.now().astimezone().isoformat(),
        "timezone": datetime.now().astimezone().tzname(),
    }


def get_health_snapshot() -> dict[str, Any]:
    backend_pid: Optional[int] = None
    backend_exit_code: Optional[int] = None
    backend_process_alive = False
    with _backend_lock:
        if _backend_process is not None:
            backend_pid = _backend_process.pid
            backend_exit_code = _backend_process.poll()
            backend_process_alive = backend_exit_code is None

    backend_state = _backend_state(_backend_ready, backend_process_alive, backend_exit_code)
    backend_message = _backend_message(_backend_ready, backend_process_alive, backend_exit_code, "")

    return {
        "status": "ok" if _backend_ready else "degraded",
        "backend": "vllm",
        "backend_ready": _backend_ready,
        "backend_state": backend_state,
        "backend_process_alive": backend_process_alive,
        "backend_url": _settings.backend_base_url,
        "backend_pid": backend_pid,
        "backend_exit_code": backend_exit_code,
        "backend_last_error": backend_message,
        "model_id": _settings.model_id,
        "model_revision": _settings.model_revision,
        "port": PORT,
        "backend_port": _settings.backend_port,
        "dtype": _settings.dtype,
        "backend_target_device": "cuda",
        "cpu_fallback": False,
        "max_model_len": _settings.max_model_len,
        "gpu_memory_utilization": _settings.gpu_memory_utilization,
        "default_query_instruction": _settings.default_query_instruction,
        "manage_backend_process": _settings.manage_backend_process,
        "preload_model": _settings.preload_model,
        "started_at": _backend_started_at,
        "server_time": datetime.now().astimezone().isoformat(),
        "timezone": datetime.now().astimezone().tzname(),
    }


async def reload_backend(new_config: dict[str, Any]) -> dict[str, Any]:
    allowed_fields = {
        "model_id",
        "model_revision",
        "dtype",
        "max_model_len",
        "gpu_memory_utilization",
        "default_query_instruction",
        "extra_args",
    }
    unknown = set(new_config) - allowed_fields
    if unknown:
        raise InputValidationError(f"Unsupported reload fields: {', '.join(sorted(unknown))}")

    with _backend_lock:
        _stop_backend_process_locked()

        if "model_id" in new_config and new_config["model_id"]:
            _settings.model_id = str(new_config["model_id"])
        if "model_revision" in new_config:
            _settings.model_revision = new_config["model_revision"] or None
        if "dtype" in new_config and new_config["dtype"]:
            _settings.dtype = str(new_config["dtype"])
        if "max_model_len" in new_config and new_config["max_model_len"] is not None:
            _settings.max_model_len = int(new_config["max_model_len"])
        if "gpu_memory_utilization" in new_config and new_config["gpu_memory_utilization"] is not None:
            _settings.gpu_memory_utilization = float(new_config["gpu_memory_utilization"])
        if "default_query_instruction" in new_config and new_config["default_query_instruction"]:
            _settings.default_query_instruction = str(new_config["default_query_instruction"])
        if "extra_args" in new_config:
            _settings.extra_args = str(new_config["extra_args"] or "")

        _settings.backend_base_url = VLLM_BASE_URL or f"http://{_settings.backend_host}:{_settings.backend_port}"
        _start_backend_process_locked()

    await wait_for_backend_ready()
    return await get_health_payload()
