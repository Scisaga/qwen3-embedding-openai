import os

import uvicorn

from app import app
from embedding_service import PRELOAD_MODEL, _apply_proxy_env


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "12302"))
    _apply_proxy_env()

    if PRELOAD_MODEL:
        print("[startup] PRELOAD_MODEL=1; app startup will wait for vLLM backend readiness.")
    else:
        print("[startup] PRELOAD_MODEL=0; backend will be started on first request.")

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
