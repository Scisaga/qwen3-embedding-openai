import contextlib
import json
import os
from typing import Literal, Optional

from fastapi import FastAPI, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from embedding_service import (
    ADMIN_TOKEN,
    DEFAULT_QUERY_INSTRUCTION,
    BackendProxyError,
    BackendUnavailableError,
    InputValidationError,
    create_embeddings,
    get_current_model_id,
    get_health_payload,
    maybe_preload_backend,
    reload_backend,
    shutdown_backend,
)
from mcp_server import create_mcp_server


class EmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str | list[str]
    model: Optional[str] = None
    dimensions: Optional[int] = Field(default=None, ge=32, le=4096)
    encoding_format: Optional[str] = None
    user: Optional[str] = None
    input_type: Optional[Literal["query", "document"]] = None
    instruction: Optional[str] = None


class ReloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: Optional[str] = None
    model_revision: Optional[str] = None
    dtype: Optional[str] = None
    max_model_len: Optional[int] = Field(default=None, ge=1)
    gpu_memory_utilization: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    default_query_instruction: Optional[str] = None
    extra_args: Optional[str] = None


def _error_response(message: str, status_code: int, error_type: str, code: Optional[int | str] = None) -> JSONResponse:
    payload = {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        }
    }
    return JSONResponse(payload, status_code=status_code)


def _authorize_admin(token: Optional[str]) -> None:
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        raise PermissionError("Invalid x-admin-token.")


def _build_index_html() -> str:
    template = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <meta name="theme-color" content="#0a1020"/>
  <link rel="icon" type="image/svg+xml" href="/static/favicon.svg"/>
  <link rel="apple-touch-icon" href="/static/logo.svg"/>
  <title>Qwen3 Embedding</title>
  <style>
    :root{
      --bg:#09111f;
      --bg2:#111d33;
      --panel:rgba(8, 15, 29, .78);
      --panel-2:rgba(14, 26, 47, .92);
      --border:rgba(148,163,184,.18);
      --text:#e2e8f0;
      --muted:#94a3b8;
      --accent:#ff7a18;
      --accent-2:#ffd166;
      --good:#22c55e;
      --danger:#ef4444;
      --shadow:0 28px 80px rgba(0,0,0,.45);
      --radius:20px;
      --radius-sm:14px;
    }
    *{box-sizing:border-box}
    html,body{margin:0;min-height:100%;font-family:"IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;background:
      radial-gradient(900px 520px at 10% -10%, rgba(255,122,24,.18), transparent 60%),
      radial-gradient(800px 420px at 100% 10%, rgba(255,209,102,.18), transparent 58%),
      linear-gradient(180deg, var(--bg), var(--bg2));color:var(--text)}
    body{padding:28px 16px 64px}
    a{color:inherit}
    code{font-family:ui-monospace, SFMono-Regular, Menlo, monospace;background:rgba(15,23,42,.58);padding:2px 8px;border-radius:999px;border:1px solid var(--border)}
    .shell{max-width:1200px;margin:0 auto}
    .hero{display:grid;grid-template-columns:1.1fr .9fr;gap:18px}
    .card{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow);backdrop-filter:blur(12px)}
    .hero-main{padding:28px}
    .badge{display:inline-flex;align-items:center;gap:8px;padding:8px 12px;border-radius:999px;background:rgba(255,122,24,.1);border:1px solid rgba(255,122,24,.18);font-size:12px;color:#fed7aa}
    .dot{width:8px;height:8px;border-radius:50%;background:var(--accent);box-shadow:0 0 0 6px rgba(255,122,24,.14)}
    h1{margin:18px 0 10px;font-size:40px;line-height:1.05;letter-spacing:-.03em}
    .hero-main p{margin:0;color:var(--muted);line-height:1.7}
    .hero-side{padding:24px;display:flex;flex-direction:column;gap:14px}
    .kv{display:flex;justify-content:space-between;gap:16px;padding:12px 0;border-bottom:1px solid rgba(148,163,184,.08)}
    .kv:last-child{border-bottom:none}
    .k{font-size:13px;color:var(--muted)}
    .v{font-size:13px;text-align:right}
    .grid{display:grid;grid-template-columns:1.06fr .94fr;gap:18px;margin-top:18px}
    .section{padding:24px}
    .section h2{margin:0 0 14px;font-size:20px}
    label{display:block;font-size:13px;color:#cbd5e1;margin:0 0 8px}
    textarea,input,select{width:100%;background:rgba(15,23,42,.64);border:1px solid var(--border);border-radius:14px;color:var(--text);padding:12px 14px;font:inherit}
    textarea{min-height:220px;resize:vertical}
    .row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    .row-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
    .hint{margin-top:8px;font-size:12px;color:var(--muted)}
    .actions{display:flex;gap:12px;margin-top:18px;flex-wrap:wrap}
    button{appearance:none;border:none;border-radius:999px;padding:12px 18px;font:600 14px/1 inherit;cursor:pointer}
    .primary{background:linear-gradient(135deg,var(--accent),var(--accent-2));color:#111827}
    .secondary{background:rgba(15,23,42,.64);color:var(--text);border:1px solid var(--border)}
    .status{display:flex;align-items:center;gap:10px;padding:14px;border-radius:16px;background:rgba(15,23,42,.45);border:1px solid var(--border);font-size:13px}
    .status .light{width:10px;height:10px;border-radius:50%;background:var(--muted)}
    .status.ok .light{background:var(--good);box-shadow:0 0 0 6px rgba(34,197,94,.16)}
    .status.bad .light{background:var(--danger);box-shadow:0 0 0 6px rgba(239,68,68,.16)}
    .metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:14px}
    .metric{padding:14px;border-radius:16px;background:rgba(15,23,42,.52);border:1px solid var(--border)}
    .metric .n{font-size:24px;font-weight:700}
    .metric .l{font-size:12px;color:var(--muted)}
    pre{margin:0;white-space:pre-wrap;word-break:break-word;background:rgba(2,6,23,.72);border:1px solid var(--border);padding:14px;border-radius:14px;max-height:480px;overflow:auto}
    .small{font-size:12px;color:var(--muted)}
    .list{display:flex;flex-direction:column;gap:10px}
    .chip{display:inline-flex;padding:6px 10px;border-radius:999px;background:rgba(255,255,255,.06);border:1px solid var(--border);font-size:12px;color:#dbeafe}
    @media (max-width: 980px){
      .hero,.grid,.row,.row-3{grid-template-columns:1fr}
      h1{font-size:32px}
      .metrics{grid-template-columns:1fr}
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="hero">
      <section class="card hero-main">
        <div class="badge"><span class="dot"></span> Self-hosted OpenAI-compatible Embeddings</div>
        <h1>Qwen3 Embedding<br/>Wrapper Service</h1>
        <p>对外暴露 <code>/v1/embeddings</code>、<code>/mcp</code>、<code>/health</code> 和内置调试页面。底层推理由同容器内的 vLLM 子进程提供，查询侧可选 Qwen instruction 扩展。</p>
      </section>
      <aside class="card hero-side">
        <div class="kv"><div class="k">Model</div><div class="v" id="modelId">__MODEL_ID__</div></div>
        <div class="kv"><div class="k">Public API</div><div class="v"><code>POST /v1/embeddings</code></div></div>
        <div class="kv"><div class="k">Backend</div><div class="v"><code>vLLM @ 127.0.0.1:8001</code></div></div>
        <div class="kv"><div class="k">Query Default</div><div class="v" id="defaultInstruction">__DEFAULT_INSTRUCTION__</div></div>
      </aside>
    </div>

    <div class="grid">
      <section class="card section">
        <h2>Generate Embeddings</h2>
        <label for="texts">Texts</label>
        <textarea id="texts" placeholder="每行一条文本。若只输入 2 行，结果区域会额外显示 cosine similarity。"></textarea>
        <div class="hint">输入会按非空行拆分。若选择 <code>query</code>，服务会在每条文本前自动拼接 instruction。</div>

        <div class="row" style="margin-top:16px">
          <div>
            <label for="inputType">Input Type</label>
            <select id="inputType">
              <option value="">document / raw</option>
              <option value="query">query</option>
              <option value="document">document</option>
            </select>
          </div>
          <div>
            <label for="dimensions">Dimensions</label>
            <input id="dimensions" type="number" min="32" max="4096" placeholder="例如 1024"/>
          </div>
        </div>

        <div class="row" style="margin-top:12px">
          <div>
            <label for="instruction">Instruction</label>
            <input id="instruction" placeholder="留空则使用默认 query instruction"/>
          </div>
          <div>
            <label for="encodingFormat">Encoding Format</label>
            <select id="encodingFormat">
              <option value="">float</option>
              <option value="float">float</option>
              <option value="base64">base64</option>
            </select>
          </div>
        </div>

        <div class="actions">
          <button class="primary" id="runBtn">Run Embedding</button>
          <button class="secondary" id="fillDemoBtn">Fill Demo</button>
          <button class="secondary" id="healthBtn">Refresh Health</button>
        </div>
      </section>

      <section class="card section">
        <h2>Runtime</h2>
        <div class="status" id="statusBox"><span class="light"></span><span id="statusText">Checking backend...</span></div>
        <div class="metrics">
          <div class="metric"><div class="n" id="metricCount">0</div><div class="l">Vectors</div></div>
          <div class="metric"><div class="n" id="metricDim">-</div><div class="l">Dimensions</div></div>
          <div class="metric"><div class="n" id="metricLatency">-</div><div class="l">Latency</div></div>
        </div>
        <div class="list" style="margin-top:16px">
          <span class="chip" id="healthModelChip">Model: __MODEL_ID__</span>
          <span class="chip" id="healthPortChip">API: /v1/embeddings</span>
          <span class="chip" id="healthBackendChip">Backend: waiting</span>
        </div>
        <div style="margin-top:16px">
          <label>Health Payload</label>
          <pre id="healthOut">loading...</pre>
        </div>
      </section>
    </div>

    <div class="grid">
      <section class="card section">
        <h2>Result Summary</h2>
        <pre id="summaryOut">尚未请求。</pre>
      </section>
      <section class="card section">
        <h2>Raw Response</h2>
        <pre id="rawOut">尚未请求。</pre>
      </section>
    </div>
  </div>

  <script>
    const els = {
      texts: document.getElementById("texts"),
      inputType: document.getElementById("inputType"),
      dimensions: document.getElementById("dimensions"),
      instruction: document.getElementById("instruction"),
      encodingFormat: document.getElementById("encodingFormat"),
      runBtn: document.getElementById("runBtn"),
      fillDemoBtn: document.getElementById("fillDemoBtn"),
      healthBtn: document.getElementById("healthBtn"),
      statusBox: document.getElementById("statusBox"),
      statusText: document.getElementById("statusText"),
      metricCount: document.getElementById("metricCount"),
      metricDim: document.getElementById("metricDim"),
      metricLatency: document.getElementById("metricLatency"),
      healthOut: document.getElementById("healthOut"),
      summaryOut: document.getElementById("summaryOut"),
      rawOut: document.getElementById("rawOut"),
      healthModelChip: document.getElementById("healthModelChip"),
      healthBackendChip: document.getElementById("healthBackendChip"),
    };

    function splitTexts(value) {
      return value.split(/\\n+/).map(v => v.trim()).filter(Boolean);
    }

    function setStatus(ok, text) {
      els.statusBox.classList.toggle("ok", ok);
      els.statusBox.classList.toggle("bad", !ok);
      els.statusText.textContent = text;
    }

    function cosineSimilarity(a, b) {
      let dot = 0;
      let normA = 0;
      let normB = 0;
      const len = Math.min(a.length, b.length);
      for (let i = 0; i < len; i += 1) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
      }
      if (!normA || !normB) return null;
      return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    function summarizeResponse(payload, latencyMs) {
      const data = Array.isArray(payload.data) ? payload.data : [];
      const first = data[0]?.embedding || [];
      const lines = [];
      lines.push(`vectors: ${data.length}`);
      lines.push(`dimensions: ${first.length || "-"}`);
      lines.push(`latency_ms: ${latencyMs}`);
      lines.push(`model: ${payload.model || "-"}`);
      if (payload.usage) {
        lines.push(`usage.prompt_tokens: ${payload.usage.prompt_tokens ?? "-"}`);
        lines.push(`usage.total_tokens: ${payload.usage.total_tokens ?? "-"}`);
      }
      if (data.length === 2 && Array.isArray(data[0].embedding) && Array.isArray(data[1].embedding)) {
        const sim = cosineSimilarity(data[0].embedding, data[1].embedding);
        if (sim !== null) {
          lines.push(`cosine_similarity: ${sim.toFixed(6)}`);
        }
      }
      lines.push("");
      data.forEach((item, idx) => {
        const preview = Array.isArray(item.embedding) ? item.embedding.slice(0, 12) : [];
        lines.push(`embedding[${idx}] head: ${JSON.stringify(preview)}`);
      });
      return lines.join("\\n");
    }

    async function refreshHealth() {
      try {
        const response = await fetch("/health");
        const payload = await response.json();
        els.healthOut.textContent = JSON.stringify(payload, null, 2);
        els.healthModelChip.textContent = `Model: ${payload.model_id || "-"}`;
        els.healthBackendChip.textContent = `Backend: ${payload.backend_ready ? "ready" : "not ready"}`;
        setStatus(Boolean(payload.backend_ready), payload.backend_ready ? "Backend ready" : (payload.backend_last_error || "Backend not ready"));
      } catch (error) {
        setStatus(false, `Health check failed: ${error}`);
      }
    }

    async function runEmbedding() {
      const texts = splitTexts(els.texts.value);
      if (!texts.length) {
        setStatus(false, "请至少输入一条文本。");
        return;
      }

      const payload = {
        input: texts.length === 1 ? texts[0] : texts,
      };

      if (els.inputType.value) payload.input_type = els.inputType.value;
      if (els.instruction.value.trim()) payload.instruction = els.instruction.value.trim();
      if (els.dimensions.value.trim()) payload.dimensions = Number(els.dimensions.value);
      if (els.encodingFormat.value) payload.encoding_format = els.encodingFormat.value;

      els.runBtn.disabled = true;
      const startedAt = performance.now();
      try {
        const response = await fetch("/v1/embeddings", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
        });
        const result = await response.json();
        const latencyMs = Math.round(performance.now() - startedAt);
        if (!response.ok) {
          setStatus(false, result?.error?.message || `HTTP ${response.status}`);
          els.rawOut.textContent = JSON.stringify(result, null, 2);
          els.summaryOut.textContent = "请求失败。";
          return;
        }

        const vectorCount = Array.isArray(result.data) ? result.data.length : 0;
        const vectorDim = result.data?.[0]?.embedding?.length || "-";
        els.metricCount.textContent = String(vectorCount);
        els.metricDim.textContent = String(vectorDim);
        els.metricLatency.textContent = `${latencyMs} ms`;
        els.summaryOut.textContent = summarizeResponse(result, latencyMs);
        els.rawOut.textContent = JSON.stringify(result, null, 2);
        setStatus(true, "Embedding completed.");
        await refreshHealth();
      } catch (error) {
        setStatus(false, `Request failed: ${error}`);
      } finally {
        els.runBtn.disabled = false;
      }
    }

    els.fillDemoBtn.addEventListener("click", () => {
      els.texts.value = "What is the capital of China?\\nThe capital of China is Beijing.";
      els.inputType.value = "query";
      els.instruction.value = "__DEFAULT_INSTRUCTION__";
      els.dimensions.value = "1024";
      els.encodingFormat.value = "float";
    });

    els.runBtn.addEventListener("click", runEmbedding);
    els.healthBtn.addEventListener("click", refreshHealth);
    refreshHealth();
  </script>
</body>
</html>
"""
    return (
        template.replace("__MODEL_ID__", get_current_model_id())
        .replace("__DEFAULT_INSTRUCTION__", DEFAULT_QUERY_INSTRUCTION)
    )


def create_application() -> FastAPI:
    mcp = create_mcp_server()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        async with mcp.session_manager.run():
            await maybe_preload_backend()
            yield
            await shutdown_backend()

    app = FastAPI(title="Qwen3 Embedding", lifespan=lifespan)
    app.mount(
        "/static",
        StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")),
        name="static",
    )
    app.mount("/mcp", mcp.streamable_http_app(), name="mcp")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _build_index_html()

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(await get_health_payload())

    @app.post("/v1/embeddings")
    async def embeddings(request: EmbeddingsRequest) -> JSONResponse:
        try:
            response_payload = await create_embeddings(request.model_dump(exclude_none=True))
        except InputValidationError as exc:
            return _error_response(str(exc), 400, "invalid_request_error", 400)
        except BackendUnavailableError as exc:
            return _error_response(str(exc), 503, "service_unavailable", 503)
        except BackendProxyError as exc:
            if exc.payload is not None:
                return JSONResponse(exc.payload, status_code=exc.status_code)
            return _error_response(str(exc), exc.status_code, "backend_error", exc.status_code)
        return JSONResponse(response_payload)

    @app.post("/admin/reload")
    async def admin_reload(request: ReloadRequest, x_admin_token: Optional[str] = Header(default=None)) -> JSONResponse:
        try:
            _authorize_admin(x_admin_token)
            payload = await reload_backend(request.model_dump(exclude_none=True))
        except PermissionError as exc:
            return _error_response(str(exc), 401, "unauthorized", 401)
        except InputValidationError as exc:
            return _error_response(str(exc), 400, "invalid_request_error", 400)
        except BackendUnavailableError as exc:
            return _error_response(str(exc), 503, "service_unavailable", 503)
        return JSONResponse(payload)

    @app.get("/openai-example", response_class=HTMLResponse)
    def openai_example() -> str:
        example = {
            "base_url": "http://localhost:12302/v1",
            "model": get_current_model_id(),
            "input": "hello world",
            "input_type": "query",
            "instruction": DEFAULT_QUERY_INSTRUCTION,
        }
        return "<pre>" + json.dumps(example, ensure_ascii=False, indent=2) + "</pre>"

    return app


app = create_application()
