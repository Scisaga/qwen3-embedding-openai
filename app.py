import contextlib
import json
import os
import asyncio
from typing import Literal, Optional

from fastapi import FastAPI, Header
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from embedding_service import (
    ADMIN_TOKEN,
    DEFAULT_QUERY_INSTRUCTION,
    MAX_DIMENSIONS,
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
from projector_service import ProjectorDependencyError, create_projector_payload


class EmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str | list[str]
    model: Optional[str] = None
    dimensions: Optional[int] = Field(default=None, ge=32, le=MAX_DIMENSIONS)
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


class ProjectorRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputs: list[str]
    labels: Optional[list[str]] = None
    model: Optional[str] = None
    input_type: Optional[Literal["query", "document"]] = None
    instruction: Optional[str] = None
    projection_method: Literal["umap", "tsne", "pca"] = "umap"
    metric: Literal["cosine", "euclidean"] = "cosine"
    neighbors_k: int = Field(default=10, ge=1, le=256)
    point_size: float = Field(default=3.5, gt=0.0, le=64.0)


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
  <meta name="theme-color" content="#0a1224"/>
  <link rel="icon" type="image/png" href="/static/logo.png"/>
  <link rel="apple-touch-icon" href="/static/logo.png"/>
  <title>Qwen3 Embedding</title>
  <style>
    :root{
      --bg0:#050913;
      --bg1:#0a1224;
      --panel:rgba(10, 18, 36, .82);
      --panel2:rgba(13, 22, 42, .92);
      --border:rgba(148,163,184,.18);
      --text:rgba(226,232,240,.94);
      --muted:rgba(148,163,184,.86);
      --muted2:rgba(148,163,184,.62);
      --accent:#ff8a1f;
      --accent2:#ffd166;
      --good:#22c55e;
      --danger:#ef4444;
      --warn:#f59e0b;
      --shadow:0 24px 80px rgba(0,0,0,.42);
      --radius:10px;
      --radius2:12px;
    }
    *{box-sizing:border-box}
    html,body{min-height:100%;margin:0}
    body{
      color:var(--text);
      font-family:"IBM Plex Sans", ui-sans-serif, system-ui, -apple-system, sans-serif;
      overflow-x:hidden;
      background:
        radial-gradient(1100px 700px at 10% -10%, rgba(255,138,31,.18), transparent 60%),
        radial-gradient(900px 600px at 100% 0%, rgba(96,165,250,.16), transparent 55%),
        linear-gradient(180deg, var(--bg0), var(--bg1));
    }
    a{color:inherit;text-decoration:none}
    code{
      font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
      background:rgba(2,6,23,.52);
      border:1px solid var(--border);
      padding:2px 8px;
      border-radius:999px;
      font-size:.92em;
    }
    .app{display:flex;min-height:100vh}
    .sidebar{
      width:250px;
      padding:16px 14px;
      border-right:1px solid var(--border);
      background:rgba(2,6,23,.54);
      backdrop-filter:blur(12px);
      position:sticky;
      top:0;
      height:100vh;
      overflow:hidden;
    }
    .brand{display:flex;gap:12px;align-items:center;padding:10px 8px 14px}
    .logo{width:40px;height:40px;border-radius:10px;display:block;object-fit:cover;box-shadow:0 12px 28px rgba(255,138,31,.26)}
    .brand-title{font-size:16px;font-weight:700;letter-spacing:.2px}
    .brand-sub{font-size:12px;color:var(--muted2);margin-top:2px}
    .nav{display:flex;flex-direction:column;gap:8px}
    .nav .nav-link{
      width:100%;
      appearance:none;
      border:none;
      text-align:left;
      display:flex;gap:10px;align-items:center;
      padding:10px 12px;border-radius:10px;border:1px solid transparent;
      background:transparent;
      color:var(--muted);transition:background .16s,border-color .16s,color .16s;
      cursor:pointer;
    }
    .nav .nav-link:hover{background:rgba(148,163,184,.08)}
    .nav .nav-link.active{
      background:rgba(255,138,31,.12);
      border-color:rgba(255,138,31,.22);
      color:var(--text);
    }
    .nav .dot{
      width:9px;height:9px;border-radius:999px;background:rgba(148,163,184,.54);
      box-shadow:0 0 0 4px rgba(148,163,184,.08)
    }
    .nav .nav-link.active .dot{
      background:var(--accent);
      box-shadow:0 0 0 4px rgba(255,138,31,.14)
    }
    .sidebar-card{
      margin-top:16px;padding:14px;border-radius:var(--radius);
      border:1px solid var(--border);background:rgba(15,23,42,.42)
    }
    .sidebar-card h3{margin:0 0 10px;font-size:13px}
    .sidebar-card p,.sidebar-card li{margin:0;color:var(--muted);font-size:12px;line-height:1.65}
    .sidebar-card ul{padding-left:16px;margin:0}
    .main{flex:1;display:flex;flex-direction:column}
    .topbar{
      position:sticky;top:0;z-index:12;
      display:flex;justify-content:space-between;align-items:center;gap:12px;
      padding:14px 18px;border-bottom:1px solid var(--border);
      background:rgba(2,6,23,.52);backdrop-filter:blur(12px)
    }
    .crumbs{display:flex;align-items:center;gap:10px;font-size:13px;color:var(--muted)}
    .crumbs strong{color:var(--text)}
    .chips{display:flex;gap:10px;flex-wrap:wrap}
    .chip{
      display:inline-flex;align-items:center;gap:8px;padding:8px 10px;border-radius:999px;
      border:1px solid var(--border);background:rgba(15,23,42,.5);font-size:12px;color:var(--muted)
    }
    .chip.ok{color:#bbf7d0;border-color:rgba(34,197,94,.26);background:rgba(34,197,94,.12)}
    .chip.warn{color:#fde68a;border-color:rgba(245,158,11,.28);background:rgba(245,158,11,.12)}
    .chip.bad{color:#fecaca;border-color:rgba(239,68,68,.26);background:rgba(239,68,68,.12)}
    .content{padding:18px 16px 32px}
    .tabbar{
      max-width:1320px;margin:0 auto 14px;display:flex;gap:8px;flex-wrap:wrap
    }
    .tab-btn{
      appearance:none;border:1px solid var(--border);background:rgba(15,23,42,.45);
      color:var(--muted);padding:8px 12px;border-radius:10px;cursor:pointer;font-size:13px;
      display:inline-flex;align-items:center
    }
    .tab-btn.active{
      color:var(--text);
      background:rgba(255,138,31,.12);
      border-color:rgba(255,138,31,.25);
    }
    .section{max-width:1320px;margin:0 auto 16px;display:none}
    .section.active{display:block}
    .section-head{margin:0 0 14px}
    .section-head h1,.section-head h2{margin:0;font-size:24px;letter-spacing:.2px}
    .section-head p{margin:8px 0 0;color:var(--muted);font-size:13px;line-height:1.7}
    .grid{display:grid;grid-template-columns:1.15fr .85fr;gap:16px}
    .grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
    .card{
      border:1px solid var(--border);border-radius:var(--radius2);background:var(--panel);
      box-shadow:var(--shadow);overflow:hidden
    }
    .card-body{padding:18px}
    .card-title{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin:0 0 14px}
    .card-title h3{margin:0;font-size:18px}
    .card-title p{margin:6px 0 0;color:var(--muted);font-size:13px}
    .status{
      display:flex;align-items:center;gap:10px;padding:12px;border-radius:10px;
      background:rgba(15,23,42,.46);border:1px solid var(--border);font-size:13px
    }
    .light{width:10px;height:10px;border-radius:999px;background:var(--muted)}
    .status.ok .light{background:var(--good);box-shadow:0 0 0 5px rgba(34,197,94,.15)}
    .status.warn .light{background:var(--warn);box-shadow:0 0 0 5px rgba(245,158,11,.16)}
    .status.bad .light{background:var(--danger);box-shadow:0 0 0 5px rgba(239,68,68,.15)}
    .error-box{
      display:none;margin-top:14px;padding:12px;border-radius:10px;
      border:1px solid rgba(239,68,68,.28);background:rgba(127,29,29,.22);color:#fecaca
    }
    .error-box.show{display:block}
    .muted{color:var(--muted)}
    .hint{margin-top:8px;font-size:12px;color:var(--muted)}
    label{display:block;margin:0 0 8px;font-size:13px;color:#dbeafe}
    textarea,input,select{
      width:100%;padding:10px 12px;border-radius:10px;border:1px solid var(--border);
      background:rgba(15,23,42,.62);color:var(--text);font:inherit
    }
    textarea{min-height:220px;resize:vertical}
    .row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
    .row-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
    .actions{display:flex;flex-wrap:wrap;gap:10px;margin-top:18px}
    .btn{
      appearance:none;border:none;border-radius:10px;padding:10px 13px;
      font:600 13px/1 inherit;cursor:pointer;transition:transform .06s,opacity .16s
    }
    .btn:hover{opacity:.94}
    .btn:active{transform:translateY(1px)}
    .btn:disabled{opacity:.55;cursor:not-allowed}
    .btn.primary{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#111827}
    .btn.secondary{background:rgba(15,23,42,.64);border:1px solid var(--border);color:var(--text)}
    .btn.ghost{background:transparent;border:1px solid var(--border);color:var(--muted)}
    .metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:14px}
    .metric{
      padding:12px;border-radius:10px;border:1px solid var(--border);background:rgba(15,23,42,.48)
    }
    .metric .n{font-size:24px;font-weight:700}
    .metric .l{font-size:12px;color:var(--muted)}
    pre{
      margin:0;white-space:pre-wrap;word-break:break-word;background:rgba(2,6,23,.72);
      border:1px solid var(--border);padding:12px;border-radius:10px;max-height:420px;overflow:auto
    }
    .toolbar{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:12px}
    .toolbar .small{font-size:12px;color:var(--muted)}
    .matrix-wrap{overflow:auto;border:1px solid var(--border);border-radius:10px;background:rgba(2,6,23,.52)}
    table{width:100%;border-collapse:collapse}
    th,td{padding:10px 12px;border-bottom:1px solid rgba(148,163,184,.08);text-align:center;font-size:12px}
    th:first-child,td:first-child{text-align:left;color:var(--muted)}
    .pill{
      display:inline-flex;padding:6px 10px;border-radius:999px;border:1px solid var(--border);
      background:rgba(255,255,255,.06);font-size:12px;color:#dbeafe
    }
    .templates{display:flex;flex-direction:column;gap:10px}
    .template-note{padding:10px 12px;border-radius:10px;border:1px solid var(--border);background:rgba(15,23,42,.42);font-size:13px;color:var(--muted)}
    .kv-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px}
    .kv-item{border:1px solid var(--border);background:rgba(15,23,42,.44);border-radius:10px;padding:10px}
    .kv-item span{display:block;font-size:12px;color:var(--muted);margin-bottom:6px}
    .kv-item strong{font-size:15px;font-weight:600;color:var(--text)}
    .api-list{list-style:none;padding:0;margin:0;display:grid;gap:8px}
    .api-list li{border:1px solid var(--border);border-radius:10px;background:rgba(15,23,42,.42);padding:10px 12px}
    .api-list p{margin:6px 0 0;color:var(--muted);font-size:12px}
    .inline-code{font-family:ui-monospace, SFMono-Regular, Menlo, monospace}
    @media (max-width: 1100px){
      .app{display:block}
      .sidebar{width:auto;height:auto;position:relative;border-right:none;border-bottom:1px solid var(--border);overflow:visible}
      .grid,.grid-2,.row,.row-3,.metrics{grid-template-columns:1fr}
      .kv-grid{grid-template-columns:1fr}
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <img class="logo" src="/static/logo.png" alt="Qwen3 Embedding"/>
        <div>
          <div class="brand-title">Qwen3 Embedding</div>
          <div class="brand-sub">Self-hosted Debug Console</div>
        </div>
      </div>

      <nav class="nav">
        <button type="button" class="nav-link active" data-tab="debug-section"><span class="dot"></span><span>调试台</span></button>
        <button type="button" class="nav-link" data-tab="results-section"><span class="dot"></span><span>结果分析</span></button>
        <button type="button" class="nav-link" data-tab="admin-section"><span class="dot"></span><span>运维管理</span></button>
        <a class="nav-link" href="/projector"><span class="dot"></span><span>Projector 视图</span></a>
      </nav>

      <div class="sidebar-card">
        <h3>当前服务</h3>
        <p>对外暴露 <code>/v1/embeddings</code>、<code>/mcp</code>、<code>/health</code>。查询侧支持 Qwen instruction 注入，底层推理由容器内的 vLLM 子进程完成。</p>
      </div>

      <div class="sidebar-card">
        <h3>调试建议</h3>
        <ul>
          <li>先看右上角和 Runtime 卡片里的 backend 状态。</li>
          <li>若显示 <span class="inline-code">starting</span>，通常是模型仍在加载到 GPU。</li>
          <li>页面默认会阻止在 backend 未就绪时发请求，避免长时间挂起。</li>
        </ul>
      </div>
    </aside>

    <main class="main">
      <div class="topbar">
        <div class="crumbs"><strong>Qwen3 Embedding</strong><span>/</span><span>Web Debug Console</span></div>
        <div class="chips">
          <span class="chip" id="topModelChip">Model: __MODEL_ID__</span>
          <span class="chip warn" id="topStatusChip">Backend: checking</span>
          <span class="chip" id="topTimeChip">Server Time: --</span>
          <span class="chip" id="topDeviceChip">Target: CUDA</span>
        </div>
      </div>

      <div class="content">
        <div class="tabbar">
          <button type="button" class="tab-btn active" data-tab="debug-section">调试台</button>
          <button type="button" class="tab-btn" data-tab="results-section">结果分析</button>
          <button type="button" class="tab-btn" data-tab="admin-section">运维管理</button>
          <a class="tab-btn" href="/projector">Projector</a>
        </div>

        <section class="section active tab-panel" id="debug-section">
          <div class="section-head">
            <h1>Embedding 调试台</h1>
            <p>更接近参考 ASR 项目的交互形态：左侧导航、模板切换、状态面板、热重载表单，以及向量下载和相似度矩阵。你可以直接在这里完成大部分接入调试。</p>
          </div>

          <div class="grid">
            <section class="card">
              <div class="card-body">
                <div class="card-title">
                  <div>
                    <h3>Generate Embeddings</h3>
                    <p>输入文本后直接调用 <code>/v1/embeddings</code>。两条以上文本会自动计算相似度矩阵。</p>
                  </div>
                  <span class="pill" id="templateChip">Template: custom</span>
                </div>

                <div class="row">
                  <div>
                    <label for="templateSelect">请求模板</label>
                    <select id="templateSelect">
                      <option value="blank">空白模板</option>
                      <option value="retrieval_pair">检索对比</option>
                      <option value="query_only">单条查询</option>
                      <option value="document_batch">文档批量</option>
                      <option value="news_similarity">新闻相似度</option>
                    </select>
                  </div>
                  <div>
                    <label>&nbsp;</label>
                    <button class="btn secondary" id="applyTemplateBtn" style="width:100%">应用模板</button>
                  </div>
                </div>

                <label for="texts" style="margin-top:16px">Texts</label>
                <textarea id="texts" placeholder="每行一条文本。"></textarea>
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
                    <input id="dimensions" type="number" min="32" max="__MAX_DIMENSIONS__" placeholder="例如 1024"/>
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
                  <button class="btn primary" id="runBtn">生成 Embedding</button>
                  <button class="btn secondary" id="healthBtn">刷新状态</button>
                  <button class="btn secondary" id="copyJsonBtn">复制 JSON</button>
                  <button class="btn secondary" id="downloadBtn">下载向量</button>
                  <button class="btn ghost" id="fillDemoBtn">填充 Demo</button>
                </div>
              </div>
            </section>

            <section class="card" id="runtime-section">
              <div class="card-body">
                <div class="card-title">
                  <div>
                    <h3>Runtime</h3>
                    <p>后端健康、模型、时区、显卡目标和当前错误都会在这里聚合展示。</p>
                  </div>
                </div>

                <div class="status warn" id="statusBox"><span class="light"></span><span id="statusText">Checking backend...</span></div>
                <div class="error-box" id="errorBox">
                  <strong>最近错误</strong>
                  <div id="errorText" style="margin-top:8px"></div>
                </div>

                <div class="metrics">
                  <div class="metric"><div class="n" id="metricCount">0</div><div class="l">Vectors</div></div>
                  <div class="metric"><div class="n" id="metricDim">-</div><div class="l">Dimensions</div></div>
                  <div class="metric"><div class="n" id="metricLatency">-</div><div class="l">Latency</div></div>
                  <div class="metric"><div class="n" id="metricState">-</div><div class="l">Backend State</div></div>
                </div>

                <div class="kv-grid">
                  <div class="kv-item"><span>模型</span><strong id="healthModelChip">__MODEL_ID__</strong></div>
                  <div class="kv-item"><span>服务时间</span><strong id="healthTimeChip">--</strong></div>
                  <div class="kv-item"><span>设备目标</span><strong id="healthDeviceChip">cuda</strong></div>
                  <div class="kv-item"><span>内层地址</span><strong id="healthBackendChip">http://127.0.0.1:8001</strong></div>
                </div>

                <div style="margin-top:16px">
                  <label>Health Payload</label>
                  <pre id="healthOut">loading...</pre>
                </div>
              </div>
            </section>
          </div>
        </section>

        <section class="section tab-panel" id="results-section">
          <div class="grid">
            <section class="card">
              <div class="card-body">
                <div class="card-title">
                  <div>
                    <h3>结果概览</h3>
                    <p>向量数量、维度、token 统计和前几维预览。</p>
                  </div>
                </div>
                <pre id="summaryOut">尚未请求。</pre>
              </div>
            </section>

            <section class="card">
              <div class="card-body">
                <div class="card-title">
                  <div>
                    <h3>模板说明</h3>
                    <p>常见调试场景可以一键填充，避免重复手输。</p>
                  </div>
                </div>
                <div class="templates" id="templateHelp">
                  <div class="template-note">选择模板后会自动填充 <code>texts</code>、<code>input_type</code>、<code>instruction</code> 和 <code>dimensions</code>。</div>
                </div>
              </div>
            </section>
          </div>

          <div class="grid" style="margin-top:16px">
            <section class="card">
              <div class="card-body">
                <div class="toolbar">
                  <div>
                    <h3 style="margin:0">Similarity Matrix</h3>
                    <div class="small">最多展示当前请求的前 8 条文本。</div>
                  </div>
                </div>
                <div class="matrix-wrap" id="matrixWrap">
                  <table id="matrixTable">
                    <tbody><tr><td style="padding:16px;color:var(--muted)">尚未请求。</td></tr></tbody>
                  </table>
                </div>
              </div>
            </section>

            <section class="card">
              <div class="card-body">
                <div class="card-title">
                  <div>
                    <h3>Raw Response</h3>
                    <p>直接展示 `/v1/embeddings` 的原始 JSON 响应，便于复制和比对。</p>
                  </div>
                </div>
                <pre id="rawOut">尚未请求。</pre>
              </div>
            </section>
          </div>
        </section>

        <section class="section tab-panel" id="admin-section">
          <div class="grid">
            <section class="card">
              <div class="card-body">
                <div class="card-title">
                  <div>
                    <h3>热重载表单</h3>
                    <p>无需重启容器即可重载模型或调整关键运行参数。</p>
                  </div>
                </div>

                <div class="row">
                  <div>
                    <label for="reloadModelId">Model ID</label>
                    <input id="reloadModelId" placeholder="Qwen/Qwen3-Embedding-8B"/>
                  </div>
                  <div>
                    <label for="reloadToken">x-admin-token</label>
                    <input id="reloadToken" placeholder="change-me"/>
                  </div>
                </div>

                <div class="row-3" style="margin-top:12px">
                  <div>
                    <label for="reloadDtype">DType</label>
                    <input id="reloadDtype" placeholder="float16"/>
                  </div>
                  <div>
                    <label for="reloadMaxModelLen">Max Model Len</label>
                    <input id="reloadMaxModelLen" type="number" placeholder="4096"/>
                  </div>
                  <div>
                    <label for="reloadGpuUtil">GPU Memory Utilization</label>
                    <input id="reloadGpuUtil" type="number" step="0.01" min="0.1" max="1" placeholder="0.72"/>
                  </div>
                </div>

                <div class="row" style="margin-top:12px">
                  <div>
                    <label for="reloadInstruction">Default Query Instruction</label>
                    <input id="reloadInstruction" placeholder="Given a web search query, retrieve relevant passages that answer the query"/>
                  </div>
                  <div>
                    <label for="reloadExtraArgs">VLLM Extra Args</label>
                    <input id="reloadExtraArgs" placeholder="--tensor-parallel-size 2"/>
                  </div>
                </div>

                <div class="actions">
                  <button class="btn primary" id="reloadBtn">提交热重载</button>
                </div>

                <div style="margin-top:16px">
                  <label>Reload Response</label>
                  <pre id="reloadOut">尚未提交。</pre>
                </div>
              </div>
            </section>

            <section class="card" id="api-section">
              <div class="card-body">
                <div class="card-title">
                  <div>
                    <h3>接口说明</h3>
                    <p>当前页面只是内置调试台，正式接入仍建议直接调用 API。</p>
                  </div>
                </div>
                <ul class="api-list">
                  <li><code>POST /v1/embeddings</code><p>OpenAI 兼容 Embeddings 接口</p></li>
                  <li><code>POST /v1/embeddings/projector</code><p>后端预计算 2D 投影 + 近邻，供 Projector 视图渲染</p></li>
                  <li><code>GET /</code><p>内置调试页面</p></li>
                  <li><code>GET /projector</code><p>Projector 点云交互页面</p></li>
                  <li><code>POST/GET /mcp</code><p>MCP Streamable HTTP 入口</p></li>
                  <li><code>GET /health</code><p>运行状态与 backend 聚合健康</p></li>
                  <li><code>GET /docs</code> / <code>GET /redoc</code><p>Swagger 与 ReDoc 文档</p></li>
                </ul>
              </div>
            </section>
          </div>
        </section>
      </div>
    </main>
  </div>

  <script>
    const REQUEST_TIMEOUT_MS = 20000;
    let lastHealthPayload = null;
    let lastEmbeddingPayload = null;

    const templates = {
      blank: {
        label: "空白模板",
        note: "保留当前输入，只重置为最基础的 document/raw 调试模式。",
        texts: "",
        inputType: "",
        instruction: "",
        dimensions: "",
        encodingFormat: "float",
      },
      retrieval_pair: {
        label: "检索对比",
        note: "最适合验证 query/document 语义是否正确，以及相似度是否符合预期。",
        texts: "What is the capital of China?\\nThe capital of China is Beijing.\\nThe Eiffel Tower is located in Paris.",
        inputType: "query",
        instruction: "__DEFAULT_INSTRUCTION__",
        dimensions: "",
        encodingFormat: "float",
      },
      query_only: {
        label: "单条查询",
        note: "用于验证 query 侧 instruction 注入是否成功。",
        texts: "How to improve retrieval quality for Chinese documents?",
        inputType: "query",
        instruction: "__DEFAULT_INSTRUCTION__",
        dimensions: "",
        encodingFormat: "float",
      },
      document_batch: {
        label: "文档批量",
        note: "把多条文档作为 document/raw 一次性编码，并查看相似度矩阵。",
        texts: "Beijing is the capital of China.\\nShanghai is a major financial center in China.\\nParis is the capital of France.",
        inputType: "document",
        instruction: "",
        dimensions: "",
        encodingFormat: "float",
      },
      news_similarity: {
        label: "新闻相似度",
        note: "用于比较多段新闻文本之间的语义距离。",
        texts: "中国央行宣布新的利率政策。\\n央行今日发布利率调整公告。\\n足球联赛将在本周末开赛。",
        inputType: "document",
        instruction: "",
        dimensions: "",
        encodingFormat: "float",
      },
    };

    const els = {
      texts: document.getElementById("texts"),
      inputType: document.getElementById("inputType"),
      dimensions: document.getElementById("dimensions"),
      instruction: document.getElementById("instruction"),
      encodingFormat: document.getElementById("encodingFormat"),
      templateSelect: document.getElementById("templateSelect"),
      applyTemplateBtn: document.getElementById("applyTemplateBtn"),
      templateChip: document.getElementById("templateChip"),
      templateHelp: document.getElementById("templateHelp"),
      runBtn: document.getElementById("runBtn"),
      fillDemoBtn: document.getElementById("fillDemoBtn"),
      healthBtn: document.getElementById("healthBtn"),
      copyJsonBtn: document.getElementById("copyJsonBtn"),
      downloadBtn: document.getElementById("downloadBtn"),
      statusBox: document.getElementById("statusBox"),
      statusText: document.getElementById("statusText"),
      errorBox: document.getElementById("errorBox"),
      errorText: document.getElementById("errorText"),
      metricCount: document.getElementById("metricCount"),
      metricDim: document.getElementById("metricDim"),
      metricLatency: document.getElementById("metricLatency"),
      metricState: document.getElementById("metricState"),
      healthOut: document.getElementById("healthOut"),
      summaryOut: document.getElementById("summaryOut"),
      rawOut: document.getElementById("rawOut"),
      matrixTable: document.getElementById("matrixTable"),
      topModelChip: document.getElementById("topModelChip"),
      topStatusChip: document.getElementById("topStatusChip"),
      topTimeChip: document.getElementById("topTimeChip"),
      topDeviceChip: document.getElementById("topDeviceChip"),
      healthModelChip: document.getElementById("healthModelChip"),
      healthTimeChip: document.getElementById("healthTimeChip"),
      healthDeviceChip: document.getElementById("healthDeviceChip"),
      healthBackendChip: document.getElementById("healthBackendChip"),
      reloadModelId: document.getElementById("reloadModelId"),
      reloadToken: document.getElementById("reloadToken"),
      reloadDtype: document.getElementById("reloadDtype"),
      reloadMaxModelLen: document.getElementById("reloadMaxModelLen"),
      reloadGpuUtil: document.getElementById("reloadGpuUtil"),
      reloadInstruction: document.getElementById("reloadInstruction"),
      reloadExtraArgs: document.getElementById("reloadExtraArgs"),
      reloadBtn: document.getElementById("reloadBtn"),
      reloadOut: document.getElementById("reloadOut"),
    };
    const tabPanels = Array.from(document.querySelectorAll(".tab-panel"));
    const tabControls = Array.from(document.querySelectorAll("[data-tab]"));

    function setActiveTab(tabId, syncHash = true) {
      const targetId = tabPanels.some(panel => panel.id === tabId) ? tabId : "debug-section";

      tabPanels.forEach((panel) => {
        panel.classList.toggle("active", panel.id === targetId);
      });
      tabControls.forEach((control) => {
        control.classList.toggle("active", control.dataset.tab === targetId);
      });

      if (syncHash) {
        history.replaceState(null, "", `#${targetId}`);
      }
    }

    function splitTexts(value) {
      return value.split(/\\n+/).map(v => v.trim()).filter(Boolean);
    }

    function showError(message) {
      els.errorBox.classList.add("show");
      els.errorText.textContent = message;
    }

    function clearError() {
      els.errorBox.classList.remove("show");
      els.errorText.textContent = "";
    }

    function setStatus(kind, text) {
      els.statusBox.classList.remove("ok", "warn", "bad");
      els.statusBox.classList.add(kind);
      els.statusText.textContent = text;
      els.topStatusChip.classList.remove("ok", "warn", "bad");
      els.topStatusChip.classList.add(kind);
      els.topStatusChip.textContent = `Backend: ${text}`;
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

    function renderMatrix(payload) {
      const data = Array.isArray(payload?.data) ? payload.data.slice(0, 8) : [];
      if (!data.length) {
        els.matrixTable.innerHTML = "<tbody><tr><td style='padding:16px;color:var(--muted)'>尚未请求。</td></tr></tbody>";
        return;
      }

      let header = "<thead><tr><th>Text</th>";
      data.forEach((_, idx) => { header += `<th>#${idx + 1}</th>`; });
      header += "</tr></thead>";

      let body = "<tbody>";
      data.forEach((row, rowIndex) => {
        const rowPreview = splitTexts(els.texts.value)[rowIndex] || `item ${rowIndex + 1}`;
        body += `<tr><td title="${rowPreview.replace(/"/g, "&quot;")}">${rowPreview.slice(0, 32)}</td>`;
        data.forEach((col) => {
          const sim = cosineSimilarity(row.embedding, col.embedding);
          body += `<td>${sim === null ? "-" : sim.toFixed(4)}</td>`;
        });
        body += "</tr>";
      });
      body += "</tbody>";
      els.matrixTable.innerHTML = header + body;
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
      lines.push("");
      data.forEach((item, idx) => {
        const preview = Array.isArray(item.embedding) ? item.embedding.slice(0, 12) : [];
        lines.push(`embedding[${idx}] head: ${JSON.stringify(preview)}`);
      });
      return lines.join("\\n");
    }

    function buildPayload() {
      const texts = splitTexts(els.texts.value);
      if (!texts.length) return null;
      const payload = {input: texts.length === 1 ? texts[0] : texts};
      if (els.inputType.value) payload.input_type = els.inputType.value;
      if (els.instruction.value.trim()) payload.instruction = els.instruction.value.trim();
      if (els.dimensions.value.trim()) payload.dimensions = Number(els.dimensions.value);
      if (els.encodingFormat.value) payload.encoding_format = els.encodingFormat.value;
      return payload;
    }

    function applyTemplate(name) {
      const selected = templates[name];
      if (!selected) return;
      els.templateChip.textContent = `Template: ${selected.label}`;
      els.templateHelp.innerHTML = `<div class="template-note">${selected.note}</div>`;
      if (name === "blank") return;
      els.texts.value = selected.texts;
      els.inputType.value = selected.inputType;
      els.instruction.value = selected.instruction;
      els.dimensions.value = selected.dimensions;
      els.encodingFormat.value = selected.encodingFormat;
    }

    function copyJson() {
      if (!lastEmbeddingPayload) {
        showError("当前还没有可复制的 JSON 结果。");
        return;
      }
      navigator.clipboard.writeText(JSON.stringify(lastEmbeddingPayload, null, 2))
        .then(() => setStatus("ok", "JSON copied to clipboard"))
        .catch((error) => showError(`复制失败：${error}`));
    }

    function downloadJson() {
      if (!lastEmbeddingPayload) {
        showError("当前还没有可下载的结果。");
        return;
      }
      const blob = new Blob([JSON.stringify(lastEmbeddingPayload, null, 2)], {type: "application/json"});
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "qwen3-embedding-response.json";
      anchor.click();
      URL.revokeObjectURL(url);
    }

    async function refreshHealth() {
      try {
        const response = await fetch("/health");
        const payload = await response.json();
        lastHealthPayload = payload;
        els.healthOut.textContent = JSON.stringify(payload, null, 2);
        els.healthModelChip.textContent = payload.model_id || "-";
        els.healthTimeChip.textContent = payload.server_time || "--";
        els.healthDeviceChip.textContent = payload.backend_target_device || "cuda";
        els.healthBackendChip.textContent = payload.backend_url || "-";
        els.topModelChip.textContent = `Model: ${payload.model_id || "-"}`;
        els.topTimeChip.textContent = `Server Time: ${payload.server_time || "--"}`;
        els.topDeviceChip.textContent = `Target: ${payload.backend_target_device || "cuda"}`;
        els.metricState.textContent = payload.backend_state || "-";
        const message = payload.backend_last_error || `Backend ${payload.backend_state || "unknown"}`;
        if (payload.backend_ready) {
          clearError();
          setStatus("ok", "ready");
        } else if (payload.backend_state === "starting") {
          showError(message);
          setStatus("warn", "starting / loading");
        } else {
          showError(message);
          setStatus("bad", payload.backend_state || "not ready");
        }
      } catch (error) {
        showError(`Health check failed: ${error}`);
        setStatus("bad", "health check failed");
      }
    }

    async function runEmbedding() {
      const payload = buildPayload();
      if (!payload) {
        showError("请至少输入一条文本。");
        setStatus("bad", "missing input");
        return;
      }

      if (!lastHealthPayload || !lastHealthPayload.backend_ready) {
        await refreshHealth();
        if (!lastHealthPayload || !lastHealthPayload.backend_ready) {
          showError(
            (lastHealthPayload && lastHealthPayload.backend_last_error)
            || "Backend 尚未 ready。若模型仍在加载 GPU，请等待片刻后重试。"
          );
          return;
        }
      }

      clearError();
      els.runBtn.disabled = true;
      setStatus("warn", "requesting");
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
      const startedAt = performance.now();

      try {
        const response = await fetch("/v1/embeddings", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
          signal: controller.signal,
        });
        const result = await response.json();
        const latencyMs = Math.round(performance.now() - startedAt);
        if (!response.ok) {
          const backendMessage = result?.error?.message || `HTTP ${response.status}`;
          if (backendMessage.includes("does not support matryoshka")) {
            els.dimensions.value = "";
            showError(`${backendMessage} 当前模型不支持自定义 dimensions，已自动清空该字段，请重试。`);
            setStatus("warn", "dimensions disabled");
          } else {
            showError(backendMessage);
            setStatus("bad", "request failed");
          }
          els.rawOut.textContent = JSON.stringify(result, null, 2);
          els.summaryOut.textContent = "请求失败。";
          renderMatrix(null);
          return;
        }

        lastEmbeddingPayload = result;
        const vectorCount = Array.isArray(result.data) ? result.data.length : 0;
        const vectorDim = result.data?.[0]?.embedding?.length || "-";
        els.metricCount.textContent = String(vectorCount);
        els.metricDim.textContent = String(vectorDim);
        els.metricLatency.textContent = `${latencyMs} ms`;
        els.summaryOut.textContent = summarizeResponse(result, latencyMs);
        els.rawOut.textContent = JSON.stringify(result, null, 2);
        renderMatrix(result);
        setStatus("ok", "embedding completed");
        await refreshHealth();
      } catch (error) {
        if (error && error.name === "AbortError") {
          showError("请求超时：backend 可能仍在加载模型，或后端响应时间过长。");
          setStatus("warn", "request timeout");
        } else {
          showError(`Request failed: ${error}`);
          setStatus("bad", "request failed");
        }
      } finally {
        clearTimeout(timer);
        els.runBtn.disabled = false;
      }
    }

    async function submitReload() {
      const payload = {};
      if (els.reloadModelId.value.trim()) payload.model_id = els.reloadModelId.value.trim();
      if (els.reloadDtype.value.trim()) payload.dtype = els.reloadDtype.value.trim();
      if (els.reloadMaxModelLen.value.trim()) payload.max_model_len = Number(els.reloadMaxModelLen.value);
      if (els.reloadGpuUtil.value.trim()) payload.gpu_memory_utilization = Number(els.reloadGpuUtil.value);
      if (els.reloadInstruction.value.trim()) payload.default_query_instruction = els.reloadInstruction.value.trim();
      if (els.reloadExtraArgs.value.trim()) payload.extra_args = els.reloadExtraArgs.value.trim();

      els.reloadBtn.disabled = true;
      els.reloadOut.textContent = "提交中...";
      try {
        const response = await fetch("/admin/reload", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "x-admin-token": els.reloadToken.value.trim(),
          },
          body: JSON.stringify(payload),
        });
        const result = await response.json();
        els.reloadOut.textContent = JSON.stringify(result, null, 2);
        if (!response.ok) {
          showError(result?.error?.message || `Reload failed with HTTP ${response.status}`);
          setStatus("bad", "reload failed");
          return;
        }
        clearError();
        setStatus("warn", "reloading backend");
        await refreshHealth();
      } catch (error) {
        els.reloadOut.textContent = String(error);
        showError(`Reload request failed: ${error}`);
      } finally {
        els.reloadBtn.disabled = false;
      }
    }

    els.applyTemplateBtn.addEventListener("click", () => applyTemplate(els.templateSelect.value));
    els.fillDemoBtn.addEventListener("click", () => applyTemplate("retrieval_pair"));
    els.healthBtn.addEventListener("click", refreshHealth);
    els.runBtn.addEventListener("click", runEmbedding);
    els.copyJsonBtn.addEventListener("click", copyJson);
    els.downloadBtn.addEventListener("click", downloadJson);
    els.reloadBtn.addEventListener("click", submitReload);
    tabControls.forEach((control) => {
      control.addEventListener("click", () => {
        setActiveTab(control.dataset.tab || "debug-section");
      });
    });

    setActiveTab(window.location.hash.replace("#", "") || "debug-section", false);
    applyTemplate("retrieval_pair");
    refreshHealth();
  </script>
</body>
</html>
"""
    return (
        template.replace("__MODEL_ID__", get_current_model_id())
        .replace("__DEFAULT_INSTRUCTION__", DEFAULT_QUERY_INSTRUCTION)
        .replace("__MAX_DIMENSIONS__", str(MAX_DIMENSIONS))
    )


def create_application() -> FastAPI:
    mcp = create_mcp_server()
    app_root = os.path.dirname(__file__)
    projector_dist_dir = os.path.join(app_root, "static", "projector")

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        async with mcp.session_manager.run():
            preload_task: Optional[asyncio.Task] = None
            preload_task = asyncio.create_task(maybe_preload_backend())
            try:
                yield
            finally:
                if preload_task is not None and not preload_task.done():
                    preload_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await preload_task
                await shutdown_backend()

    app = FastAPI(title="Qwen3 Embedding", lifespan=lifespan)
    app.mount(
        "/static",
        StaticFiles(directory=os.path.join(app_root, "static")),
        name="static",
    )
    if os.path.isdir(projector_dist_dir):
        app.mount("/projector-static", StaticFiles(directory=projector_dist_dir), name="projector-static")
    app.mount("/mcp", mcp.streamable_http_app(), name="mcp")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _build_index_html()

    @app.get("/projector", response_class=HTMLResponse)
    def projector_index():
        projector_index_path = os.path.join(projector_dist_dir, "index.html")
        if os.path.isfile(projector_index_path):
            return FileResponse(projector_index_path)
        return HTMLResponse(
            (
                "<h2>Projector frontend not built yet.</h2>"
                "<p>Run <code>npm install && npm run build</code> under <code>frontend/</code>, "
                "or rebuild the Docker image to generate <code>/projector</code> assets.</p>"
            ),
            status_code=503,
        )

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

    @app.post("/v1/embeddings/projector")
    async def projector_embeddings(request: ProjectorRequest) -> JSONResponse:
        try:
            response_payload = await create_projector_payload(
                request.model_dump(exclude_none=True),
                embedder=create_embeddings,
            )
        except InputValidationError as exc:
            return _error_response(str(exc), 400, "invalid_request_error", 400)
        except BackendUnavailableError as exc:
            return _error_response(str(exc), 503, "service_unavailable", 503)
        except BackendProxyError as exc:
            if exc.payload is not None:
                return JSONResponse(exc.payload, status_code=exc.status_code)
            return _error_response(str(exc), exc.status_code, "backend_error", exc.status_code)
        except ProjectorDependencyError as exc:
            return _error_response(str(exc), 503, "service_unavailable", 503)
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
