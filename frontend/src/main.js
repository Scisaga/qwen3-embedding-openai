import createScatterplot from "regl-scatterplot";
import "./styles.css";

const app = document.getElementById("app");

app.innerHTML = `
  <div class="page">
    <aside class="panel left">
      <h2>Projector Controls</h2>
      <div class="hint">输入文本后调用 <code>/v1/embeddings/projector</code>，后端预计算投影，前端用 WebGL 渲染点云。</div>

      <label for="inputs">Texts (one line per item)</label>
      <textarea id="inputs">What is the capital of China?
The capital of China is Beijing.
The Eiffel Tower is located in Paris.
Shanghai is a major financial center in China.</textarea>

      <div class="row">
        <div>
          <label for="projectionMethod">Projection</label>
          <select id="projectionMethod">
            <option value="umap">UMAP</option>
            <option value="tsne">t-SNE</option>
            <option value="pca">PCA</option>
          </select>
        </div>
        <div>
          <label for="metric">Metric</label>
          <select id="metric">
            <option value="cosine">cosine</option>
            <option value="euclidean">euclidean</option>
          </select>
        </div>
      </div>

      <div class="row">
        <div>
          <label for="neighborsK">Neighbors K</label>
          <input id="neighborsK" type="number" min="1" max="256" value="10" />
        </div>
        <div>
          <label for="pointSize">Point Size</label>
          <input id="pointSize" type="number" min="1" max="64" step="0.5" value="4" />
        </div>
      </div>

      <div class="row">
        <div>
          <label for="inputType">Input Type</label>
          <select id="inputType">
            <option value="">document / raw</option>
            <option value="query">query</option>
            <option value="document">document</option>
          </select>
        </div>
        <div>
          <label for="model">Model (optional)</label>
          <input id="model" placeholder="Qwen/Qwen3-Embedding-8B" />
        </div>
      </div>

      <label for="instruction">Instruction (optional)</label>
      <input id="instruction" placeholder="Given a web search query, retrieve relevant passages that answer the query" />

      <label for="labels">Labels (optional, one line per item)</label>
      <textarea id="labels" placeholder="news&#10;news&#10;travel&#10;finance"></textarea>

      <div class="actions">
        <button class="primary" id="runBtn">Run Projector</button>
        <button class="secondary" id="demoBtn">Use Demo</button>
        <button class="secondary" id="homeBtn">Back To Console</button>
      </div>

      <div class="hint">交互：滚轮缩放，拖拽平移，按住 Shift 可框选（lasso）。点击点可在右侧查看近邻。</div>
    </aside>

    <main class="panel center">
      <div class="topline">
        <div>
          <div class="title">Embedding Projector</div>
          <div class="subtitle">WebGL scatter + nearest-neighbor explorer</div>
        </div>
        <div class="badges">
          <span class="badge" id="badgeModel">model: -</span>
          <span class="badge warn" id="badgeState">status: idle</span>
          <span class="badge" id="badgeCount">points: 0</span>
          <span class="badge" id="badgeLatency">latency: -</span>
        </div>
      </div>
      <div class="canvas-wrap">
        <canvas id="plot"></canvas>
      </div>
    </main>

    <aside class="panel right">
      <div id="status" class="status warn">Ready to run projector request.</div>
      <div class="card">
        <div class="k">Selected Point</div>
        <div class="v" id="selectedPoint">None</div>
      </div>
      <div class="card">
        <div class="k">Nearest Neighbors</div>
        <div id="neighbors" class="empty">None</div>
      </div>
      <div class="card">
        <div class="k">Projection Meta</div>
        <pre id="metaOut">{}</pre>
      </div>
      <div class="card">
        <div class="k">Raw Response</div>
        <pre id="rawOut">{}</pre>
      </div>
    </aside>
  </div>
`;

const els = {
  inputs: document.getElementById("inputs"),
  projectionMethod: document.getElementById("projectionMethod"),
  metric: document.getElementById("metric"),
  neighborsK: document.getElementById("neighborsK"),
  pointSize: document.getElementById("pointSize"),
  inputType: document.getElementById("inputType"),
  model: document.getElementById("model"),
  instruction: document.getElementById("instruction"),
  labels: document.getElementById("labels"),
  runBtn: document.getElementById("runBtn"),
  demoBtn: document.getElementById("demoBtn"),
  homeBtn: document.getElementById("homeBtn"),
  plot: document.getElementById("plot"),
  status: document.getElementById("status"),
  selectedPoint: document.getElementById("selectedPoint"),
  neighbors: document.getElementById("neighbors"),
  metaOut: document.getElementById("metaOut"),
  rawOut: document.getElementById("rawOut"),
  badgeModel: document.getElementById("badgeModel"),
  badgeState: document.getElementById("badgeState"),
  badgeCount: document.getElementById("badgeCount"),
  badgeLatency: document.getElementById("badgeLatency"),
};

const state = {
  points: [],
  drawPoints: [],
  neighbors: {},
  labelsByIndex: [],
  scatterplot: null,
  running: false,
  currentResponse: null,
};

const palette = [
  "#60a5fa",
  "#34d399",
  "#f59e0b",
  "#f472b6",
  "#a78bfa",
  "#22d3ee",
  "#e879f9",
  "#f87171",
  "#84cc16",
  "#c084fc",
];

function setStatus(level, text) {
  els.status.classList.remove("ok", "warn", "bad");
  els.status.classList.add(level);
  els.status.textContent = text;

  els.badgeState.classList.remove("ok", "warn", "bad");
  els.badgeState.classList.add(level);
  els.badgeState.textContent = `status: ${text}`;
}

function splitLines(text) {
  return text
    .split(/\n+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function buildPayload() {
  const inputs = splitLines(els.inputs.value);
  if (!inputs.length) return null;

  const payload = {
    inputs,
    projection_method: els.projectionMethod.value,
    metric: els.metric.value,
    neighbors_k: Number(els.neighborsK.value || 10),
    point_size: Number(els.pointSize.value || 4),
  };

  if (els.inputType.value) payload.input_type = els.inputType.value;
  if (els.model.value.trim()) payload.model = els.model.value.trim();
  if (els.instruction.value.trim()) payload.instruction = els.instruction.value.trim();

  const labels = splitLines(els.labels.value);
  if (labels.length) {
    if (labels.length === inputs.length) {
      payload.labels = labels;
    } else {
      setStatus("warn", "labels count != inputs count, labels ignored");
    }
  }
  return payload;
}

function makeColorIndex(labels) {
  const categoryMap = new Map();
  let cursor = 0;
  return labels.map((label) => {
    const key = label || "unlabeled";
    if (!categoryMap.has(key)) {
      categoryMap.set(key, cursor++);
    }
    return categoryMap.get(key);
  });
}

function normalizeDrawPoints(points, labels) {
  const categoryIndex = makeColorIndex(labels);
  return points.map((point, index) => [
    Number.isFinite(point.normalized_x) ? point.normalized_x : point.x,
    Number.isFinite(point.normalized_y) ? point.normalized_y : point.y,
    categoryIndex[index] || 0,
    0.75,
  ]);
}

function renderSelected(index) {
  if (!Number.isInteger(index) || index < 0 || index >= state.points.length) {
    els.selectedPoint.textContent = "None";
    els.neighbors.innerHTML = "<div class='empty'>None</div>";
    return;
  }

  const point = state.points[index];
  els.selectedPoint.textContent = `[${point.index}] ${point.text}`;
  const nn = state.neighbors[String(index)] || [];
  if (!nn.length) {
    els.neighbors.innerHTML = "<div class='empty'>No neighbors</div>";
    return;
  }

  const list = nn
    .map((item) => {
      const peer = state.points[item.index];
      const preview = peer ? peer.text : `index ${item.index}`;
      return `<li><strong>#${item.index}</strong> sim=${item.similarity.toFixed(4)}<br/>${preview}</li>`;
    })
    .join("");
  els.neighbors.innerHTML = `<ul>${list}</ul>`;
}

function highlightPoint(index) {
  if (!Number.isInteger(index) || index < 0 || !state.scatterplot || !state.drawPoints.length) {
    return;
  }
  const related = [index, ...(state.neighbors[String(index)] || []).map((item) => item.index)];
  state.scatterplot.draw(state.drawPoints, { selected: related });
  renderSelected(index);
}

function bindScatterplotEvents(scatterplot) {
  scatterplot.subscribe("select", (event) => {
    const points = Array.isArray(event?.points) ? event.points : [];
    if (!points.length) return;
    highlightPoint(points[0]);
  });

  scatterplot.subscribe("pointover", (event) => {
    const point = event?.point;
    if (Number.isInteger(point)) {
      renderSelected(point);
    }
  });
}

function ensureScatterplot() {
  if (state.scatterplot) return state.scatterplot;

  const rect = els.plot.getBoundingClientRect();
  const scatterplot = createScatterplot({
    canvas: els.plot,
    width: rect.width,
    height: Math.max(500, rect.height),
    pointSize: Number(els.pointSize.value || 4),
    lassoType: "brush",
    showReticle: true,
  });
  bindScatterplotEvents(scatterplot);
  state.scatterplot = scatterplot;

  const observer = new ResizeObserver((entries) => {
    for (const entry of entries) {
      const width = entry.contentRect.width;
      const height = Math.max(500, entry.contentRect.height);
      scatterplot.set({ width, height });
    }
  });
  observer.observe(els.plot);

  return scatterplot;
}

async function runProjector() {
  if (state.running) return;
  const payload = buildPayload();
  if (!payload) {
    setStatus("bad", "inputs required");
    return;
  }

  state.running = true;
  els.runBtn.disabled = true;
  setStatus("warn", "requesting");
  const startedAt = performance.now();

  try {
    const response = await fetch("/v1/embeddings/projector", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result?.error?.message || `HTTP ${response.status}`);
    }

    state.currentResponse = result;
    state.points = Array.isArray(result.points) ? result.points : [];
    state.neighbors = result.neighbors || {};
    state.labelsByIndex = state.points.map((point) => point.label || "unlabeled");
    state.drawPoints = normalizeDrawPoints(state.points, state.labelsByIndex);

    const scatterplot = ensureScatterplot();
    scatterplot.set({
      pointSize: Number(payload.point_size || 4),
      pointColor: palette,
      colorBy: "valueA",
      opacityBy: "valueB",
      opacity: [0.25, 1],
      lassoType: "brush",
    });
    await scatterplot.draw(state.drawPoints);

    const latency = Math.round(performance.now() - startedAt);
    els.badgeModel.textContent = `model: ${result.model || "-"}`;
    els.badgeCount.textContent = `points: ${state.points.length}`;
    els.badgeLatency.textContent = `latency: ${latency} ms`;
    els.metaOut.textContent = JSON.stringify(result.projection_meta || {}, null, 2);
    els.rawOut.textContent = JSON.stringify(result, null, 2);
    setStatus("ok", "ready");

    if (state.points.length > 0) {
      highlightPoint(0);
    } else {
      renderSelected(-1);
    }
  } catch (error) {
    setStatus("bad", error instanceof Error ? error.message : String(error));
  } finally {
    state.running = false;
    els.runBtn.disabled = false;
  }
}

function fillDemo() {
  els.inputs.value = [
    "What is the capital of China?",
    "The capital of China is Beijing.",
    "The Eiffel Tower is located in Paris.",
    "Paris is the capital of France.",
    "Shanghai is a major financial center in China.",
    "How to improve retrieval quality for Chinese documents?",
  ].join("\n");
  els.labels.value = ["query", "fact", "fact", "fact", "fact", "query"].join("\n");
  els.projectionMethod.value = "umap";
  els.metric.value = "cosine";
  els.neighborsK.value = "10";
  els.pointSize.value = "4";
  els.inputType.value = "document";
}

els.runBtn.addEventListener("click", runProjector);
els.demoBtn.addEventListener("click", fillDemo);
els.homeBtn.addEventListener("click", () => {
  window.location.href = "/";
});

fillDemo();
