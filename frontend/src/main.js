import Plotly from "plotly.js-dist-min";
import "./styles.css";

const app = document.getElementById("app");

app.innerHTML = `
  <div class="page">
    <aside class="panel left">
      <h2>Projector Controls</h2>
      <div class="hint">输入文本后调用 <code>/v1/embeddings/projector</code>，后端预计算投影，前端用 3D 点云渲染。</div>

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
          <input id="pointSize" type="number" min="1" max="64" step="0.5" value="5" />
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

      <div class="hint">交互：左键拖拽旋转，滚轮缩放，右键拖拽平移。显示原点与“原点→点”箭头连线，并标注每个点编号。</div>
    </aside>

    <main class="panel center">
      <div class="topline">
        <div>
          <div class="title">Embedding Projector</div>
          <div class="subtitle">3D scatter + nearest-neighbor explorer</div>
        </div>
        <div class="badges">
          <span class="badge" id="badgeModel">model: -</span>
          <span class="badge warn" id="badgeState">status: idle</span>
          <span class="badge" id="badgeCount">points: 0</span>
          <span class="badge" id="badgeLatency">latency: -</span>
        </div>
      </div>
      <div class="canvas-wrap">
        <div id="plot"></div>
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
  plotPoints: [],
  neighbors: {},
  labelsByIndex: [],
  categoriesByIndex: [],
  running: false,
  currentResponse: null,
  pointSize: 5,
  selectedIndex: null,
  plotEventsBound: false,
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
    point_size: Number(els.pointSize.value || 5),
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

function clampToPlotRange(value) {
  if (!Number.isFinite(value)) return 0;
  return Math.max(-1, Math.min(1, value));
}

function normalizePlotPoints(points, labels) {
  const categoryIndex = makeColorIndex(labels);
  return points.map((point, index) => ({
    index,
    text: point.text,
    label: point.label,
    category: categoryIndex[index] || 0,
    x: clampToPlotRange(Number.isFinite(point.normalized_x) ? point.normalized_x : point.x),
    y: clampToPlotRange(Number.isFinite(point.normalized_y) ? point.normalized_y : point.y),
    z: clampToPlotRange(
      Number.isFinite(point.normalized_z)
        ? point.normalized_z
        : Number.isFinite(point.z)
          ? point.z
          : 0
    ),
  }));
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

function getRelatedSet(index) {
  if (!Number.isInteger(index) || index < 0) return new Set();
  const related = [index, ...(state.neighbors[String(index)] || []).map((item) => item.index)];
  return new Set(related.filter((item) => Number.isInteger(item) && item >= 0 && item < state.plotPoints.length));
}

function buildMarkerStyle(selectedIndex) {
  const relatedSet = getRelatedSet(selectedIndex);
  const sizes = [];
  const colors = [];

  for (let i = 0; i < state.plotPoints.length; i += 1) {
    const category = state.categoriesByIndex[i] || 0;
    const baseColor = palette[category % palette.length];

    let color = baseColor;
    let size = state.pointSize;

    if (Number.isInteger(selectedIndex) && i === selectedIndex) {
      color = "#f8fafc";
      size = Math.max(state.pointSize + 3, state.pointSize * 1.8);
    } else if (relatedSet.has(i)) {
      color = "#fde68a";
      size = Math.max(state.pointSize + 2, state.pointSize * 1.4);
    }

    sizes.push(size);
    colors.push(color);
  }

  return { sizes, colors };
}

function ensurePlotEvents() {
  if (state.plotEventsBound) return;

  els.plot.on("plotly_click", (event) => {
    const point = event?.points?.[0];
    const index = point?.customdata;
    if (Number.isInteger(index)) {
      highlightPoint(index);
    }
  });

  els.plot.on("plotly_hover", (event) => {
    const point = event?.points?.[0];
    const index = point?.customdata;
    if (Number.isInteger(index)) {
      renderSelected(index);
    }
  });

  state.plotEventsBound = true;
}

function buildOriginRayLineTrace() {
  const x = [];
  const y = [];
  const z = [];
  for (const point of state.plotPoints) {
    x.push(0, point.x, null);
    y.push(0, point.y, null);
    z.push(0, point.z, null);
  }
  return {
    type: "scatter3d",
    mode: "lines",
    x,
    y,
    z,
    hoverinfo: "skip",
    line: {
      color: "rgba(56,189,248,0.45)",
      width: 3,
    },
  };
}

function buildOriginRayArrowTrace() {
  const x = [];
  const y = [];
  const z = [];
  const u = [];
  const v = [];
  const w = [];

  for (const point of state.plotPoints) {
    const length = Math.sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
    if (length < 1e-6) continue;
    const ux = point.x / length;
    const uy = point.y / length;
    const uz = point.z / length;
    const arrowLength = Math.min(0.18, Math.max(0.09, length * 0.28));
    x.push(point.x);
    y.push(point.y);
    z.push(point.z);
    u.push(ux * arrowLength);
    v.push(uy * arrowLength);
    w.push(uz * arrowLength);
  }

  if (!x.length) return null;
  return {
    type: "cone",
    x,
    y,
    z,
    u,
    v,
    w,
    anchor: "tip",
    hoverinfo: "skip",
    showscale: false,
    sizemode: "absolute",
    sizeref: 0.12,
    colorscale: [
      [0, "#22d3ee"],
      [1, "#22d3ee"],
    ],
    lighting: {
      ambient: 0.72,
      diffuse: 0.9,
      specular: 0.28,
      roughness: 0.45,
      fresnel: 0.25,
    },
  };
}

function buildOriginTrace() {
  return {
    type: "scatter3d",
    mode: "markers+text",
    x: [0],
    y: [0],
    z: [0],
    text: ["Origin"],
    textposition: "top center",
    textfont: { size: 12, color: "#67e8f9" },
    hovertemplate: "Origin (0, 0, 0)<extra></extra>",
    marker: {
      size: Math.max(6, state.pointSize + 3),
      color: "#22d3ee",
      line: { width: 1.2, color: "rgba(8,47,73,0.95)" },
      opacity: 1,
    },
  };
}

async function render3DPlot(selectedIndex = null) {
  if (!state.plotPoints.length) {
    Plotly.purge(els.plot);
    state.plotEventsBound = false;
    return;
  }

  const markerStyle = buildMarkerStyle(selectedIndex);
  const x = state.plotPoints.map((item) => item.x);
  const y = state.plotPoints.map((item) => item.y);
  const z = state.plotPoints.map((item) => item.z);
  const text = state.plotPoints.map((item) => `#${item.index}`);
  const customdata = state.plotPoints.map((item) => item.index);
  const hovertext = state.plotPoints.map((item) => item.text || `#${item.index}`);
  const originRayLineTrace = buildOriginRayLineTrace();
  const originRayArrowTrace = buildOriginRayArrowTrace();
  const originTrace = buildOriginTrace();

  const data = [];
  data.push({
    type: "scatter3d",
    mode: "markers+text",
    x,
    y,
    z,
    text,
    textposition: "top center",
    textfont: { size: 11, color: "#cbd5e1" },
    customdata,
    hovertext,
    hovertemplate: "#%{customdata}<br>%{hovertext}<extra></extra>",
    marker: {
      size: markerStyle.sizes,
      color: markerStyle.colors,
      opacity: 0.96,
      line: { width: 1.2, color: "rgba(15,23,42,0.95)" },
    },
  });
  data.push(originRayLineTrace);
  if (originRayArrowTrace) data.push(originRayArrowTrace);
  data.push(originTrace);

  const layout = {
    autosize: true,
    showlegend: false,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { l: 0, r: 0, t: 0, b: 0 },
    uirevision: "projector-3d",
    scene: {
      dragmode: "orbit",
      bgcolor: "rgba(2,6,23,0.72)",
      xaxis: { visible: false, showbackground: false },
      yaxis: { visible: false, showbackground: false },
      zaxis: { visible: false, showbackground: false },
      camera: { eye: { x: 1.45, y: 1.2, z: 1.15 } },
      aspectmode: "cube",
    },
  };

  const config = {
    responsive: true,
    displaylogo: false,
    scrollZoom: true,
    modeBarButtonsToRemove: ["lasso3d", "select2d", "select3d"],
  };

  await Plotly.react(els.plot, data, layout, config);
  ensurePlotEvents();
}

function updatePlotHighlight(index) {
  if (!state.plotPoints.length) return;
  const markerStyle = buildMarkerStyle(index);
  Plotly.restyle(
    els.plot,
    {
      "marker.size": [markerStyle.sizes],
      "marker.color": [markerStyle.colors],
    },
    [0]
  );
}

function highlightPoint(index) {
  if (!Number.isInteger(index) || index < 0 || index >= state.plotPoints.length) {
    return;
  }
  state.selectedIndex = index;
  renderSelected(index);
  updatePlotHighlight(index);
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
    state.plotPoints = normalizePlotPoints(state.points, state.labelsByIndex);
    state.categoriesByIndex = state.plotPoints.map((point) => point.category || 0);
    state.pointSize = Math.max(3, Number(payload.point_size || 5));
    state.selectedIndex = null;

    await render3DPlot(null);

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
  els.pointSize.value = "5";
  els.inputType.value = "document";
}

els.runBtn.addEventListener("click", runProjector);
els.demoBtn.addEventListener("click", fillDemo);
els.homeBtn.addEventListener("click", () => {
  window.location.href = "/";
});

fillDemo();
