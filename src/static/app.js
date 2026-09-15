import { Chess } from "./vendor/chess-js/chess.js";

const API_BASE = "";
// Elo points; bars saturate at this weighted score. Logged real predictions
// (logs/predictions.jsonl) show per-side scores from ~90 up to ~845, so 400
// sits mid-range: clearly-elevated games still read as "high" without every
// ordinary game landing near full.
const SUSPICION_BAR_MAX = 400;
const AUTOPLAY_INTERVAL_MS = 1000;

const state = {
  activeTab: "paste",
  result: null,
  fenAtPly: ["start"],
  currentPly: 0,
  chart: null,
  board: null,
  autoplayTimer: null,
  attentionRanks: new Map(),
  criticalByPly: new Map(),
  criticalCounts: { white: 0, black: 0 },
};

function $(id) {
  return document.getElementById(id);
}

function setupTabs() {
  const buttons = document.querySelectorAll(".tab-button");
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const tab = button.dataset.tab;
      state.activeTab = tab;
      document.querySelectorAll(".tab-button").forEach((b) => {
        b.classList.toggle("active", b === button);
        b.setAttribute("aria-selected", b === button ? "true" : "false");
      });
      document.querySelectorAll(".tab-panel").forEach((panel) => {
        panel.classList.toggle("active", panel.dataset.panel === tab);
      });
      clearError();
    });
  });
}

function setupInputPanelToggle() {
  $("input-panel-toggle").addEventListener("click", () => {
    const panel = $("input-panel");
    const collapsed = panel.classList.toggle("collapsed");
    $("input-panel-toggle").setAttribute("aria-expanded", collapsed ? "false" : "true");
  });
}

function setupBoard() {
  state.board = window.Chessboard("board", {
    position: "start",
    pieceTheme: "/static/vendor/chessboard-js/img/chesspieces/wikipedia/{piece}.png",
  });
}

function clearError() {
  const el = $("error-message");
  el.hidden = true;
  el.textContent = "";
}

function showError(message) {
  const el = $("error-message");
  el.textContent = message;
  el.hidden = false;
}

function setLoading(isLoading) {
  const button = $("submit-button");
  button.disabled = isLoading;
  button.textContent = isLoading ? "Analyzing..." : "Analyze game";
}

async function parseErrorDetail(response) {
  try {
    const body = await response.json();
    if (body && typeof body.detail === "string") {
      return body.detail;
    }
    return JSON.stringify(body);
  } catch (err) {
    return `Request failed with status ${response.status}`;
  }
}

function currentTopK() {
  const raw = parseInt($("top-k").value, 10);
  if (Number.isNaN(raw) || raw < 1) return 5;
  return Math.min(raw, 20);
}

function currentMinPly() {
  const raw = parseInt($("min-ply").value, 10);
  if (Number.isNaN(raw) || raw < 0) return 10;
  return Math.min(raw, 100);
}

function currentBaselines() {
  const whiteRaw = $("white-baseline").value.trim();
  const blackRaw = $("black-baseline").value.trim();
  return {
    white_baseline: whiteRaw === "" ? null : Number(whiteRaw),
    black_baseline: blackRaw === "" ? null : Number(blackRaw),
  };
}

async function submitAnalysis() {
  clearError();
  const topK = currentTopK();
  const minPly = currentMinPly();
  const baselines = currentBaselines();
  const query = `top_k=${topK}&min_ply=${minPly}`;

  setLoading(true);
  let response;
  try {
    if (state.activeTab === "paste") {
      const pgn = $("pgn-text").value.trim();
      if (!pgn) {
        showError("Paste a PGN first.");
        setLoading(false);
        return;
      }
      response = await fetch(`${API_BASE}/predict/pgn?${query}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pgn, ...baselines }),
      });
    } else if (state.activeTab === "upload") {
      const fileInput = $("pgn-file");
      if (!fileInput.files || fileInput.files.length === 0) {
        showError("Choose a .pgn file first.");
        setLoading(false);
        return;
      }
      const formData = new FormData();
      formData.append("file", fileInput.files[0]);
      if (baselines.white_baseline !== null) formData.append("white_baseline", baselines.white_baseline);
      if (baselines.black_baseline !== null) formData.append("black_baseline", baselines.black_baseline);
      response = await fetch(`${API_BASE}/predict/upload?${query}`, {
        method: "POST",
        body: formData,
      });
    } else {
      const gameId = $("lichess-id").value.trim();
      if (!gameId) {
        showError("Enter a Lichess game ID or URL first.");
        setLoading(false);
        return;
      }
      response = await fetch(`${API_BASE}/predict/lichess?${query}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ game_id: gameId, ...baselines }),
      });
    }
  } catch (networkErr) {
    showError(`Could not reach the API: ${networkErr.message}`);
    setLoading(false);
    return;
  }

  if (!response.ok) {
    showError(await parseErrorDetail(response));
    setLoading(false);
    return;
  }

  const result = await response.json();
  setLoading(false);
  renderResult(result);
}

function buildFenTimeline(headers, perMove) {
  const chess = new Chess();
  const fens = ["start"];
  for (const move of perMove) {
    const uci = move.uci;
    if (!uci) {
      fens.push(fens[fens.length - 1]);
      continue;
    }
    const from = uci.slice(0, 2);
    const to = uci.slice(2, 4);
    const promotion = uci.length > 4 ? uci.slice(4, 5) : undefined;
    try {
      chess.move({ from, to, promotion });
    } catch (err) {
      // If replay ever desyncs from the model's move list, keep the board
      // showing the last good position rather than crashing the page.
    }
    fens.push(chess.fen());
  }
  return fens;
}

function buildAttentionRanks(perMove) {
  const withWeights = perMove.filter((m) => typeof m.attention_weight === "number");
  const total = withWeights.length;
  const sorted = [...withWeights].sort((a, b) => b.attention_weight - a.attention_weight);
  const map = new Map();
  sorted.forEach((m, i) => {
    const rank = i + 1;
    const percentile = Math.max(1, Math.ceil((rank / total) * 100));
    map.set(m.ply, { rank, total, percentile });
  });
  return map;
}

function buildCriticalIndex(criticalMoves) {
  const counts = { white: 0, black: 0 };
  const map = new Map();
  ["white", "black"].forEach((side) => {
    const sideMoves = criticalMoves.filter((m) => m.side === side);
    counts[side] = sideMoves.length;
    sideMoves.forEach((m, i) => {
      const entry = map.get(m.ply) || {};
      entry[side] = i + 1;
      map.set(m.ply, entry);
    });
  });
  return { map, counts };
}

function renderResult(result) {
  state.result = result;
  state.fenAtPly = buildFenTimeline(result.headers, result.per_move);
  state.currentPly = state.fenAtPly.length - 1;
  state.attentionRanks = buildAttentionRanks(result.per_move);

  const criticalIndex = buildCriticalIndex(result.critical_moves || []);
  state.criticalByPly = criticalIndex.map;
  state.criticalCounts = criticalIndex.counts;

  stopAutoplay();

  // chessboard.js sizes itself from the container's rendered width at call
  // time; the container is still `hidden` (0 width) until this point, so it
  // must be explicitly resized once the panel becomes visible.
  $("results-panel").hidden = false;
  state.board.orientation("white");
  state.board.resize();

  const headers = result.headers || {};
  $("results-title").textContent = `${headers.White || "White"} vs ${headers.Black || "Black"}`;
  $("provisional-badge").hidden = !result.provisional;
  $("min-ply-caption").textContent = result.critical_moves_min_ply;

  renderWarnings(result.warnings || []);
  renderSuspicion(result);
  renderChart(result);
  renderMoveList(result.per_move);
  renderBoardAtPly(state.currentPly);

  $("input-panel").classList.add("collapsed");
  $("input-panel-toggle").setAttribute("aria-expanded", "false");
}

function renderWarnings(warnings) {
  const box = $("warnings-box");
  if (!warnings.length) {
    box.hidden = true;
    box.innerHTML = "";
    return;
  }
  box.hidden = false;
  const items = warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("");
  box.innerHTML = `<strong>Warnings</strong><ul>${items}</ul>`;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function renderSuspicion(result) {
  const whiteScore = result.white_suspicion_score;
  const blackScore = result.black_suspicion_score;

  $("white-suspicion-bar").style.width = `${Math.min(100, (whiteScore / SUSPICION_BAR_MAX) * 100)}%`;
  $("black-suspicion-bar").style.width = `${Math.min(100, (blackScore / SUSPICION_BAR_MAX) * 100)}%`;
  $("white-suspicion-value").textContent = whiteScore.toFixed(1);
  $("black-suspicion-value").textContent = blackScore.toFixed(1);

  $("white-baseline-source").textContent = `${result.white_baseline} (${formatSource(result.white_baseline_source)})`;
  $("black-baseline-source").textContent = `${result.black_baseline} (${formatSource(result.black_baseline_source)})`;

  $("white-suspicion-explain").textContent =
    `About ${Math.round(whiteScore)} rating points of attention-weighted gap between the ` +
    `model's estimate and White's baseline.`;
  $("black-suspicion-explain").textContent =
    `About ${Math.round(blackScore)} rating points of attention-weighted gap between the ` +
    `model's estimate and Black's baseline.`;
}

function formatSource(source) {
  switch (source) {
    case "request":
      return "user-supplied";
    case "pgn_header":
      return "PGN header";
    case "self_prediction_fallback":
      return "model self-prediction, less reliable";
    default:
      return source;
  }
}

function buildSyncPlugin() {
  return {
    id: "plySync",
    afterDatasetsDraw(chart) {
      if (!state.result) return;
      const ply = state.currentPly;
      const ctx = chart.ctx;
      const area = chart.chartArea;

      // Small markers for critical plies, drawn under the current-ply dot.
      [
        ["white", 0, "#3a6ea5"],
        ["black", 1, "#a54a3a"],
      ].forEach(([side, dsIndex, color]) => {
        const meta = chart.getDatasetMeta(dsIndex);
        if (!meta || !meta.data) return;
        state.criticalByPly.forEach((info, criticalPly) => {
          if (info[side] == null) return;
          const point = meta.data[criticalPly];
          if (!point) return;
          ctx.save();
          ctx.beginPath();
          ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
          ctx.fillStyle = "#fff";
          ctx.fill();
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = color;
          ctx.stroke();
          ctx.restore();
        });
      });

      // Vertical line at the current ply.
      const meta0 = chart.getDatasetMeta(0);
      const linePoint = meta0 && meta0.data[ply];
      if (linePoint) {
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(linePoint.x, area.top);
        ctx.lineTo(linePoint.x, area.bottom);
        ctx.setLineDash([4, 3]);
        ctx.lineWidth = 1;
        ctx.strokeStyle = "rgba(60, 60, 60, 0.45)";
        ctx.stroke();
        ctx.restore();
      }

      // Current-ply dot on each rating curve.
      [0, 1].forEach((dsIndex) => {
        const meta = chart.getDatasetMeta(dsIndex);
        const point = meta && meta.data[ply];
        if (!point) return;
        ctx.save();
        ctx.beginPath();
        ctx.arc(point.x, point.y, 5, 0, Math.PI * 2);
        ctx.fillStyle = dsIndex === 0 ? "#3a6ea5" : "#a54a3a";
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = "#fff";
        ctx.fill();
        ctx.stroke();
        ctx.restore();
      });
    },
  };
}

function renderChart(result) {
  const perMove = result.per_move;
  const labels = [0, ...perMove.map((m) => m.ply)];
  const whiteRatings = [result.white_baseline, ...perMove.map((m) => m.white_rating)];
  const blackRatings = [result.black_baseline, ...perMove.map((m) => m.black_rating)];
  const whiteBaselineLine = labels.map(() => result.white_baseline);
  const blackBaselineLine = labels.map(() => result.black_baseline);

  const ctx = $("rating-chart").getContext("2d");
  if (state.chart) {
    state.chart.destroy();
  }
  state.chart = new window.Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "White rating estimate",
          data: whiteRatings,
          borderColor: "#3a6ea5",
          backgroundColor: "transparent",
          pointRadius: 0,
          borderWidth: 2,
        },
        {
          label: "Black rating estimate",
          data: blackRatings,
          borderColor: "#a54a3a",
          backgroundColor: "transparent",
          pointRadius: 0,
          borderWidth: 2,
        },
        {
          label: "White baseline",
          data: whiteBaselineLine,
          borderColor: "#3a6ea5",
          borderDash: [6, 4],
          borderWidth: 1,
          pointRadius: 0,
        },
        {
          label: "Black baseline",
          data: blackBaselineLine,
          borderColor: "#a54a3a",
          borderDash: [6, 4],
          borderWidth: 1,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { title: { display: true, text: "Ply" } },
        y: { title: { display: true, text: "Estimated rating" } },
      },
    },
    plugins: [buildSyncPlugin()],
  });
}

function jumpFromChartEvent(evt) {
  if (!state.chart) return;
  const points = state.chart.getElementsAtEventForMode(evt, "index", { intersect: false }, true);
  if (points.length) {
    renderBoardAtPly(points[0].index);
  }
}

function setupChartPointerNav() {
  const canvas = $("rating-chart");
  let dragging = false;

  canvas.addEventListener("pointerdown", (evt) => {
    dragging = true;
    jumpFromChartEvent(evt);
  });
  window.addEventListener("pointermove", (evt) => {
    if (dragging) jumpFromChartEvent(evt);
  });
  window.addEventListener("pointerup", () => {
    dragging = false;
  });
}

function clearSquareHighlights() {
  document.querySelectorAll(".highlight-from, .highlight-to").forEach((el) => {
    el.classList.remove("highlight-from", "highlight-to");
  });
}

function highlightMoveSquares(uci) {
  clearSquareHighlights();
  if (!uci || uci.length < 4) return;
  const from = uci.slice(0, 2);
  const to = uci.slice(2, 4);
  const fromEl = document.querySelector(`.square-${from}`);
  const toEl = document.querySelector(`.square-${to}`);
  if (fromEl) fromEl.classList.add("highlight-from");
  if (toEl) toEl.classList.add("highlight-to");
}

function renderPlayerBars(ply) {
  const result = state.result;
  const headers = result.headers || {};
  const whiteName = headers.White || "White";
  const blackName = headers.Black || "Black";
  const whiteCurrent = ply === 0 ? result.white_baseline : result.per_move[ply - 1].white_rating;
  const blackCurrent = ply === 0 ? result.black_baseline : result.per_move[ply - 1].black_rating;

  const perSide = {
    white: { name: whiteName, baseline: result.white_baseline, current: whiteCurrent },
    black: { name: blackName, baseline: result.black_baseline, current: blackCurrent },
  };

  const orientation = state.board.orientation();
  const topSide = orientation === "white" ? "black" : "white";
  const bottomSide = orientation === "white" ? "white" : "black";

  setPlayerBar("top", perSide[topSide]);
  setPlayerBar("bottom", perSide[bottomSide]);
}

function setPlayerBar(position, info) {
  $(`bar-${position}-name`).textContent = info.name;
  $(`bar-${position}-baseline`).textContent = Math.round(info.baseline);
  $(`bar-${position}-current`).textContent = Math.round(info.current);
}

function formatDelta(delta) {
  if (delta === null || delta === undefined || Number.isNaN(delta)) return "";
  const rounded = Math.round(delta);
  if (rounded === 0) return " (no change)";
  const sign = rounded > 0 ? "+" : "";
  return ` (${sign}${rounded})`;
}

function renderMetricsPanel(ply) {
  const container = $("metrics-content");
  const result = state.result;

  if (ply === 0) {
    container.innerHTML = `
      <p class="metrics-move-label">Start of game</p>
      <dl class="metrics-grid">
        <dt>White baseline</dt><dd>${Math.round(result.white_baseline)}</dd>
        <dt>Black baseline</dt><dd>${Math.round(result.black_baseline)}</dd>
      </dl>
      <p class="metrics-hint">Step forward to see per-move rating estimates and attention.</p>
    `;
    return;
  }

  const move = result.per_move[ply - 1];
  const prev = ply >= 2 ? result.per_move[ply - 2] : null;
  const whiteDelta = prev ? move.white_rating - prev.white_rating : null;
  const blackDelta = prev ? move.black_rating - prev.black_rating : null;
  const sideToMove = ply % 2 === 1 ? "White" : "Black";

  const attentionInfo = state.attentionRanks.get(ply);
  const attentionLine =
    attentionInfo && typeof move.attention_weight === "number"
      ? `${(move.attention_weight * 100).toFixed(2)}% of total attention ` +
        `(rank ${attentionInfo.rank} of ${attentionInfo.total}, top ${attentionInfo.percentile}%)`
      : "not available for this checkpoint";

  const critical = state.criticalByPly.get(ply) || {};
  const criticalLines = [];
  if (critical.white != null) {
    criticalLines.push(`critical for White (rank ${critical.white} of ${state.criticalCounts.white})`);
  }
  if (critical.black != null) {
    criticalLines.push(`critical for Black (rank ${critical.black} of ${state.criticalCounts.black})`);
  }
  const criticalText = criticalLines.length ? criticalLines.join(", ") : "not flagged as critical";

  container.innerHTML = `
    <p class="metrics-move-label">Ply ${ply} &middot; ${sideToMove} played ${escapeHtml(move.move || "?")}</p>
    <dl class="metrics-grid">
      <dt>White estimate</dt><dd>${Math.round(move.white_rating)}${formatDelta(whiteDelta)}</dd>
      <dt>Black estimate</dt><dd>${Math.round(move.black_rating)}${formatDelta(blackDelta)}</dd>
      <dt>White deviation from baseline</dt><dd>${move.white_deviation.toFixed(1)}</dd>
      <dt>Black deviation from baseline</dt><dd>${move.black_deviation.toFixed(1)}</dd>
      <dt>Attention</dt><dd>${attentionLine}</dd>
      <dt>Critical move</dt><dd>${criticalText}</dd>
    </dl>
  `;
}

function markActiveMoveListEntry(ply) {
  document.querySelectorAll(".move-cell.active").forEach((el) => el.classList.remove("active"));
  const match = document.querySelector(`.move-cell[data-ply="${ply}"]`);
  if (match) {
    match.classList.add("active");
    match.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}

function buildMoveCell(move, side) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = `move-cell ${side}-move`;
  btn.dataset.ply = String(move.ply);
  const critical = state.criticalByPly.get(move.ply);
  if (critical && (critical.white != null || critical.black != null)) {
    btn.classList.add("critical");
    btn.title = "Critical move";
  }
  btn.textContent = move.move || "?";
  btn.addEventListener("click", () => renderBoardAtPly(move.ply));
  return btn;
}

function renderMoveList(perMove) {
  const container = $("move-list");
  container.innerHTML = "";
  if (!perMove.length) {
    container.innerHTML = '<p class="move-list-empty">No moves to show.</p>';
    return;
  }
  for (let i = 0; i < perMove.length; i += 2) {
    const moveNumber = Math.floor(i / 2) + 1;
    const whiteMove = perMove[i];
    const blackMove = perMove[i + 1];

    const row = document.createElement("div");
    row.className = "move-row";
    row.setAttribute("role", "listitem");

    const numEl = document.createElement("span");
    numEl.className = "move-number";
    numEl.textContent = `${moveNumber}.`;
    row.appendChild(numEl);

    row.appendChild(buildMoveCell(whiteMove, "white"));
    if (blackMove) {
      row.appendChild(buildMoveCell(blackMove, "black"));
    } else {
      const empty = document.createElement("span");
      empty.className = "move-cell empty";
      row.appendChild(empty);
    }

    container.appendChild(row);
  }
}

function updatePlyIndicator() {
  $("ply-indicator").textContent = `ply ${state.currentPly} / ${state.fenAtPly.length - 1}`;
}

function renderBoardAtPly(ply, opts = {}) {
  if (!opts.fromAutoplay) {
    stopAutoplay();
  }
  const clamped = Math.max(0, Math.min(ply, state.fenAtPly.length - 1));
  state.currentPly = clamped;
  state.board.position(state.fenAtPly[clamped], false);
  updatePlyIndicator();

  if (clamped === 0) {
    clearSquareHighlights();
  } else {
    const move = state.result.per_move[clamped - 1];
    highlightMoveSquares(move.uci);
  }

  renderPlayerBars(clamped);
  renderMetricsPanel(clamped);
  markActiveMoveListEntry(clamped);
  if (state.chart) {
    state.chart.update("none");
  }
}

function stopAutoplay() {
  if (state.autoplayTimer) {
    clearInterval(state.autoplayTimer);
    state.autoplayTimer = null;
  }
  const button = $("play-pause");
  if (button) {
    button.textContent = "Play";
    button.setAttribute("aria-pressed", "false");
  }
}

function startAutoplay() {
  if (!state.result) return;
  if (state.currentPly >= state.fenAtPly.length - 1) {
    renderBoardAtPly(0);
  }
  $("play-pause").textContent = "Pause";
  $("play-pause").setAttribute("aria-pressed", "true");
  state.autoplayTimer = setInterval(() => {
    if (state.currentPly >= state.fenAtPly.length - 1) {
      stopAutoplay();
      return;
    }
    renderBoardAtPly(state.currentPly + 1, { fromAutoplay: true });
  }, AUTOPLAY_INTERVAL_MS);
}

function toggleAutoplay() {
  if (state.autoplayTimer) {
    stopAutoplay();
  } else {
    startAutoplay();
  }
}

function flipBoard() {
  state.board.flip();
  renderPlayerBars(state.currentPly);
}

function setupBoardControls() {
  $("step-start").addEventListener("click", () => renderBoardAtPly(0));
  $("step-back").addEventListener("click", () => renderBoardAtPly(state.currentPly - 1));
  $("step-forward").addEventListener("click", () => renderBoardAtPly(state.currentPly + 1));
  $("step-end").addEventListener("click", () => renderBoardAtPly(state.fenAtPly.length - 1));
  $("play-pause").addEventListener("click", toggleAutoplay);
  $("flip-board").addEventListener("click", flipBoard);
}

function setupKeyboardNav() {
  document.addEventListener("keydown", (evt) => {
    const tag = evt.target && evt.target.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    if (!state.result) return;

    switch (evt.key) {
      case "ArrowLeft":
        evt.preventDefault();
        renderBoardAtPly(state.currentPly - 1);
        break;
      case "ArrowRight":
        evt.preventDefault();
        renderBoardAtPly(state.currentPly + 1);
        break;
      case "ArrowUp":
      case "Home":
        evt.preventDefault();
        renderBoardAtPly(0);
        break;
      case "ArrowDown":
      case "End":
        evt.preventDefault();
        renderBoardAtPly(state.fenAtPly.length - 1);
        break;
      default:
        break;
    }
  });
}

function init() {
  setupTabs();
  setupInputPanelToggle();
  setupBoard();
  setupBoardControls();
  setupChartPointerNav();
  setupKeyboardNav();
  $("submit-button").addEventListener("click", submitAnalysis);
}

document.addEventListener("DOMContentLoaded", init);
