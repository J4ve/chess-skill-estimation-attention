import { Chess } from "./vendor/chess-js/chess.js";

const API_BASE = "";
const SUSPICION_BAR_MAX = 400; // Elo points; bars saturate at this weighted score.

const state = {
  activeTab: "paste",
  result: null,
  fenAtPly: ["start"],
  currentPly: 0,
  chart: null,
  board: null,
  activeCriticalMoveEl: null,
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
  const baselines = currentBaselines();

  let response;
  try {
    if (state.activeTab === "paste") {
      const pgn = $("pgn-text").value.trim();
      if (!pgn) {
        showError("Paste a PGN first.");
        return;
      }
      response = await fetch(`${API_BASE}/predict/pgn?top_k=${topK}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pgn, ...baselines }),
      });
    } else if (state.activeTab === "upload") {
      const fileInput = $("pgn-file");
      if (!fileInput.files || fileInput.files.length === 0) {
        showError("Choose a .pgn file first.");
        return;
      }
      const formData = new FormData();
      formData.append("file", fileInput.files[0]);
      if (baselines.white_baseline !== null) formData.append("white_baseline", baselines.white_baseline);
      if (baselines.black_baseline !== null) formData.append("black_baseline", baselines.black_baseline);
      response = await fetch(`${API_BASE}/predict/upload?top_k=${topK}`, {
        method: "POST",
        body: formData,
      });
    } else {
      const gameId = $("lichess-id").value.trim();
      if (!gameId) {
        showError("Enter a Lichess game ID or URL first.");
        return;
      }
      response = await fetch(`${API_BASE}/predict/lichess?top_k=${topK}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ game_id: gameId, ...baselines }),
      });
    }
  } catch (networkErr) {
    showError(`Could not reach the API: ${networkErr.message}`);
    return;
  }

  if (!response.ok) {
    showError(await parseErrorDetail(response));
    return;
  }

  const result = await response.json();
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

function renderResult(result) {
  state.result = result;
  state.fenAtPly = buildFenTimeline(result.headers, result.per_move);
  state.currentPly = state.fenAtPly.length - 1;

  // chessboard.js sizes itself from the container's rendered width at call
  // time; the container is still `hidden` (0 width) until this point, so it
  // must be explicitly resized once the panel becomes visible.
  $("results-panel").hidden = false;
  state.board.resize();

  const headers = result.headers || {};
  $("white-name").textContent = headers.White || "White";
  $("black-name").textContent = headers.Black || "Black";
  $("white-rating").textContent = `final estimate ${result.white_final_rating}`;
  $("black-rating").textContent = `final estimate ${result.black_final_rating}`;

  $("provisional-badge").hidden = !result.provisional;

  renderWarnings(result.warnings || []);
  renderSuspicion(result);
  renderChart(result);
  renderBoardAtPly(state.currentPly);
  renderCriticalMoves(result.critical_moves || []);
  updatePlyIndicator();
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

function renderChart(result) {
  const perMove = result.per_move;
  const labels = perMove.map((m) => m.ply);
  const whiteRatings = perMove.map((m) => m.white_rating);
  const blackRatings = perMove.map((m) => m.black_rating);
  const whiteBaselineLine = perMove.map(() => result.white_baseline);
  const blackBaselineLine = perMove.map(() => result.black_baseline);

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
      interaction: { mode: "index", intersect: false },
      onClick: (evt, elements, chart) => {
        const points = chart.getElementsAtEventForMode(evt, "index", { intersect: false }, true);
        if (points.length) {
          renderBoardAtPly(points[0].index + 1);
        }
      },
      scales: {
        x: { title: { display: true, text: "Ply" } },
        y: { title: { display: true, text: "Estimated rating" } },
      },
    },
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

function renderBoardAtPly(ply) {
  const clamped = Math.max(0, Math.min(ply, state.fenAtPly.length - 1));
  state.currentPly = clamped;
  state.board.position(state.fenAtPly[clamped], false);
  updatePlyIndicator();

  const perMove = state.result.per_move;
  const label = $("current-move-label");
  if (clamped === 0) {
    label.textContent = "Start of game";
    clearSquareHighlights();
  } else {
    const move = perMove[clamped - 1];
    label.textContent = `Ply ${move.ply}: ${move.move}`;
    highlightMoveSquares(move.uci);
  }
  markActiveCriticalMove(clamped);
}

function updatePlyIndicator() {
  $("ply-indicator").textContent = `ply ${state.currentPly} / ${state.fenAtPly.length - 1}`;
}

function markActiveCriticalMove(ply) {
  if (state.activeCriticalMoveEl) {
    state.activeCriticalMoveEl.classList.remove("active");
  }
  const match = document.querySelector(`.critical-move-item[data-ply="${ply}"]`);
  if (match) {
    match.classList.add("active");
    state.activeCriticalMoveEl = match;
  } else {
    state.activeCriticalMoveEl = null;
  }
}

function renderCriticalMoves(criticalMoves) {
  const list = $("critical-moves-list");
  list.innerHTML = "";
  if (!criticalMoves.length) {
    list.innerHTML = '<li class="critical-moves-empty">No critical moves to show.</li>';
    return;
  }
  criticalMoves.forEach((move) => {
    const li = document.createElement("li");
    li.className = "critical-move-item";
    li.dataset.ply = String(move.ply);
    li.innerHTML = `
      <span class="critical-move-side ${move.side}">${move.side}</span>
      <span class="critical-move-move">${escapeHtml(move.move || "?")}</span>
      <span class="critical-move-score">ply ${move.ply} &middot; score ${move.weighted_score.toFixed(2)}</span>
    `;
    li.addEventListener("click", () => renderBoardAtPly(move.ply));
    list.appendChild(li);
  });
}

function setupBoardControls() {
  $("step-start").addEventListener("click", () => renderBoardAtPly(0));
  $("step-back").addEventListener("click", () => renderBoardAtPly(state.currentPly - 1));
  $("step-forward").addEventListener("click", () => renderBoardAtPly(state.currentPly + 1));
  $("step-end").addEventListener("click", () => renderBoardAtPly(state.fenAtPly.length - 1));
}

function init() {
  setupTabs();
  setupBoard();
  setupBoardControls();
  $("submit-button").addEventListener("click", submitAnalysis);
}

document.addEventListener("DOMContentLoaded", init);
