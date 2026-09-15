import { Chess } from "./vendor/chess-js/chess.js";

const API_BASE = "";
const AUTOPLAY_INTERVAL_MS = 1000;
const HIDE_ACTUAL_RATINGS_KEY = "ratingnet.hideActualRatings";
const THEME_KEY = "ratingnet.theme";
const LIVE_RECONNECT_DELAYS_MS = [1000, 2000, 4000];
const LIVE_CLOCK_TICK_MS = 100;
const LOW_CLOCK_SECONDS = 20;

// Mirrors suspicion_labels.LABEL_TEXT in src/suspicion_labels.py. The chip
// shows the short form; the long form goes in its tooltip.
const SUSPICION_LABEL_TEXT = {
  typical: "Typical",
  unusual: "Unusual",
  highly_unusual: "Highly unusual: worth a human review",
};
const SUSPICION_LABEL_SHORT = {
  typical: "Typical",
  unusual: "Unusual",
  highly_unusual: "Highly unusual",
};

// One or two short, plain-language sentences per metric. Kept as a single
// object so wording can be edited in one place; used both for the (i)
// popovers next to each label and for the "What do these numbers mean?"
// glossary, in the same order as listed here. The page itself shows only
// short labels, so every explanation lives here.
const METRIC_INFO = {
  ratingEstimate: {
    term: "Rating estimate (est)",
    text: "The model's current guess at this player's chess rating, updated after every move. It can move up or down as the model sees more of how each side plays.",
  },
  actualRating: {
    term: "Actual rating",
    text: "The player's real rating, taken from the source game's PGN header when it recorded one. It is shown only for comparison and never fed into the estimate. The chip next to it is the estimate minus the actual rating.",
  },
  error: {
    term: "Error",
    text: "How far the estimate is from the actual rating, in rating points. Positive means the model guessed higher than the real rating, negative means lower. Final error is the error at the last move.",
  },
  baseline: {
    term: "Baseline and its source",
    text: "The pre-game rating the suspicion score compares each move against: a reviewer-entered value, the PGN header, or (least reliable) the model's own final guess.",
  },
  deviation: {
    term: "Deviation from baseline",
    text: "How many rating points this move's estimate differs from that side's baseline. Large deviations on moves the model paid a lot of attention to push the suspicion score up.",
  },
  attention: {
    term: "Attention weight and rank",
    text: "How much relative importance the model gave this move out of the whole game, and where that ranks among all moves. Higher attention means the move mattered more to the model's estimate.",
  },
  criticalMove: {
    term: "Critical move",
    text: "A move flagged for closer human review because it ranks top by attention times rating deviation. Plies before ply 10 are excluded from the ranking. It is a pointer to look at, not a verdict.",
  },
  suspicion: {
    term: "Suspicion score (S_att)",
    text: "The attention-weighted average gap, in rating points, between the model's per-move estimate and the player's baseline over the whole game. It is a supplementary flag for human review, not proof of cheating: in thesis evaluation it separated engine-substituted games only weakly (ROC-AUC 0.555 on synthetic data).",
  },
  suspicionLabel: {
    term: "Typical / Unusual / Highly unusual",
    text: "Compares this side's suspicion score with ordinary rated games from the thesis held-out test set: Typical is below the 75th percentile, Unusual is the 75th to 95th, and Highly unusual is above the 95th. It describes how uncommon the score is, not whether anyone cheated; a clean synthetic game in thesis evaluation scored well into the highly-unusual range.",
  },
  suspicionProvisional: {
    term: "Provisional cutoffs",
    text: "The percentile cutoffs behind these labels are still being computed from the full planned sample of held-out test games; this chip disappears once the full run finishes.",
  },
  clockTime: {
    term: "Clock and time spent",
    text: "Each player's remaining time, parsed from the game's own clock annotations, is one of the model's direct inputs: time pressure changes how people play. Time spent per move (shown in the move list) is derived from it. The highlighted clock is the side to move.",
  },
  liveClock: {
    term: "Live timing",
    text: "Live clocks are estimated between updates, and Lichess delays spectators by a few moves. The last-move time is how long the model took to score it on this server, which reflects hardware, not the game.",
  },
  provisional: {
    term: "Ongoing game",
    text: "Shown when the source game has no final result yet (PGN Result header is \"*\"). The analysis covers only the moves played so far and can change as the game continues.",
  },
  liveVsFull: {
    term: "Live estimate vs full-game analysis",
    text: "The live curve freezes each move's estimate the instant it was computed from a from-scratch rerun up to that move, and never revises it later. Full-game analysis reruns the whole finished game at once and is the more accurate, final view.",
  },
  testError: {
    term: "Saved test error",
    text: "This sample's actual prediction error from the thesis's held-out test evaluation, saved ahead of time so it does not depend on this deployment. The thesis model was off by about 172 rating points on average across the full test set.",
  },
  samples: {
    term: "Sample games",
    text: "Games drawn from the thesis's held-out test partition, or from its synthetic anomaly corpus where tagged. They illustrate model behaviour, not examples of fair or unfair play.",
  },
  syntheticMarkers: {
    term: "Engine-move markers",
    text: "On a synthetic sample, a marked ply is one where the move was swapped for a stronger engine's move, known for certain because the game was built that way. Both sides can still score high, because the rating model reads Maia's play as a much stronger player than its nominal band. Real games never carry this marker.",
  },
};

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
  sampleMeta: null,
  sampleMetaExpanded: false,
  substitutedPlies: new Set(),
  samplesManifest: null,
  showActualRatings: true,
  activeInfoIcon: null,
  boardInView: true,
  live: {
    active: false,
    following: true,
    mode: null, // "game" | "tv"
    gameId: null,
    finished: false,
    eventSource: null,
    frozenPerMove: [],
    reconnectAttempts: 0,
    reconnectTimer: null,
    // "idle" | "connecting" | "connected" | "reconnecting" | "disconnected" | "finished" | "stopped" | "error"
    connState: "idle",
    clockAnchor: null, // { ply, side, seconds, at }
    clockTimer: null,
    lastInferenceMs: null,
  },
};

function $(id) {
  return document.getElementById(id);
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Runs `fn` and puts the window scroll position back if anything inside it
// moved the page (re-rendering, collapsing the input bar, and so on).
function preservingWindowScroll(fn) {
  const x = window.scrollX;
  const y = window.scrollY;
  try {
    return fn();
  } finally {
    if (window.scrollX !== x || window.scrollY !== y) {
      window.scrollTo(x, y);
    }
  }
}

// --- Theme -----------------------------------------------------------------

function effectiveTheme() {
  const explicit = document.documentElement.getAttribute("data-theme");
  if (explicit === "light" || explicit === "dark") return explicit;
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function syncThemeToggleLabel() {
  const next = effectiveTheme() === "dark" ? "light" : "dark";
  $("theme-toggle").setAttribute("aria-label", `Switch to ${next} mode`);
}

function setupThemeToggle() {
  syncThemeToggleLabel();
  $("theme-toggle").addEventListener("click", () => {
    const next = effectiveTheme() === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try {
      window.localStorage.setItem(THEME_KEY, next);
    } catch (err) {
      // Storage unavailable; the choice just won't persist across reloads.
    }
    onThemeChanged();
  });
  if (window.matchMedia) {
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    const listener = () => {
      if (!document.documentElement.hasAttribute("data-theme")) onThemeChanged();
    };
    if (query.addEventListener) query.addEventListener("change", listener);
  }
}

function onThemeChanged() {
  syncThemeToggleLabel();
  if (state.result) renderChart(state.result);
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// --- Actual rating reveal/hide -------------------------------------------

function actualRatingsVisible() {
  return state.showActualRatings;
}

function loadHideActualRatingsPreference() {
  try {
    return window.localStorage.getItem(HIDE_ACTUAL_RATINGS_KEY) === "true";
  } catch (err) {
    return false;
  }
}

function saveHideActualRatingsPreference(hidden) {
  try {
    window.localStorage.setItem(HIDE_ACTUAL_RATINGS_KEY, hidden ? "true" : "false");
  } catch (err) {
    // Storage unavailable (private browsing, quota, etc.); the choice just
    // won't persist across reloads.
  }
}

function formatSignedError(value) {
  const rounded = Math.round(value);
  const sign = rounded > 0 ? "+" : "";
  return `${sign}${rounded}`;
}

function maskedOrRounded(value) {
  if (typeof value !== "number") return "-";
  if (!actualRatingsVisible()) return "hidden";
  return String(Math.round(value));
}

function actualAndErrorHtml(current, actual) {
  if (typeof actual !== "number") return "n/a";
  if (!actualRatingsVisible()) return "hidden";
  return `${Math.round(actual)}<span class="delta">${formatSignedError(current - actual)}</span>`;
}

function rerenderForActualRatingToggle() {
  if (!state.result) return;
  renderSuspicion(state.result);
  renderChart(state.result);
  renderPlayerBars(state.currentPly);
  renderMetricsPanel(state.currentPly);
  renderSampleMetaBox(state.sampleMeta);
  syncLiveClockTicker();
}

function setActualRatingsHidden(hidden) {
  state.showActualRatings = !hidden;
  $("hide-actual-toggle").checked = hidden;
  $("reveal-actual-button").hidden = !hidden;
  saveHideActualRatingsPreference(hidden);
  rerenderForActualRatingToggle();
}

function setupActualRatingToggle() {
  const hidden = loadHideActualRatingsPreference();
  state.showActualRatings = !hidden;
  $("hide-actual-toggle").checked = hidden;
  $("reveal-actual-button").hidden = !hidden;
  $("hide-actual-toggle").addEventListener("change", (evt) => setActualRatingsHidden(evt.target.checked));
  $("reveal-actual-button").addEventListener("click", () => setActualRatingsHidden(false));
}

// --- Info popovers ---------------------------------------------------------

function infoIconHtml(key) {
  const info = METRIC_INFO[key];
  if (!info) return "";
  return (
    `<button type="button" class="info-icon" data-info-key="${key}" ` +
    `aria-label="What is ${escapeHtml(info.term)}?">i</button>`
  );
}

function showInfoPopover(iconEl) {
  const info = METRIC_INFO[iconEl.dataset.infoKey];
  if (!info) return;
  const popover = $("info-popover");
  $("info-popover-term").textContent = info.term;
  $("info-popover-text").textContent = info.text;
  popover.hidden = false;

  const margin = 8;
  const iconRect = iconEl.getBoundingClientRect();
  popover.style.left = "0px";
  popover.style.top = "0px";
  const popRect = popover.getBoundingClientRect();
  const left = Math.max(margin, Math.min(iconRect.left, window.innerWidth - popRect.width - margin));
  let top = iconRect.bottom + margin;
  if (top + popRect.height > window.innerHeight - margin) {
    top = iconRect.top - popRect.height - margin;
  }
  popover.style.left = `${left}px`;
  popover.style.top = `${Math.max(margin, top)}px`;

  state.activeInfoIcon = iconEl;
  iconEl.setAttribute("aria-expanded", "true");
}

function hideInfoPopover() {
  $("info-popover").hidden = true;
  if (state.activeInfoIcon) {
    state.activeInfoIcon.setAttribute("aria-expanded", "false");
    state.activeInfoIcon = null;
  }
}

function setupInfoPopovers() {
  document.addEventListener("click", (evt) => {
    const icon = evt.target.closest(".info-icon");
    if (icon) {
      evt.preventDefault();
      evt.stopPropagation();
      showInfoPopover(icon);
      return;
    }
    if (!evt.target.closest("#info-popover")) {
      hideInfoPopover();
    }
    // Close open dropdown-style <details> (options, glossary) on outside click.
    document.querySelectorAll(".options-menu[open], .metrics-glossary[open]").forEach((el) => {
      if (!el.contains(evt.target)) el.open = false;
    });
  });
  document.addEventListener("mouseover", (evt) => {
    const icon = evt.target.closest(".info-icon");
    if (icon) showInfoPopover(icon);
  });
  document.addEventListener("mouseout", (evt) => {
    const icon = evt.target.closest(".info-icon");
    if (icon && document.activeElement !== icon) hideInfoPopover();
  });
  document.addEventListener("focusin", (evt) => {
    const icon = evt.target.closest(".info-icon");
    if (icon) showInfoPopover(icon);
  });
  document.addEventListener("focusout", (evt) => {
    const icon = evt.target.closest(".info-icon");
    if (icon) hideInfoPopover();
  });
  document.addEventListener("keydown", (evt) => {
    if (evt.key === "Escape") {
      hideInfoPopover();
      document.querySelectorAll(".options-menu[open], .metrics-glossary[open]").forEach((el) => {
        el.open = false;
      });
    }
  });
}

function renderMetricsGlossary() {
  const container = $("metrics-glossary-list");
  container.innerHTML = Object.values(METRIC_INFO)
    .map((info) => `<dt>${escapeHtml(info.term)}</dt><dd>${escapeHtml(info.text)}</dd>`)
    .join("");
}

// --- Input bar ---------------------------------------------------------------

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
      // Samples and Live have their own load/watch controls instead of the
      // shared "Analyze" button.
      $("submit-button").hidden = tab === "samples" || tab === "live";
      clearError();
      if (tab === "samples") {
        loadSamplesManifest();
      }
    });
  });
}

function setInputPanelCollapsed(collapsed) {
  const panel = $("input-panel");
  panel.classList.toggle("collapsed", collapsed);
  $("input-panel-toggle").setAttribute("aria-expanded", collapsed ? "false" : "true");
  $("input-panel-toggle-label").textContent = collapsed ? "Load game" : "Hide";
}

function setupInputPanelToggle() {
  $("input-panel-toggle").addEventListener("click", () => {
    setInputPanelCollapsed(!$("input-panel").classList.contains("collapsed"));
  });
}

function setupBoard() {
  state.board = window.Chessboard("board", {
    position: "start",
    pieceTheme: "/static/vendor/chessboard-js/img/chesspieces/wikipedia/{piece}.png",
  });
  let resizeFrame = null;
  window.addEventListener("resize", () => {
    if (resizeFrame) cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(() => {
      resizeFrame = null;
      if (!state.result) return;
      // chessboard.js only measures its container when asked; resizing also
      // rebuilds the squares, so the last-move highlight is re-applied.
      state.board.resize();
      highlightCurrentPly();
    });
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
  button.textContent = isLoading ? "Analyzing..." : "Analyze";
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

// --- Samples -----------------------------------------------------------------

async function loadSamplesManifest() {
  if (state.samplesManifest) {
    renderSamplesGrid(state.samplesManifest.samples || []);
    return;
  }
  const statusEl = $("samples-status");
  statusEl.hidden = false;
  statusEl.textContent = "Loading...";
  try {
    const response = await fetch(`${API_BASE}/static/samples/manifest.json`);
    if (!response.ok) {
      throw new Error(`Could not load sample manifest (status ${response.status}).`);
    }
    const manifest = await response.json();
    state.samplesManifest = manifest;
    renderSamplesGrid(manifest.samples || []);
  } catch (err) {
    statusEl.hidden = false;
    statusEl.textContent = `Could not load samples: ${err.message || err}`;
  }
}

function sampleBadgeLabel(sample) {
  const bits = [];
  if (sample.time_control) bits.push(sample.time_control);
  if (typeof sample.white_rating === "number" && typeof sample.black_rating === "number") {
    bits.push(`${Math.round(sample.white_rating)} vs ${Math.round(sample.black_rating)}`);
  }
  return bits.join(" · ");
}

// One-line description with a "more" toggle that expands it in place.
function buildMoreToggleLine(className, text) {
  const line = document.createElement("div");
  line.className = className;
  const span = document.createElement("span");
  span.className = `${className}-text`;
  span.textContent = text;
  line.appendChild(span);
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "more-toggle";
  toggle.textContent = "more";
  toggle.setAttribute("aria-expanded", "false");
  toggle.addEventListener("click", (evt) => {
    evt.stopPropagation();
    const expanded = line.classList.toggle("expanded");
    toggle.textContent = expanded ? "less" : "more";
    toggle.setAttribute("aria-expanded", expanded ? "true" : "false");
  });
  line.appendChild(toggle);
  return line;
}

function buildSampleCard(sample) {
  const card = document.createElement("div");
  card.className = "sample-card";
  card.dataset.sampleId = sample.id;

  const main = document.createElement("button");
  main.type = "button";
  main.className = "sample-card-main";
  main.dataset.sampleId = sample.id;
  main.title = [sample.source, sample.selection_label].filter(Boolean).join(" · ");

  const title = document.createElement("span");
  title.className = "sample-card-title";
  title.textContent = sample.title || sample.id;
  main.appendChild(title);

  const badgeLine = sampleBadgeLabel(sample);
  if (badgeLine) {
    const badge = document.createElement("span");
    badge.className = "sample-card-badge-line";
    badge.textContent = badgeLine;
    main.appendChild(badge);
  }
  main.addEventListener("click", () => loadAndAnalyzeSample(sample, card));
  card.appendChild(main);

  if (sample.description) {
    card.appendChild(buildMoreToggleLine("sample-card-desc", sample.description));
  }
  return card;
}

function renderSamplesGrid(samples) {
  const grid = $("samples-grid");
  const statusEl = $("samples-status");
  grid.innerHTML = "";
  if (!samples.length) {
    statusEl.hidden = false;
    statusEl.textContent = "No samples available.";
    return;
  }
  statusEl.hidden = true;

  const groups = new Map();
  samples.forEach((sample) => {
    const key = sample.group || "Samples";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(sample);
  });

  groups.forEach((groupSamples, groupName) => {
    const heading = document.createElement("h3");
    heading.className = "samples-group-title";
    heading.textContent = groupName;
    grid.appendChild(heading);
    groupSamples.forEach((sample) => grid.appendChild(buildSampleCard(sample)));
  });
}

async function loadAndAnalyzeSample(sample, cardEl) {
  clearError();
  stopLiveStream({ silent: true });
  const topK = currentTopK();
  const minPly = currentMinPly();
  const grid = $("samples-grid");
  grid.classList.add("loading");
  if (cardEl) cardEl.setAttribute("aria-busy", "true");
  try {
    const pgnResponse = await fetch(`${API_BASE}/static/${sample.pgn_path}`);
    if (!pgnResponse.ok) {
      throw new Error(`Could not load sample PGN (status ${pgnResponse.status}).`);
    }
    const pgnText = await pgnResponse.text();
    const query = `top_k=${topK}&min_ply=${minPly}`;
    const response = await fetch(`${API_BASE}/predict/pgn?${query}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pgn: pgnText }),
    });
    if (!response.ok) {
      throw new Error(await parseErrorDetail(response));
    }
    const result = await response.json();
    renderResult(result, { sampleMeta: sample });
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    grid.classList.remove("loading");
    if (cardEl) cardEl.removeAttribute("aria-busy");
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
  stopLiveStream({ silent: true });
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

// --- Result rendering ----------------------------------------------------------

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

function renderResult(result, opts = {}) {
  preservingWindowScroll(() => renderResultInner(result, opts));
}

function renderResultInner(result, opts) {
  const moveList = $("move-list");
  const previousListScroll = moveList.scrollTop;
  const wasLiveOnScreen = Boolean(state.result) && !$("live-badge").hidden;

  state.result = result;
  state.fenAtPly = buildFenTimeline(result.headers, result.per_move);
  state.currentPly = state.fenAtPly.length - 1;
  state.attentionRanks = buildAttentionRanks(result.per_move);

  const criticalIndex = buildCriticalIndex(result.critical_moves || []);
  state.criticalByPly = criticalIndex.map;
  state.criticalCounts = criticalIndex.counts;

  if (!opts.live || !wasLiveOnScreen) {
    state.sampleMeta = opts.sampleMeta || null;
    state.sampleMetaExpanded = false;
  }
  state.substitutedPlies = new Set((state.sampleMeta && state.sampleMeta.substituted_plies) || []);
  renderSampleMetaBox(state.sampleMeta);
  $("synthetic-marker-legend").hidden = state.substitutedPlies.size === 0;

  $("live-badge").hidden = !opts.live;
  if (!opts.live) {
    resetLiveSessionUi();
  }

  stopAutoplay();

  // chessboard.js sizes itself from the container's rendered width at call
  // time; the container is still `hidden` (0 width) until this point, so it
  // must be explicitly resized once the panel becomes visible.
  $("results-panel").hidden = false;
  if (!opts.live || !wasLiveOnScreen) {
    state.board.orientation("white");
  }
  state.board.resize();

  const headers = result.headers || {};
  $("results-title").textContent = `${headers.White || "White"} vs ${headers.Black || "Black"}`;
  $("results-title").title = $("results-title").textContent;
  $("provisional-badge").hidden = !result.provisional || Boolean(opts.live);
  METRIC_INFO.criticalMove.text = METRIC_INFO.criticalMove.text.replace(
    /before ply \d+/,
    `before ply ${result.critical_moves_min_ply}`
  );

  renderWarnings(result.warnings || []);
  renderSuspicion(result);
  renderChart(result);
  renderMoveList(result.per_move);
  if (opts.live && !state.live.following) {
    moveList.scrollTop = previousListScroll;
  }
  renderBoardAtPly(state.currentPly, { fromAutoplay: true, keepListScroll: opts.live && !state.live.following });

  $("input-panel-toggle").hidden = false;
  if (!opts.live || !wasLiveOnScreen) {
    setInputPanelCollapsed(true);
  }
}

function renderWarnings(warnings) {
  const box = $("warnings-box");
  if (!warnings.length) {
    box.hidden = true;
    box.innerHTML = "";
    return;
  }
  const wasOpen = box.open;
  box.hidden = false;
  const items = warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("");
  const label = warnings.length === 1 ? "1 warning" : `${warnings.length} warnings`;
  box.innerHTML = `<summary>&#9888; ${label}</summary><ul>${items}</ul>`;
  box.open = wasOpen;
}

function renderSampleMetaBox(sampleMeta) {
  const box = $("sample-meta-box");
  if (!sampleMeta) {
    box.hidden = true;
    box.innerHTML = "";
    return;
  }

  const tags = [sampleMeta.source, sampleMeta.selection_label]
    .filter(Boolean)
    .map((tag) => `<span class="sample-tag">${escapeHtml(tag)}</span>`)
    .join("");

  const visible = actualRatingsVisible();
  const rows = [];
  const pts = (v) => (visible ? `${v.toFixed(1)} pts` : "hidden");
  if (typeof sampleMeta.white_actual_rating === "number" || typeof sampleMeta.black_actual_rating === "number") {
    rows.push([
      `Actual rating ${infoIconHtml("actualRating")}`,
      `W ${maskedOrRounded(sampleMeta.white_actual_rating)} · B ${maskedOrRounded(sampleMeta.black_actual_rating)}`,
    ]);
  }
  if (typeof sampleMeta.white_test_error === "number" || typeof sampleMeta.black_test_error === "number") {
    const w = typeof sampleMeta.white_test_error === "number" ? pts(sampleMeta.white_test_error) : "-";
    const b = typeof sampleMeta.black_test_error === "number" ? pts(sampleMeta.black_test_error) : "-";
    rows.push([`Saved test error ${infoIconHtml("testError")}`, `W ${w} · B ${b}`]);
  }
  if (typeof sampleMeta.substitution_rate === "number") {
    rows.push([`Engine moves ${infoIconHtml("syntheticMarkers")}`, `${Math.round(sampleMeta.substitution_rate * 100)}% of moves`]);
  }
  if (sampleMeta.maia_band) {
    rows.push(["Maia band", escapeHtml(String(sampleMeta.maia_band))]);
  }
  if (sampleMeta.engine) {
    rows.push(["Engine", escapeHtml(sampleMeta.engine)]);
  }
  if (typeof sampleMeta.s_att_eval === "number") {
    rows.push([`Saved S_att ${infoIconHtml("suspicion")}`, sampleMeta.s_att_eval.toFixed(1)]);
  }

  const expanded = state.sampleMetaExpanded;
  const details = expanded
    ? `<dl class="sample-meta-details">
        ${sampleMeta.description ? `<p class="sample-meta-description">${escapeHtml(sampleMeta.description)}</p>` : ""}
        ${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}
      </dl>`
    : "";

  box.innerHTML = `
    <div class="sample-meta-header">
      <span class="sample-meta-title">${escapeHtml(sampleMeta.title || "Sample game")}</span>
      ${tags}
      <span class="card-head-spacer"></span>
      <button type="button" class="more-toggle" id="sample-meta-more" aria-expanded="${expanded}">${expanded ? "less" : "more"}</button>
    </div>
    ${details}
  `;
  box.hidden = false;
  $("sample-meta-more").addEventListener("click", () => {
    state.sampleMetaExpanded = !state.sampleMetaExpanded;
    renderSampleMetaBox(state.sampleMeta);
  });
}

function renderSuspicion(result) {
  const whiteScore = result.white_suspicion_score;
  const blackScore = result.black_suspicion_score;
  const cutoffsUsed = result.suspicion_cutoffs_used;

  $("white-suspicion-value").textContent = whiteScore.toFixed(1);
  $("black-suspicion-value").textContent = blackScore.toFixed(1);
  const gapTitle = (score, baseline, source) =>
    `About ${Math.round(score)} rating points of attention-weighted gap from the baseline ` +
    `(${maskedOrRounded(baseline)}, ${formatSource(source)})`;
  $("white-suspicion-value").title = gapTitle(whiteScore, result.white_baseline, result.white_baseline_source);
  $("black-suspicion-value").title = gapTitle(blackScore, result.black_baseline, result.black_baseline_source);

  $("suspicion-baseline-warning").hidden = ![result.white_baseline_source, result.black_baseline_source].includes(
    "self_prediction_fallback"
  );

  renderSuspicionScale("white", whiteScore, result.white_suspicion_label, cutoffsUsed, result.provisional);
  renderSuspicionScale("black", blackScore, result.black_suspicion_label, cutoffsUsed, result.provisional);

  $("suspicion-label-info").hidden = !cutoffsUsed;
  const isProvisional = Boolean(cutoffsUsed && cutoffsUsed.provisional);
  $("suspicion-label-provisional").hidden = !isProvisional;
  if (isProvisional) {
    const n = cutoffsUsed.provisional_games;
    const count = typeof n === "number" ? `Based on ${n.toLocaleString()} test games so far. ` : "";
    METRIC_INFO.suspicionProvisional.text = count + (cutoffsUsed.provisional_note || METRIC_INFO.suspicionProvisional.text);
  }
}

// Renders one side's suspicion score as a segmented Typical/Unusual/Highly
// unusual scale (colour zones sized from the resolved p75/p95 cutoffs) with
// a marker at the score's position, plus the matching label chip. The scale
// max is max(score, 1.6 * p95) so all three zones stay visible regardless of
// how far the score sits above p95.
function renderSuspicionScale(prefix, score, label, cutoffs, provisional) {
  const marker = $(`${prefix}-suspicion-marker`);
  const zoneTypical = $(`${prefix}-zone-typical`);
  const zoneUnusual = $(`${prefix}-zone-unusual`);
  const zoneHighlyUnusual = $(`${prefix}-zone-highly-unusual`);

  if (!cutoffs) {
    marker.hidden = true;
    zoneTypical.style.flexBasis = "100%";
    zoneUnusual.style.flexBasis = "0%";
    zoneHighlyUnusual.style.flexBasis = "0%";
    renderSuspicionLabelChip(prefix, label, provisional);
    return;
  }

  const p75 = cutoffs.p75;
  const p95 = cutoffs.p95;
  const scaleMax = Math.max(score, 1.6 * p95, p95 + 1);
  const typicalPct = clampPct((p75 / scaleMax) * 100);
  const unusualPct = clampPct(((p95 - p75) / scaleMax) * 100);
  const highlyUnusualPct = Math.max(0, 100 - typicalPct - unusualPct);

  zoneTypical.style.flexBasis = `${typicalPct}%`;
  zoneUnusual.style.flexBasis = `${unusualPct}%`;
  zoneHighlyUnusual.style.flexBasis = `${highlyUnusualPct}%`;

  marker.hidden = false;
  marker.style.left = `${clampPct((score / scaleMax) * 100)}%`;
  marker.title = `${score.toFixed(1)} (Typical below ${p75.toFixed(0)}, Highly unusual above ${p95.toFixed(0)})`;

  renderSuspicionLabelChip(prefix, label, provisional);
}

function clampPct(value) {
  return Math.max(0, Math.min(100, value));
}

function renderSuspicionLabelChip(prefix, label, provisional) {
  const chip = $(`${prefix}-suspicion-label-chip`);
  if (!label) {
    chip.hidden = true;
    chip.textContent = "";
    delete chip.dataset.level;
    return;
  }
  chip.hidden = false;
  chip.dataset.level = label;
  const short = SUSPICION_LABEL_SHORT[label] || label;
  const long = SUSPICION_LABEL_TEXT[label] || label;
  chip.textContent = provisional ? `${short} so far` : short;
  chip.title = provisional ? `${long}. Based on the moves so far; can change as the game continues.` : long;
}

function formatSource(source) {
  switch (source) {
    case "request":
      return "entered";
    case "pgn_header":
      return "PGN";
    case "self_prediction_fallback":
      return "self-estimate, less reliable";
    default:
      return source;
  }
}

// --- Chart ---------------------------------------------------------------------

function chartPalette() {
  return {
    white: cssVar("--white-side"),
    black: cssVar("--black-side"),
    whiteRef: cssVar("--white-side-ref"),
    blackRef: cssVar("--black-side-ref"),
    grid: cssVar("--chart-grid"),
    tick: cssVar("--chart-tick"),
    cursor: cssVar("--chart-cursor"),
    ring: cssVar("--chart-point-ring"),
    surface: cssVar("--surface"),
    synthetic: cssVar("--synthetic"),
    text: cssVar("--text"),
  };
}

function buildSyncPlugin(palette) {
  return {
    id: "plySync",
    afterDatasetsDraw(chart) {
      if (!state.result) return;
      const ply = state.currentPly;
      const ctx = chart.ctx;
      const area = chart.chartArea;

      // Small markers for critical plies, drawn under the current-ply dot.
      [
        ["white", 0, palette.white],
        ["black", 1, palette.black],
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
          ctx.fillStyle = palette.surface;
          ctx.fill();
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = color;
          ctx.stroke();
          ctx.restore();
        });
      });

      // Square markers for synthetic (engine-substituted) plies, on both curves.
      if (state.substitutedPlies.size) {
        [0, 1].forEach((dsIndex) => {
          const meta = chart.getDatasetMeta(dsIndex);
          if (!meta || !meta.data) return;
          state.substitutedPlies.forEach((substitutedPly) => {
            const point = meta.data[substitutedPly];
            if (!point) return;
            const half = 4;
            ctx.save();
            ctx.fillStyle = palette.synthetic;
            ctx.fillRect(point.x - half, point.y - half, half * 2, half * 2);
            ctx.lineWidth = 1;
            ctx.strokeStyle = palette.ring;
            ctx.strokeRect(point.x - half, point.y - half, half * 2, half * 2);
            ctx.restore();
          });
        });
      }

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
        ctx.strokeStyle = palette.cursor;
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
        ctx.fillStyle = dsIndex === 0 ? palette.white : palette.black;
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = palette.ring;
        ctx.fill();
        ctx.stroke();
        ctx.restore();
      });
    },
  };
}

function buildReferenceLineDatasets(side, color, baseline, actual, labels) {
  // Baseline and actual-rating reference lines are only plotted while
  // reveal mode is on: while hidden, omitting them entirely (rather than
  // masking their values) is what makes the chart usable for a
  // guess-the-rating demo.
  if (!actualRatingsVisible()) return [];

  const actualKnown = typeof actual === "number";
  const matchesBaseline = actualKnown && Math.abs(actual - baseline) < 0.5;
  const baselineLine = labels.map(() => baseline);

  if (matchesBaseline) {
    return [
      {
        label: `${side} actual (baseline)`,
        data: baselineLine,
        borderColor: color.baseline,
        borderDash: [6, 4],
        borderWidth: 1,
        pointRadius: 0,
      },
    ];
  }

  const datasets = [
    {
      label: `${side} baseline`,
      data: baselineLine,
      borderColor: color.baseline,
      borderDash: [6, 4],
      borderWidth: 1,
      pointRadius: 0,
    },
  ];
  if (actualKnown) {
    datasets.push({
      label: `${side} actual`,
      data: labels.map(() => actual),
      borderColor: color.actual,
      borderDash: [2, 2],
      borderWidth: 1.5,
      pointRadius: 0,
    });
  }
  return datasets;
}

function renderChart(result) {
  const palette = chartPalette();
  const perMove = result.per_move;
  const labels = [0, ...perMove.map((m) => m.ply)];
  const whiteRatings = [result.white_baseline, ...perMove.map((m) => m.white_rating)];
  const blackRatings = [result.black_baseline, ...perMove.map((m) => m.black_rating)];

  const datasets = [
    {
      label: "White estimate",
      data: whiteRatings,
      borderColor: palette.white,
      backgroundColor: "transparent",
      pointRadius: 0,
      borderWidth: 2,
    },
    {
      label: "Black estimate",
      data: blackRatings,
      borderColor: palette.black,
      backgroundColor: "transparent",
      pointRadius: 0,
      borderWidth: 2,
    },
    ...buildReferenceLineDatasets(
      "White",
      { baseline: palette.white, actual: palette.whiteRef },
      result.white_baseline,
      result.white_actual_rating,
      labels
    ),
    ...buildReferenceLineDatasets(
      "Black",
      { baseline: palette.black, actual: palette.blackRef },
      result.black_baseline,
      result.black_actual_rating,
      labels
    ),
  ];

  const ctx = $("rating-chart").getContext("2d");
  if (state.chart) {
    state.chart.destroy();
  }
  const tickFont = { size: 12 };
  state.chart = new window.Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      layout: { padding: { top: 0, right: 4 } },
      plugins: {
        legend: {
          position: "top",
          align: "end",
          labels: { color: palette.text, boxWidth: 18, boxHeight: 2, padding: 12, font: tickFont },
        },
        tooltip: {
          backgroundColor: palette.surface,
          titleColor: palette.text,
          bodyColor: palette.text,
          borderColor: palette.grid,
          borderWidth: 1,
          callbacks: { title: (items) => (items.length ? `Ply ${items[0].label}` : "") },
        },
      },
      scales: {
        x: {
          grid: { color: palette.grid },
          border: { color: palette.grid },
          ticks: { color: palette.tick, font: tickFont, maxRotation: 0, autoSkipPadding: 12 },
        },
        y: {
          grid: { color: palette.grid },
          border: { color: palette.grid },
          ticks: { color: palette.tick, font: tickFont, maxTicksLimit: 5 },
        },
      },
    },
    plugins: [buildSyncPlugin(palette)],
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

// --- Board, player bars, clocks -------------------------------------------------

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
  const fromEl = document.querySelector(`#board .square-${from}`);
  const toEl = document.querySelector(`#board .square-${to}`);
  if (fromEl) fromEl.classList.add("highlight-from");
  if (toEl) toEl.classList.add("highlight-to");
}

function highlightCurrentPly() {
  if (!state.result || state.currentPly === 0) {
    clearSquareHighlights();
    return;
  }
  highlightMoveSquares(state.result.per_move[state.currentPly - 1].uci);
}

// Mirrors format_data.parse_time_control: "{base}+{increment}" in seconds,
// e.g. "180+0". Used only for the ply-0 (no moves yet) clock display, since
// every played ply's clock comes straight from that ply's per_move record.
function parseTimeControlHeader(header) {
  if (!header) return { base: null, inc: null };
  const parts = header.split("+");
  if (parts.length !== 2 || !/^\d+$/.test(parts[0]) || !/^\d+$/.test(parts[1])) {
    return { base: null, inc: null };
  }
  return { base: parseInt(parts[0], 10), inc: parseInt(parts[1], 10) };
}

function formatClock(seconds) {
  if (typeof seconds !== "number" || Number.isNaN(seconds)) return "-";
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) {
    return `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
  }
  return `${m}:${String(sec).padStart(2, "0")}`;
}

// Ticking-clock format: m:ss, with tenths under 10 seconds, clamped at 0.
function formatLiveClock(seconds) {
  const s = Math.max(0, seconds);
  if (s >= 10) return formatClock(Math.floor(s));
  const tenths = Math.floor(s * 10) / 10;
  return `0:0${tenths.toFixed(1)}`;
}

// Plies are 1-indexed and alternate White (odd), Black (even). Returns the
// side's remaining clock as of `ply` moves played, or the TimeControl base
// allotment before either side has moved, or null if neither is derivable.
function clockSecondsForSide(result, ply, side) {
  const perMove = result.per_move || [];
  const wantWhite = side === "white";
  let targetPly = ply;
  if (ply === 0 || wantWhite !== (ply % 2 === 1)) {
    targetPly = ply === 0 ? 0 : ply - 1;
  }
  if (targetPly === 0) {
    return parseTimeControlHeader((result.headers || {}).TimeControl).base;
  }
  const record = perMove[targetPly - 1];
  return record ? record.clock_seconds : null;
}

function sideToMoveAtPly(ply) {
  return ply % 2 === 0 ? "white" : "black";
}

function barPositionForSide(side) {
  const orientation = state.board.orientation();
  const bottomSide = orientation === "white" ? "white" : "black";
  return side === bottomSide ? "bottom" : "top";
}

function gameIsOver() {
  if (!state.result) return false;
  if (state.live.mode && (state.live.active || state.live.connState !== "idle")) {
    return state.live.finished;
  }
  return !state.result.provisional;
}

function renderPlayerBars(ply) {
  const result = state.result;
  const headers = result.headers || {};
  const whiteCurrent = ply === 0 ? result.white_baseline : result.per_move[ply - 1].white_rating;
  const blackCurrent = ply === 0 ? result.black_baseline : result.per_move[ply - 1].black_rating;
  const synthesizedClock = Boolean(state.sampleMeta && state.sampleMeta.source === "synthetic corpus");
  const latestPly = state.fenAtPly.length - 1;
  const toMove = gameIsOver() && ply === latestPly ? null : sideToMoveAtPly(ply);

  const perSide = {
    white: {
      side: "white",
      name: headers.White || "White",
      current: whiteCurrent,
      actual: result.white_actual_rating,
      clockSeconds: clockSecondsForSide(result, ply, "white"),
      synthesizedClock,
      toMove: toMove === "white",
    },
    black: {
      side: "black",
      name: headers.Black || "Black",
      current: blackCurrent,
      actual: result.black_actual_rating,
      clockSeconds: clockSecondsForSide(result, ply, "black"),
      synthesizedClock,
      toMove: toMove === "black",
    },
  };

  setPlayerBar(barPositionForSide("white"), perSide.white);
  setPlayerBar(barPositionForSide("black"), perSide.black);
}

function setPlayerBar(position, info) {
  const clockEl = $(`bar-${position}-clock`);
  const clockKnown = typeof info.clockSeconds === "number";
  clockEl.hidden = !clockKnown;
  if (clockKnown) {
    clockEl.textContent = formatClock(info.clockSeconds);
    clockEl.classList.toggle("low", info.clockSeconds < LOW_CLOCK_SECONDS);
  }
  clockEl.classList.toggle("active", info.toMove);
  clockEl.title = info.synthesizedClock ? "Synthetic clock (built for the synthetic corpus)" : "";
  clockEl.dataset.side = info.side;
  $(`bar-${position}-clock-note`).hidden = !(clockKnown && info.synthesizedClock);

  $(`bar-${position}-swatch`).classList.toggle("is-black", info.side === "black");
  $(`bar-${position}-name`).textContent = info.name;
  $(`bar-${position}-name`).title = info.name;
  $(`bar-${position}-current`).textContent = Math.round(info.current);

  const actualKnown = typeof info.actual === "number";
  $(`bar-${position}-actual-wrap`).hidden = !actualKnown;
  $(`bar-${position}-error-wrap`).hidden = !actualKnown;
  if (actualKnown) {
    $(`bar-${position}-actual`).textContent = maskedOrRounded(info.actual);
    $(`bar-${position}-error`).textContent = actualRatingsVisible()
      ? formatSignedError(info.current - info.actual)
      : "hidden";
  }
}

function formatDeltaHtml(delta) {
  if (delta === null || delta === undefined || Number.isNaN(delta)) return "";
  const rounded = Math.round(delta);
  if (rounded === 0) return `<span class="delta">&plusmn;0</span>`;
  const sign = rounded > 0 ? "+" : "";
  return `<span class="delta">${sign}${rounded}</span>`;
}

function baselineText(value, source) {
  return `${maskedOrRounded(value)}<span class="delta">${escapeHtml(formatSource(source) || "")}</span>`;
}

function renderMetricsPanel(ply) {
  const container = $("metrics-content");
  const result = state.result;
  const baselineRow =
    `<dt>W baseline ${infoIconHtml("baseline")}</dt><dd>${baselineText(result.white_baseline, result.white_baseline_source)}</dd>` +
    `<dt>B baseline</dt><dd>${baselineText(result.black_baseline, result.black_baseline_source)}</dd>`;

  if (ply === 0) {
    $("metrics-title").textContent = "Start position";
    container.innerHTML = `
      <dl class="metrics-grid">
        ${baselineRow}
        <dt>W actual ${infoIconHtml("actualRating")}</dt><dd>${actualAndErrorHtml(result.white_baseline, result.white_actual_rating)}</dd>
        <dt>B actual</dt><dd>${actualAndErrorHtml(result.black_baseline, result.black_actual_rating)}</dd>
      </dl>
    `;
    return;
  }

  const move = result.per_move[ply - 1];
  const prev = ply >= 2 ? result.per_move[ply - 2] : null;
  const whiteDelta = prev ? move.white_rating - prev.white_rating : null;
  const blackDelta = prev ? move.black_rating - prev.black_rating : null;
  const sideMoved = ply % 2 === 1 ? "White" : "Black";
  const moveNumber = Math.ceil(ply / 2);
  $("metrics-title").textContent = `Ply ${ply} · ${moveNumber}${ply % 2 === 1 ? "." : "..."} ${move.move || "?"}`;

  const attentionInfo = state.attentionRanks.get(ply);
  const attentionText =
    attentionInfo && typeof move.attention_weight === "number"
      ? `${(move.attention_weight * 100).toFixed(2)}%<span class="delta">#${attentionInfo.rank}/${attentionInfo.total}</span>`
      : "n/a";

  const critical = state.criticalByPly.get(ply) || {};
  const criticalBits = [];
  if (critical.white != null) criticalBits.push(`W #${critical.white}/${state.criticalCounts.white}`);
  if (critical.black != null) criticalBits.push(`B #${critical.black}/${state.criticalCounts.black}`);
  const criticalText = criticalBits.length ? `&#9733; ${criticalBits.join(", ")}` : "no";

  const visible = actualRatingsVisible();
  const whiteDeviationText = visible ? move.white_deviation.toFixed(1) : "hidden";
  const blackDeviationText = visible ? move.black_deviation.toFixed(1) : "hidden";

  const clockText = typeof move.clock_seconds === "number" ? formatClock(move.clock_seconds) : "n/a";
  const timeSpentText =
    typeof move.time_spent_seconds === "number" ? formatTimeSpent(move.time_spent_seconds) : "n/a";

  container.innerHTML = `
    <dl class="metrics-grid">
      <dt>${sideMoved} clock ${infoIconHtml("clockTime")}</dt><dd>${clockText}</dd>
      <dt>Time spent</dt><dd>${timeSpentText}</dd>
      <dt>W est ${infoIconHtml("ratingEstimate")}</dt><dd>${Math.round(move.white_rating)}${formatDeltaHtml(whiteDelta)}</dd>
      <dt>B est</dt><dd>${Math.round(move.black_rating)}${formatDeltaHtml(blackDelta)}</dd>
      <dt>W deviation ${infoIconHtml("deviation")}</dt><dd>${whiteDeviationText}</dd>
      <dt>B deviation</dt><dd>${blackDeviationText}</dd>
      <dt>Attention ${infoIconHtml("attention")}</dt><dd>${attentionText}</dd>
      <dt>Critical ${infoIconHtml("criticalMove")}</dt><dd>${criticalText}</dd>
    </dl>
  `;
}

// Scrolls only the move list (never the window) so the active move is visible.
function markActiveMoveListEntry(ply, opts = {}) {
  const list = $("move-list");
  list.querySelectorAll(".move-cell.active").forEach((el) => el.classList.remove("active"));
  const match = list.querySelector(`.move-cell[data-ply="${ply}"]`);
  if (ply === 0 && !opts.keepListScroll) {
    list.scrollTop = 0;
  }
  if (!match) return;
  match.classList.add("active");
  if (opts.keepListScroll) return;
  const listRect = list.getBoundingClientRect();
  const cellRect = match.getBoundingClientRect();
  const top = cellRect.top - listRect.top + list.scrollTop;
  const bottom = top + cellRect.height;
  const margin = cellRect.height;
  if (top - margin < list.scrollTop) {
    list.scrollTop = Math.max(0, top - margin);
  } else if (bottom + margin > list.scrollTop + list.clientHeight) {
    list.scrollTop = bottom + margin - list.clientHeight;
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
  if (state.substitutedPlies.has(move.ply)) {
    btn.classList.add("synthetic");
    btn.title = btn.title ? `${btn.title}; engine move inserted` : "Engine move inserted";
  }
  const moveText = document.createElement("span");
  moveText.textContent = move.move || "?";
  btn.appendChild(moveText);

  if (typeof move.time_spent_seconds === "number") {
    const timeEl = document.createElement("span");
    timeEl.className = "move-cell-time";
    timeEl.textContent = formatTimeSpent(move.time_spent_seconds);
    btn.appendChild(timeEl);
  }

  btn.addEventListener("click", () => renderBoardAtPly(move.ply));
  return btn;
}

function formatTimeSpent(seconds) {
  const s = Math.max(0, Math.round(seconds));
  return s < 60 ? `${s}s` : formatClock(s);
}

function renderMoveList(perMove) {
  const container = $("move-list");
  container.innerHTML = "";
  if (!perMove.length) {
    container.innerHTML = '<p class="move-list-empty">No moves yet.</p>';
    return;
  }
  const fragment = document.createDocumentFragment();
  for (let i = 0; i < perMove.length; i += 2) {
    const moveNumber = Math.floor(i / 2) + 1;
    const whiteMove = perMove[i];
    const blackMove = perMove[i + 1];

    const row = document.createElement("div");
    row.className = "move-row";
    row.setAttribute("role", "listitem");

    const numEl = document.createElement("span");
    numEl.className = "move-number";
    numEl.textContent = `${moveNumber}`;
    row.appendChild(numEl);

    row.appendChild(buildMoveCell(whiteMove, "white"));
    if (blackMove) {
      row.appendChild(buildMoveCell(blackMove, "black"));
    } else {
      const empty = document.createElement("span");
      empty.className = "move-cell empty";
      row.appendChild(empty);
    }

    fragment.appendChild(row);
  }
  container.appendChild(fragment);
}

function updatePlyIndicator() {
  $("ply-indicator").textContent = `${state.currentPly} / ${state.fenAtPly.length - 1}`;
}

function renderBoardAtPly(ply, opts = {}) {
  if (!opts.fromAutoplay) {
    stopAutoplay();
  }
  preservingWindowScroll(() => {
    const clamped = Math.max(0, Math.min(ply, state.fenAtPly.length - 1));
    state.currentPly = clamped;
    state.board.position(state.fenAtPly[clamped], false);
    updatePlyIndicator();
    highlightCurrentPly();

    if (state.live.active || state.live.connState !== "idle") {
      state.live.following = clamped === state.fenAtPly.length - 1;
    }

    renderPlayerBars(clamped);
    renderMetricsPanel(clamped);
    markActiveMoveListEntry(clamped, { keepListScroll: opts.keepListScroll });
    if (state.chart) {
      state.chart.update("none");
    }
    updateLiveButtons();
    syncLiveClockTicker();
  });
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
  highlightCurrentPly();
  renderPlayerBars(state.currentPly);
  syncLiveClockTicker();
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

// "New move" pill: shown when a live move lands while the board is scrolled
// out of view, instead of ever moving the page for the user.
function setupBoardVisibilityWatch() {
  const pill = $("new-move-pill");
  pill.addEventListener("click", () => {
    pill.hidden = true;
    $("board-wrap").scrollIntoView({ block: "center", behavior: "smooth" });
  });
  if (!("IntersectionObserver" in window)) return;
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        state.boardInView = entry.isIntersecting && entry.intersectionRatio > 0.35;
        if (state.boardInView) pill.hidden = true;
      });
    },
    { threshold: [0, 0.35, 0.7] }
  );
  observer.observe($("board-wrap"));
}

// --- Live mode -----------------------------------------------------------
//
// Every SSE "update" carries a fresh, non-causal rerun of the whole prefix
// (see README "Live mode"), so only its *last* per-move row is new
// information; earlier rows would differ slightly run to run. To keep the
// chart from revising history, mergeLiveUpdate() accumulates a frozen
// client-side per_move array (only ever appended to) and renderResult() is
// then reused unmodified against that frozen array, exactly as it renders a
// batch result.

function extractLichessGameId(raw) {
  const trimmed = (raw || "").trim();
  const urlMatch = trimmed.match(/lichess\.org\/(?:embed\/)?([A-Za-z0-9]{8})(?:[/?#]|$)/);
  if (urlMatch) return urlMatch[1];
  if (/^[A-Za-z0-9]{8}$/.test(trimmed)) return trimmed;
  return null;
}

function mergeLiveUpdate(result) {
  const frozen = state.live.frozenPerMove;
  const incoming = result.per_move || [];
  for (let i = frozen.length; i < incoming.length; i++) {
    frozen.push(incoming[i]);
  }
  return { ...result, per_move: frozen };
}

const LIVE_STATE_TEXT = {
  connecting: "Connecting...",
  connected: "Live",
  reconnecting: "Reconnecting...",
  disconnected: "Disconnected",
  finished: "Game over",
  stopped: "Stopped",
  error: "Stopped",
};

function liveSessionOnScreen() {
  return Boolean(state.result && state.live.mode && state.live.connState !== "idle" && !$("live-badge").hidden);
}

// Single place that turns the live connection state into UI: status text,
// the board overlay (reconnecting / disconnected), the game-over badge,
// dimmed clocks, and the ticking clock.
function setLiveConnState(connState, detail) {
  state.live.connState = connState;
  const base = LIVE_STATE_TEXT[connState] || "";
  let text = detail ? `${base}: ${detail}` : base;
  if (connState === "connected") {
    const bits = ["Live"];
    if (state.live.mode === "tv" && state.live.gameId) bits.push(`TV ${state.live.gameId}`);
    if (typeof state.live.lastInferenceMs === "number") bits.push(`last move ${Math.round(state.live.lastInferenceMs)} ms`);
    if (detail) bits.push(detail);
    text = bits.join(" · ");
  }
  if (connState === "idle") text = "";

  const inputLine = $("live-status-line");
  inputLine.hidden = !text;
  inputLine.textContent = text;
  inputLine.title = text;
  $("live-status-text").textContent = text;
  $("live-status-text").title = text;
  $("live-status-panel").dataset.state = connState;

  const onScreen = liveSessionOnScreen();
  const offline = onScreen && (connState === "reconnecting" || connState === "disconnected");
  $("board-wrap").classList.toggle("is-offline", offline);
  $("board-overlay").hidden = !offline;
  $("board-overlay-spinner").hidden = connState !== "reconnecting";
  $("board-overlay-text").textContent = connState === "reconnecting" ? "Reconnecting..." : "Disconnected";
  $("board-overlay-retry").hidden = connState !== "disconnected";
  $("board-badge").hidden = !(onScreen && connState === "finished");
  $("results-panel").classList.toggle("live-offline", onScreen && connState !== "connected");

  if (onScreen && connState === "finished") {
    // Show the recorded clocks (no side to move) once the game is over.
    renderPlayerBars(state.currentPly);
  }
  updateLiveButtons();
  syncLiveClockTicker();
}

function resetLiveSessionUi() {
  clearLiveReconnectTimer();
  state.live.finished = false;
  state.live.mode = null;
  state.live.clockAnchor = null;
  setLiveConnState("idle");
}

function updateLiveButtons() {
  const live = state.live;
  $("live-stop-button").hidden = !live.active;
  $("live-follow-button").disabled = live.active;
  $("live-tv-button").disabled = live.active;

  const panelHasContent = $("live-status-text").textContent.trim().length > 0 && !$("live-badge").hidden;
  $("live-status-panel").hidden = !panelHasContent;
  $("jump-to-live").hidden = !live.mode || live.following || !state.result;
  $("show-full-analysis").hidden = !live.finished || !live.gameId;
  $("stop-live").hidden = !live.active;
}

function clearLiveReconnectTimer() {
  if (state.live.reconnectTimer) {
    clearTimeout(state.live.reconnectTimer);
    state.live.reconnectTimer = null;
  }
}

function closeLiveEventSourceQuietly() {
  if (state.live.eventSource) {
    state.live.eventSource.onerror = null;
    state.live.eventSource.onmessage = null;
    state.live.eventSource.close();
    state.live.eventSource = null;
  }
}

function stopLiveStream(opts = {}) {
  clearLiveReconnectTimer();
  if (!state.live.active && !state.live.eventSource) return;
  closeLiveEventSourceQuietly();
  state.live.active = false;
  if (!opts.silent) {
    setLiveConnState("stopped");
  } else {
    updateLiveButtons();
    syncLiveClockTicker();
  }
}

// --- Live ticking clock ---------------------------------------------------------

// Records "the side to move had `seconds` left at time `at`" from the latest
// server update, so the clock can count down locally between updates.
function setLiveClockAnchor(result) {
  const perMove = state.live.frozenPerMove;
  const latestPly = perMove.length;
  const side = sideToMoveAtPly(latestPly);
  const seconds = clockSecondsForSide({ ...result, per_move: perMove }, latestPly, side);
  if (typeof seconds !== "number") {
    state.live.clockAnchor = null;
    return;
  }
  // The update was sent after inference finished, so the clock has already
  // been running for about that long.
  const inferenceMs = typeof result.inference_ms === "number" ? result.inference_ms : 0;
  state.live.clockAnchor = { ply: latestPly, side, seconds, at: performance.now() - inferenceMs };
}

function liveClockShouldTick() {
  const live = state.live;
  return Boolean(
    live.active &&
      live.connState === "connected" &&
      !live.finished &&
      live.clockAnchor &&
      state.result &&
      !$("live-badge").hidden &&
      state.currentPly === live.clockAnchor.ply &&
      state.currentPly === state.fenAtPly.length - 1 &&
      !document.hidden
  );
}

function tickLiveClock() {
  if (!liveClockShouldTick()) {
    syncLiveClockTicker();
    return;
  }
  const anchor = state.live.clockAnchor;
  const remaining = Math.max(0, anchor.seconds - (performance.now() - anchor.at) / 1000);
  const clockEl = $(`bar-${barPositionForSide(anchor.side)}-clock`);
  clockEl.hidden = false;
  clockEl.textContent = formatLiveClock(remaining);
  clockEl.classList.toggle("low", remaining < LOW_CLOCK_SECONDS);
  clockEl.classList.add("active");
}

function syncLiveClockTicker() {
  const shouldTick = liveClockShouldTick();
  if (shouldTick && !state.live.clockTimer) {
    state.live.clockTimer = setInterval(tickLiveClock, LIVE_CLOCK_TICK_MS);
    tickLiveClock();
  } else if (!shouldTick && state.live.clockTimer) {
    clearInterval(state.live.clockTimer);
    state.live.clockTimer = null;
  }
}

function setupLiveClockVisibility() {
  document.addEventListener("visibilitychange", () => {
    // Hidden: freeze. Visible again: the anchor still holds the server value
    // and its timestamp, so the first tick resyncs to the right estimate.
    syncLiveClockTicker();
  });
}

// --- Live stream handling ------------------------------------------------------------

function handleLiveMessage(payload) {
  if (payload.type === "status") {
    switch (payload.state) {
      case "finished":
        state.live.finished = true;
        closeLiveEventSourceQuietly();
        state.live.active = false;
        setLiveConnState("finished");
        break;
      case "reconnecting":
        setLiveConnState("reconnecting");
        break;
      case "connecting":
        if (state.live.connState !== "reconnecting") setLiveConnState("connecting");
        break;
      case "connected":
        setLiveConnState("connected");
        break;
      case "capped":
        setLiveConnState("connected", "ply cap reached");
        break;
      case "unsupported":
        // Lichess TV featured a custom-position or variant game; the stream
        // stays open and waits for TV to switch.
        showError(payload.message || "This TV game cannot be analyzed.");
        setLiveConnState("connecting", "unsupported TV game, waiting for the next one");
        break;
      default:
        if (payload.message) setLiveConnState(state.live.connState, payload.state);
        break;
    }
    return;
  }

  if (payload.type === "error") {
    // Permanent: show the reason and do not reconnect.
    const detail = payload.detail || payload.message || "Live analysis stopped.";
    closeLiveEventSourceQuietly();
    clearLiveReconnectTimer();
    state.live.active = false;
    showError(detail);
    setLiveConnState("error", "see message above");
    return;
  }

  if (payload.type === "update") {
    const result = payload.result;
    const isNewGame = Boolean(
      state.live.gameId && result.lichess_game_id && result.lichess_game_id !== state.live.gameId
    );
    if (isNewGame) {
      // Lichess TV switched to a new game: start a fresh frozen history.
      state.live.frozenPerMove = [];
      state.live.following = true;
    }
    state.live.gameId = result.lichess_game_id || state.live.gameId;
    state.live.reconnectAttempts = 0;
    state.live.lastInferenceMs = typeof result.inference_ms === "number" ? result.inference_ms : null;
    if (state.live.mode === "tv") clearError();

    const previousLength = state.live.frozenPerMove.length;
    const merged = mergeLiveUpdate(result);
    const wasFollowing = state.live.following || !state.result || $("live-badge").hidden;
    const previousPly = state.currentPly;
    setLiveClockAnchor(result);
    state.live.following = wasFollowing;
    renderResult(merged, { live: true });
    if (!wasFollowing) {
      renderBoardAtPly(Math.min(previousPly, state.fenAtPly.length - 1), { keepListScroll: true });
    }
    setLiveConnState("connected");
    if (merged.per_move.length > previousLength && !state.boardInView) {
      $("new-move-pill").hidden = false;
    }
  }
}

function scheduleLiveReconnect() {
  const live = state.live;
  if (live.reconnectAttempts < LIVE_RECONNECT_DELAYS_MS.length) {
    const delay = LIVE_RECONNECT_DELAYS_MS[live.reconnectAttempts];
    live.reconnectAttempts += 1;
    setLiveConnState("reconnecting");
    clearLiveReconnectTimer();
    live.reconnectTimer = setTimeout(() => {
      live.reconnectTimer = null;
      if (live.active) connectLiveEventSource();
    }, delay);
  } else {
    live.active = false;
    setLiveConnState("disconnected");
  }
}

function connectLiveEventSource() {
  const topK = currentTopK();
  const minPly = currentMinPly();
  const url =
    state.live.mode === "game"
      ? `${API_BASE}/live/stream/${encodeURIComponent(state.live.gameId)}?top_k=${topK}&min_ply=${minPly}`
      : `${API_BASE}/live/tv?top_k=${topK}&min_ply=${minPly}`;

  if (state.live.connState !== "reconnecting") setLiveConnState("connecting");
  closeLiveEventSourceQuietly();
  const source = new EventSource(url);
  state.live.eventSource = source;
  source.onmessage = (evt) => {
    let payload;
    try {
      payload = JSON.parse(evt.data);
    } catch (err) {
      return; // Malformed event; ignore this one and keep the stream open.
    }
    handleLiveMessage(payload);
  };
  source.onerror = () => {
    if (!state.live.active || state.live.eventSource !== source) return;
    closeLiveEventSourceQuietly();
    scheduleLiveReconnect();
  };
}

function startLiveWatch(mode, gameId) {
  stopLiveStream({ silent: true });
  state.live.active = true;
  state.live.mode = mode;
  state.live.gameId = gameId || null;
  state.live.following = true;
  state.live.finished = false;
  state.live.frozenPerMove = [];
  state.live.reconnectAttempts = 0;
  state.live.clockAnchor = null;
  state.live.lastInferenceMs = null;
  updateLiveButtons();
  connectLiveEventSource();
}

// Retry after retries ran out (or after the browser came back online): keeps
// the frozen history, since the server replays the prefix on reconnect.
function retryLiveWatch() {
  if (!state.live.mode || state.live.finished) return;
  clearLiveReconnectTimer();
  state.live.active = true;
  state.live.reconnectAttempts = 0;
  setLiveConnState("reconnecting");
  connectLiveEventSource();
}

// Treat the browser going offline as a dropped connection right away, rather
// than waiting for the socket to time out.
function setupLiveNetworkEvents() {
  window.addEventListener("offline", () => {
    if (!state.live.active) return;
    closeLiveEventSourceQuietly();
    scheduleLiveReconnect();
  });
  window.addEventListener("online", () => {
    if (state.live.connState === "disconnected") retryLiveWatch();
  });
}

async function loadFullGameAnalysisForLiveGame() {
  if (!state.live.gameId) return;
  clearError();
  const topK = currentTopK();
  const minPly = currentMinPly();
  $("live-status-text").textContent = "Loading full analysis...";
  try {
    const response = await fetch(`${API_BASE}/predict/lichess?top_k=${topK}&min_ply=${minPly}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ game_id: state.live.gameId }),
    });
    if (!response.ok) {
      throw new Error(await parseErrorDetail(response));
    }
    const result = await response.json();
    closeLiveEventSourceQuietly();
    renderResult(result);
  } catch (err) {
    showError(err.message || String(err));
  }
}

function setupLiveControls() {
  $("live-follow-button").addEventListener("click", () => {
    const gameId = extractLichessGameId($("live-lichess-id").value);
    if (!gameId) {
      showError("Enter a valid Lichess game ID or URL first.");
      return;
    }
    clearError();
    startLiveWatch("game", gameId);
  });
  $("live-tv-button").addEventListener("click", () => {
    clearError();
    startLiveWatch("tv", null);
  });
  $("live-stop-button").addEventListener("click", () => stopLiveStream());
  $("stop-live").addEventListener("click", () => stopLiveStream());
  $("jump-to-live").addEventListener("click", () => renderBoardAtPly(state.fenAtPly.length - 1));
  $("show-full-analysis").addEventListener("click", loadFullGameAnalysisForLiveGame);
  $("board-overlay-retry").addEventListener("click", retryLiveWatch);
}

function init() {
  setupThemeToggle();
  setupTabs();
  setupInputPanelToggle();
  setupBoard();
  setupBoardControls();
  setupChartPointerNav();
  setupKeyboardNav();
  setupBoardVisibilityWatch();
  setupLiveControls();
  setupLiveClockVisibility();
  setupLiveNetworkEvents();
  setupActualRatingToggle();
  setupInfoPopovers();
  renderMetricsGlossary();
  $("submit-button").addEventListener("click", submitAnalysis);
}

document.addEventListener("DOMContentLoaded", init);
