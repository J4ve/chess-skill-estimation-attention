# Web prototype

[Back to the README](../README.md)


`src/static/` is a static, no-build-step page (plain HTML/CSS/JS) served by
FastAPI's `StaticFiles` at `/static`, with `GET /` returning `index.html`
directly.

The input is a one-line bar at the top with five tabs: paste PGN, upload a
`.pgn` file, enter a Lichess game ID/URL, pick a **sample game** (see "Sample
games" below), or watch a game **live** (see "Live mode" below); an
"Options" dropdown holds the per-side baseline, critical-move top-k, and
min-ply overrides shared across tabs. Once a game is analyzed the bar
collapses to a "Load game" button.

**Layout.** The results page is a compact, Lichess-like, one-screen desktop
layout: on a 1080p screen the whole analysis fits without page scrolling.
The left column is a board sized from the viewport height, with a player bar
above and below (name, rating estimate, actual rating and error, clock box)
and one row of controls. The right column holds a small header (players,
Live/Ongoing chips, the hide-ratings switch, the glossary), the two suspicion
bars, a move list that scrolls on its own, and a collapsible move-details
grid. A short rating chart runs across the full width underneath. The page
text is kept to short labels; every explanation lives in the (i) popovers
and the glossary. On narrow screens the columns stack (a dedicated phone
layout is not built yet).

**Theme.** Light and dark themes follow the operating system's
`prefers-color-scheme` by default; the sun/moon button in the top bar
switches explicitly and the choice is remembered per browser via
`localStorage`. Every colour (board, chart, suspicion zones, chips, clocks,
popovers) comes from CSS variables at the top of `src/static/styles.css`.
See "Using the analysis board" below for controls.

## Using the analysis board

**Board and player bars.** The bottom player bar always matches the board's
current orientation; the "Flip" button swaps orientation and the bars follow.
Each bar shows the player's baseline rating, the model's rating estimate at
the ply currently shown, and, when the source game recorded one, the
player's actual rating and the signed error between the estimate and that
actual rating (for example "1935 est, 1979 actual, -44"), plus the clock box
(the side to move is highlighted; under 20 seconds it turns red). The
actual rating comes only from the PGN `WhiteElo`/`BlackElo` header and is
independent of any baseline override: overriding the baseline changes only
the suspicion-score comparison, never the actual-rating display. At the last
ply the error shown is the final error; the expanded sample card shows the
sample's saved test error for comparison.

**Reveal/hide toggle.** The "Hide ratings" switch in the results header masks every actual-rating value and error across the bars, metrics
panel, chart, and sample card, showing a "Reveal" button in its place; this
supports a guess-the-rating demo where a reviewer steps through the game
watching only the live estimate before revealing the answer. Because the
baseline equals the actual rating in most games, hiding also masks baseline
numbers wherever they would give it away (shown as "baseline hidden"); the
suspicion-score bars and numbers stay visible either way, since they are not
themselves a rating value. The choice is remembered per browser via
`localStorage` (defaulting to shown) and does not affect what the API
returns, only what the page displays.

**What do these numbers mean?** Every label with a small "i" next to it
(baseline, actual rating, error, deviation, attention, critical move,
suspicion score, and more) shows a one-or-two-sentence plain-language
explanation on hover, tap, or keyboard focus. The same explanations are also
listed together in the "Glossary" dropdown ("What do these numbers mean?")
in the results header, for a reviewer who wants the full list at once. The
wording lives in one place, `METRIC_INFO` in `src/static/app.js`.

**Moves panel.** Lists every move in standard two-column notation (move
number, White, Black) with the time spent on each move. Click a move to jump
the board there. The move currently shown is highlighted and kept in view by
scrolling only the list itself, never the page. A red star marks
a critical move (see "Move details" below).

**Rating chart.** Full width, below the board and move list, like Lichess's
evaluation graph. A vertical line and a dot on each curve track the current
ply; small ring markers show critical moves. Click or drag on the chart to
jump the board to that ply. Each side's baseline is drawn as a dashed
reference line; if that side's actual rating is also known and matches the
baseline (the usual case), the two are drawn as one line labeled "actual
rating (baseline)" rather than as duplicates, and only when they differ does
a separate, visually distinct actual-rating line appear. While actual
ratings are hidden, these reference lines are omitted from the chart
entirely rather than drawn with masked values, so the rating-estimate curves
are all that remain visible.

**Controls and keys.**

| Control | Key | Effect |
| --- | --- | --- |
| `«` | `Home` / `ArrowUp` | Jump to the start of the game |
| `‹` | `ArrowLeft` | Step back one ply |
| `›` | `ArrowRight` | Step forward one ply |
| `»` | `End` / `ArrowDown` | Jump to the end of the game |
| Play/Pause | - | Autoplay forward at about one move per second; stops at the end or on any manual navigation |
| Flip | - | Swap board orientation and the player bars |

Keyboard shortcuts are ignored while typing in a text field.

**Move details panel.** A collapsible key/value grid for the currently shown
ply: the mover's clock and time spent, each side's rating estimate and its
change from the previous ply, each side's
deviation from its baseline, the move's attention weight with its rank and
percentile among all plies in the game (for example "top 5% of attention"),
and whether the move is flagged critical with its rank. At the start of the
game this shows each side's baseline instead.

**Critical moves.** Ranked per side by `attention_weight * |deviation|`,
excluding plies before a minimum ply (default 10, configurable per request;
see [docs/api.md](api.md)) since the model has little context in the opening and
early plies would otherwise dominate the list. The detector's own per-move
logits are deliberately not used here: in thesis evaluation they located
engine-substituted moves at or below chance (hit@1 about 0.19 against 0.24
for random picks on withheld rating bands).

## Suspicion score

The main suspicion number for each side is the **trained detector** (thesis
arm A3g, seed 0; `src/detector.py`). A 2-layer BiGRU reads 17 engine-free
features per ply for the first 100 plies: the rating model's estimate for
both sides, the gap from the scored side's baseline, attention, running
statistics of the estimate, captures, checks, material, and clock use. It
pools per-ply logits over the scored side's moves with gated attention and
outputs a score from 0 to 1. Each side is scored in turn against its own
resolved baseline. The score ranks games; it is not a calibrated probability
of cheating.

Measured on the thesis's synthetic anomaly corpus v2 (twin-safe grouped
split, `src/models/detector_a3g_seed0.json`): ROC-AUC 0.752 on rating bands
withheld from training (0.688 for the LightGBM detector on the same split),
0.576 and 0.606 when only 2 and 5 percent of moves were engine moves, and
0.892 at 60 percent. The feature and model code is ported from the HPC
scripts that trained it (`extract_detector_features_v2.py`,
`extra_arm_a3.py`), and `experiments/detector_parity/` shows the app path
reproducing that code to within 6e-07 on the 12 bundled samples and 8 corpus
games.

The **computed score S_att** (`src/anomaly.py`) was the first method tried:
the attention-weighted average gap, in rating points, between the per-move
estimate and the baseline, with no trained parameters. It separated
engine-substituted synthetic games only weakly in thesis evaluation (ROC-AUC
0.555). It stays on the page as one secondary line under the bars ("First
method: computed S_att"), with its own labels, and in the API as
`*_computed_score`.

## Suspicion labels

Next to each suspicion bar, a label chip classifies that side's detector
score against percentile cutoffs computed from ordinary, finished, rated
games in the thesis's held-out TEST-partition corpus (games the rating model
never trained on): **Typical** (below the 75th percentile of those games),
**Unusual** (75th to 95th percentile), or **Highly unusual: worth a human
review** (above the 95th percentile). Each bar is a segmented scale, not a
plain gradient: the track is split into green/amber/red zones sized from that
side's resolved p75/p95 cutoffs, with the score shown as a needle marker (the
number sits beside the chip). The detector's scale runs from 0 to 1. When the
server falls back to S_att as the main score, its scale max is
`max(score, 1.6 * p95)` so all three zones stay visible. The card header says
"Review aid, not proof", and the label (i) explains the comparison against
ordinary games. Cutoffs use the game's own time-control bucket
(bullet/blitz/rapid/classical/ultrabullet, derived from the PGN `TimeControl`
header) when that bucket has at least 300 sides scored; otherwise they fall
back to the overall cutoff. The boundary logic is covered in
`tests/test_suspicion_labels.py` and `tests/test_detector.py`.

Each score has its own cutoffs file, and each file names its score in a
`score` field; the API ignores a file whose `score` does not match, so the two
distributions can never be mixed up:

- `src/static/suspicion_cutoffs.json` (`"score": "detector_a3g_seed0"`):
  2,822 held-out test games (5,644 sides) from a stratified sample (up to
  700 each for bullet, blitz and rapid, all available classical and
  ultrabullet games), scored on the HPC CPU with this repo's own
  `src/detector.py`. Overall p75 0.4805 and p95 0.8227; every time control
  has its own cutoff. The full sample completed, so it is not provisional.
  Method, table with bootstrap CIs, and the sample-game sanity check are in
  `experiments/detector_cutoffs/README.md`.
- `src/static/suspicion_cutoffs_s_att.json` (`"score": "s_att"`): the S_att
  cutoffs, still **provisional** (1,200 scored games from a run stopped
  early). The secondary S_att line says "provisional" while that holds. A
  "Provisional cutoffs" chip in the suspicion header appears only if the
  main score's cutoffs are provisional.

Labels never claim cheating and never say "likely" or name a player as
suspicious. About 5 percent of ordinary games land in "Highly unusual" by
construction, and the bundled `synthetic: false alarm` sample (see "Sample
games" below) has no engine moves at all yet scores "Highly unusual" on one
side. A label describes how uncommon a score is among ordinary games, not a
verdict about the players.

The API returns the labels and the cutoffs actually applied for both scores
(see [docs/api.md](api.md)), so the labelling logic is testable independent of the
UI. `suspicion_labels.py` takes a score and a cutoffs object in the shared
shape, with no reference to either score.

## Sample games

The "Sample games" tab loads and analyzes a game in one click, reusing the
same analysis board as every other input mode. Cards are grouped by
`src/static/samples/manifest.json`'s `group` field:

- **Rating: Bullet / Blitz / Rapid** - for each time control, a
  best-predicted, typical, and worst-predicted game, picked by this
  checkpoint's held-out test error (`analysis/heldout_test_eval/`), not by
  outcome or player identity. This is an honest spread, not a highlight
  reel: the worst-predicted games are shown because the model does worse on
  them, most often master-strength games underrepresented in the training
  distribution or very short/aborted games. Descriptions only ever
  characterize level and time control; they never call out a specific
  player. Source games are real, finished, public Lichess games (CC0, no
  rights reserved), so usernames in the PGN headers are real and unchanged,
  but nothing in this tab or its descriptions labels any player's games as
  suspicious.
- **Synthetic anomaly: caught / false alarm / missed** - three games from
  this project's own synthetic anomaly corpora (a Maia policy network's
  moves with some fraction substituted for a stronger engine's,
  `src/generate_anomaly_corpus.py`), one per outcome for the main (detector)
  score against the cutoffs above: **caught** (corpus v1, 60 percent Lc0
  substitution, White scores 0.95, Highly unusual), **false alarm** (corpus
  v2, no engine moves, Black still scores 0.89, Highly unusual) and
  **missed** (corpus v2, 15 of Black's 34 eligible moves swapped for
  Stockfish 16 moves, Black scores 0.25, Typical). Both v2 games come from
  rating bands the detector never trained on. These games are synthetic end
  to end (no real player), and each substituted ply is known ground truth:
  it renders as a small square marker on the move list and the rating chart
  (next to the critical-move star/ring markers), and the sample card
  ("more") shows the substitution rate, the Maia rating band, the engine,
  the corpus suspect side, and the score saved from thesis evaluation
  (detector score for v2 games, S_att for the v1 game).

The false-alarm and missed examples picked for S_att (2026-09-15) no longer
fit those roles under the detector (the old "missed" game is now caught at
0.88). They moved to `manifest.json`'s `archived` list with the date and
reason; their PGNs stay in `src/static/samples/`, and the picker never shows
archived entries. Future sample replacements follow the same rule.

Every sample card (one line, with a "more" toggle) also shows its saved held-out test error (rating games) or saved
score (synthetic games) alongside the app's own live values, so a viewer
can see whether this deployment's numbers match the thesis evaluation's, and
the actual rating(s) used for evaluation. The "Hide ratings" switch
(see "Using the analysis board") masks the actual-rating and test-error
facts on this card the same way it masks the board and chart.

`manifest.json` documents its own schema in a `schema_notes` field; new
entries only need `id`, `title`, `description`, and `pgn_path`, with every
provenance field (`source`, `selection_label`, test errors, actual ratings,
`substituted_plies`, and the synthetic-only fields) optional.

## Live mode

The "Live" tab follows an ongoing Lichess game move by move: paste a game ID
or URL and click "Watch game", or click "Watch Lichess TV" to follow
whichever game Lichess is currently featuring (switching automatically when
TV switches games; Lichess's game export trails the TV feed by a couple of
plies, so the missing plies are filled in by a short legal-move search to
the feed's position). Both use Server-Sent Events against this app's own
`GET /live/stream/{game_id}` and `GET /live/tv` endpoints, which in turn
follow Lichess's public streaming API (`GET /api/stream/game/{id}`,
`GET /api/tv/feed`) with no token required. Each update carries
`white_actual_rating`/`black_actual_rating` straight from whatever rating
Lichess's own stream reports for that side (the game's PGN header when
following a game by ID, or the featured player's live rating when following
TV), null when the stream did not report one, and displayed and masked by
the reveal/hide toggle exactly like a batch result.

**Prefix-estimate semantics (read this before trusting the live curve).**
The model's BiLSTM is not incremental: scoring ply *t* means running the
whole model over plies 1..*t* from scratch (see the README's Limitations). Live mode
makes this workable by re-running that same batch pipeline on the growing
PGN prefix every time a new move arrives, and takes only the estimate at
the *final* ply of that run. The chart plots and freezes that one value per
move; earlier points are never revised, even though a from-scratch rerun
over a longer prefix would compute slightly different values for them. This
is why the live curve is labeled **"live (move-by-move) estimate"** and can
differ from the full-game curve you'd get by re-analyzing the finished game
normally - the full-game curve is the non-causal, most-accurate view; the
live curve is what the model could have told you in the moment. Suspicion
score and critical moves are likewise computed on the prefix so far and
marked provisional, and can shift as the game continues. Per-move inference
time on this deployment's CPU is shown in the status line after each move.

Once the game ends, the board shows a **"Game over"** badge and a **"Full
analysis"** button switches to the normal, non-live analysis view for the
finished game. On TV, a featured game that has finished gets the same badge
while the stream waits for the next featured game.

**Live clocks.** At the latest ply the side-to-move clock counts down
locally about every 100 ms (m:ss, with tenths under 10 seconds), and resyncs
to the server's recorded clocks whenever a move arrives. It freezes when the
game is over, while disconnected, while the browser tab is hidden (resyncing
on return), and when you step back to an earlier ply, which shows that ply's
recorded clocks instead. The clocks are estimates between updates.

**Delay.** Lichess itself delays a spectator's view of an ongoing game by a
few moves as an anti-cheating measure; the status line says so once
connected. This is a Lichess-side delay, not something this app can reduce.

**Following behavior.** The board auto-follows the newest move while you're
viewing the latest ply. Stepping back (via the board controls, keyboard, or
the chart) stops following and shows a "Jump to live" button; new moves
still arrive and grow the chart, but the board stays where you left it
until you jump back. Stopping, or switching to a different game or tab,
cleanly closes the stream. Live updates never move the page: if the board is
scrolled out of view, a small "New move" pill appears instead.

**Connection states.** If this app's own SSE connection drops (or the
browser goes offline), the board dims and blurs under a "Reconnecting..."
badge while the page retries with backoff (1, 2, then 4 seconds); the clocks
dim and stop. When retries run out it shows "Disconnected" with a Retry
button, and it retries automatically when the browser comes back online. A
dropped upstream connection to Lichess is retried once server-side first.
Permanent problems (a bad game ID, a game without clocks, or a game that
starts from a custom position or variant, such as a thematic arena) arrive
as an SSE error event with the reason, which the page shows without
reconnecting.

## Roadmap (not built)

- Optional Stockfish evaluation plotted alongside the rating curve, to give a
  reviewer both signals (engine correlation and rating-estimate anomaly) on
  the same timeline.

## Vendored libraries

No CDN is used, so the page works offline at the defense. Each library is
vendored under `src/static/vendor/` with its own license file:

| Library | Version | Source | License |
| --- | --- | --- | --- |
| jQuery | 3.7.1 | https://www.npmjs.com/package/jquery | MIT |
| chessboard.js | 1.0.0 | https://github.com/oakmac/chessboardjs (npm: `@chrisoakman/chessboardjs`) | MIT |
| chessboard.js piece images | v1.0.0 tag | https://github.com/oakmac/chessboardjs/tree/v1.0.0/website/img/chesspieces/wikipedia | MIT |
| chess.js | 0.13.4 | https://www.npmjs.com/package/chess.js | BSD-2-Clause |
| Chart.js | 4.4.4 | https://www.npmjs.com/package/chart.js | MIT |

chess.js 0.13.4 is the last version published as a plain ES module with no
bundler required (later versions still ship as ESM; this version was picked
because it is a small, stable release with no build step needed beyond a
native `<script type="module">` import). chessboard.js needs jQuery as a peer
dependency; both load as classic scripts before `app.js`, which imports
chess.js as a native ES module.

