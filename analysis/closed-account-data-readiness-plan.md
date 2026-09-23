# Closed-account data: preprocessing readiness plan

**Status: OBSOLETE (2026-08-28). This document assumed a Lichess-provided export, which
Lichess DECLINED on 2026-08-19** ("Unfortunately we will not be able to share that
information. Lichess cannot and does not disclose private information related to an account,
which includes bans or restrictions, to third parties"). The closed-account path pivoted
the same day to a self-service build against the PUBLIC `tosViolation` field on
`GET /api/user` / `POST /api/users` (usernames pulled from the already-preprocessed corpus,
weakly-labeled at the account level): see the `anomaly-closed-account-sample` task body for
that plan. It is not built. The format-mapping analysis below was written for an incoming
Lichess export and does not match the self-service pipeline; revisit before building.
Until the self-service build happens, Chapter 4 anomaly validation is synthetic-only (the
90,000-game bot-vs-bot corpus, `analysis/anomaly-corpus-generated.md`).

**Original framing (kept for the record).** Planning document, written before any data had
arrived. Nothing here was run against real closed-account data. No code was written and no
sample data was fabricated. Every pipeline claim below cites the file and line it was read
from.

**Trigger:** the data request to the Lichess administrators described in
`contexts/essentials/thesis_explained.md:426-439` (§7.2, fallback path 2) and
`CCS Thesis - Integrated/chapters/chapter3.tex:270`. Public search of closed-account games
was disabled on 2023-10-04 (`README.md:129` records the source verification of commit
0d421bc), so an admin export was the only path left. That request was declined on
2026-08-19 (see status above).

**Purpose:** so that when a reply lands, the only open question is "which of these four
branches is it", not "how does any of this work".

---

## 0. The gates the existing pipeline already enforces

Everything in §1 depends on these, so they are stated once here. All of them are hard
rejects: data that fails them is silently dropped, not warned about.

| Gate | Where | Effect if the export fails it |
|---|---|---|
| Every mainline ply up to 100 must carry a `[%clk H:MM:SS]` comment | `format_data.py:67`, `format_data.py:74-80`; fast-path twin at `preprocess_lichess.py:83-99` | `parse_game` returns `None` and the game is dropped. A game with *partial* clock annotation is dropped too (`format_data.py:78-80`). |
| `Variant` header must be `Standard` | `preprocess_lichess.py:68` | dropped |
| `Event` header must contain the substring `Rated` | `preprocess_lichess.py:70` | dropped. This is the most likely silent killer for a re-serialized export, since a hand-built PGN writer may not set `Event` at all. |
| `WhiteElo` and `BlackElo` must be digit strings | `preprocess_lichess.py:71-74` | dropped |
| `TimeControl` must be `<int>+<int>` | `preprocess_lichess.py:76-79` | dropped. Correspondence (`-`) is excluded by design. |
| `Time` must parse as `<int>+<int>` again at load time | `chess_rating_net.py:93` | this one is not a graceful drop, it raises. A pickle that somehow gets written with a non-conforming `Time` will crash the dataset loader, not skip. |
| Ply cap of 100 | `format_data.py:41`, `format_data.py:59` | plies past 100 are discarded, in both arms equally |

The surviving per-game record has exactly these keys (`format_data.py:85-95`): `WhiteElo`,
`BlackElo`, `White`, `Black`, `Result`, `Clocks`, `Positions`, `Moves`, `Time`.

Two consequences of that key list that matter later:

- **There is no Lichess game ID in the pickle.** `Site` is not retained. The dataset's
  `game_id` is just the pickle filename stem (`chess_rating_net.py:107`). So the existing
  ~2M-game corpus cannot be checked for overlap against an incoming sample by game ID.
  See §2.4 for the surrogate key.
- **`rating_after_last_game` is read by the loader (`chess_rating_net.py:83-86`) but is
  never written by `parse_game` (`format_data.py:85-95`).** For every pickle this pipeline
  has ever produced it is `None`. That matters for the anomaly baseline (§2.5).

---

## 1. Format-by-format mapping

### 1.1 Raw PGN, Lichess-style headers, `[%clk]` comments present

**Verdict: direct reuse, zero code changes.**

Entry point: `preprocess_lichess.py:main()` (`preprocess_lichess.py:108`), invoked exactly
as documented in its own usage block (`preprocess_lichess.py:18-22`).

Two operational notes, both of which avoid touching the source:

1. `stream_games` (`preprocess_lichess.py:52-63`) always wraps the input handle in a zstd
   decompressor (`preprocess_lichess.py:56`), so a plain `.pgn` file cannot be passed to
   `--input` as-is. Do not patch the reader. Just compress the export first
   (`zstd sample.pgn -o sample.pgn.zst`) and pass the `.zst`. One shell command, and the
   code path stays byte-identical to the one that produced the training corpus.
2. Set `--max-games` (`preprocess_lichess.py:115`) far above the export's game count. The
   reservoir appends unconditionally until it is full (`preprocess_lichess.py:149-151`), so
   an oversized reservoir keeps every eligible game and the Algorithm R replacement branch
   (`preprocess_lichess.py:152-154`) never fires. We want the whole sample, not a subsample.

**First-run smoke test, before believing anything:** read
`<output-dir>/.sampling_summary.json` (`preprocess_lichess.py:194`) and compare `scanned`,
`eligible`, and `kept` (`preprocess_lichess.py:189-191`). If `eligible` is far below
`scanned`, the export is failing one of the §0 header gates, most likely the `Event`
substring check. If `eligible` is healthy but `kept` is lower, the fast path and
`parse_game` disagreed, which prints a warning (`preprocess_lichess.py:172-173`).

**Naming trap:** output files are `game_{kept:07d}.pkl` (`preprocess_lichess.py:177`), which
collide across runs. `AGENTS.md:35` already documents that flattening several such
directories together silently overwrites. Keep the closed-account output in its own
directory and prefix it before any merge.

### 1.2 A list of account IDs or usernames, no game data

**Verdict: the fetch step is build-from-scratch, but the cheapest build is a thin filter
bolted onto the existing scan loop, not a scraper.**

Nothing in the repo fetches games by account. Two sub-options:

**(a) Cross-reference against Open Database dumps we already stream.** This is the
recommended path and it is a genuinely thin change. The scan loop at
`preprocess_lichess.py:138-154` already visits every game in a monthly archive. Adding a
`--usernames-file` set-membership test on `headers.get("White")` / `headers.get("Black")`
next to `is_eligible` (`preprocess_lichess.py:66-80`, called at line 142) is a handful of
lines. Reservoir sampling would be disabled for this use (§1.1 note 2). This also collapses
fallback path 1 and fallback path 2 from `thesis_explained.md:430-433` into one operation:
the admin reply supplies the account list, and our existing corpus infrastructure supplies
the games.

Cost, using only figures already in the repo: monthly archives run about 99M to 103M games
(`analysis/corpus-pipeline-fix-plan.md:104-105`, anchored on published per-month counts
2021-04 = 99,184,138 and 2024-01 = 98,994,760), and the post-fix observed scan rate is
"~500+ games/s" (`AGENTS.md:32`). That is roughly 2.3 days of single-core wall time per
month scanned. Scanning the whole pre-October-2023 window is therefore weeks, not hours.
Budget for it, or narrow the months using the closure dates (see email question 3.4).

A username-only filter should be able to skip movetext parsing entirely and run much faster
than that 500 games/s figure, since that figure is for a pass that still walks the mainline
for the clock check (`preprocess_lichess.py:83-99`). python-chess's visitor API is the
mechanism for skipping movetext. **Uncertain:** I could not verify the exact visitor API in
this environment because python-chess is not installed here and `prototype/requirements.txt:2`
pins no version. Verify against the installed version before assuming the speedup, and
measure it rather than assuming it.

**(b) Fetch per account from the public Lichess API.** **Uncertain, and probably dead.** The
whole premise of §7.2 is that closed-account games stopped being publicly reachable on
2023-10-04. Whether the per-user game export endpoint still serves a closed account is not
something I can check from this repo, and I will not guess. It is email question 3.3. Even
if it works, the export must be requested with clock data included or every game fails the
§0 clock gate.

Once games exist in PGN form, both sub-options rejoin §1.1 unchanged.

### 1.3 Structured export (CSV or JSON or NDJSON) with non-PGN fields

**Verdict: thin conversion layer, and the layer should target PGN text, not the pickle.**

Two possible conversion targets:

- **Convert to PGN text, then run §1.1.** Recommended. One new file under `analysis/` or
  `prototype/src/`, no changes to any existing source. The payoff is that every §0 gate,
  including the clock-alignment reject at `format_data.py:78-80`, still runs on this data.
  The closed-account sample is the arm we can least afford to get quietly wrong, so it
  should go through the same validated code path as the training corpus rather than around it.
- **Convert straight to the `parse_game` dict and pickle it.** Not recommended. It requires
  replaying the moves through `board_to_array` (`format_data.py:17-30`) to build `Positions`
  ourselves, and it bypasses the clock/position length check. The only thing it saves is a
  PGN round trip.

Minimum fields the converter needs, mapped to the §0 record: move list (SAN or UCI), a
per-ply remaining-clock value, time control as base and increment, both pre-game ratings,
result, and the variant/rated flags. Anything the export gives beyond that is bonus.

**Clock unit trap.** If the export gives clocks as centiseconds or seconds rather than a
`H:MM:SS` string, the converter must emit the string form, because `time_to_seconds`
(`format_data.py:33-36`) splits on `:` and multiplies out, and both the API path
(`api.py:157`) and the dataset path (`chess_rating_net.py:79`) call it. A bare integer would
raise, not misparse, which is at least loud.

**Clock semantics trap, already burned once on this project.** The clock feature is a
*remaining-time countdown*, normalized with mean 273 and std 380 (`AGENTS.md:54`,
`api.py:45-46`, `chess_rating_net.py:63-64`). The anomaly corpus shipped with per-move
*think-time* in that field instead, and that is a known open gap
(`thesis_explained.md:422`). If the export's time field is "seconds spent on this move",
the converter must integrate it into a countdown, not pass it through. Confirm which one it
is before writing a line of the converter: email question 3.2.

### 1.4 Something else

Ranked by how much work each implies.

- **Game IDs only, no accounts.** Same shape as §1.2, minus the account-level list. Same
  uncertainty about whether closed-account games are fetchable at all.
- **A database or spreadsheet of moderation decisions with no game data.** Then there is no
  KS test at all: no suspicion scores can be computed. This is the outcome that triggers the
  contingency already written into `chapter3.tex:270` and `chapter4.tex:111`, and it should
  be recognized as such quickly rather than worked around.
- **Pseudonymized export (opaque per-account IDs replacing usernames).** Fully workable and
  ethically preferable (§4.7). `parse_game` stores `White`/`Black` as opaque strings
  (`format_data.py:88-89`) and nothing downstream interprets them, so an opaque ID costs
  nothing except that surrogate-key de-duplication (§2.4) gets harder.
- **A partial or conditional grant** ("aggregate statistics only", "no redistribution",
  "delete after the thesis"). Not a format problem, a §4 problem. Plan for it.
- **No reply, or a decline.** The default. `thesis_explained.md:437` and `chapter4.tex:111`
  already carry the language for reporting that outcome honestly.

### 1.5 Summary

| Export shape | Mapping | Entry point |
|---|---|---|
| PGN with `[%clk]` | direct reuse | `preprocess_lichess.py:108` (compress to `.zst` first) |
| PGN without `[%clk]` | unusable as-is | rejected at `format_data.py:74-80` |
| CSV / JSON / NDJSON with moves and clocks | thin converter to PGN, then reuse | new converter, then `preprocess_lichess.py:108` |
| Account IDs or usernames only | thin filter in the existing scan loop | `preprocess_lichess.py:66-80` and the loop at `138-154` |
| Game IDs only | fetch step, feasibility unverified | same as above once PGN exists |
| Moderation records with no games | no path | contingency at `chapter3.tex:270` |

---

## 2. What has to be true for the KS comparison to be valid

The committed design is a two-sample Kolmogorov-Smirnov test on per-game suspicion scores,
closed-account against a control matched on rating distribution and time-control mix
(`thesis_explained.md:435`, `chapter3.tex:270`, `chapter3.tex:255`, `chapter1.tex:81`). The
mock numbers at `thesis_explained.md:371-378` are explicitly illustrative
(`thesis_explained.md:321-323`), so 1,247 games is not a target and should not be treated as
one.

### 2.1 Minimum fields, per game

Required, or the game cannot be scored at all:

1. Full move list.
2. A per-ply remaining-time clock for every ply up to 100 (`format_data.py:74-80`).
3. `TimeControl` as base and increment (`preprocess_lichess.py:76-79`, `chess_rating_net.py:93`).
4. Both players' pre-game ratings (`format_data.py:82-83`).

Required, or the game can be scored but not *used*:

5. **Which side is the closed account.** The detector produces separate per-side scores
   (`anomaly.py:110-111`). Without the side label the only usable output is
   `combined_score` (`anomaly.py:120`), which adds the clean opponent's score to the
   suspect's and dilutes the effect roughly by half. That directly quadruples the required
   sample size, by the arithmetic in §2.3.
6. **The closure reason.** "Closed" is not "closed for engine use". §4.3 covers the ethics
   of conflating them; statistically, mixing other closure reasons into the treatment arm is
   pure label noise and attenuates the effect exactly proportionally (§2.3).

Strongly wanted:

7. A per-game flag for which games Lichess's own analysis actually acted on. An account-level
   ban does not mean every game that account played was engine-assisted, and per §2.3 that
   dilution is the single biggest driver of required sample size.
8. The account's rating history, or at least its rating before the flagged period. See §2.5.
9. Game date. Needed for era matching and for de-duplication against the training window.

### 2.2 Unit of analysis: accounts, not games

The KS test assumes independent draws. One banned account can supply hundreds of games, and
those games are not independent: they share a player, a style, and a device. Feeding them in
as if independent inflates the effective sample size and therefore the Type I error rate,
and would produce an impressively small p-value that means nothing.

Two acceptable fixes, and both must be chosen before looking at the data:

- **One game per account**, sampled by a fixed rule. Then n is the number of distinct
  accounts, and all of §2.3 applies directly. Simplest to defend to a panel.
- **All games, with a cluster bootstrap over accounts for the p-value.** Uses more data but
  the reported n is still effectively the account count. `analysis/scripts/paired_bootstrap.py`
  is the existing in-repo pattern for a stdlib resampling test and can be mirrored.

Either way, **the sample-size numbers below are counts of distinct closed accounts, not
counts of games.** An export of 5,000 games from 12 accounts is a sample of 12.

### 2.3 Sample-size justification

For the two-sample KS test, the α = 0.05 rejection threshold is
`D_crit = c(α) * sqrt((n+m)/(n*m))` with `c(0.05) = 1.3581`. With equal groups that is
`1.3581 * sqrt(2/n)`, which sets a hard floor on what is detectable at all:

| n = m | smallest detectable D at α = 0.05 | same, Bonferroni over 25 strata |
|---|---|---|
| 100 | 0.192 | 0.263 |
| 200 | 0.136 | 0.186 |
| 300 | 0.111 | 0.152 |
| 500 | 0.086 | 0.118 |
| 1000 | 0.061 | 0.083 |

For power, requiring the observed statistic to clear that threshold with 80% probability
gives `n >= 2 * (c(α) + z_{0.8} * σ)² / Δ²` with σ = 0.7071 (the worst-case standard
deviation of the empirical CDF difference), which is `n >= 7.62 / Δ²` per group:

| true sup-distance Δ | n per group (conservative rule) | n per group (simulated 80% power floor) |
|---|---|---|
| 0.30 | 85 | 45 |
| 0.25 | 123 | 66 |
| 0.20 | 191 | 102 |
| 0.15 | 340 | 176 |
| 0.10 | 763 | 362 |

The simulated column comes from a 4,000-replication Monte Carlo of the exact two-sample KS
statistic under a deliberately unfavourable alternative (the two CDFs coincide everywhere
except a single crossing point), binary-searched to 80% power. The closed-form rule
over-delivers, at 0.98 to 1.00 power at its own recommended n, and is about 2x conservative
for that alternative family. **Plan on the conservative column and treat the simulated
column as the absolute floor.** A Type I check under the null returned 0.040 against a
nominal 0.05, which is the expected mild conservatism from the discreteness of the statistic
at these sample sizes.

**Unequal allocation.** Control games are cheap and closed-account games are not, so
oversample controls. The effective size is `n_eff = nm/(n+m)`, which caps at n as m grows,
so the ceiling on this trick is a factor of 2. At Δ = 0.20:

| allocation | closed accounts needed | controls | simulated power |
|---|---|---|---|
| 1:1 | 191 | 191 | 0.997 |
| 1:2 | 144 | 288 | 0.993 |
| 1:5 | 115 | 575 | 0.987 |
| 1:10 | 105 | 1050 | 0.979 |

**1:5 is the recommended allocation.** It captures most of the available gain (40% fewer
closed accounts needed than 1:1) and going past it buys almost nothing.

**Label noise multiplies all of this.** If only a fraction π of the closed-account games are
truly engine-assisted, the treatment arm's CDF is the mixture
`F_obs = π·F_cheat + (1-π)·F_clean`, so `sup|F_obs - F_control| = π · Δ_true` exactly. Since
required n scales as 1/Δ², **required n scales as 1/π².** At π = 0.5, four times as many
accounts. At π = 0.25, sixteen times. This is the whole reason field 7 in §2.1 (per-game
flags) is worth asking for, and it is the number to quote if anyone asks why the email is
so specific.

**The bottom line to carry into the email:** roughly 150 to 200 distinct accounts closed
specifically for engine use, at 1:5 control allocation, detects a moderate effect
(Δ = 0.20) with high confidence. Below about 45 accounts, nothing but a very large effect
is detectable and the result should not be reported as confirmatory.

### 2.4 Control construction

- **Draw controls from the held-out test split only.** The model must not have trained on
  the control games, or the control arm's scores are optimistically biased and the whole
  comparison tilts. The split is manifest-controlled and seed-fixed (`AGENTS.md:31`).
- **Check the treatment arm for training contamination too.** The corpus window is
  2021-04 to 2024-01 (`AGENTS.md:26`) inside a stated scope of 2021-04 to 2024-07
  (`chapter1.tex:81`), and pre-October-2023 closed-account games fall squarely inside it. A
  closed-account game may literally already be a training example. This is not a theoretical
  worry, it is the expected case for fallback path 1.
- **De-duplicate with a surrogate key, because there is no game ID.** As noted in §0,
  `parse_game` does not retain `Site` (`format_data.py:85-95`). The available surrogate is
  `(White, Black, Result, tuple(Moves))` from `format_data.py:88-93`, which is effectively
  unique. Build that index over the corpus pickles once and reuse it. Note that a
  pseudonymized export (§1.4) breaks the username half of this key, leaving
  `(Result, tuple(Moves))`, which is still adequate but should be spot-checked.
- **Match on strata, do not just report that the distributions "look similar".** Use the
  five rating brackets already committed in `chapter3.tex:187` (0-1200, 1201-1600,
  1601-2000, 2001-2400, 2401+) crossed with the five time-control buckets from
  `categorize_time_control` (`format_data.py:98-109`, boundaries 29 / 179 / 479 / 1499
  seconds on `base + 40*inc`). Exact 1:5 matching within each of the 25 cells makes balance
  true by construction, which is much stronger than a post-hoc balance test.
- **Bracket on the closed account's own rating**, and score that side only
  (`anomaly.py:110-111`). The unit is a (game, side) pair, not a game.
- **One pooled KS test is the pre-registered primary.** Per-cell tests are exploratory only,
  and if they are reported at all they carry the Bonferroni penalty in the §2.3 table (25
  cells moves `c(α)` from 1.3581 to 1.8585, inflating the detectable D by 37%).
- **Apply an identical minimum-ply floor to both arms** and fix it in advance. The score is
  an attention-weighted average over plies (`anomaly.py:109-111`), so short games produce
  high-variance scores. The 100-ply cap (`format_data.py:41`) already applies to both.
- **Match on era as well if the sample spans years.** Failing that, report the date
  distributions of both arms.

### 2.5 Scoring must be identical across arms, and there is a live confound

- Same checkpoint, same normalization constants. Production stays on ratings mean 1514 and
  std 366 (`AGENTS.md:34`, `api.py:43-44`), clocks 273 and 380 (`api.py:45-46`). Do not run
  the arms on different constants.
- **The current served checkpoint cannot produce suspicion scores at all.** `api.py:77-85`
  loads `model_55.pth`, which carries no anomaly branch, and `api.py:223-228` then emits
  zeros. The KS test needs a batch scoring path against an attention-enabled checkpoint (the
  `abl_*` attention runs from `AGENTS.md:31`). That path does not exist yet and is the one
  piece of real engineering that could be built now, before the data arrives, since it is
  format-independent.
- **The R_baseline choice is a genuine confound, not a detail.** `anomaly.py:9-18` defines
  `R_baseline` as the player's known pre-game Glicko-2 rating, with a fallback to the model's
  own final-ply prediction (`api.py:200-203`). Because `rating_after_last_game` is never
  written by `parse_game` (§0), that fallback is what fires today for pipeline pickles.
  Neither option is clean here:
  - Using the header rating (`WhiteElo`/`BlackElo`) is the documented intent, but **a banned
    account's rating is itself inflated by the cheating**. Their Glicko-2 number already
    climbed on engine-assisted wins, so `|prediction - rating|` can be small precisely for
    the accounts we most expect to flag. This biases the treatment arm's scores *downward*
    and works against detecting an effect.
  - Using the model's own final-ply fallback removes that bias but changes what the score
    means, and it makes the score self-referential.

  Recommendation: pre-register the header-rating version as primary (it is what Chapter 3
  describes), report the fallback version as a sensitivity check, and if the export includes
  rating history, add a third arm using the account's rating from before the flagged period.
  Whichever is chosen, both arms must use the same one.
- `scipy` is not listed in `prototype/requirements.txt` (lines 1-10). It arrives transitively
  through `scikit-learn` (line 7), but `scipy.stats.ks_2samp` is load-bearing for a headline
  result, so pin it explicitly or implement the statistic in stdlib the way
  `analysis/scripts/paired_bootstrap.py` does.
- Report the D statistic itself as the effect size, with a bootstrap confidence interval, and
  alongside it the probability of superiority (equivalently the ROC-AUC of ranking closed
  against control by suspicion score). The AUC form is directly comparable to the synthetic-corpus
  ROC-AUC results, and a p-value alone will not satisfy a panel.

---

## 3. Follow-up email to Lichess: questions ready to send

Framing for whoever sends it: lead by saying we can work with whatever format is convenient
and that we are not asking them to build anything. Everything below is a clarification, and
questions 3.1, 3.2, 3.5 and 3.11 are the ones that actually decide whether the data is
usable. If the reply budget is short, ask those four.

**Format and delivery**

1. What file format will the export take (PGN, CSV, JSON, NDJSON, database dump), roughly
   how large is it, and how would it be delivered?
2. **Does it include per-move clock times?** If so, is each value the time *remaining* on
   that player's clock after the move (the `[%clk]` convention in the public database), or
   the time *spent* on that move? Our model's clock feature is trained on the remaining-time
   convention, and games without per-move clocks cannot be used at all.
3. If the export is a list of accounts or game IDs rather than the games themselves: is there
   a supported way for us to retrieve the games, given that public search of closed-account
   games was disabled in October 2023?

**Scope and selection**

4. How many distinct accounts, how many games per account, and what date range?
5. How were the accounts and games selected (all closures in a period, a random sample, or a
   curated set)? We need to describe the selection rule in the methodology, and a curated set
   would change how we report the result.
6. Are the games restricted to rated standard chess, or do they include variants and
   correspondence? Our pipeline is scoped to rated standard with a numeric base-plus-increment
   time control.
7. What rating range and time-control mix does the sample cover? We build a matched clean
   control sample across five rating brackets and five time-control buckets, so an uneven mix
   is fine as long as we know it.

**Labels, which is where the statistical value is**

8. **What was each account closed for?** We only want accounts closed for engine assistance.
   Closures for other reasons, or self-closures, would be label noise in our comparison, and
   the sample size we need grows with the square of that contamination.
9. Is there any per-game or per-move indication of which games were actually flagged? An
   account-level ban does not mean every game that account played was engine-assisted, and a
   per-game flag would very substantially reduce the sample size we need.
10. Would you be willing to include the closure date, and the account's rating at the time of
    each game (or a rating history)? A banned account's final rating already reflects the
    assisted results, which biases our baseline comparison.

**Fields**

11. Which of the standard PGN headers will be present: `Event`, `Site`, `UTCDate`, `UTCTime`,
    `White`, `Black`, `WhiteElo`, `BlackElo`, `TimeControl`, `Termination`, `Variant`,
    `Result`? Our filter currently requires `Variant`, a rated marker in `Event`, numeric
    Elos, and a `base+increment` `TimeControl`.
12. Are `WhiteElo` and `BlackElo` the pre-game Glicko-2 ratings for that time control, as in
    the public database?
13. Do the game identifiers match the ones in the public monthly database dumps? We need to
    check whether any of these games are already in our training set and exclude them.

**Control sample, worth asking because it removes a confound**

14. Could the export also include a matched sample of games from accounts in good standing,
    from the same period, rating range, and time controls? If both arms came from the same
    export, our comparison would not have to control for a difference in data provenance.
    If not, we will build the control from the public database.

**Permissions, privacy, publication**

15. Under what terms may we use this data? The public database is CC0, but we assume a
    direct export is not automatically covered by that and would like a written statement we
    can cite.
16. May we publish aggregate results only, and are we correct in assuming we must not publish
    or redistribute the raw data, individual games, or usernames?
17. Would you prefer the usernames pseudonymized or removed before sending? We do not need
    real usernames for anything except de-duplicating against the public dumps, and we are
    happy to receive opaque identifiers.
18. Do you require deletion after a set period, or on request?
19. How would you like the dataset cited or acknowledged, and would you like to see the
    relevant chapter before it is submitted?

**Practicalities**

20. Please send only what is described above. We do not want chat logs, reports, moderator
    notes, IP or device data, or account email addresses, and would rather they were stripped
    before sending than received and deleted.
21. Our thesis timeline means data arriving after [date] cannot be included. If a full export
    is not workable, would a smaller sample, or aggregate suspicion-relevant statistics, be
    easier to provide?

---

## 4. Ethical considerations that are genuinely new

Already covered, and not repeated here: CC0 provenance and the "no PII beyond public
usernames" position (`chapter3.tex:298`); the fair-play framing as a review flag rather than
an accusation or ban mechanism (`chapter3.tex:300`); false-positive risk as a disclosed
limitation (`chapter1.tex:83`); transparency of limitations (`chapter3.tex:302`); public
release of weights, code, seeds, and splits (`chapter3.tex:304`). The points below are the
ones that only arise once the data is a private transfer of real sanctioned accounts.

**4.1 The CC0 argument does not cover this data, so the ethics paragraph as written does not
either.** `chapter3.tex:298` grounds the entire consent position in the Open Database's CC0
license. A one-off administrative export is not published under that license, and consent
cannot be inherited from it. If this data is used, `chapter3.tex:296-304` needs an added
sentence describing the separate permission under which it was obtained. This is a manuscript
change that has to happen before submission, not after, and it depends on getting email
question 3.15 answered in writing.

**4.2 We would be re-exposing something the platform deliberately stopped exposing.** Lichess
disabled public search of closed-account games in October 2023 (`README.md:129`,
`thesis_explained.md:428`). Whatever their reason, publishing anything that lets a reader
identify who was in our treatment group would work against that decision using data they
gave us in good faith. Concretely, two places in the current code would leak this by default
if pointed at the closed-account sample:

- `parse_game` writes `White` and `Black` usernames into every pickle
  (`format_data.py:88-89`), so usernames land on disk in the processed data.
- `api.py` logs the full PGN headers into `logs/predictions.jsonl` alongside the suspicion
  scores (`api.py:68-72`, `api.py:245`, `api.py:283`). That file is gitignored
  (`.gitignore:57`) so it will not be committed, but it still persists a username-to-suspicion-score
  mapping on local disk.

Neither is a bug for public database games. Both need a hashing or redaction step for this
dataset specifically.

**4.3 Closed does not mean cheated, and cheated does not mean guilty.** Accounts close for
self-requested deletion, for terms-of-service violations unrelated to engines, for ban
evasion, and because their owner died. §2.3 covers why that is a statistical problem. The
ethical half is separate: we must not describe this sample as "cheaters" anywhere in the
thesis. The defensible phrasing attributes the label to its source, something like "accounts
closed by Lichess for engine assistance, per Lichess's own moderation decision". The
manuscript already treats Kaladin and Irwin as the systems whose decisions produce this
sample (`chapter1.tex:118`, `chapter1.tex:120`, `chapter3.tex:272`), so the attribution is
consistent with what is already written.

**4.4 These subjects cannot consent, object, correct the record, or withdraw.** Closed
accounts include banned users who have no channel back to the platform and, in some cases,
deceased users. Standard human-subjects practice assumes a subject can withdraw; here nobody
can. This is new relative to both of the study's other data sources: the synthetic corpus
involves no humans at all, and the public database at least covers accounts whose owners are
still present and whose games were already public. Practical consequences: aggregate
reporting only, no per-game case studies drawn from this dataset, no worked examples in the
thesis or in the prototype demo, and no screenshots.

**4.5 The data cannot be re-fetched, so both a scientific and an ethical guarantee weaken.**
There is no second request and no correction pass. Two separate consequences:

- *Handling.* Preserve the raw export byte-for-byte with a recorded checksum, do every
  transformation as a derived copy, and keep a written chain of custody. A processing mistake
  discovered late cannot be fixed by re-downloading, which is exactly how the public database
  months are recoverable today.
- *Reproducibility.* `chapter3.tex:304` commits to releasing seeds, splits, and evaluation
  scripts so results can be reproduced. Nobody outside the team can reproduce this one,
  because nobody else can obtain the input. That has to be stated plainly in Chapter 4
  alongside the result, not buried: the synthetic bot-versus-bot result is the reproducible
  anomaly evidence, and this one is corroborating but not independently checkable. The
  `\TODO` at `chapter4.tex:111` is where that sentence belongs.

**4.6 A published low score is an implicit challenge to a real ban.** Chapter 1 and 3 already
handle false positives as a system limitation. What is new is the direction: if our detector
scores a genuinely banned account as unremarkable, and that is traceable to an individual, we
have published a quasi-exoneration of a real moderation decision we have no standing to
review. The mirror case is worse. The thesis should state that this evaluation measures our
detector against Lichess's labels and is not an audit of Lichess's moderation, and should
report no per-account results at all.

**4.7 Data minimization and repository hygiene, which need action before the data arrives.**
Ask for the minimum field set in §2.1 and nothing else, and prefer pseudonymized identifiers
(email question 3.17). Note that `.gitignore` currently covers `data/*.pgn`, `data/*.csv`,
and `data/*.jsonl` (`.gitignore:44-46`) but not `.zst`, `.ndjson`, `.json`, or any file
placed outside `data/`. An export dropped anywhere else, or in a compressed form, is not
ignored. Add the patterns before the file lands, not after, since this repository is public.

**4.8 Do not publish an evasion guide.** A detailed per-feature account of what the detector
keys on in *real* engine-assisted play is more actionable to someone trying to evade
detection than the same account for synthetic games would be. This is a small point and it
should not be used to justify vagueness about the method, but the level of operational detail
is worth a deliberate decision rather than an accident.

---

## 5. What can be done now, before any reply

Ordered by value, and none of it depends on the export's format.

1. **Build the batch anomaly-scoring path against an attention checkpoint.** Per §2.5 the
   served model cannot emit suspicion scores at all (`api.py:223-228`). This is the real
   blocker and it is entirely format-independent.
2. **Build and freeze the control sample.** The rating-bracket by time-control cell counts,
   drawn from the test split only, can be prepared now and simply subsetted once the
   treatment arm's mix is known.
3. **Build the surrogate-key overlap index** over the corpus pickles, per §2.4.
4. **Pre-register the analysis**: α, the allocation ratio, the R_baseline choice and its
   sensitivity check, the minimum-ply floor, pooled-primary versus per-cell-exploratory, and
   the effect-size reporting. `analysis/stage2-tuning-preregistration.md` is the existing
   in-repo pattern for this. Doing it before the data arrives is what makes the result
   confirmatory rather than exploratory.
5. **Add the `.gitignore` patterns** from §4.7.
6. **Pin `scipy`** in `prototype/requirements.txt`, or write the KS statistic in stdlib.

---

## 6. Open uncertainties

Stated explicitly so nobody downstream mistakes them for verified facts.

- Whether the Lichess public API can still export games for a closed account. Not checkable
  from this repo. Email question 3.3.
- The exact python-chess visitor API for skipping movetext during a header-only scan.
  python-chess is not installed in the environment this was written in and
  `prototype/requirements.txt:2` pins no version. Verify before relying on the speedup in
  §1.2(a).
- The realized throughput of a username-filter scan. The ~500 games/s figure in `AGENTS.md:32`
  is for a different pass, one that still walks the mainline. Measure it, do not assume it.
- Whether Δ = 0.20 is a realistic effect size for this detector on real engine-assisted play.
  It is used in §2.3 as a planning anchor, not as a prediction. Once the synthetic evaluation
  from `chapter3.tex:268` produces real suspicion-score distributions, recompute the required
  n from the observed separation there instead of from a round number.
