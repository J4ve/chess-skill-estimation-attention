# Schema completeness check: stored per-game fields vs. everything Chapter 4 plans to measure

Task: CLAUDE_TASK.md Part E, second-opinion pass. Written 2026-08-17. Revised 2026-08-17
after an adversarial verification pass over every load-bearing claim; the claims that did
not survive are recorded in "Considered and ruled out" at the end rather than deleted
silently.

Question this answers: the corpus pipeline deletes each month's multi-gigabyte raw
archive right after processing it. Is there any planned Chapter 4 measurement that
needs a PGN field the pipeline is not storing, such that finding out later would be
expensive or unfixable?

## Verdict up front

**No. Do not halt the pipeline.** Every test, metric, ablation, and subgroup
breakdown committed in `chapter3.tex`, scaffolded in `chapter4.tex`, listed as Tests
1 through 8 in `contexts/essentials/thesis_explained.md` §6, or prescribed in
`analysis/170k-verification.md` is computable from the nine stored fields, from
values derivable from them, or from data sources outside this corpus entirely.

There is no category (ii) finding (recoverable only by re-downloading and re-scanning)
and no category (iii) finding (not recoverable at all) against any planned
measurement. Five findings turned up, three medium and two low. None of them requires
re-preprocessing the Lichess corpus: they are fixable in documentation, in training/eval
code, or in the synthetic-corpus tooling. They are listed in "Findings" below.

Two things do need a captain decision, and both are time-sensitive only in the weak
sense that a later change makes the corpus schema non-uniform across months. Neither
is a reason to stop the run. They are in "Decisions for the captain".

## The schema, confirmed by reading the code

`parse_game` (`prototype/src/format_data.py:39-95`) returns exactly nine keys
(`:85-95`):

| Field | Type as stored | Source |
|---|---|---|
| `WhiteElo` | string of digits | `WhiteElo` header (`:82`) |
| `BlackElo` | string of digits | `BlackElo` header (`:83`) |
| `White` | string, Lichess username | `White` header (`:88`) |
| `Black` | string, Lichess username | `Black` header (`:89`) |
| `Result` | string, e.g. `1-0` | `Result` header (`:84`) |
| `Clocks` | list of raw `H:MM:SS` strings, one per stored ply | `[%clk ...]` comments (`:66-69`) |
| `Positions` | list of `torch.Tensor(12,8,8)` float32, one per stored ply | `board_to_array` (`:63`) |
| `Moves` | list of UCI strings, one per stored ply | `move.uci()` (`:64`) |
| `Time` | raw `TimeControl` header string, e.g. `180+2` | `:52`, `:94` |

`White` and `Black` are present. The task brief's field list is correct, and it
supersedes `chapter3-4-process.md:31`, which still lists a seven-key schema without
the usernames (stale since commit `08ae9f0`, 2026-08-14 23:13, "Record player names and
reject partial-clock games in parse_game").

The list is capped at 100 plies (`--max-plies` default 100,
`prototype/src/preprocess_lichess.py:117`). Nothing else from the PGN survives.

### What is discarded, measured on real data

I downloaded the first 30 MB of `lichess_db_standard_rated_2021-06.pgn.zst` by ranged
HTTP GET and decompressed it (92,522 complete game blocks). Header coverage over that
prefix:

| Discarded header | Games carrying it | Example |
|---|---|---|
| `Site` (unique game ID) | 92,522 / 92,522 (100%) | `https://lichess.org/e9ZYTaCJ` |
| `UTCDate` | 100% | `2021.06.01` |
| `UTCTime` | 100% | `00:00:30` |
| `ECO` | 100% | `C45` |
| `Opening` | 100% | `Scotch Game` |
| `Termination` | 100% | `Normal` |
| `WhiteRatingDiff` / `BlackRatingDiff` | 99.8% | `+5` / `-5` |
| `WhiteTitle` / `BlackTitle` | 578 / 575 (0.62%) | `GM`, `IM`, `FM` |

So a unique game ID, an exact timestamp, an ECO code, and a termination reason all
exist in every source record and are all being thrown away. The rest of this document
is about whether any of them is load-bearing. The short answer is that none is, for
anything currently planned.

### Month provenance is preserved, but only by path convention

Nothing inside a pickle records which month it came from. The month survives in two
places: the output directory name (`data/processed_games/{YYYY-MM}/`, written by the
pipeline) and the `{month}_` filename prefix required when flattening into the
training directory (`AGENTS.md:35`, `hpc-operations-guide.md:255-258`). Per-month
`.sampling_summary.json` files also record it explicitly
(`prototype/src/preprocess_lichess.py:182-196`). That is enough for every planned
month-granular analysis, but it is a convention, not a stored field. See Finding 5.

### The kept sample is deterministically re-derivable

This materially softens the "not fixable by retry" framing in the task brief, so it is
worth stating precisely. Recovery of a lost month does not require guessing which
30,000 games were kept. It is a deterministic re-run:

- The reservoir RNG is seeded at a fixed default of 42
  (`prototype/src/preprocess_lichess.py:119`), and neither `corpus_stream.sh` nor the
  parallel worker passes `--seed`.
- Algorithm R (`:149-154`) is deterministic given the seed and the order of the
  eligible-game stream, and that order is the archive's own byte order.
- Output filenames are assigned in reservoir order (`:177`).
- The published monthly archive is immutable, and the pipeline pins its exact byte
  size (`EXPECTED_SIZES`, `corpus_stream_parallel.sh:58-68`), so a silent republish
  would be caught.

So re-running the identical code with the identical seed on a re-downloaded archive
reproduces the same 30,000 games, in the same order, with the same filenames. Recovery
costs a ~30 GB download plus a full scan per month (roughly 50 core-hours per month at
the post-fix ~500 games/s against ~92M games), which is expensive but bounded, and it
is not a lossy re-sample. Three conditions must hold: unchanged `is_eligible` /
`has_full_clocks` / `--max-plies`, unchanged seed, and a pinned `python-chess` version
(1.10.0 per `chapter3-4-process.md:48`). The `has_full_clocks()` fast path added
2026-08-17 is documented as sampling-identical to the code it replaced (`AGENTS.md:32`,
`analysis/corpus-pipeline-fix-plan.md` §3), and I confirmed by reading both that the
eligibility predicate is the same: at least one mainline ply, and every mainline ply up
to the cap carries a `[%clk]` comment (`preprocess_lichess.py:83-99` vs.
`format_data.py:59-80`).

---

## The table

One row per planned evaluation, metric, ablation, or subgroup breakdown. "Recoverable"
column applies only to rows that are not a plain yes: (i) fixable later in
training/eval code with no re-preprocessing, (ii) fixable only by re-running the scan
on a re-downloaded archive, (iii) not recoverable at all.

| # | Planned evaluation / metric | Where it is committed | Computable from stored schema? | Recoverable |
|---|---|---|---|---|
| 1 | Test 1: test-set MAE in rating points | `thesis_explained.md:262-269`, `chapter3.tex` `tab:eval-metrics`, `chapter4.tex:84-85` | **Yes.** Labels from `WhiteElo`/`BlackElo`; inputs from `Positions` + `Clocks` | n/a |
| 2 | Test 1: RMSE (secondary) | `chapter3.tex` `tab:eval-metrics`, `chapter4.tex:84-85` | **Yes.** Same fields | n/a |
| 3 | Test 1: 95% bootstrap CI on MAE | `thesis_explained.md:264` | **Yes.** Resamples the per-game error dump | n/a |
| 4 | Test 1: naive predict-mean floor (346) | `chapter3.tex` `tab:eval-metrics` | **Yes.** `WhiteElo`/`BlackElo` only | n/a |
| 5 | Test 2: attention ablation, ΔMAE, 5 seeds, mean ± sd (n−1) | `thesis_explained.md:271-278`, `chapter3.tex` Reliability, `chapter4.tex:84-95` | **Yes.** Architecture-side; no data field involved | n/a |
| 6 | Test 2: pre-registered stage-2 deepened-CNN arm | `chapter3.tex` `sec:stage2-depth`, `chapter4.tex:33` | **Yes.** Architecture-side | n/a |
| 7 | Test 3: paired bootstrap, 10,000 resamples, over per-game absolute errors | `thesis_explained.md:280-282`, `chapter4.tex:97` | **Yes.** `test(per_game_csv=...)` already dumps `game_id, white_err, black_err, *_signed_err, time_control, white_elo, black_elo` (`chess_rating_net.py:439-445`); `game_id` is the pickle filename stem (`:107`), which is a stable within-corpus key | n/a |
| 8 | Test 4: MAE per time control (5 buckets) | `thesis_explained.md:284-286`, `chapter4.tex:126-134` | **Yes.** `Time` parsed as base+increment, bucketed by base + 40×inc (`chess_rating_net.py:93-95`, `format_data.py:98-109`) | n/a |
| 9 | Test 4: per-time-control test-game counts `n` | `chapter4.tex:128-134` `\TODO{n}` | **Yes.** Count of rows per bucket | n/a |
| 10 | Test 4: MAE per rating bracket (5 brackets), aggregated over player-observations (2N) not games | `chapter3.tex` Instrument, `chapter4.tex:154-160`, prescription in `170k-verification.md:664-679` | **Yes.** Brackets from the true `WhiteElo`/`BlackElo`; the per-game CSV already carries both sides' errors and both Elos | n/a |
| 11 | Test 4: per-bracket `n` and CIs | `170k-verification.md:681-689` | **Yes.** Same source | n/a |
| 12 | Test 4: factor-of-two subgroup threshold check | `chapter3.tex` `tab:eval-metrics`, `chapter4.tex:179` | **Yes.** Derived from rows 8 and 10 | n/a |
| 13 | Percentage of games truncated at the 100-ply cap, per time control | `chapter3.tex:229` (explicit promise), `chapter4.tex:37` `\TODO` | **Partial, needs a wording decision.** "Reached the cap" (`len(Moves) == 100`) is exact. Strictly "exceeded 100 plies" is not separable, because the original ply count is discarded. Measured gap on a real month: 0.00 to 0.55 percentage points depending on time control. See Finding 4 | (iii) for the exact original ply count; (i) for the reportable metric under a corrected definition |
| 14 | Realized per-time-control corpus composition, counts and percentage shares | `chapter3.tex:171`, `chapter4.tex:37` `\TODO` | **Yes.** From `Time` per game, cross-checkable against per-month `.sampling_summary.json` (`preprocess_lichess.py:182-196`) | n/a |
| 15 | Test 5: ROC-AUC for `S_att`, `S_max`, `S_mean` on the synthetic corpus | `thesis_explained.md:296-301`, `chapter3.tex` Anomaly Validation Protocol, `chapter4.tex:107` | **Yes in principle, blocked on two synthetic-corpus defects.** Not a Lichess-schema issue. See Findings 2 and 3 | (i) |
| 16 | Test 5: precision at calibrated 1% and 5% FPR, thresholds fit on validation only | `chapter4.tex:107` | **Yes.** Same as row 15 | (i) |
| 17 | Test 5: AUC broken down by substitution rate, by engine, by Maia band | `chapter4.tex:107`, `170k-verification.md:764-767` | **Yes.** `substitution_rate`, `engine`, `maia_band` are stored per synthetic game (`generate_anomaly_corpus.py:177-182`) | n/a |
| 18 | Test 5: "matched clean-versus-substituted pairs" and a paired bootstrap over pairs | `chapter3.tex` Synthetic cheat generation, `chapter4.tex:107`, `170k-verification.md:761-763` | **Uncertain, needs captain decision.** No cross-case pairing key exists and the generator does not produce cross-case pairs. A within-game side pairing (suspect side vs. clean opponent side) is available from `suspect_color`. See Finding 3 | (i) for the within-game reading; regeneration otherwise, ~10 to 14 h, no Lichess data involved |
| 19 | Test 5: cross-evaluation on the Kaggle Chess Cheating Dataset | `chapter3.tex` Anomaly Validation Protocol, `chapter4.tex:107` | **Yes.** External dataset with its own labels; needs its own loader, not this schema | n/a |
| 20 | Test 6: KS two-sample test, closed-account vs. clean-account suspicion scores | `thesis_explained.md:303-309` and `:426-437`, `chapter3.tex` Real-world sample, `chapter4.tex:111` | **Uncertain, gated on data availability, not on schema.** Schema-wise the corpus supports it: fallback path 1 needs pre-October-2023 dumps cross-referenced against closure notices, and the corpus months 2021-04 through 2023-09 store `White`/`Black`, which is exactly the join key that cross-reference needs | n/a |
| 21 | Test 6: control sample matched on rating bracket and time control | `chapter3.tex` Real-world sample, `chapter4.tex:111` | **Yes.** `WhiteElo`/`BlackElo` + `Time` | n/a |
| 22 | Test 7: per-move end-to-end inference latency, median / p95 / p99 / worst | `thesis_explained.md:311-313`, `chapter4.tex:183`, `170k-verification.md:1367-1373` | **Yes.** Measured on the prototype; no corpus field involved | n/a |
| 23 | Test 8: offline batch vs. live streaming agreement on 100 matched games | `thesis_explained.md:315-317`, `chapter4.tex:183` | **Yes.** A model-equivalent PGN is reconstructable from `Moves` + `Clocks` + `Time` + `Result` + Elos, and the API path is PGN-driven (`api.py:145`, `:267`, `:292`). If the 100 games must come from the held-out test partition, live mode has to be a replayed stream rather than a real Lichess subscription, because no game ID is stored. See Decision B | (i) |
| 24 | Chronological / temporal holdout: hold out the final month as a second test-only partition | `170k-verification.md:1327-1334` | **Yes, at month granularity only,** via the directory name or the `{month}_` flatten prefix, not via any stored field. Any finer temporal split is impossible. Nothing plans a finer one. See Finding 5 | (i) |
| 25 | Player-disjoint split as an honesty check, and a measured train/test player-overlap rate | `170k-verification.md:137-156` prescription 2 | **Yes.** `White`/`Black` are stored. Needs a one-time index pass over the corpus, which is a short prefix read per file rather than a full load; cost and caveats in "Item 3" below. Whether a *strict* player-disjoint 72/18/10 is achievable is a separate, unmeasured question. See Finding 1 and Decision A | (i) |
| 26 | MAE and signed bias vs. ply index; split by side to move | `170k-verification.md:1336-1345` | **Yes.** Per-ply predictions plus `Moves` ordering | n/a |
| 27 | Signed deviation reported alongside unsigned `d_t`, to separate sandbagging from engine use | `170k-verification.md:1357-1361` | **Yes.** Signed error columns are already in the per-game dump (`chess_rating_net.py:440-445`) | n/a |
| 28 | Titled-player brilliancy control set (guards "surprising = suspicious") | `170k-verification.md:1362-1365` | **Uncertain but recoverable.** `WhiteTitle`/`BlackTitle` are discarded (present on 0.62% of source games). Titles are re-lookupable from the stored usernames via the Lichess public API, or the control set can be built from a fresh targeted sample outside this corpus | (i) |
| 29 | `R_baseline` for the anomaly module: Glicko-2 rating for that time control at game start | `chapter3.tex:227` and the `d_t` definition | **Yes.** That is exactly `WhiteElo` / `BlackElo`. Note the code currently defaults to a self-prediction fallback (`api.py:199-203`), which contradicts the manuscript, but that is a code defect already logged at `170k-verification.md:721-727`, not a schema gap | n/a |
| 30 | Time-spent (hesitation) feature, tuning-stage ablation | `170k-verification.md:443-445` | **Yes.** Derivable from consecutive `Clocks` values plus the increment in `Time` | n/a |
| 31 | Rating-normalization re-fit diagnostic (corpus-actual mean/std) | `AGENTS.md:34`, `analysis/seed-rerun-results.md` | **Yes.** `WhiteElo` / `BlackElo` | n/a |
| 32 | Opening / ECO subgroup breakdown (**not currently planned**) | no manuscript commitment found | **Yes if ever wanted.** Re-derivable offline from the stored `Moves` against an open ECO table such as `lichess-org/chess-openings`. Will not be byte-identical to the discarded header unless the same table version is used | (i) |
| 33 | Engine-agreement comparator baseline, Regan-style (**not currently planned**) | Regan and Haworth are cited in `chapter3.tex` and `chapter4.tex:175` but no such measurement is committed | **Yes if ever wanted.** `Moves` replays exactly from the standard start position, so Stockfish evaluations can be recomputed. Costs compute, not data | (i) |
| 34 | Termination-reason filtering or breakdown (**not planned**) | no manuscript commitment found | **No.** `Termination` is discarded. Checkmate / stalemate / insufficient material are re-derivable by replaying `Moves`; resignation vs. timeout vs. abandonment vs. draw agreement are not | (iii) unless the month is re-scanned, but nothing plans to use it |
| 35 | Sub-month temporal analysis, or dedup by unique game ID, or a per-game provenance link back to Lichess (**not planned**) | no manuscript commitment found | **No.** `UTCDate`, `UTCTime`, `Site` are discarded | (ii): recoverable by deterministic re-run per the section above. Nothing plans to use it |

---

## Findings

Sorted by severity. Every one is category (i): none needs the Lichess corpus
re-preprocessed.

| # | Severity | What it is | Where the fix lives |
|---|---|---|---|
| 1 | Medium | Six repo sites still describe the pre-`08ae9f0` schema or promise a leakage check nothing implements; one of them is a rehearsed defense answer | Documentation and prep material, plus `.tex` and its `revised/` mirror |
| 2 | Medium | Synthetic anomaly pickles have no `Time` key, so the dataset loader raises `KeyError` on all 90,000 of them | One script over the synthetic corpus |
| 3 | Medium | "Matched clean-versus-substituted pairs" is not what the generated corpus contains | Manuscript wording plus a captain call, or regeneration |
| 4 | Low | The truncation-rate promise is not exactly computable; the error is under 0.6 pp and quantified | Chapter 3 / Chapter 4 wording |
| 5 | Low | Month provenance survives only as a path convention | Flatten step writes a manifest |

### Finding 1 (medium, category i, manuscript- and panel-facing): stale schema claims in both directions

Three manuscript sites assert that no player identifiers are stored, which was true before
commit `08ae9f0` (2026-08-14 23:13, "Record player names and reject partial-clock games in
parse_game") and is false for the 1.2M corpus, which stores `White` and `Black`
(`format_data.py:88-89`):

- `chapter3.tex:191`, dataset summary table: "Data split & game-level random (no player
  identifiers stored in per-game data)".
- `chapter3.tex:225`: "Because the per-game records do not include player identifiers,
  the split is game-level random; a player-level leakage guarantee is not feasible with
  the current data schema."
- `chapter4.tex:195`: "Because the per-game records retain no player identifiers, the
  72/18/10 partition is game-level random and no player-level leakage guarantee is
  available."

This is the good kind of wrong: the schema is richer than the manuscript claims, so
nothing has to be re-collected. But it is load-bearing for the panel, because
`170k-verification.md:137-156` prescribes exactly the upgrade these sentences now block:
report the game-level random split as primary (comparability with 182) and a
player-disjoint split as the honesty check, with the delta stated. That prescription is
now executable and the manuscript still says it is impossible.

Note the manuscript is internally inconsistent on this already: `chapter3.tex:298`
(Ethical Considerations) says "No personally identifiable information beyond public
Lichess usernames will be collected", which presumes usernames are collected.

Six further sites in the repo repeat one of the two errors. I read every one of them.
Four still say the usernames are absent:

| Site | What it says | Note |
|---|---|---|
| `contexts/essentials/panel_attacks.md:480` (entry M7) | "The per-game records carry no player identifiers, preprocessing never keeps the White/Black headers" | **Highest risk.** This is a rehearsed answer intended to be spoken at the defense |
| `chapter3-4-process.md:31` | Lists a seven-key schema, `{WhiteElo, BlackElo, Result, Time, Clocks, Positions, Moves}` | Exactly the pre-fix return dict |
| `prototype/experiments/e8_data_and_eval.py:86-89` | Prints "no username/player-ID field, so a player-disjoint split CANNOT be built from the current pickles" | Wrong as a general statement, but *not* wrong about the data it reads: `:19` globs a 2024-07 subset, and 2024-02 through 2024-07 are `PROTECTED_MONTHS` built before the fix (`corpus_stream_parallel.sh:70-72`) |
| `contexts/model-improve-findings.md:71-76` | Repeats the seven-key schema and concludes "the player-leakage check cannot be built" | Frozen finding, but the conclusion no longer holds |

Two more over-claim a leakage guarantee that nothing implements. The split is over
filenames (`load_or_create_split`, `chess_rating_net.py:548-597`, with
`train_test_split(..., random_state=split_seed)` over sorted basenames at `:583-584`):

- `contexts/essentials/thesis_explained.md:222`: "**Player-level leakage check**: no single
  Lichess username is in more than one partition."
- `revised/chapter3_v3.md:145`: "A player-level leakage check will be applied so that no
  single Lichess username appears in more than one partition." `AGENTS.md:50` requires the
  `revised/chapter*_v3.md` mirrors to stay in sync with the `.tex`, and the `.tex` now
  disclaims this, so the mirror is out of sync as well as wrong.

One correction that this document previously reported as outstanding has in fact already
been made. `170k-verification.md:157-165` (prescription 4) called the M7 panel answer "the
highest-priority correction in the panel-prep set" because it asserted the player-level
guarantee. `panel_attacks.md:478-481` has since been rewritten: M7 now opens "We do not,
and we say so rather than claiming otherwise", and `:481` explicitly warns that "An earlier
version of this playbook claimed we verify username disjointness; that claim was false and
must not be given." The audit's request was carried out. The defect that remains at that
site is the opposite one, listed in the table above: the rewritten answer now asserts that
preprocessing never keeps the White/Black headers, which stopped being true at `08ae9f0`.
Someone rehearsing M7 today would make a false statement to the panel about the thesis's
own data, in the other direction.

### Finding 2 (medium, category i, blocks Test 5 but not the corpus): the synthetic anomaly pickles have no `Time` key, so the dataset loader will raise on them

`generate_anomaly_corpus.py` writes `Positions`, `Moves`, `Clocks`, `Result`,
`move_is_substituted` (`:122-128`), then `WhiteElo`, `BlackElo`, `suspect_color`,
`engine`, `substitution_rate`, `maia_band` (`:177-182`). There is no `Time` key.

`ChessGamesDataset.__getitem__` does `initial_time, increment = map(int,
game_info["Time"].split("+"))` unconditionally (`chess_rating_net.py:93`). Scoring the
90,000-game anomaly corpus through the existing loader raises `KeyError: 'Time'` on the
first item.

This sits alongside the already-documented clock gap (`analysis/anomaly-corpus-generated.md:80-89`:
`Clocks` records engine think-time, not a Lichess-style countdown). Both are fixed in the
same place, since a synthesized countdown needs a nominal base+increment budget anyway,
which is exactly the value `Time` should hold. Cost: a small script over 90,000 files. No
regeneration, and no Lichess data involved.

### Finding 3 (medium, category i, methodology): "matched clean-versus-substituted pairs" is not what the generated corpus contains

`chapter3.tex` (Synthetic cheat generation) and `chapter4.tex:107` both describe the Test
5 evaluation as running on "matched clean-versus-substituted pairs", and
`170k-verification.md:761-763` prescribes a paired bootstrap that keeps both members of a
pair in or out of a resample together.

The generator does not produce cross-case pairs. Reading
`generate_anomaly_corpus.py:100-120` and `:170-176`: in a clean case
(`--engine none`), `use_cheater` short-circuits on `cheater_engine is not None`, so
`rng.random()` is never consumed inside the game loop; in a substituted case it is
consumed once per suspect ply. Both cases seed at 42 (`:146`, `:154`) and are launched
with identical parameters otherwise (`run_anomaly_corpus_full.sh:48-58`), so the RNG
streams diverge after the first game and `game_00001.pkl` in `1500_stockfish16_r000` has
no relationship to `game_00001.pkl` in `1500_stockfish16_r060`. There is also no pair-ID
field in the stored dicts.

This does not break ROC-AUC or precision-at-FPR, which need labelled positives and
negatives, not pairs. It affects (a) the word "matched"/"paired" in two manuscript
sentences and (b) the paired-bootstrap CI recipe.

There is a defensible pairing that *is* computable from what was stored: take the unit of
analysis to be one game-side, as `170k-verification.md:759` already specifies. Within any
substituted game the suspect side is a positive and the clean Maia opponent is a negative,
and `suspect_color` identifies which is which. That gives a genuine within-game matched
pair, controlled for opening, band, and opponent. This needs a captain or adviser call on
whether it is what the manuscript means. If they insist on cross-case pairs instead, the
corpus can be regenerated with a shared per-game seed prefix at a cost of roughly 10 to 14
hours and 23 GiB (`analysis/anomaly-corpus-pilot.md:130-134`), which touches no Lichess
data and is not affected by the raw-archive deletion at all.

Related, unverified, flagged so someone with HPC access can check it in one command: for a
given band, `{band}_stockfish16_r00` and `{band}_lc0_r00` are both generated with
`--engine none --substitution-rate 0.0 --seed 42`
(`run_anomaly_corpus_full.sh:26-29`), and Maia at `nodes=1` is deterministic. Those two
case directories may therefore be byte-identical duplicates, which would mean the 18,000
"clean control" games are 9,000 distinct games counted twice. Check:
`md5sum ~/<workdir>/data/anomaly_corpus/1500_{stockfish16,lc0}_r00/game_00000.pkl`.
If they match, the clean-control `n` in Chapter 4 needs halving, or one arm needs
regenerating under a different seed. I could not verify this without HPC access.

### Finding 4 (low, category i / iii-hybrid): the truncation-rate promise is not exactly computable, but the error is small and quantified

`chapter3.tex:229` commits: "the percentage of games truncated at 100 plies will be
reported per time control in Chapter 4", and `chapter4.tex:37` holds a `\TODO` for it.

`parse_game` keeps at most 100 plies and stores no original ply count, so a stored game
with `len(Moves) == 100` is either a game that ended exactly at ply 100 (not truncated,
nothing discarded) or a game that ran longer (truncated). Those two cases are not
separable after the fact, and the raw archive is the only place the distinction lives.

I measured how much this matters, on 92,123 eligible games from the head of the real
2021-06 archive:

| Time control | n | Reached the cap (≥100 plies) | Truly truncated (>100 plies) | Gap |
|---|---|---|---|---|
| UltraBullet | 1,364 | 3.08% | 2.86% | 0.22 pp |
| Bullet | 32,276 | 10.70% | 10.22% | 0.47 pp |
| Blitz | 45,542 | 17.07% | 16.52% | 0.55 pp |
| Rapid | 12,209 | 15.46% | 15.01% | 0.46 pp |
| Classical | 732 | 11.75% | 11.75% | 0.00 pp |
| All | 92,123 | 14.38% | 13.87% | 0.51 pp |

Caveat on those numbers: this is the head of one month, which is a chronological slice
rather than a uniform sample, so the time-control mix and the absolute rates are
indicative only. The corpus figures will come from the reservoir sample. The point that
survives is the size of the gap, which is under 0.6 percentage points everywhere.

Recommendation: report the metric as "percentage of games reaching the 100-ply cap",
which is exact under the stored schema and is also the quantity that actually matters
(it is the fraction of games where the model never sees the ending). Add one sentence
noting that games ending exactly at ply 100 are included and are a fraction of a
percent. This is a Chapter 3 and Chapter 4 wording change, not a pipeline change. I did
not make it, since `.tex` is off limits for this task.

Separately, and not a schema issue: the illustrative mock at
`thesis_explained.md:341-347` shows per-time-control truncation rates from 8%
(UltraBullet) to 58% (Classical). The real-data measurement above is nowhere near that
shape. The mock is explicitly labelled illustrative, but nobody should anchor on it
when the real table gets filled.

### Finding 5 (low, category i): month provenance survives only as a path convention

The chronological-holdout evaluation (`170k-verification.md:1327-1334`) needs to know
which month each game came from. Nothing in the pickle records it. It survives as the
per-month output directory and as the `{month}_` prefix that flattening must apply
(`AGENTS.md:35`), and `AGENTS.md:35` already documents that a bare flatten silently
collapses the corpus to ~30k files because every month reuses the same `game_*.pkl`
names.

This is fine as it stands and is backfillable at any time from the path, so it is not a
reason to change the pipeline mid-run. The cheap hardening is to make the flatten step
write a `manifest.csv` of `flat_name -> month` at flatten time, so month provenance stops
depending on a filename convention that one careless `cp` destroys.

---

## Things I specifically checked and found clean

- **Anomaly `R_baseline`.** `chapter3.tex:227` defines it as the Glicko-2 rating for the
  game's time control at game start. That is the `WhiteElo`/`BlackElo` header, stored.
  Nothing extra is needed.
- **Rating-bracket subgroups.** Both players' true ratings are stored, and the per-game
  error dump already carries both sides separately, so the 2N player-observation
  aggregation `170k-verification.md:670-679` insists on is directly computable.
- **Time-control subgroups.** `Time` is the raw header, so the base + 40×increment
  bucketing is exact and matches the manuscript thresholds.
- **Provisional ratings are not being silently dropped.** `is_eligible` requires
  `WhiteElo`/`BlackElo` to be `.isdigit()` (`preprocess_lichess.py:72-75`), which would
  drop any game whose rating header is `?`. `chapter3.tex:227` promises provisional
  ratings are retained, and Omori's motivation depends on the new-account population being
  present, so a `?` convention would have been a genuine category (ii) problem. Measured on
  92,522 real games from 2021-06: **zero** non-numeric `WhiteElo` headers. The filter is
  not excluding the provisional population.
- **Opening / ECO.** Discarded from the header, but re-derivable offline from the stored
  `Moves`, so a future opening-based breakdown is not foreclosed.
- **Exact date.** Discarded, but the only temporal analysis anyone has proposed
  (`170k-verification.md:1327-1334`) is a whole-month holdout, which the directory layout
  supports.
- **Unique game ID.** Discarded, but nothing planned joins on it, and lost months are
  deterministically re-derivable rather than requiring identification of specific games.
- **Eligibility predicate equivalence.** `has_full_clocks` (`preprocess_lichess.py:83-99`)
  and `parse_game`'s rejection logic (`format_data.py:74-80`) implement the same rule. I
  read both. Consistent.
- **Test 7 and Test 8.** Neither depends on a corpus field. Test 8 needs a PGN, which is
  reconstructable from `Moves` + `Clocks` + `Time` + `Result` + Elos, and the API is
  PGN-driven (`api.py:145`, `:267`, `:292`).

---

## Item 3: are the stored usernames actually used?

**Stored, and currently read nowhere.** A repo-wide grep for `"White"` / `"Black"` and for
`username` across `*.py`, `*.sh`, and `*.yaml` returns the writer and nothing that reads a
stored username:

- `prototype/src/format_data.py:88-89` writes them.
- `analysis/scripts/make_edge_case_pgns.py:76-77` also matches the grep, but it *writes*
  `White`/`Black` headers into synthetic test PGNs. It is not a reader of the corpus.
- `ChessGamesDataset.__getitem__` (`chess_rating_net.py:76-110`) never reads them. The
  returned item dict has no username key, and `collate_fn` (`:113-153`) has no username
  field either.
- `load_or_create_split` (`:548-597`) splits sorted file *basenames* with
  `train_test_split(..., random_state=split_seed)` (`:583-584`). No player awareness.
- Nothing else in `prototype/src/` or `analysis/scripts/` reads them.

So firstmate's summary in the task text is confirmed on the point that matters: the
player-level split is not implemented despite the usernames being stored, and implementing
it needs no re-preprocessing. That is a gap between schema and implementation.

### What building the player index actually costs

It is not a training-code one-liner, but it is also not a full read of the corpus. The
usernames sit near the head of every pickle. `parse_game` inserts the keys in the order
`WhiteElo, BlackElo, White, Black, Result, Clocks, Positions, Moves, Time`
(`format_data.py:85-95`), Python dicts preserve insertion order, `pickle.dump` writes items
in that order (`preprocess_lichess.py:178-179` uses plain `pickle.dump` at the default
protocol), and the bulk payload is `Positions`. I measured this by pickling a dict with the
same key order and a 100-ply `torch.Tensor(12,8,8)` `Positions` payload: the whole pickle
was 730,972 bytes, the `White` key appeared at byte offset 16, the two username values at
offsets 60 and 81, and `Positions` did not start until byte 328.

So the index can be built from a few hundred bytes per file (read a small prefix, parse it
with `pickletools.genops` or an incremental unpickler, stop at `Positions`). The real cost
is roughly 1.2M file opens on the shared spinning `/home` that `AGENTS.md:27` already flags
as the training bottleneck, which is a seek/IOPS problem rather than a bandwidth one. It is
a few hours once, and the result should be cached as a small CSV. Still category (i), still
no re-preprocessing.

Two caveats on coverage and on the harder half of the problem:

1. **The 170k-subset months have no username field at all.** `2024-02` through `2024-07`
   are `PROTECTED_MONTHS` (`corpus_stream_parallel.sh:70-72`), refused by the corpus run
   because they were built by the earlier 170k pass, which predates `08ae9f0` (2026-08-14
   23:13). An index over a flattened directory that includes them would be silently
   incomplete for those games. I could not open one of those pickles to confirm, so this is
   inferred from dates and from the protected-month list, not directly observed.
2. **Whether a strictly player-disjoint 72/18/10 partition is constructible is unmeasured.**
   Measuring the *leakage rate* (how many test-partition games share a player with a
   training game) is unambiguously easy once the index exists. Constructing a strict
   partition is a different problem, because each game links two players, and with heavy
   repeat play the player co-occurrence graph may have one giant connected component that
   cannot be cut into 72/18/10 without discarding a large share of games. I have not
   measured that component structure and cannot from this clone. What would settle it:
   build the index, construct the co-occurrence graph, and report the largest connected
   component's share of games. If it is small, a component-wise split works; if it
   dominates, the honest fallback is a single-side-disjoint split or a reported
   leakage-rate measurement instead of a second split. Decide after the measurement.

### Privacy read

Collection is disclosed and the field was requested deliberately, so this is a hygiene
item, not a live exposure.

- `chapter3.tex:298` and `thesis_explained.md:499` already disclose that public Lichess
  usernames are collected, and the source data is CC0.
- The field exists because the audit asked for it. `170k-verification.md:141-146`
  (prescription 1) names the exact two lines to add and warns that skipping them makes the
  player-disjoint split impossible "without a second 40-month re-stream". Commit `08ae9f0`
  is that change. "Stored with no justification" would be wrong; the justification is
  prescription 2, which has not been executed yet.
- No committed release artifact carries a username. `thesis_explained.md:519` commits to
  releasing "trained model weights, training code, data split definitions, random seeds,
  and evaluation scripts". The split definition is the manifest at
  `chess_rating_net.py:586-592`, which holds a sha256, the split seed, and three lists of
  `game_*.pkl` basenames. The evaluation artifact is the per-game dump documented at
  `chess_rating_net.py:439-445`, whose columns are `game_id, white_err, black_err,
  white_signed_err, black_signed_err, time_control, white_elo, black_elo`. Neither contains
  a username. Nothing commits to publishing the raw `.pkl` corpus.

Recommendation, in priority order:

1. Use them. Run the leakage measurement and report it. That converts a schema field with
   no consumer into the honesty check `170k-verification.md:137-156` asked for, and it
   turns Finding 1's stale manuscript sentences from an apology into a result.
2. If the raw corpus is ever released, salted-hash the usernames first. A leakage
   measurement and a player-disjoint split both need only equality of identity, not the
   plaintext handle, so a one-time pass replacing each username with
   `HMAC(salt, username)` preserves every planned use and removes the identifier. This is
   belt-and-braces for a release that is not currently planned, not a fix for anything
   live, and it can be done at any time.
3. If the captain decides against using them at all, drop them at the next re-preprocess.
   Do not drop them mid-run: it would split the corpus into two schemas for no gain.

---

## Considered and ruled out

Two claims in the first version of this document did not survive verification. They are
recorded here with the reason, so the same reasoning does not get re-derived later.

- **"Building the player index needs a ~250 to 300 GB read pass over the corpus, because
  pickle cannot be partially loaded."** False. The premise ignores key order: `White` and
  `Black` are inserted third and fourth (`format_data.py:85-95`) and therefore land in the
  first ~100 bytes of every pickle, ahead of the `Positions` payload. Measured offsets are
  in "What building the player index actually costs" above. The pass is IOPS-bound on ~1.2M
  file opens, not bandwidth-bound. The per-game size figure was also sourced wrongly: the
  ~259 KB/game at `analysis/anomaly-corpus-pilot.md:75-77` is the synthetic Maia-vs-Maia
  corpus, whose games hit the 100-ply cap more often than real Lichess games, whereas
  `170k-verification.md:406` (35 GB / 170,138 games, projecting to ~247 GB at 1.2M) implies
  about 206 KB/game for real data. Calling the two figures consistent overstated the
  agreement by about 26%. What survives from the original claim is in Item 3: not a
  one-liner, one pass over all 1.2M files, and the strict-partition feasibility question.
- **"The stored usernames are a privacy exposure, because a released derivative would carry
  roughly 2.4M raw username occurrences."** Not supported. The release commitment at
  `thesis_explained.md:519` covers weights, code, split definitions, seeds, and evaluation
  scripts. The split definitions are basename lists (`chess_rating_net.py:586-592`) and the
  evaluation dump is per-game error columns (`:439-445`); neither carries a username, and
  nothing commits to publishing the corpus pickles themselves. The related claim that the
  field has "no current justification" is also wrong: it was added on the audit's explicit
  instruction (`170k-verification.md:141-146`). The salted-hash recommendation is kept, but
  justified as a precondition for any future corpus release rather than as a fix for a live
  exposure.

---

## Decisions for the captain

### Decision A: use the usernames, or stop carrying them

Not urgent, no pipeline impact either way, but it should be logged rather than drift. See
the recommendation above. My view: use them, and hash them if the corpus is ever released.

### Decision B: whether to add `Site`, `UTCDate`, and `UTCTime` to the schema

I am raising this as an explicit decision and then recommending **no**, so that the "we
never considered it" version of this conversation cannot happen later.

The case for adding them: they cost about 40 bytes per game, roughly 48 MB across 1.2M
games against a ~247 GB corpus, so about 0.02% overhead. They would buy a true unique key
for dedup and provenance, sub-month temporal analysis, and a direct route to re-locate a
specific game in a re-downloaded archive. The decision is time-sensitive because adding
them later leaves the already-processed months permanently inconsistent, and making the
schema uniform again would mean re-downloading and re-scanning every completed month.

The case against, which is the one I land on:

1. No planned measurement needs any of the three. I checked every row in the table above.
2. Adding a field mid-run creates a two-schema corpus immediately, which is its own hazard:
   any downstream code doing `d["Site"]` breaks on the early months, and the guard has to be
   `d.get("Site")` forever.
3. The recovery story is better than the task brief assumes. The kept sample is
   deterministically re-derivable from the archive plus the fixed seed, so a late change is
   recoverable at compute cost, not as data loss.
4. Stopping a multi-week unattended run for a field nothing consumes is a worse trade than
   the residual risk.

If the captain overrules this and wants the fields, the right move is not a mid-run patch.
It is to finish the current run on the current schema and, if the fields later turn out to
matter, re-derive them for the specific months that need them using the deterministic re-run
path. That keeps one schema throughout.

---

## What I could not verify

- Whether the two 0%-substitution case directories per Maia band are byte-identical
  duplicates. Needs HPC access. Exact command in Finding 3.
- The connected-component structure of the player co-occurrence graph, which decides whether
  a strictly player-disjoint split is constructible. Needs the index pass over the corpus on
  the HPC.
- Whether the months already processed under the pre-`08ae9f0` code are in fact missing
  `White` and `Black`. The inference is strong (`08ae9f0` landed 2026-08-14 23:13; the 170k
  subset covers 2024-02 through 2024-07 and is refused outright by
  `corpus_stream_parallel.sh:70-72`; the 1.2M run relaunched 2026-08-17 per `AGENTS.md:32`),
  but I have not opened one of those pickles. It matters only if anyone runs the leakage
  measurement over a flattened directory that includes the 170k seed-rerun months. Checking
  it is one `python -c "import pickle; print(pickle.load(open(f,'rb')).keys())"` against a
  170k pickle on the HPC.
- The real per-time-control truncation rates for the reservoir-sampled corpus. My numbers
  come from the head of one archive, which is a chronological slice, not a uniform sample.

---

## Appendix: how the real-data measurements in this document were taken

Anyone can reproduce these without touching the running pipeline and without a full
archive download. Total cost is a 30 MB ranged GET.

```
curl -s -r 0-30000000 -o head.zst \
  https://database.lichess.org/standard/lichess_db_standard_rated_2021-06.pgn.zst
zstd -d --long=31 -c head.zst 2>/dev/null > head.pgn     # truncated tail is expected
```

That yields 92,522 complete game blocks (214 MB of PGN). From there:

- Header coverage table: `grep -c '^\[<Header> ' head.pgn` for each header name.
- Non-numeric Elo check: `grep '^\[WhiteElo ' head.pgn | grep -vc '^\[WhiteElo "[0-9]\+"\]$'`
  returns 0.
- Ply counts: for each game block, the ply count is exactly the number of `[%clk ...]`
  matches in the movetext, since Lichess annotates every ply. Spot-checked against a game
  whose movetext ends at `47... Ke7`: 94 clock annotations, 47 full moves, exact match.
  Apply the same `is_eligible` predicate as `preprocess_lichess.py:66-80` (Rated in Event,
  numeric Elos, `base+inc` TimeControl), then bucket by `base + 40*inc` using the thresholds
  in `format_data.py:98-109`.

The pickle byte-offset measurement in Item 3 is reproducible locally with no corpus access:

```
python3 -c "
import pickle, torch
d = {'WhiteElo':'1650','BlackElo':'1702','White':'alice_user','Black':'bob_user',
     'Result':'1-0','Clocks':['0:03:00']*100,
     'Positions':[torch.zeros(12,8,8) for _ in range(100)],
     'Moves':['e2e4']*100,'Time':'180+2'}
b = pickle.dumps(d)
print(len(b), b.find(b'White'), b.find(b'bob_user'), b.find(b'Positions'))
"
```
