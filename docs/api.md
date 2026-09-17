# API reference

[Back to the README](../README.md)


`GET /api` returns this list as JSON at runtime.

| Endpoint | Description |
| --- | --- |
| `GET /health` | Model load status, device, and checkpoint path. |
| `GET /` | The web prototype (`src/static/index.html`). |
| `GET /api` | Machine-readable endpoint list. |
| `POST /predict/pgn` | Body `{pgn, white_baseline?, black_baseline?}`. Query `top_k` (default 5), `min_ply` (default 10). |
| `POST /predict/upload` | Multipart form: `file` (`.pgn`), `white_baseline?`, `black_baseline?`. Query `top_k`, `min_ply`. |
| `POST /predict/lichess` | Body `{game_id, white_baseline?, black_baseline?}`, where `game_id` is a bare 8-character Lichess ID or a full game URL. Query `top_k`, `min_ply`. |
| `GET /live/stream/{game_id}` | Server-Sent Events. Follows one Lichess game (ongoing or just-finished) move by move. Query `top_k`, `min_ply`, `white_baseline?`, `black_baseline?`. See [Live mode](web-prototype.md#live-mode). |
| `GET /live/tv` | Server-Sent Events. Follows Lichess TV's currently featured game, switching automatically when TV switches games. Query `top_k`, `min_ply`. See [Live mode](web-prototype.md#live-mode). |

All three `predict/*` endpoints return the same shape:

- `headers`: the PGN headers (`White`, `Black`, `WhiteElo`, `Result`, ...).
- `white_baseline`, `black_baseline`: the resolved per-side baseline rating.
- `white_baseline_source`, `black_baseline_source`: one of `request` (caller
  supplied it), `pgn_header` (parsed from `WhiteElo`/`BlackElo`), or
  `self_prediction_fallback` (no usable baseline anywhere, so the model's own
  final-ply prediction was used instead; see the README's Limitations).
- `white_actual_rating`, `black_actual_rating`: the side's real rating,
  parsed from the PGN `WhiteElo`/`BlackElo` header only (`"?"` and any other
  non-numeric or missing value is `null`). Independent of the baseline: a
  `white_baseline`/`black_baseline` request override never changes this
  field, since it is a fixed ground-truth fact about the game rather than a
  reviewer-supplied value used for suspicion scoring.
- `warnings`: readable strings for anything that degrades the result (a
  fallback baseline, or an anomaly branch that is not available).
- `white_final_rating`, `black_final_rating`: the model's final-ply rating
  estimate per side.
- `suspicion_score_kind`: `detector` (the trained detector, the normal case)
  or `computed` (the detector weights are unavailable, so the main fields
  below carry S_att instead, with a warning).
- `white_suspicion_score`, `black_suspicion_score`: the main suspicion score
  per side: the detector's 0 to 1 score (see [Suspicion score](web-prototype.md#suspicion-score)).
- `white_suspicion_label`, `black_suspicion_label`: one of `typical`,
  `unusual`, or `highly_unusual`, from comparing that side's main score
  against its own percentile cutoffs on ordinary held-out test games (see
  [Suspicion labels](web-prototype.md#suspicion-labels)). `null` when the served checkpoint has no anomaly
  branch or that score has no matching cutoffs file.
- `suspicion_cutoffs_used`: the cutoffs actually applied to the main score
  (`score` names the score the cutoffs belong to, `p75`/`p95`, `source` is
  `time_control` or `overall`, `time_control` is the bucket derived from the
  PGN `TimeControl` header or `null`, plus `provisional` fields). `null` under
  the same conditions as the labels above.
- `white_computed_score`, `black_computed_score`, `combined_computed_score`:
  the computed score S_att (attention-weighted deviation from baseline, in
  rating points), always returned for comparison.
- `white_computed_label`, `black_computed_label`, `computed_cutoffs_used`:
  the same label fields for S_att, against the S_att cutoffs file.
- `per_move`: one record per ply (`ply`, `move` as SAN, `uci`, `white_rating`,
  `black_rating`, `attention_weight`, `white_deviation`, `black_deviation`,
  `clock_seconds`, `time_spent_seconds`). `clock_seconds` is that ply's mover's
  remaining clock, parsed from the PGN's `[%clk ...]` comments (one of the
  model's own inputs, not a derived value). `time_spent_seconds` is that
  side's previous remaining clock minus `clock_seconds` plus any `TimeControl`
  increment; `null` for a side's first move when `TimeControl` couldn't be
  parsed (see `format_data.compute_time_spent`).
- `critical_moves`: up to `top_k` entries per side, ranked by
  `attention_weight * |deviation|` for that side (`ply`, `move`, `side`,
  `attention_weight`, `deviation`, `weighted_score`), excluding plies before
  `min_ply`. Empty (with a warning) when the served checkpoint has no
  attention/anomaly branch.
- `critical_moves_min_ply`: the `min_ply` value actually used for the
  `critical_moves` ranking above.
- `ongoing`, `provisional`: `true` when the game has no final result yet
  (PGN `Result` header is `*`).

Bad input (a PGN with no `[%clk ...]` annotations, unparseable PGN text, or a
malformed Lichess ID) returns HTTP 422 with a readable `detail`, never a bare
500. A Lichess game that does not exist returns 404; hitting Lichess's rate
limit returns 429.

The two `/live/*` endpoints stream `text/event-stream`; each `data:` line is
one JSON object with a `type`:

- `{"type": "status", "state": ..., "message": ...}` - a status update.
  `state` is one of `connecting`, `connected`, `reconnecting`, `capped`
  (the game passed 100 plies, the model's training cap), `finished`, or
  `error`.
- `{"type": "update", "result": {...}}` - `result` has the same shape as
  the `predict/*` endpoints above, plus `inference_ms` (this move's
  inference wall time), `lichess_game_id`, and `live: true`. `per_move`
  here covers only the plies known so far; a client that wants a frozen,
  never-revised chart should keep only the newest row of each update and
  append it to its own array, as `src/static/app.js`'s `mergeLiveUpdate`
  does, rather than replacing its data with `per_move` wholesale.
- `{"type": "error", "detail": ...}` - a terminal error; the stream closes
  after this. `GET /live/stream/{game_id}` always opens the stream (HTTP 200)
  and sends this event for a setup problem (bad ID, Lichess 404/429, a
  non-standard game, missing clocks) rather than a bare HTTP error status,
  since `EventSource` cannot read an error response's body; the client should
  show `detail` and not attempt to reconnect after this event.

