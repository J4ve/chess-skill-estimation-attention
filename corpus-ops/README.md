# Corpus pipeline drivers

Two kinds of script live here. `corpus_stream_parallel.sh` and
`corpus_stream.sh` are the study's portable corpus drivers, described in the
first section. The `fm-corpus-*` files are machine specific reference copies,
described in the second.

## The corpus drivers

- `corpus_stream_parallel.sh`: the canonical cross-host coordinator, and the
  script that built the published corpus. It downloads a month, verifies it
  against a known Content-Range size, preprocesses it, syncs the result and
  deletes the archive, running several months at a time. Months are claimed
  with an atomic `mkdir` plus a heartbeat on one coordination host, so two
  machines can share a list and a claim stranded by a power cut is reclaimed
  rather than lost. Its two retry counters, `MAX_ATTEMPTS` and `MAX_STARTS`,
  are deliberately separate: see the top level README, and
  `analysis/corpus-resilience-review.md` for the eleven further gaps that were
  documented but not fixed.

  ```bash
  bash corpus_stream_parallel.sh --host hpc --scanner rust
  bash corpus_stream_parallel.sh --host hpc --status
  bash corpus_stream_parallel.sh --host hpc --scanner rust --worker 2023-05
  ```

  It resolves its own directory on startup, so it expects to sit at the root
  of the deployed tree next to `src/` and `analysis/` rather than in this
  folder. The SSH destination, key and remote working directory at the top of
  the file are placeholders.

- `corpus_stream.sh`: the superseded single-host predecessor, kept here only
  for reference. It has no claim locking, no cross-host coordination and no
  retry accounting. All corpus work after it used the parallel driver above.

## Machine specific reference copies

These are the actual orchestration scripts that built the ~2.19M-game corpus
(73 months, April 2021-present minus the protected 2024-02..07 block), copied
here for visibility/audit rather than as callable tooling. They are NOT
meant to be run as-is outside the machine they were written for: paths are
hardcoded to a specific Windows/WSL setup (`/mnt/d/firstmate/`,
`C:\Users\bacsa\Downloads\...`) and a Windows Scheduled Task drives the
`.sh`/`.vbs` pair. Full description and how they fit together:
`AGENTS.md`/`CLAUDE.md` (project root), the "Full corpus-pipeline script
inventory" entry.

- `fm-corpus-heal.sh` — the self-healing/self-pacing download+preprocess
  driver, invoked every ~10 min by the Scheduled Task. This is the version
  that actually ran; check its own header comments for the incident history
  behind its concurrency/disk-headroom gates.
- `fm-corpus-heal-trigger.vbs` — the hidden wrapper the Scheduled Task calls
  (avoids popping a visible console window each run).
- `fm-corpus-archive-month.py` — one-off compressor used to shrink completed
  months from loose `.pkl` files (~6GB/month) to `.tar.gz` (~107MB/month).
- `fm-corpus-autopreprocess.sh`, `fm-corpus-zst-cleanup.sh` — superseded,
  folded into `fm-corpus-heal.sh`; kept for reference only.

The canonical, portable, actually-run-anywhere corpus tooling is
`corpus_stream_parallel.sh` in this folder, together with
`analysis/preprocess_onepass.py`, `analysis/preprocess_fast.py` and
`src/preprocess_lichess.py`.
