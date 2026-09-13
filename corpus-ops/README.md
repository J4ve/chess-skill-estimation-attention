# Corpus pipeline ops scripts (reference copies, not portable)

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

The canonical, portable, actually-run-anywhere corpus tooling lives one level
up: `local-run/corpus_stream_parallel.sh` and `analysis/scripts/preprocess_*.py`.
