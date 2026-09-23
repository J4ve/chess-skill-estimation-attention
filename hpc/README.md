# Cluster training runners

The scripts that launched and kept alive the study's training runs on the
institutional cluster. There is no workload manager on that machine, so these
do the job a scheduler would: one run per GPU, a queue behind each GPU, and a
watcher that relaunches anything that dies.

- `hpc_launch_run.sh <session> <gpu> <experiment> <seed>`: start one
  seed-controlled rerun in its own `tmux` session, resuming from
  `models/<experiment>/latest.pth` when that file exists. The architecture
  flags are keyed off the experiment name, because passing them by hand had
  already once trained the baseline architecture under an attention run's
  name.
- `hpc_seed_queue.sh`: keep exactly one seed-rerun job per GPU, with a
  per-GPU ordered pipeline. It reads the epoch out of each run's `latest.pth`
  to decide whether a run is finished.
- `hpc_sweep_launch_run.sh <session> <gpu> <experiment> [flags...]`: the same
  idea for the stage 2 tuning sweep. Every control value is fixed in the
  script and the single swept flag is passed as extra arguments, so a cell
  differs from the control in exactly one place.
- `hpc_sweep_queue.sh`: the sweep's queue, carrying the 17 pre-registered
  cells and the one flag each of them varies.
- `hpc_training_heal.sh`: a cron driven watcher for the full corpus arms. A
  reboot once wiped the NVMe corpus store and killed three of four concurrent
  runs, which then sat idle for hours before anyone noticed. This script
  rebuilds the store and relaunches the arms from their checkpoints. It treats
  a run as finished only when its log carries the trainer's own
  `Training duration (min):` line, because the patience setting drives
  learning rate reduction rather than early stopping and no run ever exits its
  loop early.

## Before running any of these

They carry placeholder values that have to be filled in first. `~/thesis2` is
the working directory the code was deployed to, and the SSH destination and
key in `corpus-ops/corpus_stream_parallel.sh` appear as `<user>@<hpc-host>`
and `~/.ssh/<key>`.

They also assume a conda environment named `ratingnet2` with CPython 3.12 and
a CUDA build of PyTorch, `tmux`, four GPUs addressed through
`CUDA_VISIBLE_DEVICES`, and a flattened copy of the corpus on local NVMe. The
shared home directory on that machine is a spinning disk and was the training
bottleneck: about 85 seconds per epoch on NVMe against more than ten minutes
on the shared disk, which is why the data is relocated before training and why
`hpc_training_heal.sh` has to rebuild the store after a reboot.

The pre-registered sweep design and its results are in
`analysis/stage2-tuning-preregistration.md` and
`analysis/stage2-sweep-results.md`; the seed-controlled rerun is in
`analysis/seed-rerun-results.md`.
