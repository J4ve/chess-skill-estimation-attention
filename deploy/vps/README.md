# Deploying the prototype to a small ARM64 server

This is the deployment behind the live URL in the repository README. It runs the
real FastAPI application, so every input the prototype supports works, including
live mode following an ongoing Lichess game. It was set up on Ubuntu 22.04 on
aarch64 with 2 vCPUs and 11 GB of RAM, sharing the host with an unrelated
application that already owned ports 80 and 443.

## torch must come from the CPU wheel index

On aarch64 the PyPI `torch` wheel declares the `nvidia-*` CUDA runtime wheels as
dependencies, so a bare `pip install torch` pulls several gigabytes of CUDA onto
a machine with no GPU. Install it from the CPU index explicitly:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

Everything else comes from PyPI normally. `lightgbm` is needed in addition to
`requirements.txt`, which omits it, or the A0g detector method is unavailable.

## Layout

```
/opt/ratingnet/app          this repository, shallow clone
/opt/ratingnet/venv         virtual environment
/opt/ratingnet/models       the checkpoint, which is gitignored and fetched separately
```

The service runs as a dedicated system user, `ratingnet`.

## The checkpoint

`prototype/models/` is gitignored, so the checkpoint is not in the clone. Fetch
it from the release rather than copying it by hand, which also verifies it:

```bash
curl -fsSL -o /opt/ratingnet/models/attn_tuned_best.pth \
  https://github.com/J4ve/chess-skill-estimation-attention/releases/download/v0.1-weights-attn-tuned/attn_tuned_best.pth
sha256sum /opt/ratingnet/models/attn_tuned_best.pth
# ff6370e477a0ea7264933adbf517f94068ed1205126ad284d8158ae6124275a0
```

`ratingnet.service` sets `RATINGNET_CHECKPOINT` to that path. Setting it
explicitly matters: `api.py` resolves that variable first and raises if the file
is missing, so the service can never silently fall back to `model_55.pth`, which
is not one of the study's arms and reproduces no reported number.

## Install

```bash
sudo install -m 644 deploy/vps/ratingnet.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ratingnet
curl -s http://127.0.0.1:8001/health
```

The service is enabled, so it comes back after a reboot. It binds loopback only;
the reverse proxy is the only thing that should reach it.

## HTTPS

Append `deploy/vps/caddy-site.conf` to the host's `/etc/caddy/Caddyfile`, then
check the result before touching the running proxy:

```bash
sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
sudo systemctl reload caddy
```

Use `reload`, not `restart`: reload applies the new configuration with no
downtime for anything else the proxy serves. If validation fails, change nothing.

The host name is an `sslip.io` name, which resolves to the server's own address,
so a publicly trusted certificate can be issued without owning a domain.

## The prediction log

`api.py` appends one JSON line per prediction to `logs/predictions.jsonl` and never
truncates it, so on a public deployment it grows without bound and keeps the player
names from every PGN a visitor submits. Install the rotation config alongside the
service:

```bash
sudo install -m 644 deploy/vps/ratingnet-logrotate /etc/logrotate.d/ratingnet
sudo logrotate --debug /etc/logrotate.d/ratingnet   # parses and reports, changes nothing
```

It rotates on size rather than on a schedule, because the file grows with traffic
rather than with time, and keeps three compressed generations.

## Limits

`src/limits.py` holds the deployment limits, and `ratingnet.service` sets each
one as an environment variable so they retune without a code change. They are
shaped like the load rather than as a single connection cap, because one visitor
opening the page is already a dozen cheap asset requests:

| Variable | Default | What it bounds |
| --- | --- | --- |
| `RATINGNET_MAX_LIVE_STREAMS` | 20 | Live streams in total |
| `RATINGNET_MAX_LIVE_STREAMS_PER_CLIENT` | 2 | Live streams per visitor |
| `RATINGNET_MAX_ANALYSIS_RUNNING` | 2 | Analysis requests running at once |
| `RATINGNET_MAX_ANALYSIS_WAITING` | 12 | Analysis requests queued behind those |

A live stream is one viewer holding one upstream Lichess connection for the
length of a game, which is why the per-visitor cap matters more than the total.
Analysis beyond the queue depth is refused with HTTP 503 and `Retry-After`
rather than queued without bound. The unit also sets `CPUQuota` and `MemoryMax`,
which is what actually keeps an inference burst from starving the rest of the
host; a request count alone cannot guarantee that.
