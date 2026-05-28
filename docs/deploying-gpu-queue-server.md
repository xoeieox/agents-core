# Deploying gpu-queue-server

`gpu-queue-server` exposes `GPUQueue` over HTTP on port 8405. Any
Tailscale-connected host can submit, claim, and complete GPU tasks through a
network-addressable endpoint — this is the cross-node handoff that lets BRIX
(always-on) hold the canonical queue while StarHouse (sleeps between jobs)
consumes it.

## Why this exists

The GPU queue lives at `/srv/lapis/gpu-queue/` on a shared filesystem. When `/room`
relocates to BRIX, StarHouse's runner can no longer reach the queue directly.
`gpu-queue-server` on BRIX becomes the single source of truth; submitters and
the runner both reach it over the 2.5GbE Tailscale backbone - same pattern as
`mem-http` (:8404) and `weaver-http` (:8403).

During prove-out, run the server on StarHouse against the local
`/srv/lapis/gpu-queue`. After the `/room` relocation completes, redeploy on BRIX.
The library, protocol, and CLI stay identical across the cutover.

## Environment variables

### Server side

| Variable | Default | Purpose |
|---|---|---|
| `GPU_QUEUE_DIR` | `/srv/lapis/gpu-queue` | Queue directory |
| `GPU_QUEUE_BIND_HOST` | `127.0.0.1` | uvicorn bind host |
| `GPU_QUEUE_BIND_PORT` | `8405` | uvicorn bind port |
| `GPU_QUEUE_BEARER_TOKEN` | (unset) | Optional shared bearer token; omit to disable auth |
| `GPU_QUEUE_LOG_LEVEL` | `info` | uvicorn log level |

### Client side

| Variable | Default | Purpose |
|---|---|---|
| `GPU_QUEUE_SERVER` | (unset) | When set, route through HTTP; unset means local `GPUQueue` against `/srv/lapis/gpu-queue` |
| `GPU_QUEUE_BEARER_TOKEN` | (unset) | Must match server token if set |
| `GPU_QUEUE_CLIENT_TIMEOUT` | `10.0` | Per-request timeout in seconds |

## Linux - systemd (user mode)

```bash
# Install the unit
mkdir -p ~/.config/systemd/user
cp systemd/gpu-queue-server.service ~/.config/systemd/user/

# Create environment file
mkdir -p ~/.config/gpu-queue
cat > ~/.config/gpu-queue/server.env <<EOF
GPU_QUEUE_DIR=/srv/lapis/gpu-queue
GPU_QUEUE_BIND_HOST=203.0.113.10
GPU_QUEUE_BIND_PORT=8405
GPU_QUEUE_LOG_LEVEL=info
# GPU_QUEUE_BEARER_TOKEN=changeme
EOF

systemctl --user daemon-reload
systemctl --user enable gpu-queue-server
systemctl --user start gpu-queue-server

# Check status
systemctl --user status gpu-queue-server
journalctl --user -u gpu-queue-server -f
```

## Tailscale bind convention

Production deployments bind to the host's Tailscale IP rather than `127.0.0.1`:

```
GPU_QUEUE_BIND_HOST=203.0.113.10   # BRIX (post-relocation canonical host)
```

During the prove-out phase on StarHouse, bind to StarHouse's Tailscale IP
instead so submitters on BRIX can reach it over the backbone.

## Smoke test

```bash
curl http://127.0.0.1:8405/healthz
# expected: {"status":"ok","queue_dir":"...","queue_depth":0,"mode":"idle","paused":false,...}

# Submit a task
curl -X POST http://127.0.0.1:8405/v0/tasks \
  -H "Content-Type: application/json" \
  -d '{"task_type": "smoke_test", "priority": 50}'

# Check pending
curl http://127.0.0.1:8405/v0/pending

# Claim
curl -X POST http://127.0.0.1:8405/v0/claim \
  -H "Content-Type: application/json" \
  -d '{"current_model": null}'

# With bearer token:
curl -H "Authorization: Bearer <token>" http://127.0.0.1:8405/healthz
```

## Concurrency model

The server holds one `threading.Lock` that serializes all mutating operations
(`submit`, `claim`, `complete`, `fail`, `preempt`, `cancel`, `pause`, `resume`,
`update_runner_state`, `cleanup`). This makes `claim()` single-claimer-safe
under FastAPI threadpool concurrency - no task can be double-issued even with
multiple concurrent clients. Pure readers (`state`, `pending`, `active`,
`completed`, `failed`, `history`, `is_paused`) are unlocked.

## Orphaned-active detection (Invariant 9)

If StarHouse force-sleeps mid-task, the active task will remain in `active/`
indefinitely with no automatic reaper. Monitor this via `active_task_age_seconds`
on `/healthz` or `/v0/state` - a non-null, growing value indicates a stalled
task. Recovery (requeueing or failing the orphaned task on runner wake) is the
responsibility of the runner followup `gpu-queue-runner-http-mode-v0`.

## Followup: runner refactor

`gpu_queue_runner.py` switching its `claim()/complete()/fail()` calls to route
through `GPUClient` when `GPU_QUEUE_SERVER` is set is the separate followup
`gpu-queue-runner-http-mode-v0`. That bind is required before the `/room`
relocation completes but is NOT part of this service.
