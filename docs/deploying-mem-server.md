# Deploying mem-server

`mem-server` exposes `mem.db` over HTTP on port 8404. Any Tailscale-connected
host can read and write Memory state through a network-addressable endpoint.

**Topology (as of 2026-05-29):** BRIX (`203.0.113.10:8404`) is the read-write
master. StarHouse and MacBook are read-only mirrors. The master is set by the
`MEM_MASTER_HOST = "brix"` constant in `agents_core/mem.py` — not an env toggle.

## Environment variables

### Server side

| Variable | Default | Purpose |
|---|---|---|
| `MEM_DB_PATH` | `/data/memory/mem.db` | SQLite DB file location |
| `MEM_BIND_HOST` | `127.0.0.1` | uvicorn bind host |
| `MEM_BIND_PORT` | `8404` | uvicorn bind port |
| `MEM_BEARER_TOKEN` | (unset) | Optional shared bearer token; omit to disable auth |
| `MEM_LOG_LEVEL` | `info` | uvicorn log level |

### Client side

| Variable | Default | Purpose |
|---|---|---|
| `MEM_SERVER` | (unset) | When set, CLI routes through HTTP; unset means local DB |
| `MEM_BEARER_TOKEN` | (unset) | Must match server token |
| `MEM_CLIENT_TIMEOUT` | `5.0` | Per-request timeout in seconds |
| `MEM_PRINCIPAL` | (unset) | The caller's principal name, sent as the `X-Mem-Principal` header on every request (openclaw-memdb-influx-reader-v0, D1). Unset = no header = the server treats the caller as a reader (fail-closed). The BRIX-side mem CLI defaults to `brix-pm` for write verbs (promote) when this is unset (D1 / panel F7). |

## Promote verb (openclaw-memdb-influx-reader-v0, D2)

The `mem promote` subcommand promotes one row from an agent store into
mem.db with a provenance shape the server owns (the exact header line +
the batch decision key). It is an explicit, server-side verb — a local
MemoryStore-only promote is refused (loud exit 2) because it would bypass
the server's `--from` shape check + batch key (the side-effect the spec
forbids: "promotion is an explicit verb with a provenance shape, never a
side effect").

```
mem promote <key> [--content <body>] --from <agent>/<store>
             [--by <curator-principal>] [--tags tag1,tag2]
             [--rationale one-line-why] [--store atoms|machinery]
```

(The promoted body is the `--content` option, not a positional — argparse
stops consuming positionals at the first option, so the documented form
with the body as a positional after `--from` would not parse.)

- `--from` is shape-validated (path-like/ref-like token, no newlines or
  control chars, loud 400 — panel security F6).
- `--by` is the curator principal (defaults to `MEM_PRINCIPAL`, then
  `brix-pm` for the BRIX-side CLI).
- `--rationale` is the one-line rationale the D2 named decision artifact
  requires per promoted key (the batch decision key
  `decision/memdb-promotion-<YYYYMMDD>-<curator>` lists promoted keys +
  their `--from` refs + one-line rationale).
- `--store` is RESCOPED (rev-2): the machinery store is the EXISTING
  exhaust store, so the flag is MOOT at the HTTP layer (the server routes
  machine-state keys transparently). It is validated (loud ValueError on a
  typo) but does NOT change routing.

The named weekly audit command:
`mem list --tag promoted --since <7-days-ago> --limit 500` (explicit limit
— the default 50 silently truncates).

## Linux - systemd (user mode)

```bash
# Install the unit
mkdir -p ~/.config/systemd/user
cp systemd/mem-server.service ~/.config/systemd/user/

# Create environment file
mkdir -p ~/.config/mem
cat > ~/.config/mem/server.env <<EOF
MEM_DB_PATH=/data/memory/mem.db
MEM_BIND_HOST=203.0.113.10  # BRIX — read-write master
MEM_BIND_PORT=8404
MEM_LOG_LEVEL=info
# MEM_BEARER_TOKEN=changeme
EOF

systemctl --user daemon-reload
systemctl --user enable mem-server
systemctl --user start mem-server

# Check status
systemctl --user status mem-server
journalctl --user -u mem-server -f
```

## Tailscale bind convention

Production deployments bind to the host's Tailscale IP rather than `127.0.0.1`:

```
MEM_BIND_HOST=203.0.113.10   # BRIX (read-write master, 2026-05-29+)
# MEM_BIND_HOST=203.0.113.12  # StarHouse — read-only mirror; do not promote
```

## macOS - launchd / nohup pattern

```bash
# nohup background pattern
export MEM_DB_PATH=$HOME/data/memory/mem.db
export MEM_BIND_HOST=127.0.0.1
export MEM_BIND_PORT=8404
nohup mem-server > /tmp/mem-server.log 2>&1 &
echo $! > /tmp/mem-server.pid
```

For a persistent launchd plist, create `~/Library/LaunchAgents/com.lapis.mem-server.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.lapis.mem-server</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/mem-server</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>MEM_DB_PATH</key>
        <string>/home/user/data/memory/mem.db</string>
        <key>MEM_BIND_HOST</key>
        <string>127.0.0.1</string>
        <key>MEM_BIND_PORT</key>
        <string>8404</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/mem-server.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/mem-server.err</string>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.lapis.mem-server.plist
```

## Smoke test

```bash
# Local
curl http://127.0.0.1:8404/healthz
# Remote (BRIX master)
curl http://203.0.113.10:8404/healthz
# expected: {"status":"ok","db_path":"...","row_counts":{"memories":N,...}}

# With bearer token:
curl -H "Authorization: Bearer <token>" http://203.0.113.10:8404/healthz
```

## Client configuration

All non-BRIX hosts point `MEM_SERVER` at the BRIX master:

```
MEM_SERVER=http://203.0.113.10:8404
```

This is set in `~/.bashrc` and `~/.claude/settings.json` on both StarHouse and
MacBook. The `/usr/local/bin/mem` wrapper on StarHouse defaults to BRIX; rollback
shim is baked in. `mem stats` reports `mode: read-only (master is brix)` on
non-BRIX hosts and `mode: read-write` on BRIX.

The old SSH-proxy (`_remote_write()`) in `mem.py` is superseded by the HTTP
client path and can be removed once the HTTP service has been stable for ≥7 days.
