# Deploying mem-server

`mem-server` exposes `mem.db` over HTTP on port 8403. Any Tailscale-connected
host can read and write Memory state through a network-addressable endpoint.

## Environment variables

### Server side

| Variable | Default | Purpose |
|---|---|---|
| `MEM_DB_PATH` | `/data/memory/mem.db` | SQLite DB file location |
| `MEM_BIND_HOST` | `127.0.0.1` | uvicorn bind host |
| `MEM_BIND_PORT` | `8403` | uvicorn bind port |
| `MEM_BEARER_TOKEN` | (unset) | Optional shared bearer token; omit to disable auth |
| `MEM_LOG_LEVEL` | `info` | uvicorn log level |

### Client side

| Variable | Default | Purpose |
|---|---|---|
| `MEM_SERVER` | (unset) | When set, CLI routes through HTTP; unset means local DB |
| `MEM_BEARER_TOKEN` | (unset) | Must match server token |
| `MEM_CLIENT_TIMEOUT` | `5.0` | Per-request timeout in seconds |

## Linux - systemd (user mode)

```bash
# Install the unit
mkdir -p ~/.config/systemd/user
cp systemd/mem-server.service ~/.config/systemd/user/

# Create environment file
mkdir -p ~/.config/mem
cat > ~/.config/mem/server.env <<EOF
MEM_DB_PATH=/data/memory/mem.db
MEM_BIND_HOST=203.0.113.12
MEM_BIND_PORT=8403
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
MEM_BIND_HOST=203.0.113.12   # StarHouse
```

After BRIX assembly (2026-05-29), update to the BRIX Tailscale IP.

## macOS - launchd / nohup pattern

```bash
# nohup background pattern
export MEM_DB_PATH=$HOME/data/memory/mem.db
export MEM_BIND_HOST=127.0.0.1
export MEM_BIND_PORT=8403
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
        <string>8403</string>
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
curl http://127.0.0.1:8403/healthz
# expected: {"status":"ok","db_path":"...","row_counts":{"memories":N,...}}

# With bearer token:
curl -H "Authorization: Bearer <token>" http://127.0.0.1:8403/healthz
```

## SSH-proxy deprecation

The current CLI shim at `/srv/agents/scripts/mem.py` contains `_remote_write()`
which proxies write commands from non-StarHouse hosts to StarHouse via SSH. Once
`mem-server` is running in production:

1. Set `MEM_SERVER=http://<starhouse-tailscale-ip>:8403` on MacBook (and other
   non-StarHouse hosts).
2. The companion CLI-shim edit (`conductor-mem-cli-http-route-v0`) replaces the
   SSH branch with an `httpx`-based `MemClient` call.
3. Remove `_remote_write()` after ≥7 days of HTTP service in production.

The SSH proxy is NOT removed in this PR - removal is gated on the companion edit.
