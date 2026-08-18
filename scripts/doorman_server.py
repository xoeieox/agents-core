#!/usr/bin/env python3
"""Entrypoint for the doorman HTTP service (GravityWell power/lease manager).

Delegates entirely to agents_core.doorman_server.create_app(); this script
owns only the process bootstrap (env-driven host/port, logging setup,
uvicorn.run). systemd runs this directly — see systemd/doorman-server.service.
"""
import logging
import os
import sys

sys.path.insert(0, "/srv/agents")

from agents_core.doorman_server import create_app


def main():
    import uvicorn

    host = os.environ.get("DOORMAN_BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("DOORMAN_BIND_PORT", "8407"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
