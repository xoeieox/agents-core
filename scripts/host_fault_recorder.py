#!/usr/bin/env python3
"""Entrypoint for the host-fault-recorder daemon.

Delegates entirely to agents_core.host_fault_recorder.run().
systemd keeps this alive via Restart=always.
"""
import sys

sys.path.insert(0, "/srv/agents")

from agents_core.host_fault_recorder import run

if __name__ == "__main__":
    run()
