# Typed Seam for Jagged-Seam Deliberation (v0 reserved; H3 tap)

## Overview

The shared deliberation service reserves a **typed extension point** for the later jagged-seam arbiter (H3) to register additional deliberation modes without modifying the core service.

In v0, this seam is a no-op stub: a declared interface that H3 can tap when ready. No jagged-seam arbiter code runs in the shared-deliberation service; the seam exists to reserve the shape.

## Contract

### Request seam input

The `DeliberationRequest` dataclass includes a **`seam: dict | None`** field:

```python
@dataclass
class DeliberationRequest:
    # ... other fields ...
    seam: dict | None = None  # Reserved jagged-seam input (ignored in v0)
```

In v0, the `seam` field is accepted but ignored. When H3's arbiter is ready to register, it supplies seam-specific parameters in this dict (e.g., `{"mode": "jagged-arb", "config": {...}}`).

### Response seam output

The `DeliberationEnvelope` dataclass includes an **`extra_modes: list[dict]`** field:

```python
@dataclass
class DeliberationEnvelope:
    # ... other fields ...
    extra_modes: list[dict] = field(default_factory=list)  # Reserved seam output (empty in v0)
```

In v0, `extra_modes` is always an empty list. When H3's arbiter is ready, it populates this list with the results of its deliberation mode(s).

### Orchestrator seam loop

The orchestrator (`orchestrator.py`) reserves the extension point:

```python
async def run_deliberation(request: DeliberationRequest) -> DeliberationEnvelope:
    # ... Facets and Council execution ...

    # Reserved seam loop (v0: empty)
    for mode in _registered_seam_modes:
        result = await mode.run(request.seam, ...)
        envelope.extra_modes.append(result)

    return envelope
```

In v0, `_registered_seam_modes` is an empty list (no registration mechanism). H3 will add the registration contract.

## Governance: Who Can Register a Mode

**To prevent uncontrolled proliferation**, a seam-mode registration is code-level, not per-call:

1. The mode registers itself as a module-level handler in `agents_core.shared_deliberation.orchestrator`.
2. The registration handler must implement the **Mode Contract** (below).
3. **No permission gate beyond code review.** Any H3 arbiter that implements the contract can register (no PM override needed per-call).
4. If two modes conflict (same name), the last registered wins (first-write vs lazy-load convention to be documented in H3's extension).

This design allows H3 to plug in without touching this file post-registration; the tap is transparent to the service.

## Mode Contract

A registered seam mode must provide:

```python
class SeamMode:
    """Interface for a registered deliberation mode."""

    async def run(self, seam_config: dict | None, text: str, context: dict) -> dict:
        """
        Execute the mode.

        Args:
            seam_config: mode-specific config from request.seam dict
            text: the deliberation text (from request.text)
            context: caller-supplied context (from request.context)

        Returns:
            A dict with at least:
            - name: str — mode identifier (e.g. "jagged-arb")
            - ok: bool — success flag
            - result: dict — mode-specific result body
            - errors: dict — error detail if ok=False
        """
        ...
```

The mode is responsible for its own error handling, logging, and timeout. The orchestrator wraps the call in a try-except and treats any exception as mode failure (ok=False).

Example return shape:

```json
{
  "name": "jagged-arb",
  "ok": true,
  "result": {
    "some_deliberation_field": "...",
    "another_field": 42
  },
  "errors": {}
}
```

## Example: How H3 Would Register

In H3's arbiter module (hypothetical):

```python
# h3_arbiter.py
import asyncio
from agents_core.shared_deliberation import orchestrator

class JaggedArbiterMode:
    async def run(self, seam_config, text, context):
        # Run H3's deliberation logic
        ...
        return {
            "name": "jagged-arb",
            "ok": True,
            "result": {...},
            "errors": {},
        }

# Register at module load
orchestrator.register_seam_mode(JaggedArbiterMode())
```

In v0, the `register_seam_mode()` function is a no-op (stub); H3's merge will add the actual registration list.

## Future Work (H4.U5 and beyond)

1. **Multiple modes**: extend the loop to run all registered modes concurrently (per Budget).
2. **Mode discovery**: a `GET /v0/seam-modes` endpoint that lists registered modes.
3. **Per-mode timeout**: each mode gets its own timeout budget.
4. **Anti-spam guardrails**: rate-limit seam invocations or require explicit caller opt-in.
