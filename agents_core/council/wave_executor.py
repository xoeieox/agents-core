"""agents_core.council.wave_executor — concurrent seat fan-out for WaveDirector rounds.

D2 of council-wave-mode-v0. Copies the *shape* of agents_core.llm.call_swarm
(bounded ThreadPoolExecutor, order-preserving, per-item failure isolation)
rather than importing it — call_swarm targets bare prompt strings against the
lease-free swarm endpoint (SWARM_URL); council seats carry per-seat system
cards and voice through call_operator under the doorman lease and a shared
principal. Reuse the concurrency pattern, keep the council's own call path.

Supplied as the `executor` callable to lapis_engine.engine.Engine.run_waves().
Its contract (enforced by run_waves itself, mirrored here): return exactly
one response per (entity, prompt) pair, in seat order, regardless of
completion order. A seat whose act() raises must yield a SeatError in its
slot rather than propagate — one seat's failure must never shorten or
reorder the round; a short round corrupts the transcript every downstream
consumer re-parses (council-wave-mode-v0 D2).
"""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Callable

from agents_core.llm import SWARM_MAX_CONCURRENT


def concurrent_wave_executor(
    pairs: list,
    ctx,
    max_concurrent: int | None = None,
    on_seat_complete: Callable[[object, "str | object"], None] | None = None,
):
    """Bounded, order-preserving executor for one WaveDirector round.

    Args:
        pairs: list[(Entity, str)] — the round's (entity, prompt) pairs, in
            seat order (this is `director.next_wave(entities)` zipped with
            per-seat prompts, exactly as run_waves builds it).
        ctx: the round's shared RunContext, passed through to every
            entity.act() call unchanged (every seat acts blind to its
            round-mates — see WaveDirector's docstring).
        max_concurrent: bound on in-flight seats. Defaults to
            SWARM_MAX_CONCURRENT (env-overridable), mirroring call_swarm.
        on_seat_complete: optional callback fired the instant each seat's
            act() call actually returns (success or failure) — i.e. on real
            completion, not once the whole round has drained. council/cli.py
            hooks this to refresh the deliberation-hold lease per seat
            completion rather than per round (H6): if the refresh instead
            waited for this function to return, it would only fire after
            every seat in the round had already resolved, which is exactly
            the round-cadence regression H6 exists to prevent.

    Returns exactly one entry per pair, in seat order — the run_waves()
    contract. A seat whose act() raises yields a SeatError in its slot
    rather than propagating.
    """
    from lapis_engine.types import SeatError  # function-local: no module-load lapis-engine dep

    if max_concurrent is None:
        max_concurrent = SWARM_MAX_CONCURRENT

    if not pairs:
        return []

    results: list = [None] * len(pairs)

    def _call_seat(index: int, entity, prompt: str):
        try:
            response = entity.act(prompt, ctx)
        except Exception as exc:  # noqa: BLE001 - isolate any seat failure
            response = SeatError(
                seat_id=entity.id, error_class=type(exc).__name__, message=str(exc)
            )
        return index, entity, response

    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = {
            pool.submit(_call_seat, i, entity, prompt): i
            for i, (entity, prompt) in enumerate(pairs)
        }
        pending = set(futures)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                index, entity, response = future.result()
                results[index] = response
                if on_seat_complete is not None:
                    try:
                        on_seat_complete(entity, response)
                    except Exception:  # noqa: BLE001 - a hook failure must not sink the round
                        pass

    return results
