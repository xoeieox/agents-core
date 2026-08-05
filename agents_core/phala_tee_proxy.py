"""agents_core.phala_tee_proxy — local OpenAI-compatible HTTP proxy over
`agents_core.phala_tee.PhalaTeeClient`.

Turns Phala's `tee_preferred` tier into a plain OpenAI-compatible endpoint
any OpenAI-compatible client (opencode, curl, a future Hermes-style tool)
can point at without knowing anything about attestation or e2ee sealing.
`PhalaTeeClient.chat_completion` does the real confidentiality work (fetch
attestation, verify report binding, seal the request, decrypt the
response) — a client pointed straight at `https://inference.phala.com/v1`
gets an OpenAI-shaped API back but skips all of that. This module is what
turns "an OpenAI-compatible endpoint that happens to be Phala" into "the
tee_preferred privacy claim this client exists to provide."

Mirrors `agents_core/slot_server.py`'s module-shape conventions (env vars,
fail-closed non-loopback bind guard, hmac.compare_digest bearer auth).

Entry point:  python -m agents_core.phala_tee_proxy

Environment variables:
  PHALA_TEE_PROXY_BIND_HOST   — uvicorn bind host (default 127.0.0.1)
  PHALA_TEE_PROXY_BIND_PORT   — uvicorn bind port (default 8413)
  PHALA_TEE_PROXY_LOG_LEVEL   — uvicorn log level (default info)
  PHALA_TEE_PROXY_BEARER_TOKEN — optional bearer token. A blank or
    whitespace-only value is treated as equivalent to unset (still
    fail-closed on a non-loopback bind) — a config typo must never
    silently downgrade to an unauthenticated public endpoint.
  PHALA_API_KEY — read by `PhalaTeeClient` itself via its existing
    `os.environ.get("PHALA_API_KEY")` fallback; this module does not
    re-implement key loading.

Non-loopback bind guard: if PHALA_TEE_PROXY_BIND_HOST is not a loopback
address and PHALA_TEE_PROXY_BEARER_TOKEN is unset (or blank), refuse to
start — same fail-closed precedent as slot_server's SLOTS_BEARER_TOKEN
guard. v0 ships loopback-only in practice; this guard is real code, not
just a docstring warning.

Routes:
  POST /v1/chat/completions — the only real route.
  GET  /v1/models           — static catalog of empirically-good models.
  anything else             — 404 (FastAPI default; this is a narrow
                               proxy, not a general Phala gateway wrapper).

MAX_TOKENS_CEILING is a living, hand-maintained table: vendor-advertised
`max_output_length` values are not trustworthy (deepseek/deepseek-v3.2
advertises 64000 but its live serving backend hard-400s above 8192,
empirically bisected on the MacBook precedent). Add an entry whenever a
new model's real ceiling gets bisected — no automated bisection tooling
in v0.

Streaming (fake-SSE) error framing: the full response is always buffered
— decrypted and validated — before any SSE bytes are written. A failure
from `PhalaTeeClient.chat_completion` (including
`ReasoningContentDecryptionError`) is always caught before the stream
starts and surfaces as an ordinary 502 HTTP response. There is no
mid-stream error case to design for in v0: nothing is ever emitted before
success is confirmed. Do not add speculative pre-decryption chunks (e.g.
a "processing" heartbeat) without re-checking this invariant — that would
reintroduce the exact failure mode this note rules out.
"""
from __future__ import annotations

import hmac
import ipaddress
import json
import os
import sys
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from agents_core.phala_tee import PhalaTeeClient

DEFAULT_BIND_HOST = "127.0.0.1"
DEFAULT_BIND_PORT = 8413
DEFAULT_MAX_TOKENS = 4096

# Empirically-bisected per-model max_tokens ceilings — vendor-advertised
# numbers are not trustworthy, see module docstring.
MAX_TOKENS_CEILING = {
    "deepseek/deepseek-v3.2": 8192,
    "deepseek/deepseek-v4-flash": 131072,
    # Dated pin for the same backend the floating alias serves today. 131072
    # is a safe *generation* budget, deliberately NOT the measured limit: the
    # real constraint is a CONTEXT cap, prompt_tokens + max_tokens <= 2^20
    # (bisected 2026-08-05, research/phala-0731-quality-bakeoff-2026-08-05).
    # _clamp_max_tokens ASSIGNS this value when max_tokens is absent, so a
    # context-sized entry here would 400 on any non-trivial prompt.
    "deepseek/deepseek-v4-flash-0731": 131072,
}

# GOOD per research/phala-model-quality-vs-seal-2026-07-29 and
# research/phala-0731-quality-bakeoff-2026-08-05 (mem). Excludes
# google/gemma-4-31b-it (corrupts ~25% of outputs), z-ai/glm-5.2 and
# moonshotai/kimi-k2.6 (empty-answer trap) — those must never be offered
# as a default catalog entry.
MODEL_CATALOG = [
    {"id": "deepseek/deepseek-v3.2", "object": "model", "owned_by": "phala"},
    {"id": "deepseek/deepseek-v4-flash", "object": "model", "owned_by": "phala"},
    {
        "id": "deepseek/deepseek-v4-flash-0731",
        "object": "model",
        "owned_by": "phala",
        # Same backend as deepseek/deepseek-v4-flash today. Measured clean 8/8
        # on 2026-08-05. Sealing is UNAFFECTED: this id verifies and applies
        # e2ee exactly like every other, same workload_keyset_digest.
        # The flag records a LEGIBILITY gap only — unlike the floating alias,
        # this id echoes the bare routing id in response.model instead of a
        # served-model name, so there is no string to watch for drift. This
        # is a deliberate trade-off, not a defect: the pin buys immutable
        # weights at the cost of response-level auditability — do not "fix"
        # it by dropping the pin.
        # Do NOT read the alias's name as attested provenance either: Phala's
        # attestation is gateway-scoped, identical across all models, and says
        # nothing about which weights served the call
        # (finding/phala-attestation-is-gateway-scoped-not-model-scoped-2026-08-05).
        "served_model_unreported": True,
    },
    {"id": "openai/gpt-oss-120b", "object": "model", "owned_by": "phala"},
    {
        "id": "qwen/qwen3.5-122b-a10b",
        "object": "model",
        "owned_by": "phala",
        "reasoning_heavy": True,
    },
]


def _error(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def _clamp_max_tokens(model: str, body: dict) -> None:
    """Mutate `body["max_tokens"]` in place per the per-model ceiling table.

    A listed model: clamp down to the ceiling if absent or over it; never
    raise a value that's already under the ceiling. An unlisted model:
    pass through unmodified if present, default to DEFAULT_MAX_TOKENS if
    absent (avoids the empty-answer-via-uncapped-reasoning trap)."""
    ceiling = MAX_TOKENS_CEILING.get(model)
    max_tokens = body.get("max_tokens")
    if ceiling is not None:
        if max_tokens is None or max_tokens > ceiling:
            body["max_tokens"] = ceiling
    elif max_tokens is None:
        body["max_tokens"] = DEFAULT_MAX_TOKENS


def _build_stream_chunks(response: dict) -> list[dict]:
    """Build the fake-SSE chunk pair for a buffered chat-completion
    response. Forwards every field present in the real message — content,
    reasoning_content, tool_calls (each with its own index) — not just
    content. This is the regression guard for the MacBook bug
    (fix/phala-test-key-fake-sse-dropped-tool-calls-2026-07-29) where the
    first fake-SSE implementation silently dropped tool_calls while still
    reporting finish_reason: "tool_calls"."""
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason", "stop")
    response_id = response.get("id", "")
    model = response.get("model", "")
    created = response.get("created", int(time.time()))

    delta: dict[str, Any] = {"role": message.get("role", "assistant")}
    if message.get("content") is not None:
        delta["content"] = message["content"]
    if message.get("reasoning_content") is not None:
        delta["reasoning_content"] = message["reasoning_content"]
    tool_calls = message.get("tool_calls")
    if tool_calls:
        indexed_tool_calls = []
        for i, tc in enumerate(tool_calls):
            tc = dict(tc)
            tc.setdefault("index", i)
            indexed_tool_calls.append(tc)
        delta["tool_calls"] = indexed_tool_calls

    content_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }
    closing_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    return [content_chunk, closing_chunk]


def _sse_stream(response: dict):
    for chunk in _build_stream_chunks(response):
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


def create_app(client: PhalaTeeClient | None = None) -> FastAPI:
    app = FastAPI(title="phala-test-key", version="0")
    client = client or PhalaTeeClient()

    _token = os.environ.get("PHALA_TEE_PROXY_BEARER_TOKEN", "").strip()

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if _token:
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                return JSONResponse(
                    status_code=401,
                    content=_error("unauthorized", "Missing or invalid bearer token"),
                )
            presented = auth[len("Bearer "):]
            if not hmac.compare_digest(presented.encode("utf-8"), _token.encode("utf-8")):
                return JSONResponse(
                    status_code=401,
                    content=_error("unauthorized", "Missing or invalid bearer token"),
                )
        return await call_next(request)

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": MODEL_CATALOG}

    @app.post("/v1/chat/completions")
    def chat_completions(body: dict[str, Any]):
        model = body.get("model")
        if not model:
            raise HTTPException(status_code=400, detail=_error("bad_request", "missing field 'model'"))
        messages = body.get("messages")
        if not messages:
            raise HTTPException(status_code=400, detail=_error("bad_request", "missing field 'messages'"))
        stream = bool(body.get("stream", False))

        extra_body = {k: v for k, v in body.items() if k not in ("model", "messages", "stream")}
        if extra_body.get("reasoning_effort") == "none":
            del extra_body["reasoning_effort"]
        _clamp_max_tokens(model, extra_body)

        try:
            response = client.chat_completion(messages=messages, model=model, extra_body=extra_body)
        except Exception as e:
            raise HTTPException(status_code=502, detail=_error("upstream_error", str(e)))

        if not stream:
            return response
        return StreamingResponse(_sse_stream(response), media_type="text/event-stream")

    return app


def main():
    import uvicorn

    host = os.environ.get("PHALA_TEE_PROXY_BIND_HOST", DEFAULT_BIND_HOST)
    port = int(os.environ.get("PHALA_TEE_PROXY_BIND_PORT", str(DEFAULT_BIND_PORT)))
    log_level = os.environ.get("PHALA_TEE_PROXY_LOG_LEVEL", "info")
    token = os.environ.get("PHALA_TEE_PROXY_BEARER_TOKEN", "").strip()

    # Fail-closed: refuse to start if no (non-blank) bearer token and not
    # loopback-bound. Same precedent as slot_server's SLOTS_BEARER_TOKEN
    # guard — a config typo (blank/whitespace token) must never silently
    # downgrade to an unauthenticated public endpoint.
    if not token:
        try:
            addr = ipaddress.ip_address(host)
            if not addr.is_loopback:
                print(
                    f"FATAL: PHALA_TEE_PROXY_BEARER_TOKEN is unset but "
                    f"PHALA_TEE_PROXY_BIND_HOST={host!r} is non-loopback. Set "
                    f"PHALA_TEE_PROXY_BEARER_TOKEN or bind to 127.0.0.1.",
                    file=sys.stderr,
                )
                sys.exit(1)
        except ValueError:
            # Host is a hostname string, not a bare IP — can't check loopback
            # status at startup. The operator is responsible for token config.
            pass

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
