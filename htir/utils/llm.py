"""
Thin wrapper around the OpenRouter chat-completions API, with a direct-OpenAI
fallback.

OpenRouter exposes an OpenAI-compatible endpoint at
https://openrouter.ai/api/v1, so the openai SDK can be reused
with a custom base_url.  Set OPENROUTER_API_KEY in your environment
(or .env file). If that is unset but OPENAI_API_KEY is (e.g. because you only
set up Track M's live-capture key, not a separate OpenRouter one -- see
docs/live-baselines-plan.md), every LLM-backed pass here (semantic checker,
monolithic judge, PRM step-critic, Agent-as-a-Judge) transparently falls back
to calling api.openai.com directly instead. That fallback can only serve
OpenAI models: an OpenRouter-style non-"openai/" slug (e.g.
"anthropic/claude-3.5-sonnet") raises immediately rather than silently
routing to the wrong provider. The "openai/" prefix itself is stripped before
the request, since the native OpenAI API doesn't understand OpenRouter's
provider-prefixed slugs.

Supports structured JSON output via response_format.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o"   # OpenRouter model slug; change freely

_client: Optional[Any] = None
_provider: Optional[str] = None  # "openrouter" | "openai", set by get_client()


# ---------------------------------------------------------------------------
# Token accounting (SA-9: report the real token cost of an LLM-judge run so the
# arms compare on a matched, disclosed budget -- avg.tex Sec. 4.7).
#
# Every LLM-backed pass in the codebase funnels through :func:`chat`, so a single
# module-level accumulator here captures the token cost of *any* experiment
# (monolithic judge, semantic checker, PRM step-critic, Agent-as-a-Judge) with no
# per-caller plumbing. Offline (no key -> zero calls) it stays a byte-exact zero,
# so the deterministic pipeline and its numbers are untouched.
# ---------------------------------------------------------------------------

class UsageTotals(BaseModel):
    """Cumulative token/call cost since the last :func:`reset_usage`."""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, prompt: int, completion: int, total: int) -> None:
        self.calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total


_usage = UsageTotals()


def reset_usage() -> None:
    """Zero the token accumulator (call before a run you want to measure)."""
    global _usage
    _usage = UsageTotals()


def get_usage() -> UsageTotals:
    """A copy of the token totals accumulated since the last reset."""
    return _usage.model_copy(deep=True)


def _record_usage(resp: Any) -> None:
    """Add one response's token usage to the accumulator (best-effort; a missing
    or malformed ``usage`` block counts the call but zero tokens)."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        _usage.add(0, 0, 0)
        return
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", 0) or (prompt + completion))
    _usage.add(prompt, completion, total)


def get_client() -> Any:
    global _client, _provider
    if _client is None:
        # Imported lazily so the deterministic pipeline (and ``import htir``)
        # works without the optional ``llm`` extra installed. Install it with
        # ``pip install htir[llm]`` to enable semantic checks/analysis.
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "The 'openai' package is required for LLM-backed passes. "
                "Install it with: pip install 'htir[llm]'"
            ) from exc
        try:
            # Optional (same 'llm' extra as openai); populates os.environ from a
            # local .env before the getenvs below, so a .env-only key works
            # without the caller having exported it into the shell. Silently a
            # no-op without the dependency or without a .env file present.
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        if openrouter_key:
            _provider = "openrouter"
            _client = OpenAI(
                api_key=openrouter_key,
                base_url=OPENROUTER_BASE_URL,
                default_headers={
                    # Optional but recommended by OpenRouter docs
                    "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://github.com/htir/htir"),
                    "X-Title": os.getenv("OPENROUTER_APP_NAME", "HTIR"),
                },
            )
        elif openai_key:
            # Fallback: no OpenRouter key, but a direct OpenAI key is already
            # set up (e.g. for Track M's live capture). OpenAI-only models
            # ("openai/..." slugs) work as-is; non-OpenAI slugs can't be
            # served by this path and are rejected in chat() below.
            _provider = "openai"
            _client = OpenAI(api_key=openai_key)
        else:
            raise EnvironmentError(
                "Neither OPENROUTER_API_KEY nor OPENAI_API_KEY is set. "
                "Export one, or add it to your .env file."
            )
    return _client


def chat(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    response_format: Optional[dict] = None,
) -> str:  # noqa: E501
    """Send a chat-completion request and return the assistant text."""
    client = get_client()  # sets _provider as a side effect; call before using it
    resolved_model = model
    if _provider == "openai":
        if not model.startswith("openai/"):
            raise EnvironmentError(
                f"Model {model!r} needs OpenRouter (OPENROUTER_API_KEY is not set, so this "
                "fell back to calling api.openai.com directly, which can only serve "
                "OpenAI models). Export OPENROUTER_API_KEY to use non-OpenAI models."
            )
        resolved_model = model.removeprefix("openai/")

    kwargs: dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:
        # Newer reasoning-tier models (e.g. gpt-5.x) reject the legacy
        # `max_tokens` param and require `max_completion_tokens` instead.
        # Retry once with the renamed param rather than hardcode a model
        # list that will inevitably go stale.
        if "max_completion_tokens" in str(exc) and "max_tokens" in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    _record_usage(resp)
    return resp.choices[0].message.content or ""


def chat_json(
    messages: list[dict],
    schema: Type[T],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> T:
    """
    Send a chat-completion request asking for JSON output conforming to
    the given Pydantic schema.  Returns a parsed instance of that schema.
    """
    schema_str = json.dumps(schema.model_json_schema(), indent=2)
    schema_instruction = (
        "Respond ONLY with a valid JSON object that matches the following schema "
        "(no markdown fences, no extra keys):\n\n"
        f"{schema_str}"
    )

    # Merge into the caller's existing system message (if any) instead of
    # prepending a second one -- avoids sending two competing "system" role
    # messages per request.
    augmented = list(messages)
    existing_idx = next((i for i, m in enumerate(augmented) if m.get("role") == "system"), None)
    if existing_idx is not None:
        merged = augmented[existing_idx]["content"] + "\n\n" + schema_instruction
        augmented[existing_idx] = {**augmented[existing_idx], "content": merged}
    else:
        augmented.insert(0, {
            "role": "system",
            "content": "You are a precise analytical assistant. " + schema_instruction,
        })

    raw = chat(
        augmented,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    return schema.model_validate_json(raw)


def system(content: str) -> dict:
    return {"role": "system", "content": content}


def user(content: str) -> dict:
    return {"role": "user", "content": content}


def assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}
