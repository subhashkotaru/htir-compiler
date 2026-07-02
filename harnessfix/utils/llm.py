"""
Thin wrapper around the OpenRouter chat-completions API.

OpenRouter exposes an OpenAI-compatible endpoint at
https://openrouter.ai/api/v1, so the openai SDK can be reused
with a custom base_url.  Set OPENROUTER_API_KEY in your environment
(or .env file).

Supports structured JSON output via response_format.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o"   # OpenRouter model slug; change freely

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY is not set. "
                "Export it or add it to your .env file."
            )
        _client = OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                # Optional but recommended by OpenRouter docs
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://github.com/harnessfix"),
                "X-Title": os.getenv("OPENROUTER_APP_NAME", "HarnessFix"),
            },
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
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format

    resp = get_client().chat.completions.create(**kwargs)
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
    system_instruction = (
        "You are a precise analytical assistant. "
        "Respond ONLY with a valid JSON object that matches the following schema "
        "(no markdown fences, no extra keys):\n\n"
        f"{schema_str}"
    )

    augmented = [{"role": "system", "content": system_instruction}] + messages

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
