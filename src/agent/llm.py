"""Thin wrapper over the Anthropic Messages API used by the agent nodes.

The API key is read by the SDK from the environment (ANTHROPIC_API_KEY or an `ant auth login` profile);
it is never hard-coded. Any object with a compatible `.beta.messages.create(...)` can be injected
(tests use a scripted fake, so no network calls).
"""

from __future__ import annotations

from typing import Any

MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


class LLMError(RuntimeError):
    """The model refused or was cut off; the node records it and stops instead of guessing."""


class LLM:
    def __init__(self, client: Any | None = None, model: str = MODEL, effort: str = "medium",
                 max_tokens: int = 16000) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.stage = "other"                      # set by the graph to the node that is calling
        self.usage: dict[str, dict[str, int]] = {}  # tokens per stage, for the run log / UI

    def _record(self, response: Any) -> None:
        bucket = self.usage.setdefault(self.stage, {"calls": 0, **dict.fromkeys(USAGE_FIELDS, 0)})
        bucket["calls"] += 1
        u = getattr(response, "usage", None)
        for k in USAGE_FIELDS:
            bucket[k] += int(getattr(u, k, 0) or 0)

    def create(self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
               output_format: dict[str, Any] | None = None) -> Any:
        output_config: dict[str, Any] = {"effort": self.effort}
        if output_format:
            output_config["format"] = output_format
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            # stable system prompt cached across the tool loop
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
            "output_config": output_config,
            # on a safety decline the API re-runs the request on a fallback model inside the same call
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }
        if tools:
            kwargs["tools"] = tools
        response = self.client.beta.messages.create(**kwargs)
        self._record(response)
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMError(f"model refused: {getattr(details, 'category', None)} {getattr(details, 'explanation', '')}")
        if response.stop_reason == "max_tokens":
            raise LLMError("response hit max_tokens before finishing")
        return response


def text_of(response: Any) -> str:
    return "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
