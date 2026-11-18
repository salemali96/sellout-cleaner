"""Pick the AI provider: Anthropic Claude, or any OpenAI-compatible API (Gemini, Groq, OpenRouter, Mistral,
OpenAI, a local Ollama, or a custom URL). Several have free tiers.

The agent code only speaks the Anthropic Messages shape (`client.beta.messages.create(...)` returning text and
tool_use blocks). `OpenAICompatClient` translates that shape to and from `/chat/completions`, so the profiler,
cleaner, reporter and analyst run unchanged on any provider. Keys live in .env, one variable per
provider, or (hosted) in each visitor's own session.
"""

from __future__ import annotations

import json
import os
import re
from contextvars import ContextVar
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from src import env
from src.agent.llm import MODEL as CLAUDE_MODEL
from src.agent.llm import LLM

PROVIDER_VAR = "LLM_PROVIDER"


@dataclass(frozen=True)
class Provider:
    id: str
    label: str
    tier: str                  # shown in the picker: "free tier", "paid", "local, free"
    key_var: str | None        # .env variable holding the key; None = no key needed
    base_url: str | None       # None = native Anthropic API
    default_model: str
    key_url: str
    note: str = ""
    max_tokens: int = 8192

    @property
    def model_var(self) -> str:
        return f"{self.id.upper()}_MODEL"

    @property
    def url_var(self) -> str:
        return f"{self.id.upper()}_BASE_URL"


PROVIDERS: dict[str, Provider] = {p.id: p for p in [
    Provider("gemini", "Google Gemini", "free tier", "GEMINI_API_KEY",
             "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-3.5-flash",
             "https://aistudio.google.com/apikey", "Free key from Google AI Studio, no card needed."),
    Provider("groq", "Groq", "free tier", "GROQ_API_KEY", "https://api.groq.com/openai/v1",
             "llama-3.3-70b-versatile", "https://console.groq.com/keys", "Free key, very fast open models."),
    Provider("openrouter", "OpenRouter", "free models", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1",
             "meta-llama/llama-3.3-70b-instruct:free", "https://openrouter.ai/keys",
             "One key for many models. Models ending in ':free' cost nothing."),
    Provider("mistral", "Mistral", "free tier", "MISTRAL_API_KEY", "https://api.mistral.ai/v1",
             "mistral-small-latest", "https://console.mistral.ai/api-keys", "Free 'Experiment' plan."),
    Provider("huggingface", "Hugging Face", "needs credits", "HF_TOKEN", "https://router.huggingface.co/v1",
             "meta-llama/Llama-3.3-70B-Instruct", "https://huggingface.co/settings/tokens",
             "Open models hosted by Hugging Face. Calls need credits: since Oct 2026 free accounts get none (PRO "
             "includes $2/month), so a new free account gets 'payment required'."),
    Provider("ollama", "Ollama (on this computer)", "local, free", None, "http://localhost:11434/v1",
             "qwen2.5:7b", "https://ollama.com/download",
             "Runs models on your own machine. No key, and no data leaves the computer. Hugging Face GGUF models work too: "
             "ollama pull hf.co/<user>/<repo>."),
    Provider("anthropic", "Anthropic Claude", "paid · best results", "ANTHROPIC_API_KEY", None, CLAUDE_MODEL,
             "https://console.anthropic.com/settings/keys", "Pay as you go.", 16000),
    Provider("openai", "OpenAI", "paid", "OPENAI_API_KEY", "https://api.openai.com/v1", "gpt-4.1-mini",
             "https://platform.openai.com/api-keys"),
    Provider("custom", "Custom (OpenAI-compatible)", "any", "CUSTOM_API_KEY", "", "",
             "", "Any service with an OpenAI-compatible /chat/completions endpoint."),
]}


# ---- settings (keys never logged) -------------------------------------------------------------
# Local use: settings live in .env / the process environment.
# Hosted use (many visitors, one server): the app calls `use_session(store)` with a dict kept in that visitor's
# browser session. Reads check the visitor's store first, then the server's environment (the host's shared key);
# writes go only to the store, so one visitor's key or choice is never seen or used by another.

_session: ContextVar[dict[str, str] | None] = ContextVar("ai_session_settings", default=None)


def use_session(store: dict[str, str] | None) -> None:
    _session.set(store)


def _get(name: str) -> str | None:
    store = _session.get()
    if store is not None and name in store:
        return store[name] or None
    return os.environ.get(name) or None


def _set(values: dict[str, str], persist: bool) -> None:
    store = _session.get()
    if store is None:
        env.set_values(values, persist=persist)
    else:
        store.update(values)


def current() -> Provider:
    pid = _get(PROVIDER_VAR) or ""
    if pid in PROVIDERS:
        return PROVIDERS[pid]
    # no explicit choice: use the first provider that already has a key, else Gemini (free)
    return next((p for p in PROVIDERS.values() if p.key_var and _get(p.key_var)), PROVIDERS["gemini"])


def select(p: Provider) -> None:
    _set({PROVIDER_VAR: p.id}, persist=True)


def model_of(p: Provider) -> str:
    return _get(p.model_var) or p.default_model


def base_url_of(p: Provider) -> str | None:
    return _get(p.url_var) or p.base_url


def key_of(p: Provider) -> str | None:
    return _get(p.key_var) if p.key_var else None


def is_shared(p: Provider) -> bool:
    """Hosted mode, and this provider's key is the host's (not one the visitor entered)."""
    store = _session.get()
    return bool(store is not None and p.key_var and p.key_var not in store and os.environ.get(p.key_var))


def is_ready(p: Provider | None = None) -> bool:
    p = p or current()
    if p.id == "custom" and not base_url_of(p):
        return False
    return p.key_var is None or bool(key_of(p))


def masked(p: Provider) -> str:
    if is_shared(p):
        return "shared key from the host"
    k = key_of(p) or ""
    return "no key needed" if p.key_var is None else (f"{k[:6]}...{k[-4:]}" if len(k) >= 16 else "(set)")


def connect(p: Provider, key: str | None, model: str | None, base_url: str | None, remember: bool) -> None:
    values: dict[str, str] = {PROVIDER_VAR: p.id}
    if p.key_var and key:
        values[p.key_var] = env.validate_key(key)
    if model:
        values[p.model_var] = model.strip()
    if base_url:
        values[p.url_var] = base_url.strip().rstrip("/")
    _set(values, persist=remember)


def disconnect(p: Provider) -> None:
    """Forget the key. In hosted mode this drops the visitor's own key (the host's shared key, if any, remains)."""
    store = _session.get()
    if not p.key_var:
        return
    if store is None:
        env.clear_value(p.key_var)
    else:
        store.pop(p.key_var, None)


# ---- clients ----------------------------------------------------------------------------------

def openai_client(p: Provider, key: str | None = None, base_url: str | None = None) -> Any:
    import openai

    return openai.OpenAI(api_key=key or key_of(p) or "not-needed", base_url=base_url or base_url_of(p), timeout=120,
                         max_retries=2)


def make_llm(effort: str = "medium", p: Provider | None = None, model: str | None = None) -> LLM:
    p = p or current()
    model = model or model_of(p)
    if not is_ready(p) or not model:
        raise RuntimeError(f"{p.label} is not connected yet (key, model or URL missing).")
    if p.base_url is None:
        import anthropic

        return LLM(anthropic.Anthropic(api_key=key_of(p)), model=model, effort=effort, max_tokens=p.max_tokens)
    return LLM(OpenAICompatClient(openai_client(p)), model=model, effort=effort, max_tokens=p.max_tokens)


def check(p: Provider, key: str | None = None, base_url: str | None = None) -> tuple[str, str, list[str]]:
    """Free check that a key works: lists the provider's models. Returns (status, message, model ids).
    `key`/`base_url` test new values before they are saved; by default the current settings are tested."""
    try:
        if p.base_url is None:
            import anthropic

            models = [m.id for m in anthropic.Anthropic(api_key=key or key_of(p)).models.list(limit=100)]
        else:
            models = [m.id for m in openai_client(p, key, base_url).models.list()]
    except Exception as e:  # noqa: BLE001 - classify any SDK error for the UI
        code = getattr(e, "status_code", None)
        if code in (400, 401, 403):
            return "rejected", f"{p.label} rejected this key. Check that you copied all of it.", []
        if "Connection" in type(e).__name__ or "Timeout" in type(e).__name__:
            hint = " Is Ollama running?" if p.id == "ollama" else " Check your internet connection or proxy."
            return "unreachable", f"Could not reach {p.label}.{hint}", []
        return "error", f"{type(e).__name__}: {str(e)[:200]}", []
    models = [m.removeprefix("models/") for m in models if not NOT_CHAT.search(m)]
    if p.id == "openrouter":
        models = sorted(models, key=lambda m: (not m.endswith(":free"), m))
    return "ok", f"{p.label} key works.", models


# models that cannot chat with tools (embeddings, images, speech, video, robotics...)
NOT_CHAT = re.compile(r"embed|image|imagen|tts|audio|transcribe|whisper|speech|live|veo|robotics|computer-use|"
                      r"banana|moderation|dall-e|sora|aqa|guard", re.I)


def explain_error(e: Exception, p: Provider | None = None, model: str | None = None) -> str:
    """Turn an SDK error into one sentence a user can act on."""
    p = p or current()
    model = model or model_of(p)
    code = getattr(e, "status_code", None)
    body = getattr(e, "body", None)
    detail = ""
    if isinstance(body, list) and body:
        body = body[0]
    if isinstance(body, dict):
        err = body.get("error", body)
        detail = (err.get("message") if isinstance(err, dict) else str(err)) or ""
    detail = detail.strip().split("\n")[0][:300]
    name = type(e).__name__
    if code == 404:
        msg = f"{p.label} does not offer the model '{model}' to your key. Pick another model in the sidebar."
    elif code in (401, 403):
        msg = f"{p.label} rejected the key. Reconnect it in the sidebar."
    elif code == 402:
        hint = (" Hugging Face free accounts get no AI credit (PRO includes a small monthly amount); see "
                "huggingface.co/settings/billing." if p.id == "huggingface" else "")
        msg = (f"{p.label} wants payment for this request: the free credit is used up, or '{model}' is only on a "
               f"paid plan.{hint} Switch to a free provider such as Google Gemini or Groq in the sidebar.")
    elif code == 429:
        msg = f"{p.label} rate limit or free quota reached. Wait a minute, or pick a smaller model."
    elif code in (500, 502, 503, 529):
        msg = f"'{model}' is overloaded at {p.label} right now. Try again shortly, or pick another model in the sidebar."
    elif "Timeout" in name:
        msg = f"'{model}' at {p.label} did not answer in time (often overload). Try again, or pick another model."
    elif "Connection" in name:
        msg = f"Could not reach {p.label}." + (" Is Ollama running?" if p.id == "ollama" else " Check your connection.")
    else:
        msg = f"{name}: {str(e)[:300]}"
    return msg + (f" ({p.label} said: {detail})" if detail else "")


# ---- Anthropic Messages shape <-> OpenAI chat completions -------------------------------------

def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(getattr(b, "text", None) or (b.get("text", "") if isinstance(b, dict) else "") for b in content)


def _block(b: Any, field: str) -> Any:
    return b.get(field) if isinstance(b, dict) else getattr(b, field, None)


def to_openai_messages(system: Any, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": _text(system)}]
    for m in messages:
        content = m["content"]
        if m["role"] == "user" and isinstance(content, list) and content and _block(content[0], "type") == "tool_result":
            for r in content:
                text = r["content"] if isinstance(r["content"], str) else _text(r["content"])
                out.append({"role": "tool", "tool_call_id": r["tool_use_id"],
                            "content": ("ERROR: " + text) if r.get("is_error") else text})
        elif m["role"] == "assistant" and not isinstance(content, str):
            calls = [_tool_call(b) for b in content if _block(b, "type") == "tool_use"]
            msg: dict[str, Any] = {"role": "assistant", "content": _text([b for b in content if _block(b, "type") == "text"]) or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:
            out.append({"role": m["role"], "content": _text(content)})
    return out


def _tool_call(b: Any) -> dict[str, Any]:
    call: dict[str, Any] = {"id": _block(b, "id"), "type": "function",
                            "function": {"name": _block(b, "name"), "arguments": json.dumps(_block(b, "input") or {})}}
    extra = _block(b, "extra_content")
    if extra:  # e.g. Gemini 3 thought_signature: must be sent back exactly as received
        call["extra_content"] = extra
    return call


def to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                              "parameters": t["input_schema"]}} for t in tools]


def extract_json(text: str) -> str:
    """Models without strict JSON mode sometimes wrap JSON in prose or ``` fences: keep only the object."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else text


def _extra_content(call: Any) -> Any:
    """Provider-specific data on a tool call (Gemini 3 puts its thought_signature here)."""
    extra = getattr(call, "model_extra", None) or {}
    return extra.get("extra_content") if isinstance(extra, dict) else None


STOP = {"tool_calls": "tool_use", "length": "max_tokens", "content_filter": "refusal", "stop": "end_turn"}


def from_openai(resp: Any, want_json: bool) -> Any:
    choice = resp.choices[0]
    msg = choice.message
    blocks: list[Any] = []
    text = msg.content or ""
    if want_json and text:
        text = extract_json(text)
    if text:
        blocks.append(SimpleNamespace(type="text", text=text))
    for i, call in enumerate(msg.tool_calls or []):
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}  # the tool then reports the missing arguments back to the model
        blocks.append(SimpleNamespace(type="tool_use", id=call.id or f"call_{i}", name=call.function.name,
                                      input=args if isinstance(args, dict) else {},
                                      extra_content=_extra_content(call)))
    stop = "tool_use" if msg.tool_calls else STOP.get(choice.finish_reason or "stop", "end_turn")
    u = getattr(resp, "usage", None)
    prompt = getattr(u, "prompt_tokens", 0) or 0
    cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    usage = SimpleNamespace(input_tokens=prompt - cached, output_tokens=getattr(u, "completion_tokens", 0) or 0,
                            cache_read_input_tokens=cached, cache_creation_input_tokens=0)
    return SimpleNamespace(stop_reason=stop, content=blocks, usage=usage, stop_details=None)


class OpenAICompatClient:
    """Exposes `.beta.messages.create(**anthropic_kwargs)` on top of an OpenAI-compatible client."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.json_schema_ok = True  # flips off after a provider rejects json_schema; then json_object is used
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, *, model: str, max_tokens: int, system: Any, messages: list[dict[str, Any]],
               output_config: dict[str, Any] | None = None, tools: list[dict[str, Any]] | None = None,
               **_anthropic_only: Any) -> Any:
        fmt = (output_config or {}).get("format")
        sys_text = _text(system)
        if fmt:
            sys_text += ("\n\nReply with one JSON object only, no prose, matching this JSON schema:\n"
                         + json.dumps(fmt["schema"]))
        kw: dict[str, Any] = {"model": model, "max_tokens": max_tokens,
                              "messages": to_openai_messages(sys_text, messages)}
        if tools:
            kw["tools"] = to_openai_tools(tools)
        if fmt:
            return self._json_call(kw, fmt["schema"])
        return from_openai(self.client.chat.completions.create(**kw), want_json=False)

    def _json_call(self, kw: dict[str, Any], schema: dict[str, Any]) -> Any:
        if self.json_schema_ok:
            try:
                rf = {"type": "json_schema", "json_schema": {"name": "output", "schema": schema}}
                return from_openai(self.client.chat.completions.create(**kw, response_format=rf), want_json=True)
            except Exception as e:  # noqa: BLE001 - only a 400 means "json_schema not supported here"
                if getattr(e, "status_code", None) != 400:
                    raise
                self.json_schema_ok = False
        try:
            return from_openai(self.client.chat.completions.create(**kw, response_format={"type": "json_object"}),
                               want_json=True)
        except Exception as e:  # noqa: BLE001 - some models support neither mode; the prompt still asks for JSON
            if getattr(e, "status_code", None) != 400:
                raise
            return from_openai(self.client.chat.completions.create(**kw), want_json=True)
