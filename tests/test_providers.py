"""Provider layer: settings in .env, and the Anthropic <-> OpenAI-compatible translation (no network)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src import env
from src.agent import graph, providers
from src.agent.llm import LLM, LLMError
from tests.test_agent import cleaner_round_one, session  # noqa: F401 - session is a fixture

# ---- fake OpenAI-compatible client ------------------------------------------------------------


class BadRequest(Exception):
    status_code = 400


def oa_text(t: str, finish: str = "stop") -> Any:
    msg = SimpleNamespace(content=t, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish)],
                           usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7))


def oa_tools(*calls: tuple[str, dict[str, Any] | str]) -> Any:
    tc = [SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=n, arguments=a if isinstance(a, str) else json.dumps(a)))
          for i, (n, a) in enumerate(calls)]
    msg = SimpleNamespace(content=None, tool_calls=tc)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")], usage=None)


class FakeOpenAI:
    def __init__(self, script: list[Any]) -> None:
        self.script, self.calls = list(script), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step(kw) if callable(step) else step


def as_openai(anthropic_script: list[Any]) -> list[Any]:
    """Re-express the Anthropic-shaped cleaner script from test_agent as OpenAI responses."""
    out = []
    for r in anthropic_script:
        uses = [b for b in r.content if b.type == "tool_use"]
        out.append(oa_tools(*[(u.name, u.input) for u in uses]) if uses else oa_text(r.content[0].text))
    return out


# ---- translation ------------------------------------------------------------------------------

def test_messages_and_tools_are_translated() -> None:
    blocks = [SimpleNamespace(type="text", text="looking"),
              SimpleNamespace(type="tool_use", id="t1", name="get_rows", input={"row_nrs": [1]})]
    msgs = providers.to_openai_messages([{"type": "text", "text": "SYS"}], [
        {"role": "user", "content": "brief"},
        {"role": "assistant", "content": blocks},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "bad", "is_error": True}]},
    ])
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[2]["content"] == "looking" and msgs[2]["tool_calls"][0]["function"]["arguments"] == '{"row_nrs": [1]}'
    assert msgs[3] == {"role": "tool", "tool_call_id": "t1", "content": "ERROR: bad"}
    tool = providers.to_openai_tools([{"name": "x", "description": "d", "input_schema": {"type": "object"}, "strict": True}])
    assert tool == [{"type": "function", "function": {"name": "x", "description": "d", "parameters": {"type": "object"}}}]


def test_responses_become_anthropic_blocks() -> None:
    r = providers.from_openai(oa_tools(("get_rows", {"row_nrs": [2]}), ("get_rows", "{not json")), want_json=False)
    assert r.stop_reason == "tool_use" and [b.input for b in r.content] == [{"row_nrs": [2]}, {}]
    r = providers.from_openai(oa_text('Sure!\n```json\n{"a": 1}\n```'), want_json=True)
    assert json.loads(r.content[0].text) == {"a": 1} and r.usage.input_tokens == 11


def test_cut_off_and_filtered_answers_raise() -> None:
    for finish in ("length", "content_filter"):
        llm = LLM(providers.OpenAICompatClient(FakeOpenAI([oa_text("x", finish)])), model="m")
        with pytest.raises(LLMError):
            llm.create(system="s", messages=[{"role": "user", "content": "q"}])


def test_json_output_falls_back_when_schema_mode_unsupported() -> None:
    fake = FakeOpenAI([BadRequest("no json_schema"), oa_text('{"ok": true}'), oa_text('{"ok": 2}')])
    client = providers.OpenAICompatClient(fake)
    llm = LLM(client, model="m")
    fmt = {"type": "json_schema", "schema": {"type": "object"}}
    assert json.loads(llm.create(system="s", messages=[{"role": "user", "content": "q"}], output_format=fmt).content[0].text)
    llm.create(system="s", messages=[{"role": "user", "content": "q"}], output_format=fmt)
    kinds = [c.get("response_format", {}).get("type") for c in fake.calls]
    assert kinds == ["json_schema", "json_object", "json_object"]  # remembered: no second failed attempt
    assert "JSON schema" in fake.calls[0]["messages"][0]["content"]
    assert "betas" not in fake.calls[0] and "fallbacks" not in fake.calls[0]


def test_full_agent_runs_on_an_openai_compatible_provider(session) -> None:
    s, _, _ = session([], approve=True)
    fake = FakeOpenAI(as_openai(cleaner_round_one()))
    s.llm = LLM(providers.OpenAICompatClient(fake), model="free-model")
    state = graph.run_agent(s)
    assert state["status"] == "done" and [p["status"] for p in state["plan"]] == ["applied", "applied", "approved"]
    assert s.result.log["reconciliation"]["passed"]
    assert fake.calls[0]["model"] == "free-model" and fake.calls[0]["tools"][0]["type"] == "function"
    assert fake.calls[1]["messages"][-1]["role"] == "tool"
    assert s.llm.usage["cleaner"]["calls"] == 2  # lean: no follow-up summary request


# ---- settings -----------------------------------------------------------------------------------

@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    for p in providers.PROVIDERS.values():
        for var in (p.key_var, p.model_var, p.url_var):
            if var:
                monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv(providers.PROVIDER_VAR, raising=False)
    path = tmp_path / ".env"
    monkeypatch.setattr(env, "ENV_FILE", path)
    monkeypatch.setattr(env.set_values, "__defaults__", (True, path))
    monkeypatch.setattr(env.clear_value, "__defaults__", (path,))
    return path


def test_current_provider_defaults_and_detection(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert providers.current().id == "gemini" and not providers.is_ready()
    monkeypatch.setenv("GROQ_API_KEY", "gsk_123")
    assert providers.current().id == "groq" and providers.is_ready()
    assert providers.is_ready(providers.PROVIDERS["ollama"])          # no key needed
    assert not providers.is_ready(providers.PROVIDERS["custom"])      # needs a URL


def test_connect_persists_and_disconnect_clears(clean_env: Path) -> None:
    gem = providers.PROVIDERS["gemini"]
    providers.connect(gem, "AIza-test-key-123456", "gemini-x", None, remember=True)
    saved = env.parse_env(clean_env.read_text())
    assert saved == {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "AIza-test-key-123456", "GEMINI_MODEL": "gemini-x"}
    assert providers.current() is gem and providers.model_of(gem) == "gemini-x" and providers.masked(gem) == "AIza-t...3456"
    providers.disconnect(gem)
    assert env.parse_env(clean_env.read_text())["GEMINI_API_KEY"] == "" and not providers.is_ready(gem)
    with pytest.raises(ValueError):
        providers.connect(gem, "has space", None, None, remember=False)


def test_session_only_connect_writes_nothing(clean_env: Path) -> None:
    providers.connect(providers.PROVIDERS["groq"], "gsk_abc", None, None, remember=False)
    assert not clean_env.exists() and os.environ["GROQ_API_KEY"] == "gsk_abc"


def test_make_llm_picks_the_right_client(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "m-key")
    llm = providers.make_llm(p=providers.PROVIDERS["mistral"])
    assert isinstance(llm.client, providers.OpenAICompatClient) and llm.model == "mistral-small-latest"
    assert str(llm.client.client.base_url).startswith("https://api.mistral.ai/v1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-x")
    assert not isinstance(providers.make_llm(p=providers.PROVIDERS["anthropic"]).client, providers.OpenAICompatClient)
    with pytest.raises(RuntimeError):
        providers.make_llm(p=providers.PROVIDERS["openai"])


class ApiError(Exception):
    def __init__(self, code: int, body: Any = None) -> None:
        super().__init__(f"Error code: {code}")
        self.status_code, self.body = code, body


def test_errors_are_explained_in_plain_words() -> None:
    gem = providers.PROVIDERS["gemini"]
    gone = ApiError(404, [{"error": {"code": 404, "message": "This model is no longer available to new users."}}])
    msg = providers.explain_error(gone, gem, "gemini-old")
    assert "does not offer the model 'gemini-old'" in msg and "no longer available" in msg
    assert "overloaded" in providers.explain_error(ApiError(503, {"error": {"message": "high demand"}}), gem, "m")
    assert "quota" in providers.explain_error(ApiError(429), gem, "m")

    class APITimeoutError(Exception):
        pass

    assert "did not answer in time" in providers.explain_error(APITimeoutError(), gem, "m")


def test_model_list_drops_non_chat_models() -> None:
    names = ["gemini-3.5-flash", "text-embedding-004", "gemini-3.5-transcribe", "imagen-4", "gemini-omni-flash-preview"]
    assert [m for m in names if not providers.NOT_CHAT.search(m)] == ["gemini-3.5-flash", "gemini-omni-flash-preview"]


def test_gemini_thought_signature_is_sent_back() -> None:
    sig = {"google": {"thought_signature": "abc=="}}
    call = SimpleNamespace(id="c0", function=SimpleNamespace(name="get_rows", arguments='{"row_nrs": [1]}'),
                           model_extra={"extra_content": sig})
    plain = SimpleNamespace(id="c1", function=SimpleNamespace(name="get_rows", arguments='{"row_nrs": [2]}'), model_extra={})
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call, plain]),
                                                    finish_reason="tool_calls")], usage=None)
    blocks = providers.from_openai(resp, want_json=False).content
    msgs = providers.to_openai_messages("s", [{"role": "assistant", "content": blocks}])
    first, second = msgs[1]["tool_calls"]
    assert first["extra_content"] == sig and "extra_content" not in second


def test_cached_prompt_tokens_are_reported_separately() -> None:
    resp = oa_text("hi")
    resp.usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=5,
                                 prompt_tokens_details=SimpleNamespace(cached_tokens=800))
    u = providers.from_openai(resp, want_json=False).usage
    assert (u.input_tokens, u.cache_read_input_tokens) == (200, 800)


# ---- hosted: one server, many visitors ----------------------------------------------------------

def test_each_visitor_has_private_settings_over_the_hosts_shared_key(clean_env: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-host-shared-key-0001")
    gem, groq = providers.PROVIDERS["gemini"], providers.PROVIDERS["groq"]
    visitor_a: dict[str, str] = {}
    visitor_b: dict[str, str] = {}
    try:
        providers.use_session(visitor_a)
        assert providers.is_ready(gem) and providers.is_shared(gem) and providers.masked(gem) == "shared key from the host"
        providers.connect(groq, "gsk_visitor_a", "some-model", None, remember=True)
        assert providers.current() is groq and visitor_a["GROQ_API_KEY"] == "gsk_visitor_a"
        assert not clean_env.exists() and "GROQ_API_KEY" not in os.environ  # nothing written to the server

        providers.use_session(visitor_b)
        assert providers.current() is gem and not providers.is_ready(groq)  # A's key and choice are invisible to B
        providers.connect(gem, "AIza-visitor-b-key-99", None, None, remember=True)
        assert not providers.is_shared(gem) and providers.key_of(gem) == "AIza-visitor-b-key-99"
        providers.disconnect(gem)
        assert providers.key_of(gem) == "AIza-host-shared-key-0001" and providers.is_shared(gem)
    finally:
        providers.use_session(None)
    assert os.environ["GEMINI_API_KEY"] == "AIza-host-shared-key-0001"


def test_check_can_test_a_key_before_saving_it(clean_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, str]] = []

    def fake_client(p: providers.Provider, key: str | None = None, base_url: str | None = None) -> Any:
        seen.append((key or providers.key_of(p) or "", base_url or providers.base_url_of(p) or ""))
        return SimpleNamespace(models=SimpleNamespace(list=lambda: [SimpleNamespace(id="m1")]))

    monkeypatch.setattr(providers, "openai_client", fake_client)
    status, _, models = providers.check(providers.PROVIDERS["groq"], key="gsk_new")
    assert status == "ok" and models == ["m1"] and seen[0][0] == "gsk_new"
    assert not providers.is_ready(providers.PROVIDERS["groq"])  # checking did not connect
