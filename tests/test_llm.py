"""DeepSeek wrapper checks; no network or student implementations required."""

from types import SimpleNamespace

from openai.types.chat import ChatCompletion
import pytest

from cs329z_hw1 import llm

MESSAGES = [{"role": "user", "content": "Hello"}]


@pytest.fixture
def api(monkeypatch, tmp_path):
    # Ignore real credentials/configuration, and keep cache and ledger isolated.
    monkeypatch.setattr(llm, "_load_env", lambda: None)
    for name in ("CS329Z_GRADING_MODEL", "CS329Z_USER_MODEL", "CS329Z_JUDGE_MODEL", "CS329Z_BUDGET_USD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CS329Z_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    requests = []
    result = {
        "id": "test", "object": "chat.completion", "created": 0,
        "model": "deepseek-flash",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
                  "prompt_cache_hit_tokens": 60},
    }

    def client(**kwargs):
        def create(**body):
            requests.append({"client": kwargs, "body": body})
            return ChatCompletion.model_validate(result)
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr("openai.OpenAI", client)
    return requests, result


@pytest.mark.parametrize("role,model", [(None, "deepseek-flash"), ("user", "deepseek-flash"), ("judge", "deepseek-v4-pro")])
def test_roles_and_request(api, role, model):
    requests, _ = api
    assert llm.LM(role)(MESSAGES) == "OK"
    request = requests[0]
    assert request["client"]["base_url"] == "https://api.deepseek.com"
    assert request["client"]["api_key"] == "test-key"
    assert request["body"] == {
        "model": model, "messages": MESSAGES, "max_tokens": 4096,
        "temperature": 0.0, "extra_body": {"thinking": {"type": "disabled"}},
    }


def test_role_override_and_unknown_model(api, monkeypatch):
    monkeypatch.setenv("CS329Z_GRADING_MODEL", "deepseek-v4-pro")
    assert llm.LM().model == "deepseek-v4-pro"
    with pytest.raises(KeyError, match="No price on file"):
        llm.LM("unknown-model")
    assert not api[0]


def test_missing_key(api, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        llm.LM()(MESSAGES)
    assert not api[0]


@pytest.mark.parametrize("nested", [False, True])
def test_usage_cache_and_budget(api, monkeypatch, nested):
    requests, response = api
    if nested:
        response["usage"].pop("prompt_cache_hit_tokens")
        response["usage"]["prompt_tokens_details"] = {"cached_tokens": 60}
    model = llm.LM()
    assert model(MESSAGES) == "OK"
    row, = llm.read_ledger()
    expected = (40 * 0.30 + 60 * 0.006 + 10 * 1.20) / 1_000_000
    assert row["cached_tokens"] == 60
    assert row["cost_usd"] == pytest.approx(expected)
    assert model.usage()["cost_usd"] == pytest.approx(expected)
    monkeypatch.setenv("CS329Z_BUDGET_USD", str(expected / 2))
    assert model(MESSAGES) == "OK"  # cached replies remain available at the cap
    assert model.cached_calls == 1
    with pytest.raises(llm.BudgetExceeded):
        model([{"role": "user", "content": "New question"}])
    assert len(requests) == 1
    assert len(llm.read_ledger()) == 1


def test_truncation_is_charged_but_not_cached(api):
    requests, response = api
    response["choices"][0]["finish_reason"] = "length"
    model = llm.LM(max_tokens=10)
    for _ in range(2):
        with pytest.raises(llm.OutputTruncated) as exc:
            model(MESSAGES)
        assert exc.value.text == "OK"
        assert exc.value.output_tokens == 10
    assert len(requests) == 2
    assert len(llm.read_ledger()) == 2
    assert model.cached_calls == 0


def test_empty_response_not_cached(api):
    requests, response = api
    response["choices"][0]["message"]["content"] = ""
    model = llm.LM()
    assert model(MESSAGES) == ""
    assert model(MESSAGES) == ""
    assert len(requests) == 2


def test_scripted_lm_stays_offline(api):
    model = llm.ScriptedLM(["Hello"])
    assert model(MESSAGES) == "Hello"
    assert not api[0]
    assert not llm.read_ledger()
