"""Unit tests for _invalid_json_recovery_content (invalid-JSON recovery)."""

from types import SimpleNamespace

from agent.conversation_loop import _invalid_json_recovery_content


def _tc(name: str, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(function=SimpleNamespace(name=name), id=call_id)


def test_same_name_valid_sibling_is_skipped():
    """A valid sibling call to the same tool must not inherit the error."""
    calls = [_tc("web_search", "c1"), _tc("web_search", "c2")]
    results = _invalid_json_recovery_content(
        calls, [(0, "web_search", "Expecting value: line 1")]
    )

    assert results[0]["role"] == "tool"
    assert results[0]["tool_call_id"] == "c1"
    assert results[0]["content"].startswith("Error: Invalid JSON arguments.")

    assert results[1]["role"] == "tool"
    assert results[1]["tool_call_id"] == "c2"
    assert results[1]["content"] == "Skipped: other tool call in this response had invalid JSON."


def test_invalid_call_gets_error_at_its_own_index():
    """The error follows the invalid call's index, not its tool name."""
    calls = [_tc("web_search", "c1"), _tc("web_search", "c2")]
    results = _invalid_json_recovery_content(calls, [(1, "web_search", "boom")])

    assert results[0]["content"] == "Skipped: other tool call in this response had invalid JSON."
    assert results[1]["tool_call_id"] == "c2"
    assert "boom" in results[1]["content"]


def test_multiple_invalid_indices_are_isolated():
    """Each invalid index gets its own error; the middle call stays skipped."""
    calls = [_tc("alpha", "c1"), _tc("beta", "c2"), _tc("gamma", "c3")]
    results = _invalid_json_recovery_content(
        calls, [(0, "alpha", "err-one"), (2, "gamma", "err-two")]
    )

    assert "err-one" in results[0]["content"]
    assert results[0]["tool_call_id"] == "c1"
    assert results[1]["content"] == "Skipped: other tool call in this response had invalid JSON."
    assert "err-two" in results[2]["content"]
    assert results[2]["tool_call_id"] == "c3"


def test_results_are_well_formed_tool_messages():
    calls = [_tc("tool_a", "c1"), _tc("tool_b", "c2")]
    results = _invalid_json_recovery_content(calls, [(1, "tool_b", "bad json")])

    for result in results:
        assert set(result.keys()) == {"role", "name", "tool_call_id", "content"}
        assert result["role"] == "tool"
    assert results[0]["name"] == "tool_a"
    assert results[1]["name"] == "tool_b"
