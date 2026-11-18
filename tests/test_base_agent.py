import pytest

from core.base_agent import AgentError
from agents.wall_street import WallStreetAgent
from agents.wall_street.schemas import StockRecommendation
from tests.conftest import FakeLLM, response, text_block, tool_use_block


def make_agent(llm, provider, **kwargs):
    return WallStreetAgent(llm=llm, provider=provider, **kwargs)


def test_loop_runs_tool_and_returns_final_text(provider):
    llm = FakeLLM([
        response([tool_use_block("get_quote", {"ticker": "nvda"})], "tool_use"),
        response([text_block("NVDA trades at $150.")], "end_turn"),
    ])
    result = make_agent(llm, provider).ask("Price of NVDA?")

    assert result.text == "NVDA trades at $150."
    assert result.iterations == 2
    assert [c.name for c in result.tool_calls] == ["get_quote"]
    # Second call must carry: user task, assistant tool_use, user tool_result.
    second = llm.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    tool_result = second[2]["content"][0]
    assert tool_result["type"] == "tool_result" and tool_result["tool_use_id"] == "toolu_1"
    assert '"price":150.0' in tool_result["content"]


def test_parallel_tool_calls_return_in_one_message(provider):
    llm = FakeLLM([
        response([tool_use_block("get_quote", {"ticker": "AAPL"}, "a"),
                  tool_use_block("get_news", {"ticker": "AAPL"}, "b")], "tool_use"),
        response([text_block("done")], "end_turn"),
    ])
    make_agent(llm, provider).ask("AAPL?")
    results_msg = llm.calls[1]["messages"][2]
    assert [r["tool_use_id"] for r in results_msg["content"]] == ["a", "b"]


def test_tool_error_is_sent_back_not_raised(provider):
    llm = FakeLLM([
        response([tool_use_block("get_quote", {"ticker": "BAD"})], "tool_use"),
        response([text_block("Ticker not found.")], "end_turn"),
    ])
    result = make_agent(llm, provider).ask("BAD?")
    assert result.tool_calls[0].is_error
    sent = llm.calls[1]["messages"][2]["content"][0]
    assert sent["is_error"] is True and "No company data" in sent["content"]


def test_unknown_tool_is_reported_to_model(provider):
    llm = FakeLLM([
        response([tool_use_block("does_not_exist", {})], "tool_use"),
        response([text_block("ok")], "end_turn"),
    ])
    result = make_agent(llm, provider).ask("x")
    assert result.tool_calls[0].is_error


def test_structured_output_is_parsed(provider):
    payload = ('{"ticker":"AAPL","action":"hold","confidence":"medium",'
               '"time_horizon":"medium (3-12 months)","thesis":"Fair value.","model_signal":"not available",'
               '"key_risks":["China"],"what_would_change_my_mind":"Margin drop"}')
    llm = FakeLLM([response([text_block(payload)], "end_turn")])
    agent = make_agent(llm, provider)
    from core.messages import Task
    result = agent.run(Task(objective="x"), output_model=StockRecommendation)
    assert result.output.action == "hold"
    schema = llm.calls[0]["output_schema"]
    assert schema["additionalProperties"] is False


def test_iteration_cap(provider):
    llm = FakeLLM([response([tool_use_block("get_quote", {"ticker": "A"})], "tool_use")] * 3)
    with pytest.raises(AgentError, match="Gave up"):
        make_agent(llm, provider, max_iterations=3).ask("loop forever")


def test_refusal_raises(provider):
    llm = FakeLLM([response([], "refusal")])
    with pytest.raises(AgentError, match="declined"):
        make_agent(llm, provider).ask("x")


def test_web_search_is_optional(provider):
    names = lambda a: [t.get("name") for t in a.tools.api_definitions()]
    assert "web_search" not in names(make_agent(FakeLLM([]), provider))
    assert "web_search" in names(make_agent(FakeLLM([]), provider, web_search=True))
