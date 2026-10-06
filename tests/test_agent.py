"""Agent tests with a scripted fake LLM (no Ollama needed)."""
import copy
from types import SimpleNamespace as NS

from agent.agent import DataAnalystAgent
from models.response import AnalystResponse

Q = "What was our total completed revenue?"
SQL = "SELECT SUM(revenue) AS revenue FROM orders WHERE status = 'completed'"


def tool_call(sql):
    call = NS(function=NS(name="execute_sql", arguments={"sql": sql}))
    return NS(message=NS(content="", tool_calls=[call]))


def text(content):
    return NS(message=NS(content=content, tool_calls=None))


def answer_json(answer, conf=0.9):
    return text(f'{{"answer": {answer!r}, "reasoning_summary": "x", "confidence": {conf}, '
                f'"data_used": ["orders"], "requires_clarification": false}}'.replace("'", '"'))


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def chat(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        return self.replies.pop(0)


def true_revenue():
    from agent.tools import execute_sql
    from agent.grounding import format_value, infer_column_meta
    r = execute_sql(SQL)
    return format_value(r.rows[0][0], infer_column_meta("revenue", r.rows[0]))


def test_happy_path_is_verified_and_returns_executed_sql():
    rev = true_revenue()
    agent = DataAnalystAgent(client=FakeClient([tool_call(SQL), answer_json(f"Total revenue is {rev}.")]))
    r = agent.run(Q)
    assert isinstance(r, AnalystResponse)
    assert r.status == "success" and r.grounding_status == "verified"
    assert r.sql == SQL and r.executed and rev in r.answer


def test_tool_message_sent_to_llm_is_the_authoritative_block():
    rev = true_revenue()
    fake = FakeClient([tool_call(SQL), answer_json(f"Revenue: {rev}")])
    DataAnalystAgent(client=fake).run(Q)
    final_call_msgs = fake.calls[-1]["messages"]
    tool_msg = next(m for m in final_call_msgs if m["role"] == "tool")
    assert "AUTHORITATIVE QUERY RESULT" in tool_msg["content"] and "unit: INR" in tool_msg["content"]
    assert "Do not round" in final_call_msgs[-1]["content"]


def test_hallucinated_number_is_repaired():
    rev = true_revenue()
    fake = FakeClient([tool_call(SQL), answer_json("Revenue is about ₹19 crore."),
                       answer_json(f"Revenue is {rev}.")])
    r = DataAnalystAgent(client=fake).run(Q)
    assert r.grounding_status == "repaired" and r.answer == f"Revenue is {rev}." and r.answer_repairs == 1


def test_persistent_hallucination_falls_back_to_deterministic_answer():
    rev = true_revenue()
    fake = FakeClient([tool_call(SQL), answer_json("Revenue is ₹1,23,456."), answer_json("It is ₹9,99,999.")])
    r = DataAnalystAgent(client=fake).run(Q)
    assert r.grounding_status == "fallback" and rev in r.answer and "1,23,456" not in r.answer


def test_invalid_json_is_repaired():
    rev = true_revenue()
    fake = FakeClient([tool_call(SQL), text("not json at all"), answer_json(f"Revenue is {rev}.")])
    assert DataAnalystAgent(client=fake).run(Q).grounding_status == "repaired"


def test_sql_error_is_retried_once():
    rev = true_revenue()
    fake = FakeClient([tool_call("SELECT SUM(nope) FROM orders"), tool_call(SQL),
                       answer_json(f"Revenue is {rev}.")])
    r = DataAnalystAgent(client=fake).run(Q)
    assert r.status == "success" and r.sql_attempts == 2 and r.sql == SQL
    assert any(m["role"] == "tool" and "ERROR" in m["content"] for m in fake.calls[1]["messages"])


def test_dangerous_sql_is_blocked_and_never_shown_as_executed():
    fake = FakeClient([tool_call("DROP TABLE orders"), tool_call("DELETE FROM orders")])
    r = DataAnalystAgent(client=fake).run("delete everything")
    assert r.status == "failed" and not r.executed and r.sql == ""


def test_no_tool_call_means_clarification():
    fake = FakeClient([text("Which month do you want revenue for?")])
    r = DataAnalystAgent(client=fake).run("Show me revenue.")
    assert r.status == "clarification" and r.requires_clarification and not r.executed


def test_llm_outage_is_reported_not_raised():
    class Down:
        def chat(self, **kw):
            raise ConnectionError("connection refused")
    r = DataAnalystAgent(client=Down()).run(Q)
    assert r.status == "error"
