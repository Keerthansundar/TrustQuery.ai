from agent.tools import execute_sql
from config import get_settings


def test_execute_sql_success():
    r = execute_sql("SELECT COUNT(*) AS order_count FROM orders")
    assert r.success and r.columns == ["order_count"] and r.rows[0][0] > 0


def test_execute_sql_returns_exact_executed_sql():
    r = execute_sql("  SELECT 1 AS x;  ")
    assert r.success and r.sql == "SELECT 1 AS x"


def test_execute_sql_blocks_writes_before_execution():
    r = execute_sql("DELETE FROM orders")
    assert not r.success and r.error_type == "validation"
    assert execute_sql("SELECT COUNT(*) FROM orders").rows[0][0] > 0


def test_execute_sql_reports_sqlite_errors_for_retry():
    r = execute_sql("SELECT nonexistent_col FROM orders")
    assert not r.success and r.error_type == "execution"


def test_row_cap_sets_truncated(monkeypatch):
    monkeypatch.setenv("MAX_ROWS", "10")
    get_settings.cache_clear()
    r = execute_sql("SELECT order_id FROM orders")
    assert r.success and r.row_count == 10 and r.truncated


def test_timeout(monkeypatch):
    monkeypatch.setenv("QUERY_TIMEOUT_S", "0.3")
    get_settings.cache_clear()
    r = execute_sql("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT COUNT(*) FROM c")
    assert not r.success and r.error_type == "timeout"
