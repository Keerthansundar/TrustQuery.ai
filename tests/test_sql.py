import pytest

from database.db import get_connection
from database.sql_validator import validate_sql

TABLES = ["customers", "orders", "products"]


@pytest.mark.parametrize("sql", [
    "SELECT * FROM orders",
    "SELECT SUM(revenue) AS revenue FROM orders WHERE status = 'completed';",
    "WITH m AS (SELECT product_id, SUM(revenue) r FROM orders GROUP BY 1) SELECT * FROM m ORDER BY r DESC",
    "SELECT p.name, COUNT(*) FROM orders o JOIN products p ON p.product_id = o.product_id GROUP BY p.name",
    "SELECT 1 UNION SELECT 2",
])
def test_valid_queries_pass(sql):
    assert validate_sql(sql, TABLES).is_valid


@pytest.mark.parametrize("sql", [
    "INSERT INTO orders VALUES (1,1,1,'2026-01-01',1,1,'completed')",
    "UPDATE orders SET revenue = 0",
    "DELETE FROM orders",
    "DROP TABLE orders",
    "ALTER TABLE orders ADD COLUMN x INT",
    "CREATE TABLE t (a INT)",
    "PRAGMA table_info(orders)",
    "ATTACH DATABASE 'x.db' AS x",
    "SELECT * FROM orders; DROP TABLE orders",
    "SELECT * FROM sqlite_master",
    "SELECT * FROM secret_table",
    "SELECT load_extension('x')",
    "",
    "SELEC oops FROM",
])
def test_dangerous_or_invalid_queries_rejected(sql):
    assert not validate_sql(sql, TABLES).is_valid


def test_cte_name_is_not_treated_as_unknown_table():
    res = validate_sql("WITH x AS (SELECT 1 AS a) SELECT a FROM x", TABLES)
    assert res.is_valid and res.tables == []


def test_connection_is_physically_read_only():
    conn = get_connection()
    with pytest.raises(Exception):
        conn.execute("DELETE FROM orders")
