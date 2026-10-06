from agent.grounding import format_value, ground_result, indian_group, infer_column_meta, verify_answer

COLS = ["month", "revenue", "order_count"]
ROWS = [["2026-08", 23038986, 1000], ["2026-09", 17959929, 800]]


def test_indian_grouping():
    assert indian_group(4200000) == "42,00,000"
    assert indian_group(999) == "999"
    assert indian_group(-1234567) == "-12,34,567"


def test_unit_inference_and_formatting():
    assert infer_column_meta("revenue", [1]).kind == "currency"
    assert infer_column_meta("september_revenue", [1]).kind == "currency"
    assert infer_column_meta("revenue_change_pct", [1.5]).kind == "percent"
    assert infer_column_meta("order_count", [3]).kind == "count"
    assert infer_column_meta("product_id", [3]).kind == "id"
    assert infer_column_meta("product", ["a"]).kind == "text"
    assert format_value(4200000, infer_column_meta("revenue", [1])) == "₹42,00,000"
    assert format_value(-22.05, infer_column_meta("change_pct", [1.0])) == "-22.05%"


def test_block_contains_units_display_and_derived_values():
    g = ground_result(COLS, ROWS)
    assert "unit: INR" in g.text and '"₹2,30,38,986"' in g.text
    assert "COMPLETE result" in g.text
    assert "-22.05% decrease" in g.text            # derived by Python, not the LLM
    assert "MUST NOT alter" in g.text


def test_truncated_result_withholds_totals():
    g = ground_result(COLS, ROWS, db_truncated=True)
    assert "TRUNCATED" in g.text and "total across" not in g.text


def test_verify_accepts_exact_values():
    g = ground_result(COLS, ROWS)
    ok = verify_answer("Revenue fell from ₹2,30,38,986 to ₹1,79,59,929, a 22.05% decrease.", g)
    assert ok.ok, ok.violations


def test_verify_rejects_altered_or_converted_values():
    g = ground_result(COLS, ROWS)
    assert not verify_answer("Revenue was ₹1,79,59,930.", g).ok          # off by one
    assert not verify_answer("Revenue was about 1.8 crore.", g).ok       # unit conversion
    assert not verify_answer("Revenue was ₹180L.", g).ok                 # abbreviation
    assert not verify_answer("Revenue fell by 22%.", g).ok               # rounding
    assert not verify_answer("Revenue was ₹5,00,00,000.", g).ok          # invented


def test_verify_allows_years_and_question_numbers():
    g = ground_result(["product", "revenue"], [["Nova X1", 100]] * 5)
    assert verify_answer("In 2026 the top 5 products were...", g, "Show top 5 products").ok


def test_numbers_inside_product_names_are_allowed():
    g = ground_result(["product", "revenue"], [["UltraBook Pro 14", 8999900]])
    assert verify_answer("UltraBook Pro 14 earned ₹89,99,900.", g).ok
