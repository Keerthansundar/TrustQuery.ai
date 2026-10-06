# TrustQuery AI - AI Data Analyst Agent

Ask business questions in plain English. Get trustworthy answers backed by executable SQL.
Runs 100% locally (Ollama + SQLite) at zero API cost.

## Status
- [x] Phase 1 - Database + UI
- [x] Phase 2 - SQL agent (tool calling, validation, grounded answers)
- [ ] Phase 3 - RAG (chunking, embeddings, ChromaDB, re-ranking)
- [ ] Phase 4 - Reliability (planner, abstention, ambiguity handling)
- [ ] Phase 5 - Evals  - [ ] Phase 6 - Observability  - [ ] Phase 7 - LoRA experiment

## Quick start
```bash
# 1. Ollama (https://ollama.com), then pull a model
ollama pull qwen2.5:7b-instruct        # or qwen2.5:3b for low RAM (set LLM_MODEL in .env)

# 2. Python env
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# 3. Data + app
python scripts/seed_database.py
streamlit run app.py

# 4. Tests (no Ollama needed)
pytest -v
```

## How a question flows
```
Question -> context -> LLM writes SQL (execute_sql tool call)
         -> Python validates (sqlglot AST, read-only) -> SQLite (read-only, timeout, row cap)
         -> result converted to an AUTHORITATIVE block (types, units, display strings, derived values)
         -> LLM writes the answer as JSON (Pydantic) -> every number verified against the result
         -> UI: answer + the exact executed SQL in the sidebar
```

## Anti-hallucination design (result -> LLM step)
The database is the source of truth; the LLM only writes the words around it.
1. **Structured, typed result.** Columns carry a type and unit (INR / count / percent). Every value is sent as
   raw value + exact `display` string (e.g. `₹2,30,38,986`).
2. **Explicit instructions.** Values are declared authoritative; no rounding, no unit conversion
   (no "lakh/crore/K/M"), no new calculations.
3. **Math lives in Python/SQL.** Totals, shares, differences and % change are computed in Python/SQL and sent as
   DERIVED VALUES, so the model never needs to calculate.
4. **Deterministic verification.** Every number in the LLM's answer must exist in the result block
   (`agent/grounding.py::verify_answer`).
5. **Repair, then fallback.** On a mismatch the model is told which values were wrong and retries once;
   if it still fails, Python writes the answer straight from the result. The UI shows which path was used.
6. **SQL shown is SQL executed.** The model's text never supplies the SQL shown in the UI.

## SQL safety
sqlglot AST validation (single SELECT/CTE only, no DML/DDL/PRAGMA/ATTACH, known tables only) ->
read-only SQLite connection (`mode=ro` + `query_only`) -> timeout -> row cap.
