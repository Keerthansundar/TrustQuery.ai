"""The Data Analyst Agent (Phase 4: reliability).

Pipeline
  pre-flight (refuse writes / ask for clarification)      no LLM call
  context (RAG) -> abstain if clearly out of scope        no LLM call
  plan (intent + resolved date ranges + hints)            no LLM call
  LLM writes SQL via tool call
      - text-only replies with invented numbers are discarded and the model is nudged
      - SQL written as text is recovered and sent through the SAME validator
      - business-rule lint errors go back to the model once; if still wrong, run with a visible warning
  Python validates + executes (read-only) -> AUTHORITATIVE block -> LLM writes the answer (JSON)
  -> every number verified against the result -> repair once -> deterministic fallback.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

import ollama
from pydantic import ValidationError

from agent.context import build_context
from agent.grounding import ground_result, verify_answer
from agent.planner import OUT_OF_SCOPE, is_out_of_scope, make_plan, preflight
from agent.prompts import (INTERPRET_INSTRUCTIONS, LINT_FEEDBACK, NO_TOOL_NUDGE, REPAIR_INSTRUCTIONS,
                           SYSTEM_PROMPT)
from agent.tools import EXECUTE_SQL_TOOL, execute_sql, extract_sql_from_text, looks_like_made_up_results
from config import Settings, get_settings
from database.schema import get_data_window
from database.sql_lint import LintContext, lint_sql
from models.response import AnalystResponse, LLMAnswer, ToolResult

log = logging.getLogger(__name__)


class DataAnalystAgent:
    def __init__(self, client: Any = None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.client = client or ollama.Client(host=self.settings.ollama_host)

    # ------------------------------------------------------------------ public
    def run(self, question: str, followup_context: str = "") -> AnalystResponse:
        t0 = time.perf_counter()
        timings = {"context_ms": 0.0, "llm_sql_ms": 0.0, "sql_exec_ms": 0.0, "llm_answer_ms": 0.0}
        try:
            resp = self._run(question, followup_context, timings)
        except Exception as exc:  # Ollama down, model missing, DB missing, ...
            log.exception("Agent failed")
            resp = AnalystResponse(answer=f"Something went wrong while answering: {exc}",
                                   status="error", error=str(exc))
        timings["total_ms"] = (time.perf_counter() - t0) * 1000
        resp.timings_ms = {k: round(v, 1) for k, v in timings.items()}
        return resp

    # ------------------------------------------------------------------ internals
    def _chat(self, messages: list[dict], timing_key: str, timings: dict, **kwargs):
        start = time.perf_counter()
        out = self.client.chat(
            model=self.settings.llm_model, messages=messages, keep_alive=self.settings.llm_keep_alive,
            options={"temperature": self.settings.llm_temperature, "num_ctx": self.settings.llm_num_ctx},
            **kwargs)
        timings[timing_key] += (time.perf_counter() - start) * 1000
        return out

    def _run(self, question: str, followup_context: str, timings: dict) -> AnalystResponse:
        s = self.settings
        has_history = bool(followup_context)
        window = get_data_window()

        # ---- 0. pre-flight: no model needed
        pre = preflight(question, has_history, window)
        if pre:
            return AnalystResponse(answer=pre.message, status=pre.status,  # type: ignore[arg-type]
                                   requires_clarification=pre.status == "clarification")

        # ---- 1. context (RAG) + abstention
        t_ctx = time.perf_counter()
        ctx = build_context(question)
        timings["context_ms"] = (time.perf_counter() - t_ctx) * 1000
        if is_out_of_scope(question, ctx.low_confidence):
            return AnalystResponse(answer=OUT_OF_SCOPE, status="abstained", sources=ctx.sources)

        # ---- 2. plan
        plan = make_plan(question, window, has_history)
        system = SYSTEM_PROMPT.replace("{data_window}", ctx.data_window).replace("{context}", ctx.text)
        user_msg = "\n\n".join(p for p in (followup_context, plan.render(), f"CURRENT QUESTION: {question}") if p)
        messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
        lint_ctx = LintContext(window[0] if window else None, window[1] if window else None,
                               check_period=not plan.period_named and not has_history)

        # ---- 3. LLM -> execute_sql tool call
        result: Optional[ToolResult] = None
        warnings: list[str] = []
        attempts = llm_calls = 0           # attempts = SQL executions; llm_calls also counts nudges
        rescued = False
        while attempts < s.max_sql_attempts and llm_calls < s.max_sql_attempts + 1:
            llm_calls += 1
            reply = self._chat(messages, "llm_sql_ms", timings, tools=[EXECUTE_SQL_TOOL]).message
            calls = reply.tool_calls or []
            if calls:
                name, args = calls[0].function.name, dict(calls[0].function.arguments or {})
                content = reply.content or ""
            else:
                text_reply = (reply.content or "").strip()
                rescued_sql = extract_sql_from_text(text_reply)
                if rescued_sql is None:
                    if text_reply and not looks_like_made_up_results(text_reply):
                        return AnalystResponse(answer=text_reply, status="clarification",
                                               requires_clarification=True, sources=ctx.sources,
                                               plan=plan.summary(), sql_attempts=attempts,
                                               tool_call_rescued=rescued)
                    # Answered from memory (invented numbers) or said nothing: never show it.
                    log.warning("Text-only reply with result-like content discarded; forcing a tool call.")
                    messages.append({"role": "user", "content": NO_TOOL_NUDGE})
                    continue
                log.warning("Model wrote the tool call as text; recovered the SQL.")
                name, args, content, rescued = "execute_sql", {"sql": rescued_sql}, "", True
            attempts += 1
            messages.append({"role": "assistant", "content": content,
                             "tool_calls": [{"function": {"name": name, "arguments": args}}]})

            sql_text = str(args.get("sql", ""))
            issues = lint_sql(sql_text, lint_ctx) if name == "execute_sql" else []
            if name != "execute_sql":
                result = ToolResult(success=False, error=f"Unknown tool '{name}'", error_type="validation")
            elif issues and attempts < s.max_sql_attempts:
                msg = " ".join(i.message for i in issues)
                result = ToolResult(success=False, sql=sql_text, error=msg, error_type="lint")
            else:                          # clean SQL, or last attempt: run it and disclose any remaining issues
                result = execute_sql(sql_text)
                timings["sql_exec_ms"] += result.elapsed_ms
                warnings = [i.message for i in issues] if result.success else []
            if result.success:
                break
            feedback = (LINT_FEEDBACK.format(issues=result.error) if result.error_type == "lint" else
                        f"ERROR ({result.error_type}): {result.error}\nFix the problem and call execute_sql "
                        "again with ONE corrected read-only SELECT query. Do not invent columns.")
            messages.append({"role": "tool", "tool_name": name, "content": feedback})

        if result is None or not result.success:
            return AnalystResponse(
                answer="I couldn't produce a valid, safe query for that question. Try rephrasing it "
                       "or being more specific about the metric and time period.",
                status="failed", error=result.error if result else "no tool call",
                sources=ctx.sources, plan=plan.summary(), sql_attempts=attempts)

        # ---- 4. result -> AUTHORITATIVE block -> LLM interpretation -> verification
        grounded = ground_result(result.columns, result.rows, db_truncated=result.truncated,
                                 max_rows_for_llm=s.llm_max_rows)
        messages.append({"role": "tool", "tool_name": "execute_sql", "content": grounded.text})
        messages.append({"role": "user", "content": INTERPRET_INSTRUCTIONS})
        llm_answer, grounding_status, repairs = self._interpret(messages, grounded, question, timings)

        return AnalystResponse(
            answer=llm_answer.answer, sql=result.sql, executed=True,   # SQL always comes from Python
            confidence=min(llm_answer.confidence, 0.5) if warnings else llm_answer.confidence,
            reasoning_summary=llm_answer.reasoning_summary, data_used=llm_answer.data_used,
            requires_clarification=False, status="success", grounding_status=grounding_status,
            columns=result.columns, rows=result.rows, row_count=result.row_count,
            truncated=result.truncated, sources=ctx.sources, plan=plan.summary(), warnings=warnings,
            sql_attempts=attempts, answer_repairs=repairs, tool_call_rescued=rescued)

    def _interpret(self, messages, grounded, question, timings):
        schema = LLMAnswer.model_json_schema()
        max_tries = 1 + self.settings.max_answer_repairs
        for attempt in range(max_tries):
            raw = self._chat(messages, "llm_answer_ms", timings, format=schema).message.content or ""
            try:
                parsed = LLMAnswer.model_validate_json(raw)
            except ValidationError as exc:
                feedback = f"Your output was not valid JSON for the required schema: {exc.errors()[:2]}. Return valid JSON."
            else:
                check = verify_answer(parsed.answer, grounded, question)
                if check.ok:
                    return parsed, ("verified" if attempt == 0 else "repaired"), attempt
                log.warning("Answer failed numeric verification: %s", check.violations)
                feedback = REPAIR_INSTRUCTIONS.format(violations="\n".join(f"- {v}" for v in check.violations))
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": feedback})

        # Could not get a verified answer -> deterministic answer built by Python from the result.
        fallback = LLMAnswer(
            answer=grounded.fallback_answer, confidence=0.6, data_used=[],
            reasoning_summary="Wording generated directly from the query result because the model's "
                              "answer could not be verified against it.")
        return fallback, "fallback", max_tries - 1