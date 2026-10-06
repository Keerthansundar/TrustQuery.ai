"""Pydantic models: the contracts between LLM, tools and UI."""
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Return value of the execute_sql tool (never raises into the LLM loop)."""
    success: bool
    sql: str = ""                      # exact SQL that was validated/executed
    columns: list[str] = []
    rows: list[list[Any]] = []
    row_count: int = 0
    truncated: bool = False            # True if the DB row cap cut the result
    elapsed_ms: float = 0.0
    error: Optional[str] = None
    error_type: Optional[Literal["validation", "execution", "timeout"]] = None


class LLMAnswer(BaseModel):
    """What the LLM is allowed to produce. SQL is deliberately NOT here:
    the executed SQL always comes from Python, never from the model's text."""
    answer: str = Field(min_length=1)
    reasoning_summary: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    data_used: list[str] = []
    requires_clarification: bool = False


class AnalystResponse(BaseModel):
    """Validated response handed to the UI."""
    answer: str
    sql: str = ""                      # exact executed SQL ("" if nothing ran)
    executed: bool = False
    confidence: float = 0.0
    reasoning_summary: str = ""
    data_used: list[str] = []
    requires_clarification: bool = False

    status: Literal["success", "clarification", "failed", "error"] = "success"
    grounding_status: Literal["verified", "repaired", "fallback", "n/a"] = "n/a"
    error: Optional[str] = None

    columns: list[str] = []
    rows: list[list[Any]] = []
    row_count: int = 0
    truncated: bool = False
    sources: list[str] = []

    sql_attempts: int = 0
    answer_repairs: int = 0
    timings_ms: dict[str, float] = {}
