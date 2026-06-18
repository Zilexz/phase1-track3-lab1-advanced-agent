from __future__ import annotations
from typing import Literal, Optional, TypedDict
from pydantic import BaseModel, Field

class ContextChunk(BaseModel):
    title: str
    text: str

class LLMUsage(BaseModel):
    """Token + latency thực đo được từ một lời gọi LLM (hoặc mock deterministic)."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0

    def __add__(self, other: "LLMUsage") -> "LLMUsage":
        return LLMUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            latency_ms=self.latency_ms + other.latency_ms,
        )

class QAExample(BaseModel):
    qid: str
    difficulty: Literal["easy", "medium", "hard"]
    question: str
    gold_answer: str
    context: list[ContextChunk]

class JudgeResult(BaseModel):
    """Kết quả chấm điểm của Evaluator cho một câu trả lời.

    Đây là structured output (bonus `structured_evaluator`): ngoài điểm 0/1 còn
    nêu rõ bằng chứng còn thiếu và các khẳng định sai để Reflector phân tích.
    """
    score: int = Field(..., ge=0, le=1, description="1 nếu đúng, 0 nếu sai (Exact-Match có chuẩn hoá).")
    reason: str = Field(..., description="Giải thích ngắn gọn vì sao đúng/sai.")
    missing_evidence: list[str] = Field(default_factory=list, description="Bằng chứng/hop còn thiếu để trả lời đầy đủ.")
    spurious_claims: list[str] = Field(default_factory=list, description="Các khẳng định sai hoặc bịa trong câu trả lời.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Độ tự tin của evaluator (0..1).")

class ReflectionEntry(BaseModel):
    """Một mục self-reflection do Reflector tạo ra sau một lần trả lời sai."""
    attempt_id: int = Field(..., description="Lần thử thứ mấy đã sinh ra reflection này.")
    failure_reason: str = Field(..., description="Vì sao lần thử đó sai (lấy từ JudgeResult.reason).")
    lesson: str = Field(..., description="Bài học rút ra, diễn đạt tổng quát.")
    next_strategy: str = Field(..., description="Chiến thuật cụ thể cho lần thử kế tiếp.")

class AttemptTrace(BaseModel):
    attempt_id: int
    answer: str
    score: int
    reason: str
    reflection: Optional[ReflectionEntry] = None
    token_estimate: int = 0
    latency_ms: int = 0

class RunRecord(BaseModel):
    qid: str
    question: str
    gold_answer: str
    agent_type: Literal["react", "reflexion"]
    predicted_answer: str
    is_correct: bool
    attempts: int
    token_estimate: int
    latency_ms: int
    failure_mode: Literal["none", "entity_drift", "incomplete_multi_hop", "wrong_final_answer", "looping", "reflection_overfit"]
    reflections: list[ReflectionEntry] = Field(default_factory=list)
    traces: list[AttemptTrace] = Field(default_factory=list)

class ReportPayload(BaseModel):
    meta: dict
    summary: dict
    failure_modes: dict
    examples: list[dict]
    extensions: list[str]
    discussion: str

class ReflexionState(TypedDict):
    question: str
    context: list[str]
    trajectory: list[str]
    reflection_memory: list[str]
    attempt_count: int
    success: bool
    final_answer: str
