from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .runtime import FAILURE_MODE_BY_QID, actor_answer, evaluator, reflector
from .schemas import AttemptTrace, JudgeResult, LLMUsage, QAExample, ReflectionEntry, RunRecord


def compress_memory(memory: list[str], max_entries: int = 3) -> list[str]:
    """Bonus `memory_compression`: gộp/cắt reflection memory để prompt không phình.

    - Khử trùng lặp (giữ thứ tự).
    - Nếu vượt max_entries, giữ entry đầu (bài học gốc) + (max_entries-1) entry mới nhất.
    """
    seen: set[str] = set()
    deduped: list[str] = []
    for note in memory:
        key = note.strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(note.strip())
    if len(deduped) <= max_entries:
        return deduped
    return [deduped[0]] + deduped[-(max_entries - 1):]


def classify_failure(
    example: QAExample, judge: JudgeResult, traces: list[AttemptTrace], agent_type: str
) -> str:
    """Suy ra failure_mode từ kết quả chấm + lịch sử các lần thử."""
    if judge.score == 1:
        return "none"
    # Mock giữ nhãn cố định để report mock khớp kỳ vọng gốc.
    if example.qid in FAILURE_MODE_BY_QID:
        return FAILURE_MODE_BY_QID[example.qid]
    answers = [t.answer.strip().lower() for t in traces]
    # Reflexion đã thử nhiều lần mà lặp lại cùng câu trả lời sai -> looping.
    if agent_type == "reflexion" and len(answers) >= 2 and len(set(answers)) == 1:
        return "looping"
    # Từng đúng rồi đổi sang sai sau khi reflect -> reflection_overfit.
    if agent_type == "reflexion" and any(t.score == 1 for t in traces[:-1]):
        return "reflection_overfit"
    if judge.spurious_claims:
        return "entity_drift"
    if judge.missing_evidence:
        return "incomplete_multi_hop"
    return "wrong_final_answer"


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    adaptive: bool = False  # bonus `adaptive_max_attempts`
    memory_limit: int = 3   # bonus `memory_compression`

    def _effective_max_attempts(self, example: QAExample) -> int:
        """Bonus `adaptive_max_attempts`: cấp ngân sách lần thử theo độ khó."""
        if self.agent_type == "react" or not self.adaptive:
            return self.max_attempts
        if example.difficulty == "easy":
            return min(self.max_attempts, 2)
        if example.difficulty == "hard":
            return self.max_attempts + 1
        return self.max_attempts

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_judge = JudgeResult(score=0, reason="No attempt produced.")
        max_attempts = self._effective_max_attempts(example)

        for attempt_id in range(1, max_attempts + 1):
            answer, actor_usage = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            judge, eval_usage = evaluator(example, answer)
            attempt_usage: LLMUsage = actor_usage + eval_usage

            final_answer = answer
            final_judge = judge

            reflection_entry: ReflectionEntry | None = None
            if judge.score == 0 and self.agent_type == "reflexion" and attempt_id < max_attempts:
                # --- Reflexion loop: phân tích lỗi -> cập nhật bộ nhớ cho lần sau ---
                reflection_entry, refl_usage = reflector(example, attempt_id, judge)
                attempt_usage += refl_usage
                reflections.append(reflection_entry)
                reflection_memory.append(reflection_entry.next_strategy)
                reflection_memory = compress_memory(reflection_memory, self.memory_limit)

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                reflection=reflection_entry,
                token_estimate=attempt_usage.total_tokens,
                latency_ms=attempt_usage.latency_ms,
            )
            traces.append(trace)

            if judge.score == 1:
                break

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = classify_failure(example, final_judge, traces, self.agent_type)
        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_judge.score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )


class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3, adaptive: bool = False, memory_limit: int = 3) -> None:
        super().__init__(
            agent_type="reflexion",
            max_attempts=max_attempts,
            adaptive=adaptive,
            memory_limit=memory_limit,
        )
