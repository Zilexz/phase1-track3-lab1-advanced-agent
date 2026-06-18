"""Mock runtime: phản hồi LLM giả lập, hoàn toàn deterministic.

Dùng để hiểu flow và để autograde chạy KHÔNG cần API key (bonus
`mock_mode_for_autograding`). Mỗi hàm trả về (kết quả, LLMUsage) — usage là
con số tổng hợp deterministic, đồng bộ interface với llm_runtime.
"""
from __future__ import annotations

from .schemas import JudgeResult, LLMUsage, QAExample, ReflectionEntry
from .utils import normalize_answer

FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}


def _usage(prompt: int, completion: int, latency: int) -> LLMUsage:
    return LLMUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion, latency_ms=latency)


def actor_answer(
    example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]
) -> tuple[str, LLMUsage]:
    usage = _usage(prompt=260 + 20 * len(reflection_memory), completion=24, latency=120 + 30 * attempt_id)
    if example.qid not in FIRST_ATTEMPT_WRONG:
        return example.gold_answer, usage
    if agent_type == "react":
        return FIRST_ATTEMPT_WRONG[example.qid], usage
    if attempt_id == 1 and not reflection_memory:
        return FIRST_ATTEMPT_WRONG[example.qid], usage
    return example.gold_answer, usage


def evaluator(example: QAExample, answer: str) -> tuple[JudgeResult, LLMUsage]:
    usage = _usage(prompt=140, completion=40, latency=90)
    if normalize_answer(example.gold_answer) == normalize_answer(answer):
        return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization."), usage
    if normalize_answer(answer) == "london":
        return (
            JudgeResult(
                score=0,
                reason="The answer stopped at the birthplace city and never completed the second hop to the river.",
                missing_evidence=["Need to identify the river that flows through London."],
                spurious_claims=[],
            ),
            usage,
        )
    return (
        JudgeResult(
            score=0,
            reason="The final answer selected the wrong second-hop entity.",
            missing_evidence=["Need to ground the answer in the second paragraph."],
            spurious_claims=[answer],
        ),
        usage,
    )


def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> tuple[ReflectionEntry, LLMUsage]:
    usage = _usage(prompt=180, completion=60, latency=110)
    strategy = (
        "Do the second hop explicitly: birthplace city -> river through that city."
        if example.qid == "hp2"
        else "Verify the final entity against the second paragraph before answering."
    )
    entry = ReflectionEntry(
        attempt_id=attempt_id,
        failure_reason=judge.reason,
        lesson="A partial first-hop answer is not enough; the final answer must complete all hops.",
        next_strategy=strategy,
    )
    return entry, usage
