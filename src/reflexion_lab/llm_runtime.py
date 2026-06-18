"""LLM-backed runtime: thay thế mock bằng lời gọi LLM thật.

Mỗi hàm trả về (kết quả, LLMUsage) để agents.py cộng dồn token/latency THỰC.
Hàm parse JSON chịu lỗi (model đôi khi bọc ```json ... ```), và có fallback an
toàn để một câu hỏi hỏng không làm sập cả benchmark.
"""
from __future__ import annotations

import json
import os
import re

from .llm_client import get_client
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, LLMUsage, QAExample, ReflectionEntry
from .utils import normalize_answer

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_ANSWER_RE = re.compile(r"final answer\s*:\s*(.+)", re.IGNORECASE)


def _role_model(role_env: str) -> str | None:
    """Model riêng cho từng vai trò; rỗng -> dùng LLM_MODEL mặc định.

    Cho phép cấu hình "critic mạnh hơn actor": ví dụ actor yếu (gpt-oss-20b)
    + evaluator/reflector mạnh (GLM-4.7) -> lỗi đa dạng & reflection đáng tin.
    """
    return os.getenv(role_env) or None


def _extract_json(text: str) -> dict:
    """Tách object JSON đầu tiên trong text (bỏ code-fence / lời dẫn thừa)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_RE.search(text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _format_context(example: QAExample) -> str:
    return "\n".join(f"[{c.title}] {c.text}" for c in example.context)


def actor_answer(
    example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]
) -> tuple[str, LLMUsage]:
    context = _format_context(example)
    parts = [f"QUESTION:\n{example.question}", f"\nCONTEXT:\n{context}"]
    if reflection_memory:
        notes = "\n".join(f"- {m}" for m in reflection_memory)
        parts.append(f"\nREFLECTION NOTES (from your previous wrong attempts):\n{notes}")
    user = "\n".join(parts)

    resp = get_client().chat(ACTOR_SYSTEM, user, model=_role_model("LLM_ACTOR_MODEL"))
    match = _ANSWER_RE.search(resp.text)
    answer = match.group(1).strip() if match else resp.text.strip().splitlines()[-1].strip()
    answer = answer.strip().strip('"').strip()
    return answer, resp.usage


def evaluator(example: QAExample, answer: str) -> tuple[JudgeResult, LLMUsage]:
    user = (
        f"QUESTION:\n{example.question}\n\n"
        f"GOLD ANSWER:\n{example.gold_answer}\n\n"
        f"PREDICTED ANSWER:\n{answer}"
    )
    resp = get_client().chat(EVALUATOR_SYSTEM, user, json_mode=True, model=_role_model("LLM_EVAL_MODEL"))
    data = _extract_json(resp.text)

    # Fallback: nếu LLM không trả JSON hợp lệ, dùng so khớp chuẩn hoá.
    if "score" not in data:
        correct = normalize_answer(example.gold_answer) == normalize_answer(answer)
        judge = JudgeResult(
            score=1 if correct else 0,
            reason="Fallback exact-match (evaluator JSON không parse được).",
            confidence=0.5,
        )
        return judge, resp.usage

    try:
        score = int(data.get("score", 0))
    except (TypeError, ValueError):
        score = 0
    judge = JudgeResult(
        score=1 if score == 1 else 0,
        reason=str(data.get("reason", "")) or "(no reason)",
        missing_evidence=[str(x) for x in data.get("missing_evidence", []) or []],
        spurious_claims=[str(x) for x in data.get("spurious_claims", []) or []],
        confidence=float(data.get("confidence", 1.0) or 1.0),
    )
    return judge, resp.usage


def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> tuple[ReflectionEntry, LLMUsage]:
    user = (
        f"QUESTION:\n{example.question}\n\n"
        f"WRONG ANSWER:\n(the answer that was just graded as incorrect)\n\n"
        f"GRADER REASON:\n{judge.reason}\n"
        f"MISSING EVIDENCE: {judge.missing_evidence}\n"
        f"SPURIOUS CLAIMS: {judge.spurious_claims}"
    )
    resp = get_client().chat(REFLECTOR_SYSTEM, user, json_mode=True, model=_role_model("LLM_REFLECT_MODEL"))
    data = _extract_json(resp.text)
    entry = ReflectionEntry(
        attempt_id=attempt_id,
        failure_reason=str(data.get("failure_reason") or judge.reason),
        lesson=str(data.get("lesson") or "Complete every reasoning hop before answering."),
        next_strategy=str(
            data.get("next_strategy")
            or "Re-read the context, resolve each intermediate entity, then answer the final hop."
        ),
    )
    return entry, resp.usage
