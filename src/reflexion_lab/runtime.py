"""Dispatcher chọn backend mock hay LLM thật.

Bonus `mock_mode_for_autograding`: đặt biến môi trường LAB_MODE=mock (mặc định)
để chạy không cần API key; LAB_MODE=llm để gọi FPT/GLM-4.7 thật.

agents.py chỉ import từ module này -> đổi backend không phải sửa agent.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

from . import mock_runtime
from .mock_runtime import FAILURE_MODE_BY_QID  # re-export cho agents.py

# Nạp .env sớm để LAB_MODE / LLM_* có hiệu lực trước khi đọc biến môi trường.
load_dotenv()

__all__ = ["get_mode", "actor_answer", "evaluator", "reflector", "FAILURE_MODE_BY_QID"]


def get_mode() -> str:
    return os.getenv("LAB_MODE", "mock").strip().lower()


def _backend():
    if get_mode() == "llm":
        from . import llm_runtime

        return llm_runtime
    return mock_runtime


def actor_answer(example, attempt_id, agent_type, reflection_memory):
    return _backend().actor_answer(example, attempt_id, agent_type, reflection_memory)


def evaluator(example, answer):
    return _backend().evaluator(example, answer)


def reflector(example, attempt_id, judge):
    return _backend().reflector(example, attempt_id, judge)
