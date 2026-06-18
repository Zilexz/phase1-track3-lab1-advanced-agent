"""Thin OpenAI-compatible LLM client.

Mặc định trỏ tới FPT AI Marketplace (endpoint OpenAI-compatible), nhưng có thể
đổi sang OpenAI/Gemini-proxy/Ollama bằng biến môi trường — không cần sửa code.

Biến môi trường (đọc từ .env):
    LLM_API_KEY      : API key (bắt buộc khi LAB_MODE=llm)
    LLM_BASE_URL     : base URL, mặc định https://mkp-api.fptcloud.com/v1
    LLM_MODEL        : tên model, mặc định GLM-4.7
    LLM_TEMPERATURE  : nhiệt độ sinh, mặc định 0.0
    LLM_MAX_RETRIES  : số lần retry khi lỗi mạng, mặc định 3
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv

from .schemas import LLMUsage

load_dotenv()

DEFAULT_BASE_URL = "https://mkp-api.fptcloud.com/v1"
DEFAULT_MODEL = "GLM-4.7"


@dataclass
class ChatResponse:
    text: str
    usage: LLMUsage


class LLMClient:
    """Bao bọc openai.OpenAI; trả về cả text lẫn token/latency thực đo."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
        self.model = model or os.getenv("LLM_MODEL", DEFAULT_MODEL)
        self.temperature = temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0.0"))
        self.max_retries = max_retries if max_retries is not None else int(os.getenv("LLM_MAX_RETRIES", "3"))
        if not self.api_key:
            raise RuntimeError(
                "Thiếu LLM_API_KEY. Hãy thêm vào file .env (xem .env.example) hoặc chạy ở LAB_MODE=mock."
            )
        # import muộn để LAB_MODE=mock không cần cài openai
        from openai import OpenAI

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(self, system: str, user: str, json_mode: bool = False, model: str | None = None) -> ChatResponse:
        kwargs: dict = {
            "model": model or self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            start = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(**kwargs)
                latency_ms = int((time.perf_counter() - start) * 1000)
                text = (resp.choices[0].message.content or "").strip()
                u = getattr(resp, "usage", None)
                usage = LLMUsage(
                    prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                    total_tokens=getattr(u, "total_tokens", 0) or 0,
                    latency_ms=latency_ms,
                )
                return ChatResponse(text=text, usage=usage)
            except Exception as err:  # noqa: BLE001 - retry mọi lỗi tạm thời
                last_err = err
                # json_mode đôi khi không được model hỗ trợ -> bỏ và thử lại
                if json_mode and "response_format" in kwargs:
                    kwargs.pop("response_format", None)
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM call thất bại sau {self.max_retries} lần: {last_err}")


_CLIENT: LLMClient | None = None


def get_client() -> LLMClient:
    """Singleton client để tái dùng connection trong cả benchmark."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient()
    return _CLIENT
