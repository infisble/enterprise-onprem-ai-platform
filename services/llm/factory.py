from __future__ import annotations

from functools import lru_cache

from packages.core.config import get_settings
from services.llm.base import LLMProvider
from services.llm.mock import build_mock
from services.llm.openai_compat import OpenAIProvider, VLLMProvider


@lru_cache
def get_llm() -> LLMProvider:
    s = get_settings()
    if s.llm_provider == "vllm":
        return VLLMProvider(s.vllm_base_url, s.vllm_model, s.vllm_api_key, s.llm_timeout_s)
    if s.llm_provider == "openai":
        if not s.openai_api_key:
            raise RuntimeError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
        return OpenAIProvider(s.openai_base_url, s.openai_model, s.openai_api_key, s.llm_timeout_s)
    return build_mock()
