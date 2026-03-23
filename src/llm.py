from __future__ import annotations

import os

from beartype import beartype
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@beartype
def build_llm(model_env_var: str, temperature: float = 0.0) -> BaseChatModel:
    """Build a ChatOpenAI model pointed at OpenRouter.

    The model name is read from the given environment variable.
    Falls back to the generic MODEL env var, then to openai/gpt-4o.
    """
    api_key = os.environ["OPENROUTER_API_KEY"]
    model: str = os.getenv(model_env_var) or os.getenv("MODEL") or "openai/gpt-4o"
    return ChatOpenAI(  # type: ignore[call-arg]
        model=model,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base=OPENROUTER_BASE_URL,
        max_retries=12,
        request_timeout=180,
    )


@beartype
def build_emotion_agent_llm() -> BaseChatModel:
    """LLM used by each of the 5 Ekman emotion agents."""
    return build_llm("EMOTION_AGENT_MODEL")


@beartype
def build_aggregate_llm() -> BaseChatModel:
    """LLM used by the aggregate agent that synthesises emotion outputs."""
    return build_llm("AGGREGATE_MODEL")


@beartype
def build_assessor_llm() -> BaseChatModel:
    """LLM used as the GPT-4-style quality assessor for ERG evaluation."""
    return build_llm("ASSESSOR_MODEL")
