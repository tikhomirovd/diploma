from __future__ import annotations

"""Single-LLM baseline for ERC and ERG (no multi-agent layer).

Used via --mode baseline in run_erc.py / run_erg.py to replicate the
"Baseline" column from the InsideOut paper's comparison tables.
"""

import json
import re
from typing import Any

from beartype import beartype
from langchain_core.messages import HumanMessage, SystemMessage

from src.llm import build_emotion_agent_llm
from src.prompts import (
    baseline_erc_system,
    baseline_erc_user,
    baseline_erg_system,
    baseline_erg_user,
)


@beartype
def _parse_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        result: Any = json.loads(cleaned)
        return dict(result)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            result2: Any = json.loads(match.group())
            return dict(result2)
        return {"raw": text}


@beartype
def run_erc_baseline(conversation: str) -> dict[str, Any]:
    """Single LLM call for ERC — no multi-agent graph.

    Returns a dict with the same keys as the InsideOut ERC graph:
    final_emotion, reasoning.
    """
    llm = build_emotion_agent_llm()
    messages = [
        SystemMessage(content=baseline_erc_system()),  # type: ignore[call-arg]
        HumanMessage(content=baseline_erc_user(conversation)),  # type: ignore[call-arg]
    ]
    response = llm.invoke(messages)
    parsed = _parse_json(str(response.content))
    return {
        "final_emotion": str(parsed.get("final_emotion", "")).strip().lower(),
        "reasoning": str(parsed.get("reasoning", "")),
        "assessments": {},
    }


@beartype
def run_erg_baseline(conversation: str) -> dict[str, Any]:
    """Single LLM call for ERG — no multi-agent graph.

    Returns a dict with the same keys as the InsideOut ERG graph:
    final_response, reasoning, assumed_emotion, proposed_responses.
    """
    llm = build_emotion_agent_llm()
    messages = [
        SystemMessage(content=baseline_erg_system()),  # type: ignore[call-arg]
        HumanMessage(content=baseline_erg_user(conversation)),  # type: ignore[call-arg]
    ]
    response = llm.invoke(messages)
    parsed = _parse_json(str(response.content))
    return {
        "final_response": str(parsed.get("final_response", str(response.content))).strip(),
        "reasoning": str(parsed.get("reasoning", "")),
        "assumed_emotion": "",
        "proposed_responses": {},
    }
