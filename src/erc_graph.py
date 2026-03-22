from __future__ import annotations

import json
import re
from typing import Annotated, Any

from beartype import beartype
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from src.llm import build_aggregate_llm, build_emotion_agent_llm
from src.prompts import (
    EKMAN_EMOTIONS,
    erc_aggregate_system,
    erc_aggregate_user,
    erc_emotion_agent_system,
    erc_emotion_agent_user,
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@beartype
def _merge_assessments(
    existing: dict[str, dict[str, Any]],
    update: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Reducer: merge parallel emotion-agent outputs into a single dict."""
    return {**existing, **update}


class ERCState(TypedDict):
    conversation: str
    assessments: Annotated[dict[str, dict[str, Any]], _merge_assessments]
    final_emotion: str
    reasoning: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Node factories
# ---------------------------------------------------------------------------


@beartype
def _make_emotion_agent_node(
    emotion: str,
) -> Any:
    """Return a LangGraph node function for the given Ekman emotion."""

    @beartype
    def node(state: ERCState) -> dict[str, Any]:
        llm = build_emotion_agent_llm()
        messages = [
            SystemMessage(content=erc_emotion_agent_system(emotion)),
            HumanMessage(content=erc_emotion_agent_user(state["conversation"])),
        ]
        response = llm.invoke(messages)
        parsed = _parse_json(str(response.content))
        return {"assessments": {emotion: parsed}}

    node.__name__ = f"{emotion}_agent"
    return node


@beartype
def _aggregate_node(state: ERCState) -> dict[str, Any]:
    """Aggregate all five emotion-agent assessments into a final label."""
    llm = build_aggregate_llm()
    assessments: dict[str, dict[str, str | float]] = {
        k: {ik: iv for ik, iv in v.items()} for k, v in state["assessments"].items()
    }
    messages = [
        SystemMessage(content=erc_aggregate_system()),
        HumanMessage(content=erc_aggregate_user(assessments)),
    ]
    response = llm.invoke(messages)
    parsed = _parse_json(str(response.content))
    return {
        "final_emotion": str(parsed.get("final_emotion", "")),
        "reasoning": str(parsed.get("reasoning", "")),
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


@beartype
def build_erc_graph() -> Any:
    """Build and compile the InsideOut ERC LangGraph."""
    builder: StateGraph[ERCState] = StateGraph(ERCState)

    # Add one node per Ekman emotion
    for emotion in EKMAN_EMOTIONS:
        builder.add_node(f"{emotion}_agent", _make_emotion_agent_node(emotion))

    # Add the aggregate node
    builder.add_node("aggregate", _aggregate_node)

    # Fan-out: START → all five emotion agents (parallel)
    for emotion in EKMAN_EMOTIONS:
        builder.add_edge(START, f"{emotion}_agent")

    # Fan-in: all five emotion agents → aggregate
    for emotion in EKMAN_EMOTIONS:
        builder.add_edge(f"{emotion}_agent", "aggregate")

    # aggregate → END
    builder.add_edge("aggregate", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@beartype
def run_erc(conversation: str) -> dict[str, Any]:
    """Run the ERC graph on a formatted conversation string.

    Returns a dict with keys: assessments, final_emotion, reasoning.
    """
    graph = build_erc_graph()
    result: dict[str, Any] = graph.invoke(
        {
            "conversation": conversation,
            "assessments": {},
            "final_emotion": "",
            "reasoning": "",
        }
    )
    return result
