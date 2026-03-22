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
    erg_aggregate_system,
    erg_aggregate_user,
    erg_assessor_system,
    erg_assessor_user,
    erg_emotion_agent_system,
    erg_emotion_agent_user,
)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@beartype
def _merge_responses(
    existing: dict[str, str],
    update: dict[str, str],
) -> dict[str, str]:
    """Reducer: merge parallel emotion-agent response proposals."""
    return {**existing, **update}


class ERGState(TypedDict):
    conversation: str
    assumed_emotion: str
    proposed_responses: Annotated[dict[str, str], _merge_responses]
    final_response: str
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
# Nodes
# ---------------------------------------------------------------------------


@beartype
def _emotion_assessor_node(state: ERGState) -> dict[str, Any]:
    """Zero-shot emotion assessment before distributing to agents."""
    llm = build_emotion_agent_llm()
    messages = [
        SystemMessage(content=erg_assessor_system()),
        HumanMessage(content=erg_assessor_user(state["conversation"])),
    ]
    response = llm.invoke(messages)
    parsed = _parse_json(str(response.content))
    return {"assumed_emotion": str(parsed.get("emotion", "neutral"))}


@beartype
def _make_erg_agent_node(emotion: str) -> Any:
    """Return a LangGraph node function for the given Ekman emotion in ERG."""

    @beartype
    def node(state: ERGState) -> dict[str, Any]:
        llm = build_emotion_agent_llm()
        messages = [
            SystemMessage(content=erg_emotion_agent_system(emotion)),
            HumanMessage(
                content=erg_emotion_agent_user(
                    state["assumed_emotion"],
                    state["conversation"],
                )
            ),
        ]
        response = llm.invoke(messages)
        parsed = _parse_json(str(response.content))
        reply = str(parsed.get("response", str(response.content)))
        return {"proposed_responses": {emotion: reply}}

    node.__name__ = f"{emotion}_responder"
    return node


@beartype
def _aggregate_node(state: ERGState) -> dict[str, Any]:
    """Select or synthesise the most empathetic response from the five proposals."""
    llm = build_aggregate_llm()
    messages = [
        SystemMessage(content=erg_aggregate_system()),
        HumanMessage(
            content=erg_aggregate_user(
                state["conversation"],
                state["assumed_emotion"],
                state["proposed_responses"],
            )
        ),
    ]
    response = llm.invoke(messages)
    parsed = _parse_json(str(response.content))
    return {
        "final_response": str(parsed.get("final_response", str(response.content))),
        "reasoning": str(parsed.get("reasoning", "")),
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


@beartype
def build_erg_graph() -> Any:
    """Build and compile the InsideOut ERG LangGraph."""
    builder: StateGraph[ERGState] = StateGraph(ERGState)

    # Emotion assessor (sequential, runs first)
    builder.add_node("emotion_assessor", _emotion_assessor_node)

    # Five parallel emotion-specific responders
    for emotion in EKMAN_EMOTIONS:
        builder.add_node(f"{emotion}_responder", _make_erg_agent_node(emotion))

    # Aggregate node
    builder.add_node("aggregate", _aggregate_node)

    # START → emotion_assessor
    builder.add_edge(START, "emotion_assessor")

    # emotion_assessor → all five responders (parallel fan-out)
    for emotion in EKMAN_EMOTIONS:
        builder.add_edge("emotion_assessor", f"{emotion}_responder")

    # All five responders → aggregate (fan-in)
    for emotion in EKMAN_EMOTIONS:
        builder.add_edge(f"{emotion}_responder", "aggregate")

    # aggregate → END
    builder.add_edge("aggregate", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@beartype
def run_erg(conversation: str) -> dict[str, Any]:
    """Run the ERG graph on a formatted conversation string.

    Returns a dict with keys: assumed_emotion, proposed_responses,
    final_response, reasoning.
    """
    graph = build_erg_graph()
    result: dict[str, Any] = graph.invoke(
        {
            "conversation": conversation,
            "assumed_emotion": "",
            "proposed_responses": {},
            "final_response": "",
            "reasoning": "",
        }
    )
    return result
