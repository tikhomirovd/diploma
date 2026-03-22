from __future__ import annotations

"""Unit tests for ERC and ERG graph structure using a mocked LLM."""

import json
from typing import Any
from unittest.mock import MagicMock, patch

from src.erc_graph import build_erc_graph
from src.erg_graph import build_erg_graph
from src.prompts import EKMAN_EMOTIONS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_llm(content: str) -> MagicMock:
    """Return a mock LLM whose .invoke() returns a message with `content`."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = content
    mock_llm.invoke.return_value = mock_response
    return mock_llm


ERC_AGENT_RESPONSE = json.dumps(
    {"label": "sad", "confidence": 0.8, "rationale": "The speaker sounds sad."}
)
ERC_AGGREGATE_RESPONSE = json.dumps(
    {"final_emotion": "sad", "reasoning": "Majority of agents agree."}
)

ERG_ASSESSOR_RESPONSE = json.dumps({"emotion": "sad", "reasoning": "Sounds sad."})
ERG_AGENT_RESPONSE = json.dumps(
    {"response": "I'm sorry to hear that.", "rationale": "Empathetic reply."}
)
ERG_AGGREGATE_RESPONSE = json.dumps(
    {"final_response": "I'm really sorry to hear that.", "reasoning": "Most empathetic."}
)


# ---------------------------------------------------------------------------
# ERC graph tests
# ---------------------------------------------------------------------------


class TestERCGraph:
    @patch("src.erc_graph.build_emotion_agent_llm")
    @patch("src.erc_graph.build_aggregate_llm")
    def test_erc_graph_returns_final_emotion(
        self,
        mock_agg_factory: MagicMock,
        mock_agent_factory: MagicMock,
    ) -> None:
        mock_agent_factory.return_value = _make_mock_llm(ERC_AGENT_RESPONSE)
        mock_agg_factory.return_value = _make_mock_llm(ERC_AGGREGATE_RESPONSE)

        graph = build_erc_graph()
        result: dict[str, Any] = graph.invoke(
            {
                "conversation": "Speaker: I am really upset today.",
                "assessments": {},
                "final_emotion": "",
                "reasoning": "",
            }
        )

        assert result["final_emotion"] == "sad"
        assert "Majority" in result["reasoning"]

    @patch("src.erc_graph.build_emotion_agent_llm")
    @patch("src.erc_graph.build_aggregate_llm")
    def test_erc_graph_collects_all_five_assessments(
        self,
        mock_agg_factory: MagicMock,
        mock_agent_factory: MagicMock,
    ) -> None:
        mock_agent_factory.return_value = _make_mock_llm(ERC_AGENT_RESPONSE)
        mock_agg_factory.return_value = _make_mock_llm(ERC_AGGREGATE_RESPONSE)

        graph = build_erc_graph()
        result: dict[str, Any] = graph.invoke(
            {
                "conversation": "Speaker: Life feels heavy.",
                "assessments": {},
                "final_emotion": "",
                "reasoning": "",
            }
        )

        assert set(result["assessments"].keys()) == set(EKMAN_EMOTIONS)

    @patch("src.erc_graph.build_emotion_agent_llm")
    @patch("src.erc_graph.build_aggregate_llm")
    def test_erc_graph_handles_malformed_json(
        self,
        mock_agg_factory: MagicMock,
        mock_agent_factory: MagicMock,
    ) -> None:
        mock_agent_factory.return_value = _make_mock_llm("not valid json at all")
        mock_agg_factory.return_value = _make_mock_llm(ERC_AGGREGATE_RESPONSE)

        graph = build_erc_graph()
        result: dict[str, Any] = graph.invoke(
            {
                "conversation": "Speaker: Something happened.",
                "assessments": {},
                "final_emotion": "",
                "reasoning": "",
            }
        )

        # Should not raise; assessments contain raw fallback
        assert len(result["assessments"]) == len(EKMAN_EMOTIONS)


# ---------------------------------------------------------------------------
# ERG graph tests
# ---------------------------------------------------------------------------


class TestERGGraph:
    @patch("src.erg_graph.build_emotion_agent_llm")
    @patch("src.erg_graph.build_aggregate_llm")
    def test_erg_graph_returns_final_response(
        self,
        mock_agg_factory: MagicMock,
        mock_agent_factory: MagicMock,
    ) -> None:
        # emotion_assessor and responder nodes both use build_emotion_agent_llm
        mock_agent_factory.return_value = _make_mock_llm(ERG_ASSESSOR_RESPONSE)
        # Make agent responses return agent JSON after first call
        agent_llm = _make_mock_llm(ERG_AGENT_RESPONSE)
        assessor_llm = _make_mock_llm(ERG_ASSESSOR_RESPONSE)

        def side_effect() -> MagicMock:
            # First call from assessor node, rest from agent nodes
            return assessor_llm if side_effect.count == 0 else agent_llm  # type: ignore[attr-defined]

        side_effect.count = 0  # type: ignore[attr-defined]

        mock_agent_factory.return_value = _make_mock_llm(ERG_AGENT_RESPONSE)
        mock_agg_factory.return_value = _make_mock_llm(ERG_AGGREGATE_RESPONSE)

        graph = build_erg_graph()
        result: dict[str, Any] = graph.invoke(
            {
                "conversation": "Speaker: I feel hopeless.",
                "assumed_emotion": "",
                "proposed_responses": {},
                "final_response": "",
                "reasoning": "",
            }
        )

        assert result["final_response"] != ""

    @patch("src.erg_graph.build_emotion_agent_llm")
    @patch("src.erg_graph.build_aggregate_llm")
    def test_erg_graph_collects_five_proposed_responses(
        self,
        mock_agg_factory: MagicMock,
        mock_agent_factory: MagicMock,
    ) -> None:
        mock_agent_factory.return_value = _make_mock_llm(ERG_AGENT_RESPONSE)
        mock_agg_factory.return_value = _make_mock_llm(ERG_AGGREGATE_RESPONSE)

        graph = build_erg_graph()
        result: dict[str, Any] = graph.invoke(
            {
                "conversation": "Speaker: Nobody understands me.",
                "assumed_emotion": "",
                "proposed_responses": {},
                "final_response": "",
                "reasoning": "",
            }
        )

        assert set(result["proposed_responses"].keys()) == set(EKMAN_EMOTIONS)


# ---------------------------------------------------------------------------
# Prompt constant tests
# ---------------------------------------------------------------------------


def test_ekman_emotions_count() -> None:
    assert len(EKMAN_EMOTIONS) == 5


def test_ekman_emotions_set() -> None:
    assert set(EKMAN_EMOTIONS) == {"anger", "disgust", "fear", "happiness", "sadness"}
