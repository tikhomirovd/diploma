from __future__ import annotations

import json

from beartype import beartype

# The five Ekman emotions used by InsideOut agents.
EKMAN_EMOTIONS: list[str] = ["anger", "disgust", "fear", "happiness", "sadness"]

# All 32 EmpatheticDialogues emotion labels, shown to the aggregate ERC agent
# so it always picks from the canonical label set.
EMOTIONS_32_LIST = (
    "afraid, angry, annoyed, anticipating, anxious, apprehensive, ashamed, "
    "caring, confident, content, devastated, disappointed, disgusted, "
    "embarrassed, excited, faithful, furious, grateful, guilty, hopeful, "
    "impressed, jealous, joyful, lonely, nostalgic, prepared, proud, sad, "
    "sentimental, surprised, terrified, trusting"
)


# ---------------------------------------------------------------------------
# ERC prompts
# ---------------------------------------------------------------------------


@beartype
def erc_emotion_agent_system(emotion: str) -> str:
    """System prompt for one Ekman emotion agent in the ERC task."""
    return (
        f"You are an empathetic conversation analyst who perceives the world "
        f"through the lens of {emotion.upper()}. "
        f"Your task is to read a conversation and determine the primary emotion "
        f"the SPEAKER (the person sharing their story) is feeling. "
        f"Because you specialise in {emotion}, you are especially sensitive to "
        f"any signals of {emotion} in what the speaker says.\n\n"
        f"You must respond with a JSON object and nothing else:\n"
        f'{{"label": "<emotion>", "confidence": <0.0-1.0>, "rationale": "<brief explanation>"}}\n\n'
        f"The label must be one of the 32 EmpatheticDialogues emotions: "
        f"{EMOTIONS_32_LIST}."
    )


@beartype
def erc_emotion_agent_user(conversation: str) -> str:
    """User message for one Ekman emotion agent in the ERC task."""
    return (
        f"Conversation:\n{conversation}\n\n"
        f"What is the speaker's primary emotion? "
        f"Respond with a JSON object only."
    )


@beartype
def erc_aggregate_system() -> str:
    """System prompt for the ERC aggregate agent."""
    return (
        "You are an expert emotion analyst. "
        "You have received assessments from five specialised emotional agents "
        "(anger, disgust, fear, happiness, sadness). "
        "Your task is to synthesise their outputs and decide the single most "
        "accurate emotion label for the speaker's primary emotional state.\n\n"
        "You must respond with a JSON object and nothing else:\n"
        '{"final_emotion": "<emotion>", "reasoning": "<brief explanation>"}\n\n'
        f"The label must be one of: {EMOTIONS_32_LIST}."
    )


@beartype
def erc_aggregate_user(assessments: dict[str, dict[str, str | float]]) -> str:
    """User message for the ERC aggregate agent."""
    formatted = json.dumps(assessments, indent=2, ensure_ascii=False)
    return (
        f"Agent assessments:\n{formatted}\n\n"
        f"Synthesise these assessments and choose the final emotion label. "
        f"Respond with a JSON object only."
    )


# ---------------------------------------------------------------------------
# ERG prompts
# ---------------------------------------------------------------------------


@beartype
def erg_assessor_system() -> str:
    """System prompt for the zero-shot emotion assessor in ERG."""
    return (
        "You are an empathetic conversation analyst. "
        "Read the conversation below and identify the primary emotion "
        "the SPEAKER is experiencing right now. "
        "Respond with a JSON object and nothing else:\n"
        '{"emotion": "<emotion>", "reasoning": "<brief explanation>"}\n\n'
        f"The emotion must be one of: {EMOTIONS_32_LIST}."
    )


@beartype
def erg_assessor_user(conversation: str) -> str:
    """User message for the zero-shot emotion assessor in ERG."""
    return (
        f"Conversation:\n{conversation}\n\n"
        f"What is the speaker's current emotional state? "
        f"Respond with a JSON object only."
    )


@beartype
def erg_emotion_agent_system(emotion: str) -> str:
    """System prompt for one Ekman emotion agent in the ERG task."""
    return (
        f"You are an empathetic conversational assistant who approaches every "
        f"interaction through the emotional lens of {emotion.upper()}. "
        f"Your goal is to generate a compassionate, contextually appropriate "
        f"reply that acknowledges the speaker's feelings and gently steers the "
        f"conversation toward positive well-being.\n\n"
        f"Respond with a JSON object and nothing else:\n"
        f'{{"response": "<your empathetic reply>", '
        f'"rationale": "<why this response helps the speaker>"}}'
    )


@beartype
def erg_emotion_agent_user(
    assumed_emotion: str,
    conversation: str,
) -> str:
    """User message for one Ekman emotion agent in the ERG task."""
    return (
        f"The speaker's current emotional state has been identified as: "
        f"{assumed_emotion}.\n\n"
        f"Conversation so far:\n{conversation}\n\n"
        f"Generate an empathetic listener response. "
        f"Respond with a JSON object only."
    )


@beartype
def erg_aggregate_system() -> str:
    """System prompt for the ERG aggregate agent."""
    return (
        "You are an expert in empathetic communication. "
        "Five specialised emotional agents (anger, disgust, fear, happiness, "
        "sadness) have each proposed a listener response to the conversation. "
        "Your task is to select or synthesise the single most empathetic, "
        "contextually appropriate, and helpful response.\n\n"
        "Respond with a JSON object and nothing else:\n"
        '{"final_response": "<the best empathetic reply>", '
        '"reasoning": "<why this response is most effective>"}'
    )


@beartype
def erg_aggregate_user(
    conversation: str,
    speaker_emotion: str,
    proposed: dict[str, str],
) -> str:
    """User message for the ERG aggregate agent."""
    formatted = json.dumps(proposed, indent=2, ensure_ascii=False)
    return (
        f"Speaker's emotional state: {speaker_emotion}\n\n"
        f"Conversation so far:\n{conversation}\n\n"
        f"Proposed responses from emotional agents:\n{formatted}\n\n"
        f"Select or synthesise the most empathetic response. "
        f"Respond with a JSON object only."
    )


# ---------------------------------------------------------------------------
# ERG assessor (quality evaluation by LLM-as-judge)
# ---------------------------------------------------------------------------


@beartype
def llm_judge_system() -> str:
    """System prompt for the LLM-as-judge quality assessor (ERG evaluation)."""
    return (
        "You are an expert evaluator of empathetic dialogue systems. "
        "You will be given a conversation, the speaker's emotional state, and "
        "a generated listener response. Score the response on five dimensions, "
        "each from 1 to 10:\n"
        "- Fluency (F): coherence and grammatical smoothness\n"
        "- Identification (I): how well it addresses the speaker's problem\n"
        "- Empathy (E): how well it understands and validates the speaker's feelings\n"
        "- Suggestion (S): quality and helpfulness of any advice or suggestions\n"
        "- Overall (O): overall effectiveness as emotional support\n\n"
        "Respond with a JSON object and nothing else:\n"
        '{"F": <1-10>, "I": <1-10>, "E": <1-10>, "S": <1-10>, "O": <1-10>}'
    )


@beartype
def llm_judge_user(
    conversation: str,
    speaker_emotion: str,
    response: str,
) -> str:
    """User message for the LLM-as-judge quality assessor."""
    return (
        f"Speaker's emotional state: {speaker_emotion}\n\n"
        f"Conversation:\n{conversation}\n\n"
        f"Generated listener response:\n{response}\n\n"
        f"Score this response. Respond with a JSON object only."
    )
