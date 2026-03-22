from __future__ import annotations

from src.data import (
    Conversation,
    Utterance,
    _unescape,
    load_split,
    map_to_18_classes,
)


def test_unescape() -> None:
    assert _unescape("hello_comma_ world") == "hello, world"
    assert _unescape("no replacement here") == "no replacement here"


def test_map_to_18_classes_known() -> None:
    assert map_to_18_classes("afraid") == "fear"
    assert map_to_18_classes("angry") == "anger"
    assert map_to_18_classes("joyful") == "joy"
    assert map_to_18_classes("disgusted") == "disgust"
    assert map_to_18_classes("sad") == "sadness"


def test_map_to_18_classes_unknown_passthrough() -> None:
    assert map_to_18_classes("nonexistent_emotion") == "nonexistent_emotion"


def test_load_split_returns_conversations() -> None:
    convs = load_split("test")
    assert len(convs) > 0
    assert all(isinstance(c, Conversation) for c in convs)


def test_conversation_has_utterances() -> None:
    convs = load_split("test")
    for conv in convs[:10]:
        assert len(conv.utterances) > 0
        assert all(isinstance(u, Utterance) for u in conv.utterances)
        assert conv.emotion != ""
        assert conv.conv_id != ""


def test_format_history_nonempty() -> None:
    convs = load_split("test")
    conv = convs[0]
    history = conv.format_history()
    assert isinstance(history, str)
    assert len(history) > 0
    assert "Speaker" in history or "Listener" in history


def test_format_history_for_erg() -> None:
    convs = load_split("test")
    conv = convs[0]
    history, last_speaker = conv.format_history_for_erg()
    assert isinstance(history, str)
    assert isinstance(last_speaker, str)


def test_utterances_sorted_by_index() -> None:
    convs = load_split("test")
    for conv in convs[:20]:
        indices = [u.utterance_idx for u in conv.utterances]
        assert indices == sorted(indices)


def test_no_raw_comma_escapes_in_text() -> None:
    convs = load_split("test")
    for conv in convs[:50]:
        assert "_comma_" not in conv.emotion
        assert "_comma_" not in conv.prompt
        for utt in conv.utterances:
            assert "_comma_" not in utt.text
