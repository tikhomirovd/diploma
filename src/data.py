from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from beartype import beartype

DATA_DIR = Path(__file__).parent.parent / "data" / "empatheticdialogues"

# EmpatheticDialogues has 32 emotion labels.
EMOTIONS_32: list[str] = [
    "afraid",
    "angry",
    "annoyed",
    "anticipating",
    "anxious",
    "apprehensive",
    "ashamed",
    "caring",
    "confident",
    "content",
    "devastated",
    "disappointed",
    "disgusted",
    "embarrassed",
    "excited",
    "faithful",
    "furious",
    "grateful",
    "guilty",
    "hopeful",
    "impressed",
    "jealous",
    "joyful",
    "lonely",
    "nostalgic",
    "prepared",
    "proud",
    "sad",
    "sentimental",
    "surprised",
    "terrified",
    "trusting",
]

# Coarser 18-class mapping used in Table 2 of the paper.
MAPPING_32_TO_18: dict[str, str] = {
    "afraid": "fear",
    "angry": "anger",
    "annoyed": "anger",
    "anticipating": "anticipation",
    "anxious": "fear",
    "apprehensive": "fear",
    "ashamed": "sadness",
    "caring": "joy",
    "confident": "joy",
    "content": "joy",
    "devastated": "sadness",
    "disappointed": "sadness",
    "disgusted": "disgust",
    "embarrassed": "sadness",
    "excited": "joy",
    "faithful": "trust",
    "furious": "anger",
    "grateful": "joy",
    "guilty": "sadness",
    "hopeful": "anticipation",
    "impressed": "surprise",
    "jealous": "anger",
    "joyful": "joy",
    "lonely": "sadness",
    "nostalgic": "sadness",
    "prepared": "anticipation",
    "proud": "joy",
    "sad": "sadness",
    "sentimental": "sadness",
    "surprised": "surprise",
    "terrified": "fear",
    "trusting": "trust",
}


@dataclass(frozen=True)
class Utterance:
    utterance_idx: int
    speaker_idx: str
    text: str


@dataclass(frozen=True)
class Conversation:
    conv_id: str
    emotion: str  # ground-truth label (32 classes)
    prompt: str  # speaker's situation description
    utterances: list[Utterance]

    @beartype
    def format_history(self) -> str:
        """Format the dialogue as a plain-text string for prompt injection."""
        lines: list[str] = []
        for utt in self.utterances:
            role = "Speaker" if utt.speaker_idx != "0" else "Listener"
            lines.append(f"{role}: {utt.text}")
        return "\n".join(lines)

    @beartype
    def format_history_for_erg(self) -> tuple[str, str]:
        """Return (history_without_last, last_speaker_utterance) for ERG.

        The ERG task: given dialogue history, generate the listener's reply
        to the last speaker turn.
        """
        history_lines: list[str] = []
        last_speaker_text = ""
        for utt in self.utterances:
            role = "Speaker" if utt.speaker_idx != "0" else "Listener"
            text = f"{role}: {utt.text}"
            history_lines.append(text)
            if utt.speaker_idx != "0":
                last_speaker_text = utt.text
        # Drop the last utterance from history if it's the listener's reply
        # (we want to predict it); keep everything up to last speaker turn.
        history = "\n".join(history_lines[:-1]) if history_lines else ""
        return history, last_speaker_text


@beartype
def _unescape(text: str) -> str:
    """Replace the dataset's _comma_ placeholder with a real comma."""
    return text.replace("_comma_", ",")


@beartype
def load_split(
    split: Literal["train", "valid", "test"],
) -> list[Conversation]:
    """Load one split of EmpatheticDialogues into Conversation objects."""
    path = DATA_DIR / f"{split}.csv"
    rows: dict[str, list[dict[str, str]]] = {}

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cid = row["conv_id"]
            if cid not in rows:
                rows[cid] = []
            rows[cid].append(row)

    conversations: list[Conversation] = []
    for cid, conv_rows in rows.items():
        conv_rows_sorted = sorted(conv_rows, key=lambda r: int(r["utterance_idx"]))
        first = conv_rows_sorted[0]
        emotion = _unescape(first["context"].strip())
        prompt = _unescape(first["prompt"].strip())
        utterances = [
            Utterance(
                utterance_idx=int(r["utterance_idx"]),
                speaker_idx=r["speaker_idx"].strip(),
                text=_unescape(r["utterance"].strip()),
            )
            for r in conv_rows_sorted
        ]
        conversations.append(
            Conversation(
                conv_id=cid,
                emotion=emotion,
                prompt=prompt,
                utterances=utterances,
            )
        )
    return conversations


@beartype
def map_to_18_classes(emotion: str) -> str:
    """Map a 32-class emotion label to the coarser 18-class set."""
    return MAPPING_32_TO_18.get(emotion, emotion)
