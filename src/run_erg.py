from __future__ import annotations

"""Batch ERG experiment runner.

Usage:
    uv run python -m src.run_erg [--split test] [--n 100] \
        [--out data/results/erg_results.json] [--judge]
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from beartype import beartype
from dotenv import load_dotenv

# load_dotenv before importing project modules so env vars are available
load_dotenv()

from src.data import Conversation, load_split
from src.erg_graph import run_erg
from src.evaluation import (
    ERGResult,
    compute_erg_metrics,
    compute_judge_scores,
    save_erg_results,
)
from src.llm import build_assessor_llm
from src.prompts import llm_judge_system, llm_judge_user


@beartype
def _parse_json(text: str) -> dict[str, object]:
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        result: Any = json.loads(cleaned)
        return dict(result)  # type: ignore[arg-type]
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            result2: Any = json.loads(match.group())
            return dict(result2)  # type: ignore[arg-type]
        return {}


@beartype
def _run_judge(conversation: str, emotion: str, response: str) -> dict[str, float]:
    """Ask the assessor LLM to score one generated response."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = build_assessor_llm()
    messages = [
        SystemMessage(content=llm_judge_system()),
        HumanMessage(content=llm_judge_user(conversation, emotion, response)),
    ]
    result = llm.invoke(messages)
    parsed = _parse_json(str(result.content))
    return {k: float(v) for k, v in parsed.items() if isinstance(v, (int, float))}


@beartype
def _format_erg_table(metrics: dict[str, float]) -> str:
    lines = ["", "=" * 50, " ERG Results (classic metrics)", "=" * 50]
    for key, val in metrics.items():
        lines.append(f"  {key:<12}: {val:.4f}")
    lines.append("=" * 50)
    return "\n".join(lines)


@beartype
def _format_judge_table(scores: dict[str, float]) -> str:
    if not scores:
        return ""
    lines = ["", "=" * 50, " ERG Judge Scores (LLM-as-assessor, /10)", "=" * 50]
    for key, val in scores.items():
        lines.append(f"  {key:<12}: {val:.4f}")
    lines.append("=" * 50)
    return "\n".join(lines)


@beartype
def run(
    split: str = "test",
    n: int | None = None,
    out: str = "data/results/erg_results.json",
    judge: bool = False,
) -> None:
    print(f"Loading '{split}' split …")
    conversations: list[Conversation] = load_split(split)  # type: ignore[arg-type]
    if n is not None:
        conversations = conversations[:n]

    total = len(conversations)
    print(f"Running ERG on {total} conversations …\n")

    results: list[ERGResult] = []
    for i, conv in enumerate(conversations, 1):
        # For ERG we predict the listener's reply to the last speaker turn.
        history, _last_speaker_text = conv.format_history_for_erg()
        # Ground-truth reference: last utterance in the conversation
        # (the listener's actual reply, if it exists).
        last_utt = conv.utterances[-1]
        reference = last_utt.text if last_utt.speaker_idx == "0" else ""

        try:
            output = run_erg(history)
            generated = str(output.get("final_response", "")).strip()
            assumed_emotion = str(output.get("assumed_emotion", ""))
            proposed = dict(output.get("proposed_responses", {}))
            reasoning = str(output.get("reasoning", ""))
        except Exception as exc:
            print(f"  [ERROR] conv {conv.conv_id}: {exc}", file=sys.stderr)
            generated = ""
            assumed_emotion = ""
            proposed = {}
            reasoning = ""

        judge_scores: dict[str, float] = {}
        if judge and generated and history:
            try:
                judge_scores = _run_judge(history, assumed_emotion, generated)
            except Exception as exc:
                print(f"  [JUDGE ERROR] conv {conv.conv_id}: {exc}", file=sys.stderr)

        results.append(
            ERGResult(
                conv_id=conv.conv_id,
                reference=reference,
                generated=generated,
                assumed_emotion=assumed_emotion,
                proposed_responses=proposed,
                reasoning=reasoning,
                judge_scores=judge_scores,
            )
        )

        print(
            f"  [{i:>4}/{total}] {conv.conv_id} | "
            f"emotion={assumed_emotion:<15} | "
            f"response={generated[:60]!r}"
        )

    metrics = compute_erg_metrics(results)
    print(_format_erg_table(metrics))

    if judge:
        judge_agg = compute_judge_scores(results)
        print(_format_judge_table(judge_agg))

    save_erg_results(results, out)
    print(f"\nResults saved → {out}")


@beartype
def main() -> None:
    parser = argparse.ArgumentParser(description="Run InsideOut ERG evaluation")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--n", type=int, default=None, help="Max conversations to process")
    parser.add_argument("--out", default="data/results/erg_results.json")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Run LLM-as-judge scoring (uses ASSESSOR_MODEL env var)",
    )
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    run(split=args.split, n=args.n, out=args.out, judge=args.judge)


if __name__ == "__main__":
    main()
