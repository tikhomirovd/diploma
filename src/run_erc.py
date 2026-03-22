from __future__ import annotations

"""Batch ERC experiment runner.

Usage:
    uv run python -m src.run_erc [--split test] [--n 100] [--out data/results/erc_results.json]
"""

import argparse
import sys
from pathlib import Path

from beartype import beartype
from dotenv import load_dotenv

# load_dotenv before importing project modules so env vars are available
load_dotenv()

from src.baseline import run_erc_baseline
from src.data import Conversation, load_split
from src.erc_graph import run_erc
from src.evaluation import ERCResult, compute_erc_accuracy, save_erc_results


@beartype
def _format_table(metrics: dict[str, float]) -> str:
    lines = ["", "=" * 40, " ERC Results", "=" * 40]
    lines.append(f"  Accuracy (32 classes): {metrics['acc_32']:.2f}%")
    lines.append(f"  Accuracy (18 classes): {metrics['acc_18']:.2f}%")
    lines.append("=" * 40)
    return "\n".join(lines)


@beartype
def run(
    split: str = "test",
    n: int | None = None,
    out: str = "data/results/erc_results.json",
    mode: str = "insideout",
) -> None:
    print(f"Loading '{split}' split …")
    conversations: list[Conversation] = load_split(split)  # type: ignore[arg-type]
    if n is not None:
        conversations = conversations[:n]

    total = len(conversations)
    print(f"Running ERC [{mode}] on {total} conversations …\n")

    results: list[ERCResult] = []
    for i, conv in enumerate(conversations, 1):
        history = conv.format_history()
        try:
            if mode == "baseline":
                output = run_erc_baseline(history)
            else:
                output = run_erc(history)
            predicted = str(output.get("final_emotion", "")).strip().lower()
            reasoning = str(output.get("reasoning", ""))
            assessments = dict(output.get("assessments", {}))
        except Exception as exc:
            print(f"  [ERROR] conv {conv.conv_id}: {exc}", file=sys.stderr)
            predicted = ""
            reasoning = ""
            assessments = {}

        results.append(
            ERCResult(
                conv_id=conv.conv_id,
                ground_truth=conv.emotion,
                predicted=predicted,
                assessments=assessments,
                reasoning=reasoning,
            )
        )

        status = "✓" if predicted == conv.emotion else "✗"
        print(
            f"  [{i:>4}/{total}] {status} "
            f"gt={conv.emotion:<15} pred={predicted:<15} | {conv.conv_id}"
        )

    metrics = compute_erc_accuracy(results)
    print(_format_table(metrics))

    save_erc_results(results, out)
    print(f"\nResults saved → {out}")


@beartype
def main() -> None:
    parser = argparse.ArgumentParser(description="Run InsideOut ERC evaluation")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--n", type=int, default=None, help="Max conversations to process")
    parser.add_argument("--out", default="data/results/erc_results.json")
    parser.add_argument(
        "--mode",
        default="insideout",
        choices=["insideout", "baseline"],
        help="insideout: full multi-agent graph; baseline: single LLM call",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    run(split=args.split, n=args.n, out=args.out, mode=args.mode)


if __name__ == "__main__":
    main()
