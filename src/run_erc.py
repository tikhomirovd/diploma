from __future__ import annotations

"""Batch ERC experiment runner.

Usage:
    uv run python -m src.run_erc [--split test] [--n 200] [--mode insideout] \
        [--out data/results/test/erc_insideout.json] [--checkpoint 50]
"""

import argparse
import json
import sys
import time
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
def _load_checkpoint(path: str) -> dict[str, ERCResult]:
    """Load existing results from disk (for resume support)."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {
            r["conv_id"]: ERCResult(
                conv_id=r["conv_id"],
                ground_truth=r["ground_truth"],
                predicted=r["predicted"],
                assessments=r.get("assessments", {}),
                reasoning=r.get("reasoning", ""),
            )
            for r in raw
        }
    except Exception:
        return {}


@beartype
def run(
    split: str = "test",
    n: int | None = None,
    out: str = "data/results/erc_results.json",
    mode: str = "insideout",
    checkpoint: int = 50,
    sleep_s: float = 0.0,
) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading '{split}' split …")
    conversations: list[Conversation] = load_split(split)  # type: ignore[arg-type]
    if n is not None:
        conversations = conversations[:n]

    # Resume: skip already-processed conversations
    done = _load_checkpoint(out)
    if done:
        print(f"Resuming — {len(done)} conversations already done.")

    total = len(conversations)
    print(f"Running ERC [{mode}] on {total} conversations …\n")

    results: list[ERCResult] = list(done.values())
    new_count = 0

    for i, conv in enumerate(conversations, 1):
        if conv.conv_id in done:
            print(f"  [{i:>4}/{total}] SKIP {conv.conv_id}")
            continue

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

        result = ERCResult(
            conv_id=conv.conv_id,
            ground_truth=conv.emotion,
            predicted=predicted,
            assessments=assessments,
            reasoning=reasoning,
        )
        results.append(result)
        new_count += 1

        status = "✓" if predicted == conv.emotion else "✗"
        print(
            f"  [{i:>4}/{total}] {status} "
            f"gt={conv.emotion:<15} pred={predicted:<15} | {conv.conv_id}"
        )

        if new_count % checkpoint == 0:
            save_erc_results(results, out)
            print(f"  [checkpoint] saved {len(results)} results → {out}")

        if sleep_s > 0:
            time.sleep(sleep_s)

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
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=50,
        help="Save intermediate results every N conversations (default: 50)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        dest="sleep_s",
        help="Seconds to sleep between conversations (default: 0)",
    )
    args = parser.parse_args()

    run(
        split=args.split,
        n=args.n,
        out=args.out,
        mode=args.mode,
        checkpoint=args.checkpoint,
        sleep_s=args.sleep_s,
    )


if __name__ == "__main__":
    main()
