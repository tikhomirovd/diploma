from __future__ import annotations

"""Display all experiment results as formatted tables and update data/results/README.md.

Usage:
    uv run python scripts/show_results.py            # print tables
    uv run python scripts/show_results.py --update   # also rewrite README.md
"""

import argparse
import json
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import (
    ERCResult,
    ERGResult,
    compute_erc_accuracy,
    compute_erg_metrics,
)

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"
SPLITS = ["test", "valid"]
MODES = ["baseline", "insideout"]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_erc(path: Path) -> list[ERCResult] | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        ERCResult(
            conv_id=r["conv_id"],
            ground_truth=r["ground_truth"],
            predicted=r["predicted"],
            assessments=r.get("assessments", {}),
            reasoning=r.get("reasoning", ""),
        )
        for r in raw
    ]


def load_erg(path: Path) -> list[ERGResult] | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        ERGResult(
            conv_id=r["conv_id"],
            reference=r["reference"],
            generated=r["generated"],
            assumed_emotion=r.get("assumed_emotion", ""),
            proposed_responses=r.get("proposed_responses", {}),
            reasoning=r.get("reasoning", ""),
            judge_scores=r.get("judge_scores", {}),
        )
        for r in raw
    ]


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

COL_W = 10  # column width for metric values


def _cell(val: float | None, width: int = COL_W) -> str:
    if val is None:
        return "—".center(width)
    return f"{val:.2f}".rjust(width)


def _row(*cells: str, widths: list[int]) -> str:
    return " | ".join(c.ljust(w) for c, w in zip(cells, widths))


def _sep(widths: list[int]) -> str:
    return "-+-".join("-" * w for w in widths)


# ---------------------------------------------------------------------------
# ERC table
# ---------------------------------------------------------------------------


def erc_table() -> str:
    headers = ["Split", "Mode", "N", "ACC-32 (%)", "ACC-18 (%)"]
    widths = [6, 10, 6, 10, 10]
    lines: list[str] = [
        _row(*headers, widths=widths),
        _sep(widths),
    ]
    for split in SPLITS:
        for mode in MODES:
            path = RESULTS_DIR / split / f"erc_{mode}.json"
            results = load_erc(path)
            if results is None:
                n_str = "—"
                acc32 = acc18 = None
            else:
                n_str = str(len(results))
                m = compute_erc_accuracy(results)
                acc32 = m.get("acc_32")
                acc18 = m.get("acc_18")
            lines.append(
                _row(
                    split,
                    mode,
                    n_str.rjust(widths[2]),
                    _cell(acc32, widths[3]),
                    _cell(acc18, widths[4]),
                    widths=widths,
                )
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ERG table
# ---------------------------------------------------------------------------


def erg_table() -> str:
    headers = ["Split", "Mode", "N", "B-1", "B-2", "B-3", "B-4", "R-1", "R-2", "Dist-1"]
    widths = [6, 10, 6, 7, 7, 7, 7, 7, 7, 8]
    lines: list[str] = [
        _row(*headers, widths=widths),
        _sep(widths),
    ]
    for split in SPLITS:
        for mode in MODES:
            path = RESULTS_DIR / split / f"erg_{mode}.json"
            results = load_erg(path)
            if results is None:
                n_str = "—"
                m: dict[str, float] = {}
            else:
                n_str = str(len(results))
                m = compute_erg_metrics(results)
            lines.append(
                _row(
                    split,
                    mode,
                    n_str.rjust(widths[2]),
                    _cell(m.get("bleu_1"), widths[3]),
                    _cell(m.get("bleu_2"), widths[4]),
                    _cell(m.get("bleu_3"), widths[5]),
                    _cell(m.get("bleu_4"), widths[6]),
                    _cell(m.get("rouge_1"), widths[7]),
                    _cell(m.get("rouge_2"), widths[8]),
                    _cell(m.get("distinct_1"), widths[9]),
                    widths=widths,
                )
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# README update
# ---------------------------------------------------------------------------

_ERC_MARKER_START = "| Split | Mode      | ACC-32 (%) | ACC-18 (%) |"
_ERC_MARKER_END = "\n\n---"
_ERG_MARKER_START = "| Split | Mode      | B-1 | B-2 | B-3 | B-4 | R-1 | R-2 | Dist-1 |"


def _md_erc_table() -> str:
    header = "| Split | Mode      | N    | ACC-32 (%) | ACC-18 (%) |"
    sep = "|-------|-----------|------|------------|------------|"
    rows = []
    for split in SPLITS:
        for mode in MODES:
            path = RESULTS_DIR / split / f"erc_{mode}.json"
            results = load_erc(path)
            if results is None:
                rows.append(f"| {split:<5} | {mode:<9} | —    | —          | —          |")
            else:
                m = compute_erc_accuracy(results)
                rows.append(
                    f"| {split:<5} | {mode:<9} | {len(results):<4} "
                    f"| {m['acc_32']:>10.2f} | {m['acc_18']:>10.2f} |"
                )
    return "\n".join([header, sep] + rows)


def _md_erg_table() -> str:
    header = "| Split | Mode      | N    |   B-1  |   B-2  |   B-3  |   B-4  |   R-1  |   R-2  | Dist-1 |"
    sep =    "|-------|-----------|------|--------|--------|--------|--------|--------|--------|--------|"
    rows = []
    for split in SPLITS:
        for mode in MODES:
            path = RESULTS_DIR / split / f"erg_{mode}.json"
            results = load_erg(path)
            if results is None:
                rows.append(f"| {split:<5} | {mode:<9} | —    | —      | —      | —      | —      | —      | —      | —      |")
            else:
                m = compute_erg_metrics(results)
                rows.append(
                    f"| {split:<5} | {mode:<9} | {len(results):<4} "
                    f"| {m.get('bleu_1', 0):>6.2f} | {m.get('bleu_2', 0):>6.2f} "
                    f"| {m.get('bleu_3', 0):>6.2f} | {m.get('bleu_4', 0):>6.2f} "
                    f"| {m.get('rouge_1', 0):>6.2f} | {m.get('rouge_2', 0):>6.2f} "
                    f"| {m.get('distinct_1', 0):>6.4f} |"
                )
    return "\n".join([header, sep] + rows)


def update_readme() -> None:
    readme = RESULTS_DIR / "README.md"
    if not readme.exists():
        print("README.md not found, skipping update.")
        return

    model = Path(__file__).parent.parent / ".env"
    model_name = "unknown"
    if model.exists():
        for line in model.read_text().splitlines():
            if line.startswith("MODEL="):
                model_name = line.split("=", 1)[1].strip()

    content = f"""# Experiment Results

Model: `{model_name}`  
Temperature: 0.0  
Dataset: EmpatheticDialogues  

Run `python scripts/show_results.py --update` to regenerate this file.

---

## ERC — Emotion Recognition in Conversation

{_md_erc_table()}

---

## ERG — Empathetic Response Generation

{_md_erg_table()}

---

## Result Files

```
data/results/
  test/
    erc_baseline.json      ← ERC, single-LLM baseline, test split
    erc_insideout.json     ← ERC, InsideOut multi-agent, test split
    erg_baseline.json      ← ERG, single-LLM baseline, test split
    erg_insideout.json     ← ERG, InsideOut multi-agent, test split
  valid/
    erc_baseline.json
    erc_insideout.json
    erg_baseline.json
    erg_insideout.json
```

Each JSON file is a list of per-conversation result objects.  
ERC fields: `conv_id`, `ground_truth`, `predicted`, `assessments`, `reasoning`  
ERG fields: `conv_id`, `reference`, `generated`, `assumed_emotion`, `proposed_responses`, `reasoning`, `judge_scores`
"""
    readme.write_text(content, encoding="utf-8")
    print(f"README updated → {readme}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Show experiment results")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Rewrite data/results/README.md with current numbers",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" ERC — Emotion Recognition in Conversation")
    print("=" * 70)
    print(erc_table())

    print("\n" + "=" * 70)
    print(" ERG — Empathetic Response Generation")
    print("=" * 70)
    print(erg_table())
    print()

    if args.update:
        update_readme()


if __name__ == "__main__":
    main()
