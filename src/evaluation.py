from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import nltk
from beartype import beartype
from rouge_score import rouge_scorer

from src.data import map_to_18_classes

# Download tokenizer data once (safe to call multiple times)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ---------------------------------------------------------------------------
# Data classes for results
# ---------------------------------------------------------------------------


@dataclass
class ERCResult:
    conv_id: str
    ground_truth: str
    predicted: str
    assessments: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class ERGResult:
    conv_id: str
    reference: str  # last listener utterance from dataset
    generated: str  # InsideOut final_response
    assumed_emotion: str = ""
    proposed_responses: dict[str, str] = field(default_factory=dict)
    reasoning: str = ""
    judge_scores: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ERC metrics
# ---------------------------------------------------------------------------


@beartype
def compute_erc_accuracy(results: list[ERCResult]) -> dict[str, float]:
    """Return accuracy over 32 and 18 emotion classes."""
    if not results:
        return {"acc_32": 0.0, "acc_18": 0.0}

    correct_32 = sum(
        1 for r in results if r.predicted.strip().lower() == r.ground_truth.strip().lower()
    )
    correct_18 = sum(
        1
        for r in results
        if map_to_18_classes(r.predicted.strip().lower())
        == map_to_18_classes(r.ground_truth.strip().lower())
    )
    n = len(results)
    return {
        "acc_32": round(100.0 * correct_32 / n, 2),
        "acc_18": round(100.0 * correct_18 / n, 2),
    }


# ---------------------------------------------------------------------------
# ERG metrics
# ---------------------------------------------------------------------------


@beartype
def _tokenize(text: str) -> list[str]:
    return list(nltk.word_tokenize(text.lower()))


@beartype
def _bleu_n(reference_tokens: list[str], hypothesis_tokens: list[str], n: int) -> float:
    """Compute BLEU-n (precision of n-grams) * 100."""
    if len(hypothesis_tokens) < n:
        return 0.0

    ref_ngrams: dict[tuple[str, ...], int] = {}
    for i in range(len(reference_tokens) - n + 1):
        ng = tuple(reference_tokens[i : i + n])
        ref_ngrams[ng] = ref_ngrams.get(ng, 0) + 1

    matches = 0
    hyp_ngrams: dict[tuple[str, ...], int] = {}
    for i in range(len(hypothesis_tokens) - n + 1):
        ng = tuple(hypothesis_tokens[i : i + n])
        hyp_ngrams[ng] = hyp_ngrams.get(ng, 0) + 1

    for ng, count in hyp_ngrams.items():
        matches += min(count, ref_ngrams.get(ng, 0))

    total_hyp = len(hypothesis_tokens) - n + 1
    return round(100.0 * matches / total_hyp, 4) if total_hyp > 0 else 0.0


@beartype
def _distinct_1(texts: list[str]) -> float:
    """Compute Distinct-1: ratio of unique unigrams to total unigrams * 100."""
    tokens: list[str] = []
    for t in texts:
        tokens.extend(_tokenize(t))
    if not tokens:
        return 0.0
    return round(100.0 * len(set(tokens)) / len(tokens), 4)


@beartype
def compute_erg_metrics(results: list[ERGResult]) -> dict[str, float]:
    """Compute BLEU-1/2/3/4, ROUGE-1/2, and Distinct-1 for ERG results."""
    if not results:
        return {}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=False)

    bleu: dict[int, list[float]] = {1: [], 2: [], 3: [], 4: []}
    rouge1_list: list[float] = []
    rouge2_list: list[float] = []
    generated_texts: list[str] = []

    for r in results:
        ref_tokens = _tokenize(r.reference)
        hyp_tokens = _tokenize(r.generated)

        for n in (1, 2, 3, 4):
            bleu[n].append(_bleu_n(ref_tokens, hyp_tokens, n))

        scores = scorer.score(r.reference, r.generated)
        rouge1_list.append(scores["rouge1"].fmeasure * 100)
        rouge2_list.append(scores["rouge2"].fmeasure * 100)
        generated_texts.append(r.generated)

    def avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "bleu_1": avg(bleu[1]),
        "bleu_2": avg(bleu[2]),
        "bleu_3": avg(bleu[3]),
        "bleu_4": avg(bleu[4]),
        "rouge_1": avg(rouge1_list),
        "rouge_2": avg(rouge2_list),
        "distinct_1": _distinct_1(generated_texts),
    }


@beartype
def compute_judge_scores(results: list[ERGResult]) -> dict[str, float]:
    """Average the LLM-judge scores (F, I, E, S, O) across all results."""
    keys = ["F", "I", "E", "S", "O"]
    totals: dict[str, float] = {k: 0.0 for k in keys}
    count = 0
    for r in results:
        if r.judge_scores:
            for k in keys:
                totals[k] += float(r.judge_scores.get(k, 0))
            count += 1
    if count == 0:
        return {}
    return {k: round(totals[k] / count, 4) for k in keys}


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


@beartype
def save_erc_results(results: list[ERCResult], path: str) -> None:
    """Write ERC results to a JSON file."""
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            [
                {
                    "conv_id": r.conv_id,
                    "ground_truth": r.ground_truth,
                    "predicted": r.predicted,
                    "assessments": r.assessments,
                    "reasoning": r.reasoning,
                }
                for r in results
            ],
            fh,
            indent=2,
            ensure_ascii=False,
        )


@beartype
def save_erg_results(results: list[ERGResult], path: str) -> None:
    """Write ERG results to a JSON file."""
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            [
                {
                    "conv_id": r.conv_id,
                    "reference": r.reference,
                    "generated": r.generated,
                    "assumed_emotion": r.assumed_emotion,
                    "proposed_responses": r.proposed_responses,
                    "reasoning": r.reasoning,
                    "judge_scores": r.judge_scores,
                }
                for r in results
            ],
            fh,
            indent=2,
            ensure_ascii=False,
        )
