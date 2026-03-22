# Experiment Results

Model: see `.env` → `MODEL` (or `EMOTION_AGENT_MODEL` / `AGGREGATE_MODEL` for per-role overrides)  
Temperature: 0.0  
Dataset: EmpatheticDialogues  

Run `python scripts/show_results.py` to regenerate the tables below.

---

## ERC — Emotion Recognition in Conversation

Metric: Accuracy over 32-class and 18-class (coarser Plutchik grouping) label sets.

| Split | Mode      | ACC-32 (%) | ACC-18 (%) |
|-------|-----------|------------|------------|
| test  | baseline  | —          | —          |
| test  | insideout | —          | —          |
| valid | baseline  | —          | —          |
| valid | insideout | —          | —          |

---

## ERG — Empathetic Response Generation

Metrics: BLEU-1/2/3/4 (×100), ROUGE-1/2 (×100), Distinct-1 (0–1 fraction).

| Split | Mode      | B-1 | B-2 | B-3 | B-4 | R-1 | R-2 | Dist-1 |
|-------|-----------|-----|-----|-----|-----|-----|-----|--------|
| test  | baseline  | —   | —   | —   | —   | —   | —   | —      |
| test  | insideout | —   | —   | —   | —   | —   | —   | —      |
| valid | baseline  | —   | —   | —   | —   | —   | —   | —      |
| valid | insideout | —   | —   | —   | —   | —   | —   | —      |

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
