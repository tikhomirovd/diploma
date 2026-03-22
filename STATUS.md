# Project Status

_Last updated: 2026-03-22_

## Current phase

`[x] Research` → `[ ] Prototype` → `[ ] MVP` → `[ ] Thesis writing`

## Active sprint

> InsideOut prototype complete — ready for experiments

---

## Implementation checklist

### Infrastructure
- [x] Repository setup
- [x] Python environment (uv + langchain + langgraph + openrouter)
- [x] Dataset placed at `data/empatheticdialogues/{train,valid,test}.csv`
- [x] Add evaluation dependencies: `nltk`, `rouge-score`
- [x] Update `.env.example` with per-role model env vars

### Core modules
- [x] `src/llm.py` — multi-model LLM factory (per-role env vars → OpenRouter)
- [x] `src/data.py` — load EmpatheticDialogues CSVs into typed `Conversation` dataclasses
- [x] `src/prompts.py` — all prompt templates (ERC + ERG agents + aggregators)

### LangGraph graphs
- [x] `src/erc_graph.py` — ERC StateGraph: 5 parallel emotion agents → aggregate → emotion label
- [x] `src/erg_graph.py` — ERG StateGraph: emotion assessor → 5 parallel agents → aggregate → response

### Evaluation & runners
- [x] `src/evaluation.py` — ACC (32/18-class), BLEU-1/2/3/4, ROUGE-1/2, Distinct-1
- [x] `src/run_erc.py` — batch ERC experiment runner (CLI, `--n`, JSON results)
- [x] `src/run_erg.py` — batch ERG experiment runner (CLI, `--n`, JSON results)

### Tests
- [x] `tests/test_data.py` — unit tests for data loading and conversation grouping
- [x] `tests/test_graphs.py` — unit tests for ERC/ERG graph structure (mocked LLM)

---

## Completed

- [x] Repository setup
- [x] Python environment (uv + langchain + langgraph + openrouter)
- [x] Architecture design (InsideOut ERC + ERG via LangGraph)
- [x] All core modules implemented (llm, data, prompts, erc_graph, erg_graph, evaluation)
- [x] Experiment runners (run_erc.py, run_erg.py)
- [x] Unit tests — 16 passing

---

## Blockers

_None_

## Notes

- OpenRouter key: set `OPENROUTER_API_KEY` in `.env`
- Per-role model env vars: `EMOTION_AGENT_MODEL`, `AGGREGATE_MODEL`, `ASSESSOR_MODEL`
- Dataset: EmpatheticDialogues, 32 emotion classes; 18-class mapping applied for coarser eval
- ERC task: predict `context` column (emotion label) from conversation history
- ERG task: generate empathetic response to the speaker's last utterance
