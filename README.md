# InsideOut — Diploma Project

> Implementation of the **InsideOut** emotional intelligence framework for LLMs
> ([Mozikov et al., 2024](https://ebooks.iospress.nl/doi/10.3233/FAIA240793))
> using LangGraph and OpenRouter.

## Stack

| Layer | Technology |
|---|---|
| Runtime | Python 3.12 |
| Package manager | [uv](https://github.com/astral-sh/uv) |
| LLM orchestration | LangChain + LangGraph |
| LLM gateway | [OpenRouter](https://openrouter.ai) |
| Runtime type safety | [beartype](https://github.com/beartype/beartype) |
| Static type checker | [pyright](https://github.com/microsoft/pyright) |
| Linter + formatter | [ruff](https://github.com/astral-sh/ruff) |
| Tests | pytest |

## Quick start

```bash
# Install dependencies (creates .venv automatically)
uv sync

# Copy env template and fill in your OpenRouter key + model names
cp .env.example .env

# Run smoke test
uv run python -m src.main
```

## Running experiments

```bash
# ERC — emotion recognition, first 100 test conversations
uv run python -m src.run_erc --split test --n 100

# ERG — empathetic response generation, first 100 test conversations
uv run python -m src.run_erg --split test --n 100

# ERG with LLM-as-judge scoring (uses ASSESSOR_MODEL)
uv run python -m src.run_erg --split test --n 100 --judge
```

Results are saved to `data/results/erc_results.json` and `data/results/erg_results.json`.

## Project structure

```
diploma/
├── src/
│   ├── llm.py          # LLM factory (OpenRouter, per-role env vars)
│   ├── data.py         # Load EmpatheticDialogues CSVs → typed dataclasses
│   ├── prompts.py      # All prompt templates (ERC + ERG)
│   ├── erc_graph.py    # InsideOut ERC graph (emotion recognition)
│   ├── erg_graph.py    # InsideOut ERG graph (empathetic response)
│   ├── evaluation.py   # Metrics: ACC, BLEU, ROUGE, Distinct-1, judge scores
│   ├── run_erc.py      # ERC experiment runner (CLI)
│   └── run_erg.py      # ERG experiment runner (CLI)
├── tests/
│   ├── test_data.py    # Data loading unit tests
│   └── test_graphs.py  # ERC/ERG graph tests (mocked LLM)
├── data/
│   ├── empatheticdialogues/  # train/valid/test.csv
│   └── results/              # experiment outputs (git-ignored)
├── .env.example        # Env variable template
├── STATUS.md           # Task tracker
└── pyproject.toml
```

## Development toolchain

### Type safety — two complementary layers

| Tool | When it runs | What it catches |
|---|---|---|
| **beartype** | At runtime (when functions are called) | Wrong types passed to functions during actual execution |
| **pyright** | At development time (static analysis) | Type errors before the code ever runs |

They are **not alternatives** — they complement each other:
- pyright finds bugs while you write code (IDE integration, CI)
- beartype is a runtime safety net that catches anything static analysis missed

### Commands

```bash
# Run tests
uv run pytest

# Static type checking (pyright)
uv run pyright

# Lint (check only)
uv run ruff check .

# Lint + auto-fix
uv run ruff check --fix .

# Format
uv run ruff format .

# Check formatting without changing files (for CI)
uv run ruff format --check .
```

### Run all checks at once

```bash
uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest
```
