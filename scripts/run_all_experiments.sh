#!/usr/bin/env bash
# Run all 8 experiments (ERC+ERG × baseline+insideout × test+valid) sequentially.
# Designed to run overnight — supports resume (re-running the script safely re-uses
# checkpoints and skips already-processed conversations).
#
# Usage:
#   bash scripts/run_all_experiments.sh            # all experiments, n=200 per run
#   bash scripts/run_all_experiments.sh 500        # override n (conversations per run)
#   nohup bash scripts/run_all_experiments.sh > logs/run_all.log 2>&1 &

set -euo pipefail

N="${1:-200}"           # conversations per experiment (default 200)
CHECKPOINT=50           # checkpoint every N conversations
LOG_DIR="logs"
RESULTS_DIR="data/results"

mkdir -p "$LOG_DIR" "$RESULTS_DIR/test" "$RESULTS_DIR/valid"

run_experiment() {
    local task="$1"    # erc or erg
    local mode="$2"    # baseline or insideout
    local split="$3"   # test or valid
    local out="$RESULTS_DIR/$split/${task}_${mode}.json"
    local log="$LOG_DIR/${task}_${mode}_${split}.log"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [$task / $mode / $split]  n=$N  →  $out"
    echo "  Log: $log"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    uv run python -m "src.run_${task}" \
        --split "$split" \
        --n "$N" \
        --mode "$mode" \
        --out "$out" \
        --checkpoint "$CHECKPOINT" \
        2>&1 | tee "$log"

    echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
}

echo "=================================================="
echo " InsideOut Experiment Suite"
echo " Model: $(grep '^MODEL=' .env 2>/dev/null | cut -d= -f2)"
echo " n per experiment: $N"
echo " Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="

# --- test split (primary evaluation) ---
run_experiment erc baseline test
run_experiment erc insideout test
run_experiment erg baseline test
run_experiment erg insideout test

# --- valid split ---
run_experiment erc baseline valid
run_experiment erc insideout valid
run_experiment erg baseline valid
run_experiment erg insideout valid

echo ""
echo "=================================================="
echo " All experiments done: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="

echo ""
echo "Generating results summary …"
uv run python scripts/show_results.py --update
