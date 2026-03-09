#!/usr/bin/env bash
# launch_new_experiments.sh
#
# Launches new training jobs needed for professor feedback revisions:
#   1. NO-PAST-1K fix: 2 additional seeds (42, 123) — fixes missing std in Table 11
#   2. Additive ablation: 3 new conditions × 3 seeds = 9 jobs — for appendix table
#   3. Structured + smart-selection combos: 4 conditions × 5 seeds = 20 jobs — Table 9 additions
#
# Usage:
#   ./experiments/training/launch_new_experiments.sh [tier]
#
# Tiers:
#   fix       — Only the NO-PAST-1K fix (2 jobs) — run first
#   additive  — Additive ablation (9 jobs) — requires data on HF Hub
#   table9    — Struct + smart selection (20 jobs) — requires combined data on HF Hub
#   all       — All of the above sequentially
#
# Prerequisites:
#   - hf auth login
#   - For 'additive' and 'table9' tiers: run create_new_datasets.py first to upload data

set -uo pipefail

HF_USER="JosueG"
DATASET_REPO="${HF_USER}/adja-nmt-splits"
RESULTS_REPO="${HF_USER}/adja-nmt-results"
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/hf_job_train.py"
MODEL_600M="facebook/nllb-200-distilled-600M"

DELAY=5
RETRY_WAIT=65
MAX_RETRIES=3
JOB_COUNT=0
FAIL_COUNT=0

launch_job() {
    local experiment=$1
    local condition=$2
    local seed=$3
    local hardware=$4
    local model=$5
    local timeout=${6:-"3h"}
    local results_repo=${7:-"$RESULTS_REPO"}

    JOB_COUNT=$((JOB_COUNT + 1))
    echo "[Job #${JOB_COUNT}] ${experiment}/${condition}/seed${seed} on ${hardware}"

    local attempt=0
    while true; do
        if hf jobs uv run \
            --flavor "$hardware" \
            --timeout "$timeout" \
            --detach \
            --secrets HF_TOKEN \
            -e EXPERIMENT="$experiment" \
            -e CONDITION="$condition" \
            -e SEED="$seed" \
            -e MODEL="$model" \
            -e DATASET_REPO="$DATASET_REPO" \
            -e RESULTS_REPO="$results_repo" \
            "$SCRIPT_PATH" 2>&1; then
            break
        else
            attempt=$((attempt + 1))
            if [ $attempt -ge $MAX_RETRIES ]; then
                echo "FAILED after ${MAX_RETRIES} retries"
                FAIL_COUNT=$((FAIL_COUNT + 1))
                break
            fi
            echo "  Rate limited — waiting ${RETRY_WAIT}s..."
            sleep "$RETRY_WAIT"
        fi
    done
    sleep "$DELAY"
}

# ============================================================================
# Tier: Fix — NO-PAST-1K missing seeds (2 jobs)
# ============================================================================
tier_fix() {
    echo ""
    echo "=========================================="
    echo "FIX: NO-PAST-1K missing seeds (2 jobs)"
    echo "=========================================="
    echo "Data already on HF Hub from seed456 run."

    launch_job "ablations/module_size_ctrl" "NO-PAST-1K" 42  "t4-small" "$MODEL_600M" "2h"
    launch_job "ablations/module_size_ctrl" "NO-PAST-1K" 123 "t4-small" "$MODEL_600M" "2h"

    echo "Done. After jobs complete, sync results and update tab:module_size_ctrl in acl_latex.tex."
}

# ============================================================================
# Tier: Additive — Additive ablation (9 jobs)
# Conditions build up modules: M1+M2, M1+M2+M3, M1+M2+M3+M4
# (ADD-M1 = BASE-ONLY already exists; ADD-FULL = FULL already exists)
# ============================================================================
tier_additive() {
    echo ""
    echo "=========================================="
    echo "ADDITIVE ABLATION (9 jobs)"
    echo "=========================================="
    echo "NOTE: Requires ADD-M1M2, ADD-M1M2M3, ADD-M1M2M3M4 data on HF Hub."
    echo "Run 'python experiments/training/create_additive_ablation_data.py' first."

    local conditions=("ADD-M1M2" "ADD-M1M2M3" "ADD-M1M2M3M4")
    local seeds=(42 123 456)

    for cond in "${conditions[@]}"; do
        for seed in "${seeds[@]}"; do
            launch_job "ablations/additive" "$cond" "$seed" "t4-small" "$MODEL_600M" "3h"
        done
    done

    echo "Done. After jobs complete, run:"
    echo "  python experiments/analysis/compute_additive_ablation_table.py"
}

# ============================================================================
# Tier: Table9 — Structured + smart-selection combinations (20 jobs)
# ============================================================================
tier_table9() {
    echo ""
    echo "=========================================="
    echo "TABLE 9 NEW ROWS (20 jobs)"
    echo "=========================================="
    echo "NOTE: Requires combined dataset files on HF Hub."
    echo "Run 'python experiments/training/create_table9_datasets.py' first."

    local conditions=("STRUCT4K-TFIDF2K" "STRUCT4K-LENGTH2K" "STRUCT4K-VOCAB2K" "STRUCT4K-ALL-BASELINES")
    local seeds=(42 123 456 789 2024)

    for cond in "${conditions[@]}"; do
        for seed in "${seeds[@]}"; do
            launch_job "baselines" "$cond" "$seed" "t4-small" "$MODEL_600M" "3h"
        done
    done

    echo "Done. After jobs complete, sync and update tab:baselines in acl_latex.tex."
}

# ============================================================================
# Main dispatch
# ============================================================================
TIER="${1:-all}"

case "$TIER" in
    fix)       tier_fix ;;
    additive)  tier_additive ;;
    table9)    tier_table9 ;;
    all)
        tier_fix
        echo ""
        echo "=== fix jobs launched. Run additive/table9 after creating datasets. ==="
        ;;
    *)
        echo "Unknown tier: $TIER"
        echo "Usage: $0 [fix|additive|table9|all]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Total jobs launched: ${JOB_COUNT}"
echo "Total failures:      ${FAIL_COUNT}"
echo "=========================================="
