#!/usr/bin/env bash
# launch_jobs.sh — Launch all experiment training runs on HuggingFace Jobs.
#
# Prerequisites:
#   1. hf auth login
#   2. Data uploaded to DATASET_REPO
#   3. hf_job_train.py in experiments/training/
#
# Usage:
#   ./launch_jobs.sh [tier]
#
# Tiers:
#   smoke  — T1: Exp1 × seed 42 only (6 jobs)
#   core   — T2: Exp1 remaining seeds + baselines (39 jobs)
#   scale  — T3: Exp2 all conditions × 5 seeds (110 jobs)
#   ablate — T4: Ablations × 3 seeds (78 jobs)
#   arch   — T5: Architecture variants (6 jobs)
#   all    — T1 through T5 sequentially

set -uo pipefail

# ============================================================================
# Configuration — EDIT THESE
# ============================================================================
HF_USER="JosueG"
DATASET_REPO="${HF_USER}/adja-nmt-splits"
RESULTS_REPO="${HF_USER}/adja-nmt-results"
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/hf_job_train.py"
MODEL_600M="facebook/nllb-200-distilled-600M"
MODEL_1_3B="facebook/nllb-200-1.3B"
MODEL_MBART="facebook/mbart-large-50-many-to-many-mmt"

SEEDS_5=(42 123 456 789 2024)
SEEDS_3=(42 123 456)
DELAY=5   # seconds between job launches
RETRY_WAIT=65  # seconds to wait on 429 (rate limit window is 5 min)
MAX_RETRIES=3

# ============================================================================
# Job launcher (with retry on 429 rate limit)
# ============================================================================
JOB_COUNT=0
FAIL_COUNT=0

launch_job() {
    local experiment=$1
    local condition=$2
    local seed=$3
    local hardware=$4
    local model=$5
    local timeout=${6:-"3h"}

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
            -e RESULTS_REPO="$RESULTS_REPO" \
            "$SCRIPT_PATH" 2>&1; then
            break
        else
            attempt=$((attempt + 1))
            if [ $attempt -ge $MAX_RETRIES ]; then
                echo "FAILED after ${MAX_RETRIES} retries: ${experiment}/${condition}/seed${seed}"
                FAIL_COUNT=$((FAIL_COUNT + 1))
                break
            fi
            echo "  Rate limited — waiting ${RETRY_WAIT}s before retry ${attempt}/${MAX_RETRIES}..."
            sleep "$RETRY_WAIT"
        fi
    done

    sleep "$DELAY"
}

# ============================================================================
# Tier 1: Smoke test — Exp1, seed 42 only
# ============================================================================
tier_smoke() {
    echo ""
    echo "=========================================="
    echo "TIER 1: SMOKE TEST (Exp1 × seed 42)"
    echo "=========================================="

    local conditions=(
        "RANDOM-10K"
        "RANDOM-6K_STRUCTURED-4K"
        "RANDOM-10K_STRUCTURED-4K"
        "STRUCTURED-4K-ONLY"
        "RANDOM-4K"
        "STRUCTURED-2K"
    )

    for cond in "${conditions[@]}"; do
        launch_job "exp1" "$cond" 42 "t4-small" "$MODEL_600M"
    done

    echo ""
    echo "Smoke test: ${#conditions[@]} jobs launched."
    echo "Check results before proceeding to next tier."
}

# ============================================================================
# Tier 2: Core — Exp1 remaining seeds + baselines
# ============================================================================
tier_core() {
    echo ""
    echo "=========================================="
    echo "TIER 2: CORE (Exp1 seeds 123-2024 + baselines)"
    echo "=========================================="

    local exp1_conditions=(
        "RANDOM-10K"
        "RANDOM-6K_STRUCTURED-4K"
        "RANDOM-10K_STRUCTURED-4K"
        "STRUCTURED-4K-ONLY"
        "RANDOM-4K"
        "STRUCTURED-2K"
    )

    # Exp1 remaining seeds (123, 456, 789, 2024)
    for cond in "${exp1_conditions[@]}"; do
        for seed in 123 456 789 2024; do
            launch_job "exp1" "$cond" "$seed" "t4-small" "$MODEL_600M"
        done
    done

    # Baselines × 5 seeds
    local baselines=(
        "LENGTH-STRATIFIED"
        "VOCAB-MAXIMIZED"
        "TF-IDF-DIVERSE"
    )

    for cond in "${baselines[@]}"; do
        for seed in "${SEEDS_5[@]}"; do
            launch_job "baselines" "$cond" "$seed" "t4-small" "$MODEL_600M"
        done
    done

    echo ""
    echo "Core tier launched."
}

# ============================================================================
# Tier 3: Scaling curves — Exp2 all conditions × 5 seeds
# ============================================================================
tier_scale() {
    echo ""
    echo "=========================================="
    echo "TIER 3: SCALING CURVES (Exp2 × 5 seeds)"
    echo "=========================================="

    # Structured scaling
    for size in 200 500 1000 2000 3000 4000; do
        for seed in "${SEEDS_5[@]}"; do
            launch_job "exp2" "STRUCTURED-${size}" "$seed" "t4-small" "$MODEL_600M"
        done
    done

    # Random scaling
    for size in 200 500 1000 2000 4000 6000 8000 10000; do
        for seed in "${SEEDS_5[@]}"; do
            launch_job "exp2" "RANDOM-${size}" "$seed" "t4-small" "$MODEL_600M"
        done
    done

    # Combined additive
    for struct_size in 500 1000 2000 4000; do
        for seed in "${SEEDS_5[@]}"; do
            launch_job "exp2" "RANDOM-6K_STRUCTURED-${struct_size}" "$seed" "t4-small" "$MODEL_600M"
        done
    done

    # Combined replacement
    for struct_size in 500 1000 2000 4000; do
        local random_size=$((10000 - struct_size))
        for seed in "${SEEDS_5[@]}"; do
            launch_job "exp2" "REPLACE-R${random_size}_S${struct_size}" "$seed" "t4-small" "$MODEL_600M"
        done
    done

    echo ""
    echo "Scaling tier launched."
}

# ============================================================================
# Tier 4: Ablations × 3 seeds
# ============================================================================
tier_ablate() {
    echo ""
    echo "=========================================="
    echo "TIER 4: ABLATIONS (× 3 seeds)"
    echo "=========================================="

    # Module leave-one-out
    local module_loo=(
        "FULL" "NO-NEGATION" "NO-PAST" "NO-FUTURE" "NO-QUESTIONS" "BASE-ONLY"
    )
    for cond in "${module_loo[@]}"; do
        for seed in "${SEEDS_3[@]}"; do
            launch_job "ablations/module_loo" "$cond" "$seed" "t4-small" "$MODEL_600M"
        done
    done

    # Size-controlled module
    local module_sc=(
        "FULL-1K" "NO-NEG-1K" "NO-PAST-1K" "NO-FUT-1K" "NO-QUEST-1K" "BASE-1K"
    )
    for cond in "${module_sc[@]}"; do
        for seed in "${SEEDS_3[@]}"; do
            launch_job "ablations/module_size_ctrl" "$cond" "$seed" "t4-small" "$MODEL_600M"
        done
    done

    # Pronoun coverage
    local pronouns=("ALL-8" "REDUCED-4" "SINGULAR-3" "MINIMAL-1")
    for cond in "${pronouns[@]}"; do
        for seed in "${SEEDS_3[@]}"; do
            launch_job "ablations/pronoun" "$cond" "$seed" "t4-small" "$MODEL_600M"
        done
    done

    # Verb diversity
    local verbs=("10-VERBS" "5-VERBS-a" "5-VERBS-b" "5-VERBS-c" "3-VERBS-a" "3-VERBS-b" "3-VERBS-c" "1-VERB")
    for cond in "${verbs[@]}"; do
        for seed in "${SEEDS_3[@]}"; do
            launch_job "ablations/verb" "$cond" "$seed" "t4-small" "$MODEL_600M"
        done
    done

    # Minimal pairs
    local pairs=("PAIRS-INTACT" "PAIRS-BROKEN")
    for cond in "${pairs[@]}"; do
        for seed in "${SEEDS_3[@]}"; do
            launch_job "ablations/minimal_pairs" "$cond" "$seed" "t4-small" "$MODEL_600M"
        done
    done

    echo ""
    echo "Ablation tier launched."
}

# ============================================================================
# Tier 5: Architecture variants
# ============================================================================
tier_arch() {
    echo ""
    echo "=========================================="
    echo "TIER 5: ARCHITECTURE VARIANTS"
    echo "=========================================="

    # NLLB-1.3B on A10G (3 seeds, best structured condition)
    for seed in "${SEEDS_3[@]}"; do
        launch_job "exp1" "RANDOM-10K_STRUCTURED-4K" "$seed" "a10g-small" "$MODEL_1_3B" "4h"
    done

    # mBART-50 on T4 (3 seeds)
    for seed in "${SEEDS_3[@]}"; do
        launch_job "exp1" "RANDOM-10K_STRUCTURED-4K" "$seed" "t4-small" "$MODEL_MBART" "3h"
    done

    echo ""
    echo "Architecture tier launched."
}

# ============================================================================
# Main
# ============================================================================
TIER=${1:-smoke}

echo "HF Jobs Launcher — Adja NMT Experiments"
echo "Dataset repo: ${DATASET_REPO}"
echo "Results repo: ${RESULTS_REPO}"
echo "Script: ${SCRIPT_PATH}"
echo ""

case "$TIER" in
    smoke)  tier_smoke ;;
    core)   tier_core ;;
    scale)  tier_scale ;;
    ablate) tier_ablate ;;
    arch)   tier_arch ;;
    all)
        tier_smoke
        echo ""
        echo ">>> Launching remaining tiers (review smoke results separately) <<<"
        echo ""
        tier_core
        tier_scale
        tier_ablate
        tier_arch
        ;;
    *)
        echo "Unknown tier: $TIER"
        echo "Usage: $0 [smoke|core|scale|ablate|arch|all]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Total jobs launched: ${JOB_COUNT}"
if [ "${FAIL_COUNT:-0}" -gt 0 ]; then
    echo "FAILED submissions: ${FAIL_COUNT}"
fi
echo "Monitor at: https://huggingface.co/settings/jobs"
echo "=========================================="
