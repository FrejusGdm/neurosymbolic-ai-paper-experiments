#!/bin/bash
#SBATCH -J lrec-nmt
#SBATCH -o /dartfs/rc/lab/R/RCoto/godeme/lrec-baselines/logs/lrec-%A_%a.out
#SBATCH -e /dartfs/rc/lab/R/RCoto/godeme/lrec-baselines/logs/lrec-%A_%a.err
#SBATCH -p gpuq
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=1-60
# NOTE: All paths are hardcoded — $(dirname "$0") does NOT work in SBATCH directives
#       because SLURM copies the script to /var/spool/slurmd/ at submission time.
#
# Job layout:
#   Lines  1-15 in jobs.tsv → fr2adj, random split
#   Lines 16-30 in jobs.tsv → fr2adj, stratified split
#   Lines 31-45 in jobs.tsv → adj2fr, random split
#   Lines 46-60 in jobs.tsv → adj2fr, stratified split
#   Each block: NLLB-600M, mBART-50 (fr-init), ByT5-base × 5 seeds
#
# To run a subset:
#   sbatch --array=1-30   submit_array.sbatch   # fr2adj only (already done)
#   sbatch --array=31-60  submit_array.sbatch   # adj2fr only (new)
#   sbatch --array=31-45  submit_array.sbatch   # adj2fr random only
#   sbatch --test-only --array=31-60 submit_array.sbatch   # dry-run check

set -euo pipefail

# ── Hardcoded paths ──────────────────────────────────────────────────────────
HPC_DIR="/dartfs/rc/lab/R/RCoto/godeme/lrec-baselines"
JOBS_FILE="${HPC_DIR}/jobs.tsv"
TRAIN_SCRIPT="${HPC_DIR}/train_lrec.py"
CONTAINER="/dartfs/rc/lab/R/RCoto/godeme/adja-nmt-hpc/nmt-training.sif"
DATA_BASE_DIR="${HPC_DIR}/data"
MODELS_DIR="${HPC_DIR}/models"
RESULTS_DIR="${HPC_DIR}/results"

# ── Sanity-check that critical files exist before doing any work ─────────────
if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found: ${CONTAINER}"
    echo "  The existing adja-nmt-hpc container should be reused."
    exit 1
fi

if [ ! -f "$JOBS_FILE" ]; then
    echo "ERROR: jobs.tsv not found: ${JOBS_FILE}"
    echo "  Run: python generate_jobs_tsv.py --output ${JOBS_FILE}"
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: Training script not found: ${TRAIN_SCRIPT}"
    echo "  Upload: scp train_lrec.py ${TRAIN_SCRIPT}"
    exit 1
fi

# ── Read job params from jobs.tsv (line $SLURM_ARRAY_TASK_ID) ────────────────
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS_FILE")
if [ -z "$LINE" ]; then
    echo "ERROR: No line ${SLURM_ARRAY_TASK_ID} in ${JOBS_FILE}"
    exit 1
fi

JOB_ID=$(echo "$LINE"       | cut -f1)
SEED=$(echo "$LINE"         | cut -f2)
MODEL_KEY=$(echo "$LINE"    | cut -f3)
RESULTS_SUBDIR=$(echo "$LINE" | cut -f4)
SIMILAR_LANG=$(echo "$LINE" | cut -f5)
MODEL_TYPE=$(echo "$LINE"   | cut -f6)
SPLIT=$(echo "$LINE"        | cut -f7)
DIRECTION=$(echo "$LINE"    | cut -f8)

# ── Map model_key → local model path ─────────────────────────────────────────
case "$MODEL_KEY" in
    nllb-600m)  MODEL_PATH="${MODELS_DIR}/nllb-200-distilled-600M" ;;
    mbart-50)   MODEL_PATH="${MODELS_DIR}/mbart-large-50" ;;
    byt5-base)  MODEL_PATH="${MODELS_DIR}/byt5-base" ;;
    *)          echo "Unknown model_key: $MODEL_KEY"; exit 1 ;;
esac

# ── Data dir for this split ───────────────────────────────────────────────────
DATA_DIR="${DATA_BASE_DIR}/${SPLIT}"

echo "========================================"
echo "SLURM Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Job ID:           ${JOB_ID}"
echo "Seed:             ${SEED}"
echo "Model key:        ${MODEL_KEY} -> ${MODEL_PATH}"
echo "Model type:       ${MODEL_TYPE}"
echo "Similar lang:     ${SIMILAR_LANG}"
echo "Split:            ${SPLIT}"
echo "Direction:        ${DIRECTION}"
echo "Data dir:         ${DATA_DIR}"
echo "Results subdir:   ${RESULTS_SUBDIR}"
echo "========================================"

# ── Sanity-check model directory ─────────────────────────────────────────────
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model directory not found: ${MODEL_PATH}"
    echo "  Run download_models.sh first."
    exit 1
fi

# ── Sanity-check data directory ──────────────────────────────────────────────
for f in train.tsv val.tsv test.tsv; do
    if [ ! -f "${DATA_DIR}/${f}" ]; then
        echo "ERROR: Data file not found: ${DATA_DIR}/${f}"
        echo "  Run prepare_splits.py locally and scp the splits to ${DATA_BASE_DIR}/"
        exit 1
    fi
done

# ── Skip if results already exist ────────────────────────────────────────────
RESULT_FILE="${RESULTS_DIR}/${SPLIT}/${RESULTS_SUBDIR}/${DIRECTION}/seed${SEED}/test_metrics.json"
if [ -f "$RESULT_FILE" ]; then
    echo "SKIP: Results already exist at ${RESULT_FILE}"
    exit 0
fi

# ── Create output directory ───────────────────────────────────────────────────
mkdir -p "$(dirname "$RESULT_FILE")"

# ── Run training in container ─────────────────────────────────────────────────
apptainer exec --nv \
    --bind "${DATA_DIR}:/data:ro" \
    --bind "${MODEL_PATH}:/model:ro" \
    --bind "${RESULTS_DIR}:/results" \
    --bind "${TRAIN_SCRIPT}:/train.py:ro" \
    "$CONTAINER" \
    python3 /train.py \
        --model-type   "$MODEL_TYPE" \
        --seed         "$SEED" \
        --model-path   /model \
        --data-dir     /data \
        --results-dir  "/results/${SPLIT}/${RESULTS_SUBDIR}" \
        --similar-lang "$SIMILAR_LANG" \
        --direction    "$DIRECTION"

echo "Job ${JOB_ID} (split=${SPLIT}, direction=${DIRECTION}, model=${MODEL_KEY}, seed=${SEED}) completed."
