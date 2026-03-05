#!/bin/bash
#SBATCH -J nmt-train
#SBATCH -o /dartfs/rc/lab/R/RCoto/godeme/adja-nmt-hpc/logs/nmt-%A_%a.out
#SBATCH -e /dartfs/rc/lab/R/RCoto/godeme/adja-nmt-hpc/logs/nmt-%A_%a.err
#SBATCH -p gpuq
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_3g.40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=41-699
# NOTE: Lines 1-40 in jobs.tsv are already-completed NLLB-1.3B jobs.
#       Adjust --array range to skip completed or run subsets.
#       e.g. --array=41-233   = remaining 1.3B only
#            --array=234-466  = mBART French-init only
#            --array=467-699  = mBART Random-init only
#            --array=41-699   = all remaining (default)

set -euo pipefail

# ── Paths ──
# NOTE: Cannot use $(dirname "$0") — SLURM copies the script to
# /var/spool/slurmd/ so $0 points there, not the submission dir.
HPC_DIR="/dartfs/rc/lab/R/RCoto/godeme/adja-nmt-hpc"
JOBS_FILE="${HPC_DIR}/jobs.tsv"
TRAIN_SCRIPT="${HPC_DIR}/hf_job_train_hpc.py"
CONTAINER="${HPC_DIR}/nmt-training.sif"
DATA_DIR="${HPC_DIR}/data"
MODELS_DIR="${HPC_DIR}/models"
RESULTS_DIR="${HPC_DIR}/results"

# ── Read job params from jobs.tsv (line $SLURM_ARRAY_TASK_ID) ──
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS_FILE")
if [ -z "$LINE" ]; then
    echo "ERROR: No line ${SLURM_ARRAY_TASK_ID} in ${JOBS_FILE}"
    exit 1
fi

JOB_ID=$(echo "$LINE" | cut -f1)
EXPERIMENT=$(echo "$LINE" | cut -f2)
CONDITION=$(echo "$LINE" | cut -f3)
SEED=$(echo "$LINE" | cut -f4)
MODEL_KEY=$(echo "$LINE" | cut -f5)
RESULTS_SUBDIR=$(echo "$LINE" | cut -f6)
SIMILAR_LANG=$(echo "$LINE" | cut -f7)

# Map model_key to local model path
case "$MODEL_KEY" in
    nllb-1.3b)     MODEL_PATH="${MODELS_DIR}/nllb-200-1.3B" ;;
    mbart-50)      MODEL_PATH="${MODELS_DIR}/mbart-large-50" ;;
    *)             echo "Unknown model_key: $MODEL_KEY"; exit 1 ;;
esac

echo "========================================"
echo "SLURM Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Job ID:           ${JOB_ID}"
echo "Experiment:       ${EXPERIMENT}"
echo "Condition:        ${CONDITION}"
echo "Seed:             ${SEED}"
echo "Model:            ${MODEL_KEY} -> ${MODEL_PATH}"
echo "Results subdir:   ${RESULTS_SUBDIR}"
echo "Similar lang:     ${SIMILAR_LANG}"
echo "========================================"

# ── Skip if results already exist ──
RESULT_FILE="${RESULTS_DIR}/${RESULTS_SUBDIR}/${EXPERIMENT}/${CONDITION}/seed${SEED}/test_metrics.json"
if [ -f "$RESULT_FILE" ]; then
    echo "SKIP: Results already exist at ${RESULT_FILE}"
    exit 0
fi

# ── Create output directory ──
mkdir -p "$(dirname "$RESULT_FILE")"

# ── Run training in container ──
apptainer exec --nv \
    --bind "${DATA_DIR}:/data:ro" \
    --bind "${MODEL_PATH}:/model:ro" \
    --bind "${RESULTS_DIR}:/results" \
    --bind "${TRAIN_SCRIPT}:/train.py:ro" \
    "$CONTAINER" \
    python3 /train.py \
    --experiment "$EXPERIMENT" \
    --condition "$CONDITION" \
    --seed "$SEED" \
    --model-path /model \
    --data-dir /data \
    --results-dir "/results/${RESULTS_SUBDIR}" \
    --similar-lang "$SIMILAR_LANG"

echo "Job ${JOB_ID} completed successfully."
