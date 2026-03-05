#!/bin/bash
# download_models.sh — Download models for the LREC dataset-paper baselines.
#
# Run this ONCE interactively on the HPC login node (NOT via SLURM).
# $(dirname "$0") is safe here because this script is run directly, not batch-submitted.
#
# Requires:
#   - nmt-training.sif already built in adja-nmt-hpc/
#   - HF_TOKEN environment variable set (or in ~/.bashrc)
#
# Usage:
#   bash download_models.sh [models_dir]
#
# What it downloads to lrec-baselines/models/:
#   nllb-200-distilled-600M  (~1.2 GB) — NEW
#   byt5-base                (~1.2 GB) — NEW
#   mbart-large-50           → symlink to adja-nmt-hpc/models/mbart-large-50 (already there)
#
# NOTE: If the symlink approach fails (filesystem doesn't support cross-dir symlinks),
#       uncomment the FALLBACK block below to download mBART fresh instead.

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
HPC_DIR="/dartfs/rc/lab/R/RCoto/godeme"
LREC_DIR="${HPC_DIR}/lrec-baselines"
COMPOSITION_DIR="${HPC_DIR}/adja-nmt-hpc"
CONTAINER="${COMPOSITION_DIR}/nmt-training.sif"
MODELS_DIR="${1:-${LREC_DIR}/models}"

mkdir -p "$MODELS_DIR"

if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found at ${CONTAINER}"
    echo "  The container from the composition paper experiment should already be built."
    exit 1
fi

echo "Downloading models to: ${MODELS_DIR}"
echo "Using container:       ${CONTAINER}"
echo ""

# ── NLLB-200-distilled-600M (~1.2 GB) ────────────────────────────────────────
echo "=== NLLB-200-distilled-600M ==="
if [ -d "${MODELS_DIR}/nllb-200-distilled-600M" ] && \
   [ -f "${MODELS_DIR}/nllb-200-distilled-600M/config.json" ]; then
    echo "  Already exists, skipping."
else
    apptainer exec --bind "${MODELS_DIR}:/models" "$CONTAINER" \
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/nllb-200-distilled-600M', local_dir='/models/nllb-200-distilled-600M')
"
    echo "  Done."
fi
echo ""

# ── ByT5-base (~1.2 GB) ──────────────────────────────────────────────────────
echo "=== google/byt5-base ==="
if [ -d "${MODELS_DIR}/byt5-base" ] && \
   [ -f "${MODELS_DIR}/byt5-base/config.json" ]; then
    echo "  Already exists, skipping."
else
    apptainer exec --bind "${MODELS_DIR}:/models" "$CONTAINER" \
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('google/byt5-base', local_dir='/models/byt5-base')
"
    echo "  Done."
fi
echo ""

# ── mBART-large-50 → symlink to existing download ────────────────────────────
echo "=== mBART-large-50 (symlink from adja-nmt-hpc/models/) ==="
MBART_TARGET="${COMPOSITION_DIR}/models/mbart-large-50"
MBART_LINK="${MODELS_DIR}/mbart-large-50"

if [ -e "$MBART_LINK" ]; then
    echo "  Already exists or symlinked, skipping."
elif [ -d "$MBART_TARGET" ]; then
    ln -s "$MBART_TARGET" "$MBART_LINK"
    echo "  Symlinked: ${MBART_LINK} -> ${MBART_TARGET}"
else
    echo "  WARNING: ${MBART_TARGET} not found. Falling back to fresh download."
    # FALLBACK: download fresh
    apptainer exec --bind "${MODELS_DIR}:/models" "$CONTAINER" \
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/mbart-large-50-many-to-many-mmt', local_dir='/models/mbart-large-50')
"
    echo "  Done (fresh download)."
fi
echo ""

echo "All models ready in ${MODELS_DIR}/"
ls -lh "${MODELS_DIR}/"
echo ""
echo "Next steps:"
echo "  1. Create logs dir: mkdir -p ${LREC_DIR}/logs"
echo "  2. Verify: sbatch --test-only --array=1-30 ${LREC_DIR}/submit_array.sbatch"
echo "  3. Submit: sbatch --array=1-30 ${LREC_DIR}/submit_array.sbatch"
