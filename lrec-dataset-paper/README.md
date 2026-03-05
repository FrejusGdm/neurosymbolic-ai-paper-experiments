# LREC Dataset Paper — Baseline Experiments

This folder contains all scripts for retraining the MT baselines for the LREC
dataset paper revision. It is **completely separate** from the composition-paper
experiments in `experiments/`.

## Why retrain?

- The existing `shared/test.tsv` (1,455 pairs) belongs to the composition paper.
  The dataset paper needs its own 80/10/10 split of the full 10K Tatoeba corpus.
- Reviewer #1 asked whether splits are stratified. We run both random and
  length-stratified splits and report comparison statistics in the paper.
- Improved preprocessing pipeline gives higher BLEU — results are not comparable
  to the original submission without retraining.

## Models

| Model                    | Key        | Notes                          |
|--------------------------|------------|-------------------------------|
| NLLB-200-distilled-600M  | nllb-600m  | Ewe-init for aj_Latn token    |
| mBART-large-50 (fr-init) | mbart-fr   | French-init for aj_Latn token |
| ByT5-base                | byt5       | Character-level, no lang tokens |

5 seeds × 2 splits × 3 models = **30 HPC jobs** total.

## Quick Start

### Step 1 — Prepare data splits (local)

```bash
cd lrec-dataset-paper/data/
python prepare_splits.py \
    --input /path/to/raw_10k_fr_adja.tsv \
    --output-dir splits/
```

Produces `splits/random/` and `splits/stratified/`, each with
`train.tsv` (~8K), `val.tsv` (~1K), `test.tsv` (~1K), and `split_stats.txt`.

### Step 2 — Generate jobs.tsv (local)

```bash
cd lrec-dataset-paper/training/hpc/
python generate_jobs_tsv.py --output jobs.tsv
```

### Step 3 — Upload to HPC

```bash
HPC_BASE="f006g5b@discovery.dartmouth.edu:/dartfs/rc/lab/R/RCoto/godeme"

# Create remote directories
ssh f006g5b@discovery.dartmouth.edu \
    "mkdir -p /dartfs/rc/lab/R/RCoto/godeme/lrec-baselines/{data,models,results,logs}"

# Upload both data splits (run from lrec-dataset-paper/data/)
scp -r splits/random splits/stratified \
    ${HPC_BASE}/lrec-baselines/data/

# Upload scripts (run from lrec-dataset-paper/training/hpc/)
scp train_lrec.py jobs.tsv submit_array.sbatch download_models.sh \
    ${HPC_BASE}/lrec-baselines/
```

### Step 4 — Download models (on HPC login node)

```bash
# On the HPC:
cd /dartfs/rc/lab/R/RCoto/godeme/lrec-baselines/
bash download_models.sh
mkdir -p logs/
```

### Step 5 — Dry run + Submit

```bash
# Verify paths before submitting
sbatch --test-only --array=1-30 submit_array.sbatch

# Submit all 30 jobs
sbatch --array=1-30 submit_array.sbatch

# Monitor
squeue -u $USER
tail -f logs/lrec-<JOBID>_1.out
```

### Step 6 — Collect results (local, after jobs complete)

```bash
# Download results
scp -r f006g5b@discovery.dartmouth.edu:/dartfs/rc/lab/R/RCoto/godeme/lrec-baselines/results/ ./results/

# Aggregate
cd lrec-dataset-paper/evaluation/
python collect_results.py --results-dir ../results/ --output all_results.csv
```

## File Map

```
lrec-dataset-paper/
├── README.md                        ← this file
├── data/
│   └── prepare_splits.py            ← 80/10/10 random + stratified splits
├── training/
│   └── hpc/
│       ├── train_lrec.py            ← training script (NLLB, mBART, ByT5)
│       ├── generate_jobs_tsv.py     ← generates 30-line jobs.tsv
│       ├── submit_array.sbatch      ← SLURM array (hardcoded paths, --array=1-30)
│       └── download_models.sh       ← downloads NLLB-600M + ByT5-base
└── evaluation/
    └── collect_results.py           ← walk results/ → all_results.csv + summary
```

## HPC Path Reference

| Component          | HPC Path                                                    |
|--------------------|-------------------------------------------------------------|
| Working dir        | `/dartfs/rc/lab/R/RCoto/godeme/lrec-baselines/`             |
| Container (reused) | `/dartfs/rc/lab/R/RCoto/godeme/adja-nmt-hpc/nmt-training.sif` |
| Data               | `.../lrec-baselines/data/{random,stratified}/`              |
| Models             | `.../lrec-baselines/models/{nllb-200-distilled-600M,...}/`  |
| Results            | `.../lrec-baselines/results/{random,stratified}/{model}/`   |
| Logs               | `.../lrec-baselines/logs/lrec-<JOBID>_<TASK>.{out,err}`     |

## Addressing Reviewer Feedback

| Reviewer | Comment | Response |
|----------|---------|----------|
| #1 | Are splits stratified by vocabulary/length? | `split_stats.txt` shows length distribution per stratum — include as table in Section 3.3 |
| #1 | Report chrF++ (not just chrF) | `test_chrfpp` recorded in all `test_metrics.json` |
| #2 | How were Tatoeba sentences sampled? | Add to paper: random sample, deduplicated |
| #3 | ByT5 results not discussed | Add discussion of ByT5 vs NLLB comparison |
