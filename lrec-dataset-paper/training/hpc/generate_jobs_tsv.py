"""
generate_jobs_tsv.py — Generate jobs.tsv for LREC dataset-paper SLURM array.

Produces 60 lines (no header — line N = SLURM task N):
  Lines  1-15 : fr2adj, random split    (3 models × 5 seeds)
  Lines 16-30 : fr2adj, stratified split (3 models × 5 seeds)
  Lines 31-45 : adj2fr, random split    (3 models × 5 seeds)
  Lines 46-60 : adj2fr, stratified split (3 models × 5 seeds)

Lines 1-30 (fr2adj) are ALREADY SUBMITTED — submit only --array=31-60 for adj2fr.

Columns (tab-separated):
  job_id  seed  model_key  results_subdir  similar_lang  model_type  split  direction

Usage:
    python generate_jobs_tsv.py [--output jobs.tsv]
"""

import argparse

SEEDS = [42, 123, 456, 789, 1024]

# (model_key, results_subdir, similar_lang, model_type)
MODELS = [
    ("nllb-600m",  "nllb-600m",  "ewe_Latn",  "nllb"),
    ("mbart-50",   "mbart-fr",   "fr_XX",      "mbart"),
    ("byt5-base",  "byt5",       "none",       "byt5"),
]

SPLITS = ["random", "stratified"]
DIRECTIONS = ["fr2adj", "adj2fr"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="jobs.tsv")
    args = parser.parse_args()

    lines = []
    job_id = 0

    for direction in DIRECTIONS:
        for split in SPLITS:
            for model_key, results_subdir, similar_lang, model_type in MODELS:
                for seed in SEEDS:
                    job_id += 1
                    lines.append(
                        f"{job_id}\t{seed}\t{model_key}\t{results_subdir}\t"
                        f"{similar_lang}\t{model_type}\t{split}\t{direction}"
                    )

    assert len(lines) == 60, f"Expected 60 jobs, got {len(lines)}"

    with open(args.output, "w") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Generated {args.output} with {len(lines)} jobs")
    print(f"  Lines  1-15 : fr2adj, random split     (3 models × 5 seeds)")
    print(f"  Lines 16-30 : fr2adj, stratified split  (3 models × 5 seeds)")
    print(f"  Lines 31-45 : adj2fr, random split     (3 models × 5 seeds)")
    print(f"  Lines 46-60 : adj2fr, stratified split  (3 models × 5 seeds)")
    print()
    print("  NOTE: Lines 1-30 already submitted (fr2adj).")
    print("  Submit adj2fr with: sbatch --array=31-60 submit_array.sbatch")
    print()
    print("Preview (first 3 lines):")
    for line in lines[:3]:
        print(f"  {line}")
    print("  ...")
    print("Preview (lines 31-33, adj2fr):")
    for line in lines[30:33]:
        print(f"  {line}")


if __name__ == "__main__":
    main()
