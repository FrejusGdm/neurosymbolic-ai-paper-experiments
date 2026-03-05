"""
collect_results.py — Aggregate test_metrics.json files from LREC baseline runs.

Walks the results directory (scp'd from HPC) and produces:
  - all_results.csv : one row per (split, model, direction, seed)
  - summary to stdout: mean ± std per (split, model, direction) for BLEU, chrF, chrF++

Usage:
    python collect_results.py --results-dir /path/to/results [--output all_results.csv]

Expected directory layout (from submit_array.sbatch):
    results/
      random/nllb-600m/fr2adj/seed42/test_metrics.json
      random/nllb-600m/fr2adj/seed123/test_metrics.json
      ...
      random/nllb-600m/adj2fr/seed42/test_metrics.json
      ...
      stratified/nllb-600m/fr2adj/seed42/test_metrics.json
      ...
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict


METRIC_COLS = ["test_bleu", "test_chrf", "test_chrfpp", "test_ter"]
META_COLS   = ["split", "model_key", "direction", "seed", "train_size", "val_size",
               "actual_steps", "best_val_chrf", "training_time_seconds", "timestamp"]


def find_metrics(results_dir):
    """Walk results/ and return list of dicts with split/model_key/direction/seed/metrics."""
    rows = []
    for root, dirs, files in os.walk(results_dir):
        if "test_metrics.json" not in files:
            continue
        path = os.path.join(root, "test_metrics.json")
        try:
            with open(path) as f:
                m = json.load(f)
        except Exception as e:
            print(f"  WARNING: could not read {path}: {e}")
            continue

        # Infer from directory structure:
        #   results/{split}/{model_key}/{direction}/seed{N}/test_metrics.json
        rel = os.path.relpath(root, results_dir)
        parts = rel.replace("\\", "/").split("/")
        if len(parts) < 4:
            print(f"  WARNING: unexpected path structure: {rel}")
            continue

        split     = parts[0]
        model_key = parts[1]
        direction = parts[2]  # "fr2adj" or "adj2fr"
        seed_str  = parts[3]  # e.g. "seed42"
        seed      = int(seed_str.replace("seed", "")) if seed_str.startswith("seed") else -1

        rows.append({
            "split":     split,
            "model_key": model_key,
            "direction": direction,
            "seed":      seed,
            **{k: m.get(k, "") for k in METRIC_COLS},
            "train_size":              m.get("train_size", ""),
            "val_size":                m.get("val_size", ""),
            "actual_steps":            m.get("actual_steps", ""),
            "best_val_chrf":           m.get("best_val_chrf", ""),
            "training_time_seconds":   m.get("training_time_seconds", ""),
            "timestamp":               m.get("timestamp", ""),
        })

    return sorted(rows, key=lambda r: (r["split"], r["direction"], r["model_key"], r["seed"]))


def print_summary(rows):
    """Print mean ± std per (split, direction, model_key) for BLEU, chrF, chrF++."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["direction"], row["model_key"])].append(row)

    print("\n" + "=" * 82)
    print(f"{'Split':<12} {'Dir':<8} {'Model':<14} {'N':>2}  {'BLEU':>8}  {'chrF':>8}  {'chrF++':>8}  {'TER':>8}")
    print("-" * 82)

    for (split, direction, model_key), group in sorted(grouped.items()):
        def mean_std(metric):
            vals = [r[metric] for r in group if isinstance(r[metric], (int, float))]
            if not vals:
                return "  -"
            mean = sum(vals) / len(vals)
            std  = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            return f"{mean:5.1f}±{std:.1f}"

        print(
            f"{split:<12} {direction:<8} {model_key:<14} {len(group):>2}  "
            f"{mean_std('test_bleu'):>8}  {mean_std('test_chrf'):>8}  "
            f"{mean_std('test_chrfpp'):>8}  {mean_std('test_ter'):>8}"
        )

    print("=" * 82)


def main():
    parser = argparse.ArgumentParser(description="Aggregate LREC baseline results")
    parser.add_argument("--results-dir", required=True,
                        help="Path to results/ directory (scp'd from HPC)")
    parser.add_argument("--output", default="all_results.csv",
                        help="Output CSV path (default: all_results.csv)")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: results directory not found: {args.results_dir}")
        sys.exit(1)

    print(f"Scanning: {args.results_dir}")
    rows = find_metrics(args.results_dir)
    print(f"Found {len(rows)} result files")

    if not rows:
        print("No results found. Check that HPC jobs completed and results were scp'd.")
        sys.exit(1)

    # Write CSV
    all_cols = ["split", "model_key", "direction", "seed"] + METRIC_COLS + \
               ["train_size", "val_size", "actual_steps", "best_val_chrf",
                "training_time_seconds", "timestamp"]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written: {args.output}")

    # Print summary table
    print_summary(rows)

    # Count by split + direction
    for split in sorted({r["split"] for r in rows}):
        for direction in sorted({r["direction"] for r in rows}):
            n = sum(1 for r in rows if r["split"] == split and r["direction"] == direction)
            if n:
                print(f"  {split}/{direction}: {n} result files")

    expected = 60
    if len(rows) < expected:
        missing = expected - len(rows)
        print(f"\nWARNING: Expected {expected} results but found {len(rows)} ({missing} missing).")
        print("  Check SLURM logs: cat /dartfs/rc/lab/R/RCoto/godeme/lrec-baselines/logs/*.err")


if __name__ == "__main__":
    main()
