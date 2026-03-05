"""
prepare_splits.py — Create 80/10/10 train/val/test splits from the 10K Tatoeba FR-Adja corpus.

Produces TWO splits from the same 10K data:
  - random/     : straight shuffle → 80/10/10 slice
  - stratified/ : split within sentence-length bins (short ≤8 / medium 9-16 / long >16 words)
                  so all three sets share the same length distribution

Both splits write a split_stats.txt for inclusion as a paper table (addresses Reviewer #1).

Usage:
    python prepare_splits.py --input /path/to/cleaned_v2_normalized.csv --output-dir splits/

Input format: CSV with header row containing columns "French" and "Translation".
    ID,French,Translation,original_id,dataset_source
    1,Je mange.,ŋ du.,1,dataset1
    ...

Output format (TSV, no header):
    french_sentence\tadja_sentence
"""

import argparse
import csv
import os
import random
import sys


SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# Sentence-length bins (counted in whitespace-split French tokens)
SHORT_MAX  = 8
MEDIUM_MAX = 16
# long = > MEDIUM_MAX


def load_csv(path):
    """Load FR-Adja pairs from CSV (columns: French, Translation) or plain TSV."""
    pairs = []

    # Detect format: if first line has a comma and looks like a header, treat as CSV
    with open(path, encoding="utf-8") as f:
        first_line = f.readline()

    if "French" in first_line and "Translation" in first_line:
        # CSV with header
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for lineno, row in enumerate(reader, 2):
                fr = row.get("French", "").strip()
                aj = row.get("Translation", "").strip()
                if not fr or not aj:
                    print(f"  WARNING: skipping empty pair at line {lineno}")
                    continue
                pairs.append((fr, aj))
    else:
        # Fall back to plain TSV (tab-separated, no header)
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    print(f"  WARNING: skipping malformed line {lineno}: {repr(line)}")
                    continue
                pairs.append((parts[0].strip(), parts[1].strip()))

    return pairs


def write_split(pairs, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for fr, aj in pairs:
            f.write(f"{fr}\t{aj}\n")


def length_bin(fr_sentence):
    n = len(fr_sentence.split())
    if n <= SHORT_MAX:
        return "short"
    elif n <= MEDIUM_MAX:
        return "medium"
    else:
        return "long"


def split_list(lst, train_r, val_r):
    """Slice list into train/val/test proportions."""
    n = len(lst)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)
    return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]


def write_stats(train, val, test, out_dir):
    """Write split_stats.txt with length distribution per stratum × split."""
    lines = []
    lines.append(f"{'Split':<12} {'Total':>6} {'Short(≤8)':>10} {'Medium(9-16)':>13} {'Long(>16)':>10} {'Mean':>6} {'Median':>7} {'Max':>5} {'Vocab':>6}")
    lines.append("-" * 75)

    for split_name, pairs in [("train", train), ("val", val), ("test", test)]:
        lengths   = [len(fr.split()) for fr, _ in pairs]
        shorts    = sum(1 for l in lengths if l <= SHORT_MAX)
        mediums   = sum(1 for l in lengths if SHORT_MAX < l <= MEDIUM_MAX)
        longs     = sum(1 for l in lengths if l > MEDIUM_MAX)
        mean_l    = sum(lengths) / len(lengths) if lengths else 0
        sorted_l  = sorted(lengths)
        median_l  = sorted_l[len(sorted_l) // 2] if lengths else 0
        max_l     = max(lengths) if lengths else 0
        vocab     = len({tok for fr, _ in pairs for tok in fr.split()})
        lines.append(
            f"{split_name:<12} {len(pairs):>6} {shorts:>10} {mediums:>13} {longs:>10} "
            f"{mean_l:>6.1f} {median_l:>7} {max_l:>5} {vocab:>6}"
        )

    lines.append("")
    lines.append(f"Total pairs in dataset: {len(train) + len(val) + len(test)}")

    stats_path = os.path.join(out_dir, "split_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Stats: {stats_path}")
    for line in lines:
        print(f"    {line}")


def make_random_split(pairs, out_dir):
    """Shuffle all pairs and slice 80/10/10."""
    shuffled = pairs[:]
    random.seed(SEED)
    random.shuffle(shuffled)
    train, val, test = split_list(shuffled, TRAIN_RATIO, VAL_RATIO)

    write_split(train, os.path.join(out_dir, "train.tsv"))
    write_split(val,   os.path.join(out_dir, "val.tsv"))
    write_split(test,  os.path.join(out_dir, "test.tsv"))

    print(f"\n[random split]  train={len(train)}  val={len(val)}  test={len(test)}")
    write_stats(train, val, test, out_dir)


def make_stratified_split(pairs, out_dir):
    """Split within length bins so train/val/test have matching length distributions."""
    bins = {"short": [], "medium": [], "long": []}
    for pair in pairs:
        bins[length_bin(pair[0])].append(pair)

    all_train, all_val, all_test = [], [], []
    for bin_name in ("short", "medium", "long"):
        subset = bins[bin_name][:]
        random.seed(SEED)
        random.shuffle(subset)
        tr, vl, te = split_list(subset, TRAIN_RATIO, VAL_RATIO)
        all_train.extend(tr)
        all_val.extend(vl)
        all_test.extend(te)
        print(f"  {bin_name}: {len(subset)} total  → train={len(tr)} val={len(vl)} test={len(te)}")

    # Shuffle the concatenated splits so bins aren't contiguous
    random.seed(SEED)
    random.shuffle(all_train)
    random.shuffle(all_val)
    random.shuffle(all_test)

    write_split(all_train, os.path.join(out_dir, "train.tsv"))
    write_split(all_val,   os.path.join(out_dir, "val.tsv"))
    write_split(all_test,  os.path.join(out_dir, "test.tsv"))

    print(f"\n[stratified split]  train={len(all_train)}  val={len(all_val)}  test={len(all_test)}")
    write_stats(all_train, all_val, all_test, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Prepare random + stratified 80/10/10 splits")
    parser.add_argument("--input",      required=True, help="Raw 10K TSV (fr\\tadja)")
    parser.add_argument("--output-dir", required=True, help="Root output directory")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: input file not found: {args.input}")
        sys.exit(1)

    print(f"Loading pairs from: {args.input}")
    pairs = load_csv(args.input)
    print(f"  Loaded {len(pairs)} pairs")

    # Length distribution overview
    shorts  = sum(1 for fr, _ in pairs if length_bin(fr) == "short")
    mediums = sum(1 for fr, _ in pairs if length_bin(fr) == "medium")
    longs   = sum(1 for fr, _ in pairs if length_bin(fr) == "long")
    print(f"  Length bins: short={shorts} medium={mediums} long={longs}")

    random_dir      = os.path.join(args.output_dir, "random")
    stratified_dir  = os.path.join(args.output_dir, "stratified")

    print(f"\n--- Random split ---")
    make_random_split(pairs, random_dir)

    print(f"\n--- Stratified split (by sentence length) ---")
    make_stratified_split(pairs, stratified_dir)

    print(f"\nDone. Splits written to: {args.output_dir}/")
    print(f"  {random_dir}/train.tsv, val.tsv, test.tsv, split_stats.txt")
    print(f"  {stratified_dir}/train.tsv, val.tsv, test.tsv, split_stats.txt")


if __name__ == "__main__":
    main()
