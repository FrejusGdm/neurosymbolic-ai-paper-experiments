"""
upload_to_hf.py — Upload the French-Adja parallel corpus to Hugging Face Hub.

Usage:
    huggingface-cli login   # first time only
    python lrec-dataset-paper/upload_to_hf.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value

REPO_ID = "JosueG/french-adja-parallel-corpus"
BASE_DIR = Path(__file__).resolve().parent

# Paths
SPLITS_DIR = BASE_DIR / "data" / "splits" / "random"
README_PATH = BASE_DIR / "dataset_card.md"


def load_split(path):
    """Load a TSV split file (no header, two columns: french \\t adja)."""
    df = pd.read_csv(path, sep="\t", header=None, names=["fr", "adj"],
                      dtype=str, na_filter=False)
    return df


def main():
    print(f"Loading splits from: {SPLITS_DIR}")

    train = load_split(SPLITS_DIR / "train.tsv")
    val = load_split(SPLITS_DIR / "val.tsv")
    test = load_split(SPLITS_DIR / "test.tsv")

    print(f"  Train: {len(train):,} pairs")
    print(f"  Val:   {len(val):,} pairs")
    print(f"  Test:  {len(test):,} pairs")
    print(f"  Total: {len(train) + len(val) + len(test):,} pairs")

    features = Features({
        "fr": Value("string"),
        "adj": Value("string"),
    })

    ds = DatasetDict({
        "train": Dataset.from_pandas(train, features=features, preserve_index=False),
        "validation": Dataset.from_pandas(val, features=features, preserve_index=False),
        "test": Dataset.from_pandas(test, features=features, preserve_index=False),
    })

    # Read dataset card
    if README_PATH.exists():
        readme_content = README_PATH.read_text(encoding="utf-8")
        print(f"  Dataset card: {len(readme_content):,} chars")
    else:
        print(f"  WARNING: No dataset card found at {README_PATH}")
        readme_content = None

    print(f"\nPushing to: https://huggingface.co/datasets/{REPO_ID}")
    ds.push_to_hub(
        REPO_ID,
        private=False,
        commit_message="Initial upload: 10,000 French-Adja parallel sentence pairs",
    )

    # Upload the dataset card separately (push_to_hub doesn't handle README)
    if readme_content:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(README_PATH),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
        print("  Dataset card uploaded.")

    print(f"\nDone! View at: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
