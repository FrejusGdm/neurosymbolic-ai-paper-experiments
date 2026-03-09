"""
hf_job_train_hpc.py — HPC-adapted training script for Dartmouth SLURM cluster.

Identical training logic to hf_job_train.py but reads data from local paths
and writes results to local filesystem (no HuggingFace Hub dependency).

Usage (inside Apptainer container):
    python3 hf_job_train_hpc.py \
        --experiment exp1 \
        --condition RANDOM-10K \
        --seed 42 \
        --model-path /model \
        --data-dir /data \
        --results-dir /results/nllb-1.3b \
        --similar-lang ewe_Latn

Data layout expected:
    {data_dir}/{experiment}/{condition}/train.tsv
    {data_dir}/{experiment}/{condition}/val.tsv
    {data_dir}/shared/test.tsv

Results output:
    {results_dir}/{experiment}/{condition}/seed{seed}/test_metrics.json
"""

import argparse
import gc
import json
import os
import random
import re
import sys
import time
import typing as tp
import unicodedata
from datetime import datetime, timezone

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import sacrebleu
import torch
from sacremoses import MosesPunctNormalizer
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
    set_seed,
)
from transformers.optimization import Adafactor


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="NMT training for HPC")
    p.add_argument("--experiment", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--model-path", required=True, help="Local path to model directory")
    p.add_argument("--data-dir", required=True, help="Root dir with {experiment}/{condition}/ splits")
    p.add_argument("--results-dir", required=True, help="Output dir for test_metrics.json")
    p.add_argument("--similar-lang", default=None, help="Lang token to copy embeddings from, or 'none'")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--num-beams", type=int, default=5)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--log-steps", type=int, default=50)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Text preprocessing (from NLLB pretraining pipeline)
# ---------------------------------------------------------------------------

_mpn = MosesPunctNormalizer(lang="en")
_mpn.substitutions = [(re.compile(r), sub) for r, sub in _mpn.substitutions]


def _get_non_printing_char_replacer(replace_by: str = " ") -> tp.Callable[[str], str]:
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }
    def replace_non_printing_char(line: str) -> str:
        return line.translate(non_printable_map)
    return replace_non_printing_char


_replace_nonprint = _get_non_printing_char_replacer(" ")


def preproc(text: str) -> str:
    clean = _mpn.normalize(text)
    clean = _replace_nonprint(clean)
    clean = unicodedata.normalize("NFKC", clean)
    return clean


# ---------------------------------------------------------------------------
# Language token setup
# ---------------------------------------------------------------------------

def fix_tokenizer(tokenizer, new_lang="aj_Latn"):
    extra = list(getattr(tokenizer, "additional_special_tokens", []))
    if new_lang not in extra:
        tokenizer.add_special_tokens({"additional_special_tokens": extra + [new_lang]})

    new_id = tokenizer.convert_tokens_to_ids(new_lang)

    if hasattr(tokenizer, "lang_code_to_id") and isinstance(tokenizer.lang_code_to_id, dict):
        tokenizer.lang_code_to_id[new_lang] = new_id
    if hasattr(tokenizer, "id_to_lang_code") and isinstance(tokenizer.id_to_lang_code, dict):
        tokenizer.id_to_lang_code[new_id] = new_lang

    init = getattr(tokenizer, "init_kwargs", {})
    if isinstance(init.get("lang_code_to_id", None), dict):
        init["lang_code_to_id"][new_lang] = new_id
    if isinstance(init.get("id_to_lang_code", None), dict):
        init["id_to_lang_code"][str(new_id)] = new_lang

    return new_id


def init_lang_token(model, tokenizer, new_lang="aj_Latn", similar_lang="ewe_Latn"):
    fix_tokenizer(tokenizer, new_lang)
    model.resize_token_embeddings(len(tokenizer))

    added_token_id = tokenizer.convert_tokens_to_ids(new_lang)
    assert added_token_id != tokenizer.unk_token_id, f"{new_lang} was not registered"

    if similar_lang.lower() == "none":
        print(f"Initialized {new_lang} with random embedding (no similar lang)")
    else:
        similar_lang_id = tokenizer.convert_tokens_to_ids(similar_lang)
        assert similar_lang_id != tokenizer.unk_token_id, f"{similar_lang} not found in tokenizer"

        with torch.no_grad():
            emb_in = model.get_input_embeddings().weight
            emb_in[added_token_id].copy_(emb_in[similar_lang_id])

            out_head = model.get_output_embeddings()
            if out_head is not None and hasattr(out_head, "weight") and out_head.weight.shape[0] == emb_in.shape[0]:
                out_head.weight[added_token_id].copy_(out_head.weight[similar_lang_id])

        print(f"Initialized {new_lang} from {similar_lang} (id {added_token_id} <- {similar_lang_id})")

    model.config.forced_bos_token_id = None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pairs(file_path):
    pairs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                pairs.append((preproc(parts[0]), preproc(parts[1])))
    return pairs


def get_batch(data, batch_size):
    batch = random.choices(data, k=batch_size)
    src_texts = [p[0] for p in batch]
    tgt_texts = [p[1] for p in batch]
    return src_texts, tgt_texts


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def evaluate_val(model, tokenizer, val_data, device, src_lang, tgt_lang,
                 batch_size, max_length, num_beams, num_samples=200):
    model.eval()
    val_losses = []
    all_preds = []
    all_refs = []

    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)

    n_batches = max(1, min(len(val_data), num_samples) // batch_size)
    sample_data = random.sample(val_data, min(len(val_data), n_batches * batch_size))

    with torch.no_grad():
        for i in range(0, len(sample_data), batch_size):
            batch = sample_data[i:i + batch_size]
            batch_src = [p[0] for p in batch]
            batch_tgt = [p[1] for p in batch]

            tokenizer.src_lang = src_lang
            tokenizer.tgt_lang = tgt_lang
            x = tokenizer(batch_src, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_length).to(device)

            with tokenizer.as_target_tokenizer():
                y = tokenizer(batch_tgt, return_tensors="pt", padding=True,
                              truncation=True, max_length=max_length).to(device)

            labels = y.input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100

            loss = model(**x, labels=labels).loss
            val_losses.append(loss.item())

            generated = model.generate(
                **x,
                forced_bos_token_id=forced_bos,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=1.0,
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            all_preds.extend(decoded)
            all_refs.extend(batch_tgt)

    avg_loss = np.mean(val_losses)
    bleu = sacrebleu.corpus_bleu(all_preds, [all_refs]).score
    chrf = sacrebleu.corpus_chrf(all_preds, [all_refs]).score

    model.train()
    return avg_loss, bleu, chrf


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------

def evaluate_test(model, tokenizer, test_file, src_lang, tgt_lang,
                  batch_size, max_length, num_beams):
    model.eval()
    device = model.device

    sources, references = [], []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                sources.append(preproc(parts[0]))
                references.append(preproc(parts[1]))

    predictions = []
    tokenizer.src_lang = src_lang
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)

    for i in range(0, len(sources), batch_size):
        batch_src = sources[i:i + batch_size]
        inputs = tokenizer(
            batch_src, return_tensors="pt", max_length=max_length,
            truncation=True, padding=True,
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=1.0,
            )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        predictions.extend(decoded)

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references])
    chrfpp = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
    ter = sacrebleu.corpus_ter(predictions, [references])

    metrics = {
        "test_bleu": bleu.score,
        "test_chrf": chrf.score,
        "test_chrfpp": chrfpp.score,
        "test_ter": ter.score,
        "test_bleu_signature": str(bleu),
        "test_n_samples": len(predictions),
    }

    # ROUGE-L (optional — requires rouge-score in container)
    try:
        from rouge_score import rouge_scorer as rouge_scorer_lib
        scorer = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=False)
        rouge_l_scores = [scorer.score(ref, pred)["rougeL"].fmeasure
                          for pred, ref in zip(predictions, references)]
        metrics["test_rougeL"] = sum(rouge_l_scores) / len(rouge_l_scores) * 100.0
    except ImportError:
        pass  # rouge-score not available in this container

    # Perplexity (NLL-based)
    try:
        import math as _math
        total_nll = 0.0
        total_tokens = 0
        model.eval()
        tokenizer.src_lang = tgt_lang
        for i in range(0, len(sources), batch_size):
            batch_src = sources[i:i + batch_size]
            batch_ref = references[i:i + batch_size]
            tgt_enc = tokenizer(
                batch_ref, return_tensors="pt", max_length=max_length,
                truncation=True, padding=True,
            ).to(device)
            tokenizer.src_lang = src_lang
            src_enc = tokenizer(
                batch_src, return_tensors="pt", max_length=max_length,
                truncation=True, padding=True,
            ).to(device)
            tokenizer.src_lang = tgt_lang  # restore for next iter
            labels = tgt_enc["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            with torch.no_grad():
                out = model(**src_enc, labels=labels)
            n_tokens = (labels != -100).sum().item()
            total_nll += out.loss.item() * n_tokens
            total_tokens += n_tokens
        tokenizer.src_lang = src_lang  # restore after loop
        if total_tokens > 0:
            metrics["test_perplexity"] = _math.exp(total_nll / total_tokens)
    except Exception:
        pass  # don't let perplexity failure break the whole job

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Auto-detect model family from path/name
    model_name_lower = args.model_path.lower()
    if "mbart" in model_name_lower:
        src_default, sim_default = "fr_XX", "fr_XX"
    else:
        src_default, sim_default = "fra_Latn", "ewe_Latn"

    src_lang = src_default
    tgt_lang = "aj_Latn"
    similar_lang = args.similar_lang if args.similar_lang else sim_default

    print("=" * 60)
    print("HPC Training Script")
    print(f"  Experiment:    {args.experiment}")
    print(f"  Condition:     {args.condition}")
    print(f"  Seed:          {args.seed}")
    print(f"  Model path:    {args.model_path}")
    print(f"  Data dir:      {args.data_dir}")
    print(f"  Results dir:   {args.results_dir}")
    print(f"  SRC_LANG:      {src_lang}")
    print(f"  TGT_LANG:      {tgt_lang}")
    print(f"  SIMILAR_LANG:  {similar_lang}")
    print(f"  LR:            {args.lr}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Max epochs:    {args.max_epochs}")
    print(f"  Warmup:        {args.warmup_steps} steps")
    print(f"  Eval every:    {args.eval_steps} steps")
    print(f"  Patience:      {args.patience} evals")
    print(f"  Date:          {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    set_seed(args.seed)

    # Load data from local paths
    train_file = os.path.join(args.data_dir, args.experiment, args.condition, "train.tsv")
    val_file = os.path.join(args.data_dir, args.experiment, args.condition, "val.tsv")
    test_file = os.path.join(args.data_dir, "shared", "test.tsv")

    for f in [train_file, val_file, test_file]:
        if not os.path.exists(f):
            print(f"ERROR: Data file not found: {f}")
            sys.exit(1)

    # Load model + tokenizer from local path
    print(f"\nLoading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)

    # Register aj_Latn token
    init_lang_token(model, tokenizer, tgt_lang, similar_lang)

    # Load data
    train_data = load_pairs(train_file)
    val_data = load_pairs(val_file)
    print(f"  Train: {len(train_data)} pairs")
    print(f"  Val:   {len(val_data)} pairs")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  Device: {device}")

    # Optimizer + scheduler
    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=args.lr,
        clip_threshold=1.0,
        weight_decay=1e-3,
    )

    max_steps = (len(train_data) // args.batch_size) * args.max_epochs
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    print(f"\n  Max training steps: {max_steps}")
    print(f"\nStarting training...")

    # Training loop
    model.train()
    start_time = time.time()
    losses = []
    best_chrf = -1.0
    best_model_state = None
    evals_without_improvement = 0
    actual_steps = 0

    for step in range(1, max_steps + 1):
        batch_src, batch_tgt = get_batch(train_data, args.batch_size)

        try:
            tokenizer.src_lang = src_lang
            tokenizer.tgt_lang = tgt_lang
            x = tokenizer(batch_src, return_tensors="pt", padding=True,
                          truncation=True, max_length=args.max_length).to(device)

            with tokenizer.as_target_tokenizer():
                y = tokenizer(batch_tgt, return_tensors="pt", padding=True,
                              truncation=True, max_length=args.max_length).to(device)

            y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100

            loss = model(**x, labels=y.input_ids).loss
            loss.backward()
            losses.append(loss.item())

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        except RuntimeError as e:
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Step {step} | RuntimeError: {e}")
            continue

        actual_steps = step

        if step % args.log_steps == 0:
            avg_loss = np.mean(losses[-args.log_steps:])
            elapsed = time.time() - start_time
            print(f"Step {step}/{max_steps} | loss: {avg_loss:.4f} | elapsed: {elapsed:.0f}s")

        if step % args.eval_steps == 0:
            val_loss, val_bleu, val_chrf = evaluate_val(
                model, tokenizer, val_data, device, src_lang, tgt_lang,
                args.batch_size, args.max_length, args.num_beams
            )
            print(f"  Val loss: {val_loss:.4f} | Val BLEU: {val_bleu:.1f} | Val chrF: {val_chrf:.1f}")

            if val_chrf > best_chrf:
                best_chrf = val_chrf
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                evals_without_improvement = 0
                print(f"  -> New best chrF: {best_chrf:.1f}")
            else:
                evals_without_improvement += 1
                print(f"  -> No improvement ({evals_without_improvement}/{args.patience})")

            if evals_without_improvement >= args.patience:
                print(f"\nEarly stopping at step {step} (patience={args.patience})")
                break

            model.train()

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.0f}s ({training_time / 3600:.1f}h)")
    print(f"  Actual steps: {actual_steps}")
    print(f"  Best val chrF: {best_chrf:.1f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
        print("Restored best model checkpoint.")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_test(
        model, tokenizer, test_file, src_lang, tgt_lang,
        args.batch_size, args.max_length, args.num_beams
    )
    test_metrics["training_time_seconds"] = training_time
    test_metrics["experiment"] = args.experiment
    test_metrics["condition"] = args.condition
    test_metrics["seed"] = args.seed
    test_metrics["model"] = args.model_path
    test_metrics["train_size"] = len(train_data)
    test_metrics["val_size"] = len(val_data)
    test_metrics["learning_rate"] = args.lr
    test_metrics["batch_size"] = args.batch_size
    test_metrics["max_epochs"] = args.max_epochs
    test_metrics["actual_steps"] = actual_steps
    test_metrics["best_val_chrf"] = best_chrf
    test_metrics["warmup_steps"] = args.warmup_steps
    test_metrics["src_lang"] = src_lang
    test_metrics["tgt_lang"] = tgt_lang
    test_metrics["similar_lang"] = similar_lang
    test_metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
    test_metrics["platform"] = "dartmouth-hpc"

    print(f"\nTest metrics:")
    print(f"  BLEU:   {test_metrics['test_bleu']:.1f}")
    print(f"  chrF:   {test_metrics['test_chrf']:.1f}")
    print(f"  chrF++: {test_metrics['test_chrfpp']:.1f}")
    print(f"  TER:    {test_metrics['test_ter']:.1f}")

    # Save results to local filesystem
    result_dir = os.path.join(args.results_dir, args.experiment, args.condition, f"seed{args.seed}")
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, "test_metrics.json")

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nResults saved to: {result_path}")
    print("Done.")


if __name__ == "__main__":
    main()
