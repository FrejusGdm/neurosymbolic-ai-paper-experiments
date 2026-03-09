"""
train_lrec.py — LREC dataset-paper baseline training script.

Trains NLLB-200-distilled-600M, mBART-large-50, or ByT5-base on the Tatoeba 10K
French-Adja corpus (pre-split into train/val/test.tsv).

Adapted from experiments/training/hpc/hf_job_train_hpc.py — same training loop,
same preprocessing, same Adafactor optimizer. Key differences:
  - No --experiment/--condition flags (single dataset condition per paper).
  - Add --model-type flag (nllb | mbart | byt5) to select language-token logic.
  - ByT5: skip language token registration; character-level, max_length=512.
  - Simpler data layout: {data_dir}/train.tsv, val.tsv, test.tsv.
  - Results output: {results_dir}/{model_key}/seed{seed}/test_metrics.json.

Usage (inside Apptainer container):
    python3 train_lrec.py \
        --model-type nllb \
        --seed 42 \
        --model-path /model \
        --data-dir /data \
        --results-dir /results/nllb-600m \
        --similar-lang ewe_Latn

Data layout:
    {data_dir}/train.tsv
    {data_dir}/val.tsv
    {data_dir}/test.tsv
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
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LREC dataset-paper NMT baseline training")
    p.add_argument("--model-type", required=True, choices=["nllb", "mbart", "byt5"],
                   help="Model family: nllb | mbart | byt5")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--model-path", required=True, help="Local path to model weights")
    p.add_argument("--data-dir", required=True,
                   help="Directory containing train.tsv, val.tsv, test.tsv")
    p.add_argument("--results-dir", required=True,
                   help="Output directory for test_metrics.json")
    p.add_argument("--similar-lang", default=None,
                   help="Language token to copy embeddings from (NLLB/mBART only). "
                        "Use 'none' for random init.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max-length", type=int, default=None,
                   help="Token limit (default: 128 for nllb/mbart, 512 for byt5)")
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--num-beams", type=int, default=5)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--log-steps", type=int, default=50)
    p.add_argument("--direction", default="fr2adj", choices=["fr2adj", "adj2fr"],
                   help="Translation direction: fr2adj (French→Adja) or adj2fr (Adja→French)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Text preprocessing (identical to hf_job_train_hpc.py)
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
# Language token setup (NLLB / mBART only)
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
            if out_head is not None and hasattr(out_head, "weight") and \
               out_head.weight.shape[0] == emb_in.shape[0]:
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
    return [p[0] for p in batch], [p[1] for p in batch]


# ---------------------------------------------------------------------------
# Tokenization helpers — handles NLLB/mBART vs ByT5
# ---------------------------------------------------------------------------

def tokenize_batch(tokenizer, src_texts, tgt_texts, model_type, src_lang, tgt_lang,
                   max_length, device, direction="fr2adj"):
    """Returns (encoder_inputs, labels_tensor) for the given model type."""
    if model_type == "byt5":
        # ByT5 is character-level; no language tokens needed.
        # Use a direction-specific prefix so the model learns the correct mapping.
        task_prefix = "translate Adja to French: " if direction == "adj2fr" else "translate French to Adja: "
        prefixed = [f"{task_prefix}{s}" for s in src_texts]
        x = tokenizer(prefixed, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length).to(device)
        y = tokenizer(tgt_texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length).to(device)
    else:
        # NLLB / mBART: set src/tgt language codes.
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang
        x = tokenizer(src_texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length).to(device)
        with tokenizer.as_target_tokenizer():
            y = tokenizer(tgt_texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_length).to(device)

    labels = y.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return x, labels


def generate_batch(model, tokenizer, inputs, model_type, tgt_lang,
                   max_length, num_beams, device):
    """Run beam search generation; returns decoded strings."""
    gen_kwargs = dict(max_length=max_length, num_beams=num_beams, length_penalty=1.0)

    if model_type != "byt5":
        forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
        gen_kwargs["forced_bos_token_id"] = forced_bos

    generated = model.generate(**inputs, **gen_kwargs)
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def evaluate_val(model, tokenizer, val_data, device, model_type, src_lang, tgt_lang,
                 batch_size, max_length, num_beams, num_samples=200, direction="fr2adj"):
    model.eval()
    val_losses = []
    all_preds, all_refs = [], []

    n_batches = max(1, min(len(val_data), num_samples) // batch_size)
    sample_data = random.sample(val_data, min(len(val_data), n_batches * batch_size))

    with torch.no_grad():
        for i in range(0, len(sample_data), batch_size):
            batch = sample_data[i:i + batch_size]
            batch_src = [p[0] for p in batch]
            batch_tgt = [p[1] for p in batch]

            x, labels = tokenize_batch(tokenizer, batch_src, batch_tgt, model_type,
                                       src_lang, tgt_lang, max_length, device, direction)
            loss = model(**x, labels=labels).loss
            val_losses.append(loss.item())

            decoded = generate_batch(model, tokenizer, x, model_type, tgt_lang,
                                     max_length, num_beams, device)
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

def evaluate_test(model, tokenizer, test_file, model_type, src_lang, tgt_lang,
                  batch_size, max_length, num_beams, direction="fr2adj"):
    model.eval()
    device = model.device

    # TSV is always col1=French, col2=Adja. Swap for adj2fr.
    sources, references = [], []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                fr, aj = preproc(parts[0]), preproc(parts[1])
                if direction == "adj2fr":
                    sources.append(aj)
                    references.append(fr)
                else:
                    sources.append(fr)
                    references.append(aj)

    if model_type != "byt5":
        tokenizer.src_lang = src_lang

    predictions = []
    for i in range(0, len(sources), batch_size):
        batch_src = sources[i:i + batch_size]
        batch_tgt = references[i:i + batch_size]  # dummy, not used for generation

        if model_type == "byt5":
            task_prefix = "translate Adja to French: " if direction == "adj2fr" else "translate French to Adja: "
            prefixed = [f"{task_prefix}{s}" for s in batch_src]
            inputs = tokenizer(prefixed, return_tensors="pt", max_length=max_length,
                               truncation=True, padding=True).to(device)
        else:
            inputs = tokenizer(batch_src, return_tensors="pt", max_length=max_length,
                               truncation=True, padding=True).to(device)

        with torch.no_grad():
            decoded = generate_batch(model, tokenizer, inputs, model_type, tgt_lang,
                                     max_length, num_beams, device)
        predictions.extend(decoded)

    bleu    = sacrebleu.corpus_bleu(predictions, [references])
    chrf    = sacrebleu.corpus_chrf(predictions, [references])
    chrfpp  = sacrebleu.corpus_chrf(predictions, [references], word_order=2)
    ter     = sacrebleu.corpus_ter(predictions, [references])

    return {
        "test_bleu":           bleu.score,
        "test_chrf":           chrf.score,
        "test_chrfpp":         chrfpp.score,
        "test_ter":            ter.score,
        "test_bleu_signature": str(bleu),
        "test_n_samples":      len(predictions),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Set defaults based on model type (always fr2adj perspective first, swap below if needed)
    if args.model_type == "byt5":
        src_lang = None       # ByT5 doesn't use language codes
        tgt_lang = None
        similar_lang = "none"
        max_length = args.max_length or 512
    elif args.model_type == "mbart":
        src_lang = "fr_XX"
        tgt_lang = "aj_Latn"
        similar_lang = args.similar_lang or "fr_XX"
        max_length = args.max_length or 128
    else:  # nllb
        src_lang = "fra_Latn"
        tgt_lang = "aj_Latn"
        similar_lang = args.similar_lang or "ewe_Latn"
        max_length = args.max_length or 128

    # Swap src/tgt for the reverse direction
    if args.direction == "adj2fr" and args.model_type != "byt5":
        src_lang, tgt_lang = tgt_lang, src_lang

    print("=" * 60)
    print("LREC Dataset Paper — Baseline Training")
    print(f"  Model type:    {args.model_type}")
    print(f"  Seed:          {args.seed}")
    print(f"  Model path:    {args.model_path}")
    print(f"  Data dir:      {args.data_dir}")
    print(f"  Results dir:   {args.results_dir}")
    print(f"  Direction:     {args.direction}")
    print(f"  SRC_LANG:      {src_lang}")
    print(f"  TGT_LANG:      {tgt_lang}")
    print(f"  SIMILAR_LANG:  {similar_lang}")
    print(f"  LR:            {args.lr}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Max epochs:    {args.max_epochs}")
    print(f"  Max length:    {max_length}")
    print(f"  Warmup:        {args.warmup_steps} steps")
    print(f"  Eval every:    {args.eval_steps} steps")
    print(f"  Patience:      {args.patience} evals")
    print(f"  Date:          {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    set_seed(args.seed)

    # Verify data files exist
    train_file = os.path.join(args.data_dir, "train.tsv")
    val_file   = os.path.join(args.data_dir, "val.tsv")
    test_file  = os.path.join(args.data_dir, "test.tsv")
    for f in [train_file, val_file, test_file]:
        if not os.path.exists(f):
            print(f"ERROR: Data file not found: {f}")
            sys.exit(1)

    # Load model + tokenizer
    print(f"\nLoading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)

    # Register aj_Latn language token (NLLB / mBART only).
    # Always register aj_Latn regardless of direction — it must be in the vocab
    # whether it's the source or target language.
    if args.model_type != "byt5":
        aj_lang = "aj_Latn"
        init_lang_token(model, tokenizer, aj_lang, similar_lang)

    # Load data — TSV is always col1=French, col2=Adja.
    # For adj2fr swap the columns so (src, tgt) = (Adja, French).
    train_data = load_pairs(train_file)
    val_data   = load_pairs(val_file)
    if args.direction == "adj2fr":
        train_data = [(aj, fr) for fr, aj in train_data]
        val_data   = [(aj, fr) for fr, aj in val_data]
    print(f"  Train: {len(train_data)} pairs")
    print(f"  Val:   {len(val_data)} pairs")

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
            x, labels = tokenize_batch(tokenizer, batch_src, batch_tgt, args.model_type,
                                       src_lang, tgt_lang, max_length, device, args.direction)
            loss = model(**x, labels=labels).loss
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
                model, tokenizer, val_data, device, args.model_type,
                src_lang, tgt_lang, args.batch_size, max_length, args.num_beams,
                direction=args.direction
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

    # Test evaluation
    print("\nEvaluating on test set...")
    test_metrics = evaluate_test(
        model, tokenizer, test_file, args.model_type, src_lang, tgt_lang,
        args.batch_size, max_length, args.num_beams, args.direction
    )
    test_metrics.update({
        "training_time_seconds": training_time,
        "model_type":            args.model_type,
        "model_path":            args.model_path,
        "seed":                  args.seed,
        "train_size":            len(train_data),
        "val_size":              len(val_data),
        "learning_rate":         args.lr,
        "batch_size":            args.batch_size,
        "max_epochs":            args.max_epochs,
        "max_length":            max_length,
        "actual_steps":          actual_steps,
        "best_val_chrf":         best_chrf,
        "warmup_steps":          args.warmup_steps,
        "src_lang":              src_lang,
        "tgt_lang":              tgt_lang,
        "similar_lang":          similar_lang,
        "data_dir":              args.data_dir,
        "timestamp":             datetime.now(timezone.utc).isoformat(),
        "platform":              "dartmouth-hpc",
        "direction":             args.direction,
    })

    print(f"\nTest metrics:")
    print(f"  BLEU:   {test_metrics['test_bleu']:.1f}")
    print(f"  chrF:   {test_metrics['test_chrf']:.1f}")
    print(f"  chrF++: {test_metrics['test_chrfpp']:.1f}")
    print(f"  TER:    {test_metrics['test_ter']:.1f}")

    # Save
    result_path = os.path.join(args.results_dir, args.direction, f"seed{args.seed}", "test_metrics.json")
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nResults saved to: {result_path}")
    print("Done.")


if __name__ == "__main__":
    main()
