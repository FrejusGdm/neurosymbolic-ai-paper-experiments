# Professor Feedback Revisions — 2026-03-06
## Last updated: 2026-03-08

## Status: IN PROGRESS — Paper fixes done; data prep done; jobs launching

---

## Quick Summary (read this first)

| # | Task | Status | Notes |
|---|------|--------|-------|
| 1 | Add chrF++ to 4 tables | ✅ DONE | Pasted into acl_latex.tex |
| 2 | Add COMET/BertScore explanation | ✅ DONE | §Evaluation + §Limitations |
| 3 | hf_job_train.py: add ROUGE-L + Perplexity | ✅ DONE | All new jobs will report these |
| 4 | NO-PAST-1K seeds 42 + 123 (HF Jobs) | ✅ LAUNCHED | Job IDs: 69acff9cc6af61df4cdb741c, 69ad004ec6af61df4cdb742a |
| 5 | Data prep script for new experiments | ✅ DONE | `experiments/data/prepare_new_experiments.py` |
| 6 | Additive ablation splits created locally | ✅ DONE | `experiments/data/splits/new_experiments/` |
| 7 | Combo splits created locally | ✅ DONE | `experiments/data/splits/new_experiments/` |
| 8 | Upload new splits to HF Hub | ⏳ PENDING USER | Run: `python experiments/data/prepare_new_experiments.py` |
| 9 | Launch 9 additive ablation jobs | ✅ LAUNCHED (HF Jobs) | Job IDs: 69adc38e…69adc3b5 + 2 more (rate-limited, auto-retry) |
| 10 | Launch 20 Table 9 combo jobs | ⏳ PENDING HPC submit | Via `submit_table9.sbatch` on Dartmouth Discovery |
| 11 | Update Table 11 (NO-PAST-1K std) | ⏳ PENDING jobs | After seeds 42+123 complete |
| 12 | Add appendix additive ablation table | ⏳ PENDING jobs | After 9 additive jobs complete |
| 13 | Add Table 9 new rows | ⏳ PENDING jobs | After 20 combo jobs complete |

---

## What YOU need to do right now

### Step 1 — Upload new data splits to HF Hub
```bash
cd /Users/josuegodeme/Downloads/neurosymbolic-ai-paper-experiments
export HF_TOKEN=hf_...   # your token
python experiments/data/prepare_new_experiments.py
```
Creates and uploads 7 new condition directories (3 additive + 4 combo) to `JosueG/adja-nmt-splits`.

### Step 2 — Launch 9 additive ablation jobs (HF Jobs) ✅ DONE
Already running — rate limiting is handled automatically by the launcher.

### Step 3 — Upload Table 9 data to HPC + submit SLURM array
```bash
# 3a. Upload the 4 new combo splits to HPC (from repo root)
rsync -av \
  experiments/data/splits/new_experiments/baselines/ \
  f006g5b@discovery.dartmouth.edu:/dartfs/rc/lab/R/RCoto/godeme/adja-nmt-hpc/data/baselines/

# 3b. Upload the new HPC scripts
scp experiments/training/hpc/jobs_table9.tsv \
    experiments/training/hpc/submit_table9.sbatch \
    experiments/training/hpc/hf_job_train_hpc.py \
    f006g5b@discovery.dartmouth.edu:/dartfs/rc/lab/R/RCoto/godeme/adja-nmt-hpc/

# 3c. SSH in and submit
ssh f006g5b@discovery.dartmouth.edu
cd /dartfs/rc/lab/R/RCoto/godeme/adja-nmt-hpc
sbatch submit_table9.sbatch
```

### Step 4 — After all jobs complete: sync results
```bash
# For HF Jobs results (NLLB-600M):
python experiments/evaluation/collect_hf_results.py
# For HPC jobs (if any ran there):
rsync -av --include="*/test_metrics.json" --include="*/" --exclude="*" \
  f006g5b@discovery.dartmouth.edu:/dartfs/rc/lab/R/RCoto/godeme/adja-nmt-hpc/results/ \
  experiments/results/hpc_new/
```

---

## Completed Changes (detail)

### ✅ 1. Added chrF++ to 4 tables (2026-03-06)
- `tab:scaling` — expanded to `{rccccccc}`, added chrF++ for Structured + Random
- `tab:module_loo` — expanded to `{lrrrrr}`, added chrF++ column with ±std
- `tab:structure_matters` — expanded to `{lrrrr}`, updated `\multicolumn{5}`
- `tab:pronoun` — expanded to `{llrrr}`, added chrF++ column
- Values from: `experiments/results/summary/all_results.csv` via `compute_chrfpp_for_tables.py`

### ✅ 2. Added COMET/BertScore explanation (2026-03-06)
- §Evaluation Metrics: added paragraph explaining why COMET/BertScore excluded (Adja absent from training data of those models)
- §Limitations: updated to reference the evaluation section
- Added ROUGE-L and Perplexity to reported metrics list

### ✅ 3. hf_job_train.py: ROUGE-L + Perplexity (2026-03-06)
- Added `rouge-score` to uv script dependencies
- `evaluate_test()` now computes `test_rougeL` and `test_perplexity`
- Perplexity: NLL-based, batched, uses `tokenizer.src_lang = TGT_LANG` (not deprecated `as_target_tokenizer`)
- All new training jobs (29 total) will report these metrics automatically

### ✅ 4. NO-PAST-1K jobs launched (2026-03-08)
- Seed 42: job ID `69acff9cc6af61df4cdb741c`
- Seed 123: job ID `69ad004ec6af61df4cdb742a`
- Once complete: check `JosueG/adja-nmt-results`, then update `tab:module_size_ctrl` row from `$20.1$` to `$xx.x \pm x.x$`

### ✅ 5. New data splits created (2026-03-08)
Script: `experiments/data/prepare_new_experiments.py`
Local output: `experiments/data/splits/new_experiments/`

**Additive ablation splits** (group-aware 90/10 split on `base_sentence_id`):
| Condition | Total | Train | Val |
|-----------|-------|-------|-----|
| ADD-M1M2 (M1+M2) | 1,448 | 1,292 | 156 |
| ADD-M1M2M3 (M1+M2+M3) | 2,183 | 1,944 | 239 |
| ADD-M1M2M3M4 (M1+M2+M3+M4) | 2,919 | 2,597 | 322 |

**Combo splits** (structured_train.tsv + baseline TSV, deduped, simple 90/10):
| Condition | Total | Train | Val |
|-----------|-------|-------|-----|
| STRUCT4K-TFIDF2K | 5,262 | 4,735 | 527 |
| STRUCT4K-LENGTH2K | 5,262 | 4,735 | 527 |
| STRUCT4K-VOCAB2K | 5,262 | 4,735 | 527 |
| STRUCT4K-ALL-BASELINES | 7,859 | 7,073 | 786 |

---

## Pending Changes (what happens after jobs complete)

### ⏳ Table 11 (tab:module_size_ctrl) — fix NO-PAST-1K std
After seeds 42+123 complete:
- Collect results → recompute mean±std for NO-PAST-1K
- Update LaTeX row from `$20.1$` to `$xx.x \pm x.x$`

### ⏳ Appendix: Additive Ablation Table
New table `tab:additive_ablation` showing cumulative gain as modules added:
```
ADD-M1 (= BASE-ONLY)     | ~815  train | BLEU | chrF++ | ROUGE-L | PPL
ADD-M1M2                 | ~1292 train | ...
ADD-M1M2M3               | ~1944 train | ...
ADD-M1M2M3M4             | ~2597 train | ...   ← expected largest jump
ADD-FULL (= FULL)        | ~3462 train | ...
```
Expected result: M4 (future tense) should produce the largest single-step jump.

### ⏳ Table 9 (tab:baselines) — new combo rows
After 20 STRUCT4K-* jobs complete, add rows:
```
Struct-4K + TF-IDF-2K    | ~4735 train | BLEU | chrF | chrF++ | ROUGE-L | PPL
Struct-4K + Length-2K    | ~4735 train | ...
Struct-4K + Vocab-2K     | ~4735 train | ...
Struct-4K + All-3-methods| ~7073 train | ...
```

---

## Overview of What Professor Requested

Professor provided feedback on the paper draft requiring:
1. **Consistency**: Add chrF++ everywhere it was missing (4 tables)
2. **Missing std**: Fix NO-PAST-1K row in Table 11 (only 1 seed ran)
3. **Metric justification**: Explain why COMET/BertScore not used
4. **New metrics**: Add ROUGE-L and Perplexity to all tables
5. **New experiments**: Additive ablation → appendix table
6. **New experiments**: Structured + smart-selection combos → Table 9

---

## Codebase Structure Reference

```
neurosymbolic-ai-paper-experiments/
├── acl_latex.tex                                ← Main paper (two-column ACL)
├── latex_experiments/
│   └── 2026-03-06_professor-feedback.md        ← THIS FILE
├── experiments/
│   ├── results/
│   │   ├── summary/
│   │   │   ├── all_results.csv                 ← All NLLB-600M results (source of truth)
│   │   │   └── robustness_table.csv            ← Multi-model summary
│   │   ├── ablations/                          ← module_loo, module_size_ctrl, minimal_pairs, verb, pronoun
│   │   ├── baselines/                          ← TF-IDF, LENGTH-STRAT, VOCAB-MAX (5 seeds each)
│   │   ├── exp1/                               ← Main 6-condition experiment (5 seeds)
│   │   ├── exp2/                               ← Scaling curves (5 seeds)
│   │   └── hpc_new/                            ← NLLB-1.3B, mBART results (rsync from HPC)
│   ├── data/
│   │   ├── splits/
│   │   │   ├── shared/                         ← structured_train.csv/.tsv, test.tsv
│   │   │   ├── baselines/                      ← TF-IDF-DIVERSE, LENGTH-STRATIFIED, VOCAB-MAXIMIZED
│   │   │   └── new_experiments/                ← NEW: additive + combo splits (created 2026-03-08)
│   │   └── prepare_new_experiments.py          ← NEW: creates + uploads all 7 new conditions
│   ├── analysis/
│   │   ├── compute_chrfpp_for_tables.py        ← Extracts chrF++ values for LaTeX tables
│   │   ├── compute_robustness_table.py         ← Multi-model aggregation
│   │   ├── generate_paper_figures.py           ← Main paper figures (fig2–fig8)
│   │   ├── generate_appendix_figure.py         ← Appendix heatmap
│   │   └── generate_robustness_figure.py       ← Robustness bar chart
│   └── training/
│       ├── hf_job_train.py                     ← HF Jobs training (now includes ROUGE-L + PPL)
│       └── launch_new_experiments.sh           ← NEW: launcher for fix/additive/table9 tiers
```

---

## Table Numbering (Rendered PDF)

| LaTeX Label | Table # | Content | Status |
|-------------|---------|---------|--------|
| tab:positioning | 1 | Related work comparison | OK |
| tab:modules | 2 | Five-module design | OK |
| tab:hyperparams | 3 | Training config | OK |
| tab:exp1_conditions | 4 | Experiment conditions | OK |
| tab:exp1_results | 5 | Main results | OK |
| tab:scaling | 6 | Scaling curves | ✅ chrF++ added |
| tab:replacement | 7 | Budget allocation | OK |
| tab:additive | 8 | Additive curve | OK |
| tab:baselines | **9** | Smart-selection baselines | ⏳ NEW ROWS PENDING |
| tab:module_loo | **10** | Leave-one-out ablation | ✅ chrF++ added |
| tab:module_size_ctrl | **11** | Size-controlled ablation | ⏳ NO-PAST-1K std pending |
| tab:structure_matters | **12** | Minimal pairs + verb diversity | ✅ chrF++ added |
| tab:pronoun | **13** | Pronoun coverage | ✅ chrF++ added |
| tab:robustness | 14 | Multi-model robustness | OK |
| tab:additive_ablation | APP | Additive ablation (new) | ⏳ PENDING JOBS |
