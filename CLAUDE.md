<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating whether **data composition and structure matter more than quantity** in extremely low-resource neural machine translation (NMT). Language pair: French → Adja (Gbe language, ~1M speakers). Preliminary finding: 5,000 systematically-structured sentences achieve BLEU ~20, vs BLEU 2-3 from 10,000 random sentences.

The approach is a **linguistically-motivated curriculum-style corpus design** applied at the data-collection stage. This is applied neurosymbolic AI: symbolic linguistic structure (grammar rules, minimal pairs) combined with neural generation (GPT-4 transformations).

## Running the Pipeline

Scripts are standalone Python files run sequentially from within a `code-run-*` directory. There is no build system, package manager, or test suite.

```bash
# Full pipeline (each script depends on the previous output):
python generation-1.py              # Module 1: base present-tense sentences → module1_base.csv
python generation-2.py              # Module 2: negation → module2_negation.csv
python generation-3.py              # Modules 3-5: past/future/questions via GPT-4 API (requires OPENAI_API_KEY in .env)
python deduplicate_module5.py       # Dedup questions → module5_questions_dedup.csv
python combine_final_dataset.py     # Assemble final dataset → ADJA_*.csv/.xlsx/.txt
```

**Dependencies** (no requirements.txt exists): `pandas`, `openpyxl`, `openai` (v1+ API), `python-dotenv`, `tqdm`.

**Environment**: `OPENAI_API_KEY` must be set in a `.env` file for `generation-3.py` (uses `python-dotenv`). The model used is `gpt-4.1-mini-2025-04-14`.

## Data Pipeline Architecture

Five-module curriculum design, each building on Module 1 base sentences:

| Module | Content | Generation Method | Output Structure |
|--------|---------|-------------------|------------------|
| 1 | Present tense SVO | Pure Python (combinatorial: 8 pronouns × 10 verbs × ~5 objects each) | `affirmative_present` |
| 2 | Negation (ne...pas) | Pure Python string manipulation | `negation_present` |
| 3 | Past tense (passé composé) | GPT-4 API, batches of 15 | `affirmative_past` |
| 4 | Future tense (aller + inf) | GPT-4 API, batches of 15 | `affirmative_future` |
| 5 | Questions (yes/no + wh) | GPT-4 API, batches of 15 | `question_yn` / `question_wh` |

**Minimal pairs**: Every Module 2-5 sentence links to exactly one Module 1 sentence via `base_sentence_id`. Only one grammatical feature changes per transformation.

**API batch processing**: 15 sentences per call, 3 retries per batch, 1.5s rate limiting. Failed batches saved to `failed_batches_*.txt` for manual review.

## Key Design Decisions

- **Stratified sampling** in Module 1: `groupby(['pronoun', 'verb']).apply(sample)` ensures balanced coverage when targeting a sentence count below the full combinatorial product.
- **Sentence ID format**: `M{module}_{sequential:04d}` (e.g., `M1_0001`, `M3_0042`).
- **Multiple export formats**: CSV (full metadata), Excel (translator-friendly with empty Adja column), TXT (numbered, by-module, translation template).
- **Two code runs** (`code-run-1/`, `code-run-2/`): parallel experimental iterations with different parameterizations.

## Repository Structure

- `project-info/` — Research proposal, implementation notes, literature review (~2000 lines, 100+ citations)
- `scripts-from-another-workspace/code-run-{1,2}/` — Two iterations of the generation pipeline
- `.mcp.json` — MCP server config for NIA and Exa (gitignored, contains API keys)

## HF Jobs Gotchas (learned the hard way)

- **`hf jobs uv run` — script path must be LAST argument**, after all `--flag` and `-e` options. Putting it elsewhere silently breaks.
- **Stdout is fully buffered in containers.** Add `sys.stdout.reconfigure(line_buffering=True)` at the top of any training script, otherwise `print()` output won't appear in the HF Jobs log viewer until the script exits.
- **Rate limits:** HF allows 1,000 API calls per 5-min window (free) or 2,500 (PRO). Each `hf jobs uv run` makes ~3-4 API calls (whoami, upload, create). With DELAY=5s between submissions, you'll hit 429 every ~15 jobs. The launcher (`launch_jobs.sh`) has retry logic built in — it waits 65s and retries up to 3 times.
- **`Seq2SeqTrainer` is fragile across transformers versions.** The custom training loop in `hf_job_train.py` avoids it entirely — uses Adafactor + manual step loop. Don't go back to Trainer.
- **`as_target_tokenizer()` is deprecated** in transformers 4.44+ but still works. Pin `transformers==4.44.2`. If it breaks in a future version, replace with: set `tokenizer.src_lang = TGT_LANG`, tokenize, then restore `src_lang`.
- **`aj_Latn` is a custom token** — must call `fix_tokenizer()` + `model.resize_token_embeddings()` every time the model/tokenizer is loaded.

## Critical Constraints

- **Data is private**: Generated Adja translation data must never be committed. The language community's consent governs data sharing.
- **No data files in repo**: All CSV/Excel/TXT outputs are generated locally and kept outside version control.
- **API keys**: `.mcp.json` and `.env` are gitignored. Never commit credentials.
