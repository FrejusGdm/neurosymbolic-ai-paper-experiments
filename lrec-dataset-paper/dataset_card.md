---
language:
- fr
- ajg
license: cc-by-nc-sa-4.0
task_categories:
- translation
tags:
- machine-translation
- low-resource
- african-languages
- gbe-languages
- parallel-corpus
- adja
- benin
- togo
- west-africa
pretty_name: French-Adja Parallel Corpus
extra_gated_prompt: >-
  This dataset contains French-Adja parallel text created in collaboration with
  the Adja-speaking community in Benin. Access requires agreement to the terms
  below. Please fill out all fields.
extra_gated_fields:
  Name: text
  Email: text
  Affiliation: text
  Intended use: text
  I agree to use this dataset only for research purposes and to respect the rights of the Adja-speaking community: checkbox
size_categories:
- 1K<n<10K
source_datasets:
- original
annotations_creators:
- expert-generated
language_creators:
- expert-generated
multilinguality:
- translation-pair
dataset_info:
  features:
  - name: fr
    dtype: string
  - name: adj
    dtype: string
  splits:
  - name: train
    num_examples: 8000
  - name: validation
    num_examples: 1000
  - name: test
    num_examples: 1000
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
---

# French-Adja Parallel Corpus

**The first publicly available parallel text corpus for Adja machine translation, targeting an under-resourced Gbe language spoken by approximately 1,000,000 people in Benin and Togo.**

## Dataset Description

This dataset contains **10,000 French-Adja sentence pairs** created through a six-month collaborative translation effort with native Adja speakers in the Couffo region of Benin. It is designed to serve as a foundation for machine translation research and other NLP tasks for the Adja language.

### About Adja

Adja (ISO 639-3: `ajg`) is a Gbe language of the Niger-Congo family, closely related to Fon, Ewe, and Gen. It is spoken by approximately 1 million people, primarily in the Couffo and Mono departments of southern Benin and in southeastern Togo. Despite its significant speaker population, Adja had **no publicly available text-based NLP resources** prior to this work --- no parallel corpora, no machine translation systems, and no labeled computational datasets. Concurrent work by Justin et al. (2025) introduced Eyaa-Tom, a multi-language dataset for Togolese languages that includes Adja among its targets, but the publicly released Adja data consists only of a small amount of audio — no French-Adja parallel text for MT has been made available.

### Languages

| | Language | ISO 639 | Script |
|---|----------|---------|--------|
| `fr` | French | fra | Latin |
| `adj` | Adja | ajg | Latin (with diacritics: ɔ, ɛ, ɖ, ŋ, tonal marks) |

## Dataset Creation

### Translation Process

1. **Source sentences**: 10,000 French sentences selected through uniform random sampling from [Tatoeba](https://tatoeba.org), a collaborative platform of community-contributed translations
2. **Translation team**: 5 native Adja speakers from the Couffo region of Benin:
   - 2 translators (1 government-accredited Adja language instructor + 1 experienced fluent speaker)
   - 3 transcribers (all native speakers)
3. **Process**: French sentences were read aloud, discussed for comprehension, then translated orally into Adja. Transcribers recorded the spoken Adja in writing. Translations were then typed and formatted for computational use
4. **Duration**: 6 months of collaborative work
5. **Quality control**: A dedicated cleaning pipeline was applied:
   - Unicode normalization (NFKC)
   - Spacing normalization
   - Non-standard character replacement
   - Bidirectional punctuation consistency checks
   - Quotation mark matching

### Why Oral Translation?

Adja is primarily a spoken language. Having translators produce Adja translations **orally** before transcription preserves natural spoken Adja and avoids the artificiality that can arise from written-first translation in a language with limited written tradition.

## Dataset Structure

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `fr` | string | French source sentence |
| `adj` | string | Adja translation |

### Data Splits

| Split | Count | Percentage |
|-------|-------|------------|
| `train` | 8,000 | 80% |
| `validation` | 1,000 | 10% |
| `test` | 1,000 | 10% |

Splits were created through uniform random sampling (seed 42).

### Examples

| French | Adja |
|--------|------|
| Je sue tous les jours. | ŋ kɔ nɔ ade tɛgbɛ ɛ. |
| Je pense que tu devrais voir ça. | ŋ bumɔ́ wɔ a kpɔɛ alo wɔ a nya. |
| Tu te perds. | E búbu ɔ deki. |
| Cesse de rêver et ouvre les yeux. | Mi edrɔ kukú ahùn ŋkuvi wo. |
| Le gouvernement a reçu son autorité de l'empereur. | Eju tatɔ xɔ acɛ ega ɖuɖu tɔ. |

## Corpus Statistics

| Metric | French | Adja |
|--------|--------|------|
| Total tokens | 66,245 | 66,661 |
| Vocabulary size | 11,385 | 12,560 |
| Type-Token Ratio | 0.172 | 0.188 |
| Hapax legomena | 7,196 (63%) | 8,318 (66%) |
| Mean sentence length | 6.62 words | 6.67 words |
| Median sentence length | 6.0 | 6.0 |
| Sentence length std | 2.92 | 3.18 |
| Min -- Max length | 1 -- 63 | 1 -- 68 |

Adja exhibits higher lexical diversity (TTR 0.188 vs 0.172), reflecting the morphological richness characteristic of Gbe languages.

## Baseline Results

We fine-tuned three models on this corpus and report mean results over 5 random seeds on the **random test split**:

| Model | Direction | BLEU | chrF++ |
|-------|-----------|------|--------|
| **NLLB-600M** | FR → ADJ | 4.5 ± 0.2 | 26.1 ± 0.3 |
| **NLLB-600M** | ADJ → FR | 11.8 ± 0.6 | 30.3 ± 0.5 |
| **mBART-50** | FR → ADJ | 3.4 ± 0.3 | 22.6 ± 0.5 |
| **mBART-50** | ADJ → FR | 8.7 ± 0.1 | 26.1 ± 0.3 |
| **ByT5-base** | FR → ADJ | 4.3 ± 0.5 | 26.1 ± 0.4 |
| **ByT5-base** | ADJ → FR | 11.5 ± 0.9 | 31.2 ± 0.9 |

All models used Adafactor (lr=1e-4), batch size 16, and early stopping on validation chrF (patience 10).

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("JosueG/french-adja-parallel-corpus")

# Access a sentence pair
print(dataset["train"][0])
# {'fr': 'Je sue tous les jours.', 'adj': 'ŋ kɔ nɔ ade tɛgbɛ ɛ.'}

# Fine-tune a translation model
for example in dataset["train"]:
    src = example["fr"]
    tgt = example["adj"]
```

## Ethical Considerations

- This corpus was created in direct collaboration with native Adja speakers in Benin. The translation team was compensated for their work
- The dataset is released under a **non-commercial license** (CC BY-NC-SA 4.0) to protect against exploitative use while enabling academic research
- Adja is primarily a spoken language; this written corpus does not capture the full richness of spoken Adja, including tonal variation and dialectal differences across communities
- The Tatoeba source sentences reflect global French usage and may not represent the specific variety of French spoken in Benin and Togo

## Citation

If you use this dataset, please cite:

```bibtex
@inproceedings{godeme2026french-adja,
    title     = {A 10,000-Sentence French-Adja Parallel Corpus for Machine Translation},
    author    = {Godeme, Josue and Coto-Solano, Rolando},
    booktitle = {Proceedings of the 2026 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2026)},
    year      = {2026},
    note      = {To appear}
}
```

## Acknowledgements

We are deeply grateful to the Adja-speaking community members in the Couffo region of Benin who dedicated their time and expertise to translating and transcribing this corpus over a six-month period. Their commitment to documenting and advancing their language made this work possible.
