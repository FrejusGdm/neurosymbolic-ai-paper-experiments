# Data Composition vs. Quantity in Low-Resource NMT: French → Adja

**Research question:** Does structured data composition matter more than quantity in extremely low-resource neural machine translation?

**Key finding:** ~4K systematically-structured sentences + 10K random sentences → BLEU ~20, vs. BLEU 2–3 from 10K random sentences alone. The structured subset—designed as a linguistically-motivated curriculum—appears to drive the improvement, not the extra data.

## Language

**Adja** (also Aja) is a Gbe language spoken by ~1M people in southern Benin and Togo. It has no prior NMT systems and minimal digital resources. This project pairs it with French, the region's official language.

## Approach

We design the training corpus itself as a curriculum. Five modules build systematically on a base set of present-tense SVO sentences:

| Module | Content | Method |
|--------|---------|--------|
| 1 | Present tense (base) | Combinatorial generation (Python) |
| 2 | Negation | Rule-based transformation |
| 3 | Past tense (passé composé) | GPT-4 transformation |
| 4 | Future tense (aller + inf) | GPT-4 transformation |
| 5 | Questions (yes/no + wh-) | GPT-4 transformation |

Every sentence in Modules 2–5 links to exactly one Module 1 sentence, changing only one grammatical feature (minimal pairs). This gives the model structured signal about how French grammar works.

## Repository Structure

```
.
├── README.md                          ← You are here
├── CLAUDE.md                          ← AI assistant instructions
├── AGENTS.md                          ← Agent coordination spec
├── experiments/                       ← Experiment framework
│   ├── README.md                      ← Experiment tracker (conditions, baselines, ablations)
│   ├── GUIDE.md                       ← Step-by-step guide to running experiments
│   ├── BASELINES.md                   ← Baseline construction recipes
│   ├── ABLATIONS.md                   ← Ablation conditions
│   ├── REVIEWER_DEFENSE.md            ← Anticipated reviewer objections + responses
│   ├── EXPERIMENT_LOG.md              ← Run-by-run log with metrics
│   ├── configs/                       ← YAML experiment configs
│   ├── data/                          ← Split preparation scripts + manifests
│   ├── preprocessing/                 ← Data cleaning scripts (pre-split)
│   ├── training/                      ← Training scripts
│   ├── evaluation/                    ← Metric computation scripts
│   ├── analysis/                      ← Statistical analysis + visualization
│   ├── notebooks/                     ← Jupyter notebooks for exploration
│   ├── templates/                     ← Config templates
│   └── results/                       ← Outputs (gitignored)
├── scripts-from-another-workspace/    ← Corpus generation pipeline (two runs)
│   ├── code-run-1/                    ← Run 1: 10 verbs, ~1,982 sentences
│   └── code-run-2/                    ← Run 2: 10 different verbs, ~2,284 sentences
├── project-info/                      ← Research proposal, literature review, notes
└── openspec/                          ← Change proposal spec system
```

## Getting Started

1. See [`experiments/GUIDE.md`](experiments/GUIDE.md) for the step-by-step experiment workflow
2. See [`experiments/README.md`](experiments/README.md) for the full experiment tracker
3. See [`experiments/preprocessing/README.md`](experiments/preprocessing/README.md) for data cleaning steps

### Prerequisites

```
pip install -r experiments/requirements.txt
```

Core dependencies: `transformers`, `datasets`, `sacrebleu`, `pandas`, `pyyaml`, `tqdm`.

## Data Privacy

Adja translation data is **not included** in this repository. The language community's consent governs data sharing. All data files (`*.csv`, `*.xlsx`, `*.tsv`, `*.parquet`) are gitignored.

The French-only sentence generation scripts and their outputs are included for reproducibility.

## Target Venues

- AfricaNLP Workshop
- LoResMT Workshop
- EMNLP/ACL Findings

## Citation

```
(citation placeholder — to be added upon publication)
```

## License

(license placeholder — to be determined)
