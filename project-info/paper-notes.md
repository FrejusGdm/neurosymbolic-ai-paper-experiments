# Paper Notes — French-Adja NMT via Linguistically-Informed Corpus Design

Prepared for advisor discussion. Last updated: February 2026.

---

## Research Question & Hypothesis

For extremely low-resource language pairs where parallel data must be created from scratch, does the **composition and structure** of the training corpus matter more than its **size**? We hypothesize that a small (~4K sentence), linguistically-structured corpus — designed as a grammatical curriculum covering present tense, negation, past, future, and questions — will outperform a larger (~10K sentence) randomly-sourced corpus for neural machine translation. Preliminary evidence: the structured corpus combined with 10K random Tatoeba sentences achieves BLEU ~20, while the random data alone yields BLEU 2–3.

---

## Method Summary

We design the parallel corpus itself as a **grammatical curriculum** with five modules:

1. **Module 1 — Present tense**: Systematic SVO sentences across 8 subject pronouns, 20 verbs, and ~5 objects each. Generated combinatorially (pure Python), then stratified-sampled for balance.
2. **Module 2 — Negation**: Each M1 sentence transformed with "ne...pas" negation. Only one grammatical feature changes (minimal pair).
3. **Module 3 — Past tense**: M1 sentences converted to passé composé via GPT-4 API.
4. **Module 4 — Future tense**: M1 sentences converted to near future (aller + infinitive) via GPT-4 API.
5. **Module 5 — Questions**: Yes/no and wh-questions derived from M1 sentences via GPT-4 API.

Every sentence in Modules 2–5 links to exactly one Module 1 base sentence, creating **minimal pairs** that differ in exactly one grammatical dimension. The resulting ~4,250 French sentences were then translated into Adja by a native speaker/translator.

The key insight is **not** about training curriculum (reordering existing data during training) but about **designing the data collection itself** as a curriculum. This is upstream of the model — the structured corpus is model-agnostic.

---

## Related Work & Positioning

Our work sits at the intersection of three research threads. Understanding where we fit — and where we diverge — is critical for framing the paper.

### Thread 1: Data quality over data quantity in NMT

The "more data is better" assumption (Halevy et al., 2009; Banko & Brill, 2001) held for high-resource NMT but has been decisively challenged in low-resource settings. Khayrallah & Koehn (2018) showed NMT is disproportionately sensitive to noise — even small amounts of misaligned data trigger hallucinations. Junczys-Dowmunt (2018) demonstrated that aggressively filtered subsets of noisy web corpora outperform much larger unfiltered ones. The WMT Parallel Corpus Filtering shared tasks (Koehn et al., 2019) institutionalized this: selecting the top few percent by quality is often better than using everything. NLLB (2022) achieved state-of-the-art across 200 languages through careful data filtering, not just scale. Ranathunga et al. (2024) confirmed that for low-resource web-mined corpora, aggressive quality filtering consistently helps.

**Where we fit:** We take data quality one step further. Instead of *filtering* an existing corpus, we *design* the corpus from scratch with quality and structure built in. Our work is downstream of the quality-vs-quantity insight but upstream of the model — the structured corpus is model-agnostic.

### Thread 2: Curriculum learning for NMT

Bengio et al. (2009) formalized curriculum learning: models train better when examples progress from simple to complex. Platanios et al. (2019) introduced competence-based CL for NMT, showing models benefit from mastering high-frequency patterns before harder ones. Zhou et al. (2020) showed uncertainty-aware curricula prevent low-resource models from getting trapped in poor local minima.

**The gap we fill:** All existing CL work assumes you already have a dataset and reorders/reweights it during training. Nobody has asked: what if you design the *data collection itself* as a curriculum? We embed the curriculum into the dataset topology — present tense first, then negation, past, future, questions — rather than imposing it at training time. The distinction matters: our approach is a data construction methodology, not a training strategy. It works with any model and any training schedule.

### Thread 3: Lexical coverage, linguistic structure, and African languages

Jones et al. (2023) argued in GATITOS that carefully curated lexica outperform larger noisy ones for low-resource MT — ensuring coverage of key semantic concepts and grammatical functions gives stable baselines that random web-crawling cannot. Nielsen et al. (2025) identified the "Alligator Problem": models trained on sparse data confuse distributionally similar words due to insufficient co-occurrence evidence, a dominant failure mode across 122 low-resource languages. Groshan et al. (2025) showed linguistically motivated data augmentation only helps when synthetic examples match natural distributions; poorly grounded augmentation actively harms.

For African languages specifically: Masakhane (Orife et al., 2020) built community-driven MT systems for 30+ African languages, showing what's possible with modest parallel data. Dossou & Emezue (2021) worked on English-Fon/French-Fon MT for a closely related Gbe language. FFR v1.1 provides a French-Fon parallel corpus. Adelani et al. (2022) with MENYO-20k created a benchmark for Yoruba-English. Wang et al. (2024) introduced AfriCOMET, showing standard metrics like BLEU correlate weakly with human judgments for African languages. Ojo & Ogueji (2023) demonstrated even commercial LLMs perform poorly on African language translation.

**Where we fit:** Our work bridges computational NLP and linguistic fieldwork. Rather than mining the web for a language with near-zero digital presence, we partner with a native speaker to create a structured corpus guided by grammatical principles. This aligns with GATITOS's insight about curated lexica but goes further: we don't just curate vocabulary, we curate *grammatical structure*.

### The specific gap

| Existing approach | What it does | What it assumes |
|---|---|---|
| Corpus filtering (Junczys-Dowmunt, 2018) | Remove bad data from large corpus | Large corpus exists |
| Curriculum learning (Platanios et al., 2019) | Reorder data during training | Dataset already collected |
| Data augmentation (Groshan et al., 2025) | Generate synthetic variants | Seed corpus exists |
| Cross-lingual transfer (NLLB, 2022) | Leverage related languages | Pretrained multilingual model |
| **Our approach** | **Design the corpus from scratch as a curriculum** | **Access to a native speaker/translator** |

Nobody has proposed designing the dataset itself as a structured curriculum for a language with no existing parallel data. That's the gap.

---

## Key Contributions

1. **First work to design the dataset itself as a curriculum** for low-resource NMT. Prior curriculum learning work reorders existing data during training; we structure the data collection process from scratch.

2. **Reproducible, language-agnostic framework** for low-resource corpus design. The five-module curriculum can be applied to any language pair with SVO-like structure. The generation scripts are open-source.

3. **Empirical evidence that composition beats quantity at equal total size.** The core comparison: ~4K structured + 10K random (BLEU ~20) vs. 10K random alone (BLEU 2–3). Ablations isolate the contribution of each module.

4. **Data efficiency story**: Half the data, order-of-magnitude better performance. This is directly relevant for endangered/low-resource language communities where translation effort is the bottleneck.

---

## Limitations (honest self-assessment)

1. **Single language pair.** All experiments are French→Adja. We cannot claim the approach generalizes across typologies (SOV languages, tonal interactions, morphologically rich targets). This is a proof-of-concept.

2. **Small scale.** ~14K total sentences, BLEU ~20. This is not a production-quality system. The value is in the methodology, not the absolute quality.

3. **Artificial grammar coverage.** Only SVO structures with 20 high-frequency verbs. Real language use involves relative clauses, conditionals, passive voice, embedded questions — none of which are covered.

4. **Data quality confound.** Structured sentences are inherently cleaner and more predictable than random Tatoeba sentences. The improvement may partially reflect data quality rather than curriculum structure. The baselines (length-stratified, vocab-maximized, TF-IDF diverse random selections) help control for this but don't eliminate the confound entirely.

5. **Minimal-pair redundancy risk.** If most of the benefit comes from the M1 base sentences alone, the minimal-pair structure adds cost without value. The BASE-ONLY ablation is critical to validate this.

6. **Single translator.** All Adja translations come from one native speaker. Inter-annotator agreement is not measured.

---

## What to Emphasize

- **Data efficiency**: The headline result. Fieldworkers and language communities have limited translation budgets — showing that *what* you translate matters more than *how much* is directly actionable.
- **Curriculum design novelty**: Nobody else has proposed structuring the corpus creation itself as a linguistic curriculum. This is a clear gap in the literature.
- **Fieldwork integration**: This bridges NLP and language documentation. The framework gives fieldworkers a principled protocol for what sentences to prioritize.
- **Reproducible framework**: Scripts are open-source. Another team working on a different low-resource pair can adapt the pipeline.

## What to Downplay

- **"Neurosymbolic" framing**: The approach is better described as "linguistically-informed corpus design." The symbolic component is just grammar rules used to generate sentences — this is a standard approach in applied linguistics. Calling it neurosymbolic oversells the AI angle and may invite unfair scrutiny from reviewers expecting formal logic integration or differentiable symbolic reasoning.
- **Absolute translation quality**: BLEU ~20 is not good by any standard benchmark. The story is the *comparative* improvement, not the absolute score.
- **Broad generalizability**: Until replicated on other pairs, this is a case study. Present it that way.

---

## Recommended Framing

### Draft Abstract

> Low-resource neural machine translation typically assumes that more parallel data leads to better models. We challenge this assumption for the case of creating parallel corpora from scratch. We introduce a linguistically-informed corpus design framework that structures data collection as a five-module grammatical curriculum covering present tense, negation, past tense, future tense, and questions. Applied to French→Adja (a Gbe language with ~1M speakers), our ~4,000 structured sentences combined with 10,000 random parallel sentences achieve BLEU ~20, compared to BLEU 2–3 for the random data alone. Ablation studies isolate the contribution of each grammatical module and show that corpus composition dominates quantity in this regime. Our framework is language-agnostic, reproducible, and designed for integration into language documentation workflows.

### Draft Introduction Framing

Position the paper at the intersection of three threads:
1. **Data-centric AI** (the idea that data quality > model architecture, cf. Andrew Ng's framing)
2. **Low-resource NMT** (Masakhane, specific African language pairs)
3. **Curriculum learning** (but distinguish: curriculum *during training* vs. curriculum *at data creation time*)

The gap: everyone focuses on model architectures, training tricks, or data augmentation for low-resource NMT. Nobody has asked: *if you're creating the training data from scratch, what should you create?*

---

## Anticipated Reviewer Objections

### 1. "The improvement could just be from cleaner data, not structure."

**Response:** This is a valid concern and we address it with multiple baselines. We compare against not just random sampling but also length-stratified, vocabulary-maximized, and TF-IDF diverse subsets of the random data — all at the same size as the structured corpus. If the structured corpus outperforms all three, the advantage cannot be attributed to cleanliness alone. Additionally, the module ablations (removing negation, past, future, or questions independently) show that specific grammatical components contribute, which is inconsistent with a pure data-quality explanation.

### 2. "BLEU ~20 is too low to be useful."

**Response:** The contribution is methodological, not system-level. BLEU ~20 is indeed far from production quality. The value is in the *comparative* result (10x improvement over the random baseline at half the data) and in the reproducible framework that fieldworkers can use. We are explicit about this limitation.

### 3. "Single language pair — how do we know this generalizes?"

**Response:** We don't claim it does. This paper is a proof-of-concept for a specific approach to corpus design. Future work should test on typologically diverse pairs (SOV languages, agglutinative morphology, non-SVO structures). We provide the framework and scripts to facilitate replication.

### 4. "Why not compare against back-translation, cross-lingual transfer, or multilingual pretraining?"

**Response:** These are complementary, not competing approaches. Our structured corpus can be *combined* with back-translation or used to fine-tune a multilingual model (e.g., NLLB). We compare against data selection baselines because our contribution is about *what data to collect*, not about model architecture or training strategy. We include an NLLB fine-tuning baseline to show the approach works with pretrained models too.

### 5. "The minimal pairs might introduce harmful redundancy."

**Response:** This is precisely why we include the BASE-ONLY ablation. If Module 1 alone performs similarly to the full 5-module curriculum, then minimal pairs add cost without value. If the full curriculum significantly outperforms BASE-ONLY, the pairs contribute genuine signal. Either result is informative.

---

## Suggested Paper Outline

| Section | Content | Status |
|---------|---------|--------|
| 1. Introduction | Motivation, gap, contribution summary | Can draft now |
| 2. Related Work | Masakhane, curriculum learning, data-centric AI, low-resource NMT, Gbe languages | Can draft now (see lit review in `project-info/`) |
| 3. Method | 5-module curriculum design, generation pipeline, translation protocol | Can draft now |
| 3.1 Corpus Design | Module descriptions, minimal-pair structure, verb/pronoun coverage | Can draft now |
| 3.2 Translation Protocol | Fieldwork integration, single-translator setup, quality checks | Can draft now |
| 4. Experimental Setup | | |
| 4.1 Data | Structured (~4.2K), random Tatoeba (10K), splits, statistics | Can draft now |
| 4.2 Model | NLLB-200-distilled-600M fine-tuning, hyperparameters | Can draft after experiments |
| 4.3 Baselines | Random, length-stratified, vocab-maximized, TF-IDF diverse | Can draft now |
| 4.4 Ablations | Module LOO, pronoun coverage, verb diversity, minimal-pair structure | Can draft now |
| 4.5 Evaluation | BLEU, chrF++, (optional: AfriCOMET if available) | Can draft after experiments |
| 5. Results | Tables, scaling curves, ablation results | Needs experiment runs |
| 6. Analysis | What modules matter most, data efficiency curves, error analysis | Needs experiment runs |
| 7. Discussion | Implications for fieldwork, limitations, future directions | Can draft skeleton now |
| 8. Conclusion | Summary of contributions | After results |

**What can be written now:** Sections 1–3, 4.1, 4.3–4.4, and the skeleton of 7. That's roughly 60% of the paper.

**What needs experiment results:** Sections 4.2, 4.5, 5, 6, and finalizing 7–8.

---

## Target Venues

1. **AfricaNLP Workshop** (co-located with ICLR or ACL) — best fit, community-aligned
2. **LoResMT Workshop** (low-resource MT, co-located with MT Summit or ACL)
3. **EMNLP/ACL Findings** — if results are strong and well-ablated
4. **LREC-COLING** — good venue for data/resource papers

Page limits and formatting vary. AfricaNLP is typically 4–8 pages; Findings is 8 pages + references.
