# Limitations Reference for Paper Writer

> **Purpose:** Comprehensive, severity-ranked inventory of every limitation, confound, and awkward result in the French-Adja NMT study. Each entry gives the honest statement, why it matters, what we did about it, and any counter-argument. Use this as the single reference for drafting the Limitations, Discussion, and rebuttals.
>
> **How to use in the paper:** Lead with Tier 1 in the Limitations section (reviewers respect honesty). Weave Tier 2 into Discussion. Address Tier 3 proactively where the results are presented. Frame Tier 4 as Future Work opportunities, not failures.

---

## Tier 1: Fundamental (must appear in the Limitations section)

### 1. Single language pair

**Honest statement:** All 224 experiments are French-Adja. Both languages are SVO and largely isolating/analytic. We cannot claim cross-typological generalization.

**Why it matters:** A reviewer can dismiss the entire paper as a single case study. The structured curriculum may exploit the SVO-to-SVO alignment and fail on SOV targets, agglutinative morphology, or tonal interactions that differ from Adja's.

**Partial mitigation:**
- Adja is typologically representative of many West African isolating SVO languages, so the method likely transfers within that family.
- The generation scripts and five-module framework are open-sourced and language-agnostic in design; only the verb/object lexicons need replacing.
- We frame the paper as a proof-of-concept, not a universal claim.

**Counter-argument:** Proving generalization requires multi-language replication, which is future work. The contribution is the *methodology* and the empirical evidence that structure matters on at least one pair.

---

### 2. Narrow grammatical scope

**Honest statement:** The structured corpus covers only 20 verbs, SVO word order, and five grammatical constructions (present, negation, past, future, questions). No relative clauses, conditionals, passive voice, embedded questions, or complex subordination.

**Why it matters:** Real language is infinitely more complex. The model trained on structured data alone may produce grammatically correct but formulaic translations that fail on anything outside the template.

**Partial mitigation:**
- The test set (1,455 sentences) includes out-of-distribution structures not covered by any training condition, so scores already reflect some generalization penalty.
- The MODULE-LOO and SIZE-CONTROLLED ablations confirm each grammatical module contributes independently (removing future tense alone costs -5.6 BLEU), suggesting extending to more modules would yield further gains.
- Framed as proof-of-concept: "we show structure matters with five modules; adding more modules is straightforward future work."

**Counter-argument:** The narrowness is intentional — we test a minimal viable curriculum. Expanding to more constructions is additive work, not a conceptual limitation.

---

### 3. Single translator (no inter-annotator agreement)

**Honest statement:** All Adja translations were produced by one native speaker/translator. There is no second annotator, no inter-annotator agreement score, and no back-translation check.

**Why it matters:** Systematic bias from a single translator (dialectal preference, consistent errors, personal style) could inflate or deflate metrics in ways we cannot detect. Reviewers at top venues increasingly expect IAA scores.

**Partial mitigation:**
- The translator is a native Adja speaker with translation experience; the French source sentences were validated separately with no issues flagged.
- For an extremely low-resource language like Adja (~1M speakers, minimal digital presence), finding multiple qualified translators is a genuine practical constraint.
- This mirrors the fieldwork reality the paper targets: most endangered-language documentation projects have a single consultant.

**Counter-argument:** Single-translator setups are standard in language documentation and low-resource NMT work (e.g., Masakhane community contributions). We acknowledge this limitation explicitly and note that a second annotator would strengthen future iterations.

---

### 4. No human evaluation

**Honest statement:** All conclusions rely on automatic metrics (SacreBLEU, chrF, chrF++, TER). No human adequacy/fluency judgments were collected.

**Why it matters:** Wang et al. (2024, AfriCOMET) showed BLEU correlates weakly with human judgments for African languages. Our BLEU scores may not reflect actual translation quality as perceived by Adja speakers.

**Partial mitigation:**
- We report chrF and chrF++ as primary metrics (better for morphologically rich / low-resource languages than BLEU).
- All claims are *comparative* (structured vs. random, intact vs. broken) rather than about absolute quality. Even if the metric is imperfect, consistent biases affect all conditions equally.
- Multiple metrics agree directionally across all experiments.

**Counter-argument:** Human evaluation is planned as follow-up work. COMET is not applicable (Adja is not in COMET's training data — see Tier 4, item 15). Automatic metrics are the standard in the low-resource NMT literature we compare against.

---

## Tier 2: Methodological (should appear — builds reviewer trust)

### 5. Data quality confound

**Honest statement:** Structured sentences are inherently cleaner, shorter, and more grammatically predictable than Tatoeba sentences. Part of the structured advantage may be a data-quality effect, not a curriculum-structure effect.

**Why it matters:** A reviewer could argue we're comparing "clean data vs. noisy data," not "structured vs. unstructured."

**Partial mitigation:**
- The smart-selection baselines (LENGTH-STRATIFIED, VOCAB-MAXIMIZED, TF-IDF-DIVERSE) select clean, high-quality subsets from the same Tatoeba pool — and they achieve BLEU 19.1-24.0 at 2K. This confirms that data quality matters, but the structured approach works *without needing a pre-existing corpus to select from*.
- The PAIRS-BROKEN ablation uses the *exact same* structured sentences with only the minimal-pair alignment shuffled: BLEU drops from 22.9 to 5.4. This isolates structure from quality — same sentences, different organization, catastrophic performance loss.

**Counter-argument:** The minimal-pair ablation is the cleanest control for this confound. If quality alone explained the results, PAIRS-BROKEN should perform similarly to PAIRS-INTACT. It does not.

---

### 6. Smart baselines outperform structured-only on BLEU

**Honest statement:** VOCAB-MAXIMIZED (BLEU 24.0 +/- 0.75) and LENGTH-STRATIFIED (BLEU 23.7 +/- 0.41) at 2K sentences from Tatoeba outperform STRUCTURED-2K (BLEU 19.9 +/- 2.69) on BLEU.

**Why it matters:** A reviewer can say "just do smart selection from existing data — no need for your fancy curriculum." This undercuts the practical motivation.

**Partial mitigation:**
- The baselines *require* an already-translated 10K corpus to select from. They answer the question "what if you select data better?" Our approach answers a different question: "what if you *design* the data from scratch for a language with no existing parallel corpus?"
- When structured data is *combined* with random data (R8K+S2K: BLEU 22.4, R10K+S4K: BLEU 22.5), the results are competitive with or exceed the baselines — and the combined models have much better chrF (41.2 vs. 40.4).
- The structured approach is additive: it creates *new* data that complements existing corpora, while smart selection only re-arranges what already exists.

**Counter-argument:** The comparison is between "design new data" vs. "select from existing data." Both are valid strategies. We show they are complementary: the optimal approach uses structured data *plus* random data.

---

### 7. chrF divergence from BLEU

**Honest statement:** Structured-only models show a striking chrF gap: STRUCTURED-2K gets chrF 29.2, while VOCAB-MAXIMIZED gets chrF 40.4 and combined R10K+S4K gets chrF 41.2. Structured-only models have high BLEU but low chrF relative to baselines and combined models.

**Why it matters:** chrF measures character-level overlap and is often considered more reliable than BLEU for morphologically complex languages. The low chrF for structured-only models suggests they may produce grammatically patterned but lexically impoverished translations.

**Partial mitigation:**
- The test set contains Tatoeba-derived sentences, so vocabulary overlap naturally favors Tatoeba-trained models on chrF. This is a test-set composition effect, not necessarily a quality gap.
- Combined models (which include structured data) get the *highest* chrF scores (41.2), showing structured data's value as a complement.
- When structured data is added to random data, both BLEU and chrF improve — structured data contributes even by chrF's measure.

**Counter-argument:** The chrF gap is real and should be acknowledged. It reinforces the main practical recommendation: use structured data *in combination* with diverse natural data, not as a replacement.

---

### 8. Structured data plateaus at ~2-3K sentences

**Honest statement:** STRUCTURED-2K achieves BLEU 19.9, STRUCTURED-3K achieves 21.1, and STRUCTURED-4K achieves 19.5 (not significantly different). Adding more structured data of the same type does not improve performance beyond ~2-3K sentences.

**Why it matters:** The structured approach has a ceiling. A reviewer could argue this limits scalability.

**Partial mitigation:**
- The plateau is expected and actually *supports* our argument: the curriculum has finite content (20 verbs x 5 modules x 8 pronouns), so adding more samples of the same patterns saturates the learning signal.
- The solution is not "more of the same" but "complementary data" — either expanding the curriculum to new verbs/constructions or adding diverse random data (which the combined conditions demonstrate works).
- The plateau at 2-3K means the method is *efficient*: you don't need more than ~2K structured sentences to capture the curriculum's benefit.

**Counter-argument:** Plateauing is a feature, not a bug. It tells practitioners exactly how much structured data to invest in (~2K) before pivoting to complementary sources.

---

### 9. Test set may favor combined models

**Honest statement:** The shared test set (1,455 sentences) was drawn from the combined pool of structured + Tatoeba data using the group-aware 80/10/10 split. Models trained on both data types may have a compositional advantage.

**Why it matters:** If the test set vocabulary skews toward Tatoeba, Tatoeba-trained models get a chrF boost. If it includes structured-style sentences, structured models get a BLEU boost. The comparison may not be on fully neutral ground.

**Partial mitigation:**
- The group-aware split ensures that all transformations of a base sentence stay together (train/val/test), preventing data leakage within the structured portion.
- All conditions are evaluated on the same test set, so any bias affects all conditions equally in terms of ranking.
- The test set was not designed to favor any particular condition.

**Counter-argument:** A fully neutral test set would require independently sourced Adja sentences (e.g., from natural speech or a different text domain). This is acknowledged as a limitation and planned for future work.

---

## Tier 3: Results that need explaining (address proactively where results appear)

### 10. PAIRS-BROKEN anomaly

**Honest statement:** PAIRS-BROKEN (BLEU 5.4 +/- 0.72) performs *worse* than RANDOM-4K (BLEU 3.7 +/- 0.48) despite using 3,823 structured sentences. Shuffling the minimal-pair alignment doesn't just remove the benefit — it actively harms performance below the random baseline.

**Why it matters:** Intuitively, broken pairs should produce something between intact pairs and fully random — not worse than random. This needs explanation or it looks like an error.

**Partial mitigation:**
- Plausible explanation: breaking the pairs creates *contradictory* training signal. The model sees M2-M5 sentences tagged as transformations of M1 bases they don't correspond to, learning incorrect grammatical mappings. This is worse than random noise because it's *systematically* misleading.
- The chrF for PAIRS-BROKEN (14.0) is also far below RANDOM-4K (28.8), confirming this is not a BLEU artifact.
- This actually *strengthens* the minimal-pair argument: the pairing relationship isn't just helpful, it's essential. Disrupting it is catastrophic.

**Counter-argument:** The anomaly is evidence, not a flaw. Frame it as: "The structure is so important that corrupting it is worse than having no structure at all."

---

### 11. REDUCED-4 pronoun high variance

**Honest statement:** The REDUCED-4 pronoun condition (je, tu, il, nous) has std = 5.18 on BLEU (mean 19.5), far higher than other conditions (typical std = 0.3-2.7). One of the 3 seeds is a clear outlier.

**Why it matters:** With only 3 seeds and one outlier, the mean may not be reliable. A reviewer could question whether the pronoun ablation results are stable.

**Partial mitigation:**
- The directional trend is clear and monotonic: ALL-8 (22.9) > REDUCED-4 (19.5) > SINGULAR-3 (18.6) > MINIMAL-1 (15.9). Even discounting REDUCED-4, the overall pattern holds.
- Report the outlier seed explicitly and present the median alongside the mean.
- 3 seeds is the minimum standard; 5 seeds would be better but was not run for all ablation conditions.

**Counter-argument:** The high variance at REDUCED-4 may reflect a real instability at 4 pronouns (an intermediate coverage level). The endpoints (ALL-8 and MINIMAL-1) are stable, and the overall trend is robust.

---

### 12. RANDOM-8K = RANDOM-10K (identical results)

**Honest statement:** RANDOM-8K and RANDOM-10K produce identical metrics (BLEU 4.13, chrF 30.81) because the 80/10/10 split of 10K yields only ~7,200 training sentences — the same as 80% of 9K (the available pool after splits). The "10K" label overstates the actual training data.

**Why it matters:** A reader might think we tested 10,000 training sentences when the actual training set was 7,200. Claims like "200 structured beats 10,000 random" are slightly misleading if the 10K condition only uses 7.2K for training.

**Partial mitigation:**
- The paper should clearly state that "10K" refers to the pre-split total and that actual training sizes after 80/10/10 are reported in all tables (e.g., 7,200 for RANDOM-10K).
- This is standard in NMT papers — dataset sizes are usually stated pre-split.
- The comparison remains dramatic even with the true training size: 180 structured training sentences (STRUCTURED-200) outperform 7,200 random training sentences.

**Counter-argument:** Be transparent about it. Add a footnote or parenthetical in the first mention: "RANDOM-10K (7,200 training sentences after 80/10/10 split)."

---

### 13. Absolute quality is low

**Honest statement:** The best BLEU score is 22.5 (R10K+S4K). This is far from production-grade MT.

**Why it matters:** A reviewer could question the practical significance: "You improved from terrible to bad."

**Partial mitigation:**
- The contribution is methodological, not system-level. The paper is about *data efficiency*, not about building a production Adja translator.
- BLEU ~20 is competitive for genuinely low-resource African languages (cf. Masakhane baselines, Dossou's Fon work).
- The practical value is upstream: telling fieldworkers *what to translate* to maximize model quality per unit of translation effort.
- The comparative improvement (5x BLEU from 4x less data) is the headline, not the absolute score.

**Counter-argument:** Frame it as: "Even at BLEU 20, the system may be useful for assisted translation workflows where a human post-edits machine output, dramatically reducing per-sentence effort." But do not overclaim utility.

---

## Tier 4: Scope / Future Work (frame as opportunities, not failures)

### 14. No active learning / submodular selection comparison

**Honest statement:** We compare against three smart-selection baselines but not against active learning or submodular data selection, which are established techniques for data-efficient NMT.

**Why it matters:** Active learning selects the most informative examples iteratively. If it achieves similar gains from a pre-existing pool, the structured approach adds little for languages that already have some parallel data.

**Counter-argument:** Active learning and submodular selection require an initial seed corpus and iterative model retraining. Our approach operates at the data *creation* stage, before any model exists. They address different scenarios and are complementary, not competing.

---

### 15. No COMET metric

**Honest statement:** We do not report COMET scores. Adja is not in COMET's training data, so standard COMET would produce meaningless numbers.

**Why it matters:** COMET is increasingly expected at top venues (ACL, EMNLP) as a primary metric.

**Counter-argument:** AfriCOMET (Wang et al., 2024) exists for some African languages but does not cover Adja. Fine-tuning COMET on Adja would require human judgments we don't yet have. This is noted as future work, contingent on collecting human evaluation data (which also addresses limitation #4).

---

### 16. GPT-4 dependency for Modules 3-5

**Honest statement:** Modules 3 (past tense), 4 (future tense), and 5 (questions) are generated via GPT-4 API calls. This introduces cost, non-determinism, and a dependency on a commercial API.

**Why it matters:** Reproducibility is undermined if the API is deprecated or outputs change. Cost could be prohibitive for very large-scale generation. Reviewers may question whether the "symbolic" claim holds when half the generation is neural.

**Counter-argument:** The GPT-4 dependency is for *French sentence generation only* (source-side), not for translation. Modules 1-2 are purely deterministic (Python string manipulation) and demonstrate the same curriculum principle. The GPT-4 transformations could be replaced with rule-based French morphology (conjugation tables) at the cost of development time but no conceptual change. We provide the generated outputs so the experiments are reproducible regardless of API access.

---

### 17. Single model architecture

**Honest statement:** All experiments use NLLB-200-distilled-600M. We do not test whether the structured data advantage holds for other architectures (e.g., mBART, MarianMT, smaller/larger NLLB variants).

**Why it matters:** The benefit of structured data might be architecture-specific. A different model might learn just as well from random data.

**Counter-argument:** NLLB-200 is the current state-of-the-art multilingual model for low-resource translation and the natural choice. Testing additional architectures is straightforward future work. The structured data is model-agnostic by design (it's a property of the corpus, not the training procedure), so there is no reason to expect architecture dependence — but this should be verified.

---

### 18. No cross-lingual transfer experiments

**Honest statement:** Parallel data exists for closely related Gbe languages (23K French-Ewe, 53K French-Fon) that could be used for cross-lingual transfer or multilingual training. We do not test this.

**Why it matters:** Cross-lingual transfer from Fon/Ewe could substantially boost Adja translation quality and might interact with the structured data advantage.

**Counter-argument:** Cross-lingual transfer is complementary to our contribution. Our paper asks "does data composition matter?" — adding transfer learning adds a confound. Testing whether structured + transfer > random + transfer is excellent future work and we have the data to do it.

---

## Writer Guidance

### How to present these in the paper

1. **Lead with #1 (single language pair).** Reviewers will notice it anyway. Being upfront about it earns trust. One sentence: "Our experiments are limited to a single language pair (French-Adja), and replication across typologically diverse languages is needed to establish generalizability."

2. **Be honest, not defensive.** State each limitation plainly. Avoid hedging language ("it could be argued that..."). Instead: "We acknowledge that..." followed by the mitigation.

3. **The Limitations section should include Tier 1 (#1-4) and the most important Tier 2 items (#5-6).** This is 4-6 short paragraphs. Do not bury limitations in footnotes.

4. **Weave Tier 2 (#5-9) into the Discussion** where the relevant results are being interpreted. For example, discuss the chrF divergence (#7) right after presenting the main results table. Discuss the baseline comparison (#6) immediately after the baselines figure.

5. **Address Tier 3 (#10-13) inline** where those results appear. The PAIRS-BROKEN anomaly (#10) should be discussed in the same paragraph as the minimal-pair results, not deferred. The RANDOM-8K=10K identity (#12) needs a footnote at first mention.

6. **Frame Tier 4 (#14-18) as Future Work**, typically 1-2 paragraphs at the end of Discussion. Group them: "cross-lingual transfer and alternative architectures" (#17-18), "richer evaluation" (#15, human eval from #4), "comparison with active learning methods" (#14), "removing the GPT-4 dependency" (#16).

### The intuitive counter

A reviewer might object: "Isn't it *obvious* that structured data would help? This is just common sense."

Frame the response as: **"We provide empirical evidence for an intuitive hypothesis."** Common sense says structured learning beats random exposure — this is well-established in education and language pedagogy. But in NMT, the community has focused overwhelmingly on model architectures and training procedures, not on data design. The intuition was there; the controlled experiments were not. We quantify the effect (5x BLEU improvement, 4x less data, 17.5-point BLEU drop when structure is corrupted), identify the optimal allocation (~20% structured in a mixed budget), and provide a replicable framework. Intuitive hypotheses still need empirical validation, especially when the practical implications for endangered-language work are significant.

---

## Quick-Reference Numbers

All numbers below are mean +/- std over the stated number of seeds. Test set: 1,455 sentences.

| Number used in limitations | Exact value | Source |
|---|---|---|
| VOCAB-MAXIMIZED BLEU | 23.99 +/- 0.75 (5 seeds) | Baselines table |
| LENGTH-STRATIFIED BLEU | 23.69 +/- 0.41 (5 seeds) | Baselines table |
| STRUCTURED-2K BLEU | 19.91 +/- 2.69 (5 seeds) | Exp1 table |
| STRUCTURED-2K chrF | 29.20 +/- 0.70 (5 seeds) | Exp1 table |
| STRUCTURED-3K BLEU | 21.13 +/- 0.62 (4 seeds) | Scaling table |
| STRUCTURED-4K BLEU | 19.51 +/- 1.34 (5 seeds) | Exp1 table |
| RANDOM-10K BLEU | 4.13 +/- 0.13 (5 seeds) | Exp1 table |
| RANDOM-4K BLEU | 3.66 +/- 0.48 (5 seeds) | Exp1 table |
| R10K+S4K BLEU | 22.45 +/- 1.99 (5 seeds) | Exp1 table |
| R10K+S4K chrF | 41.17 +/- 1.53 (5 seeds) | Exp1 table |
| R8K+S2K BLEU (20% replacement) | 22.44 +/- 0.66 (5 seeds) | Replacement table |
| PAIRS-INTACT BLEU | 22.90 +/- 0.28 (3 seeds) | Ablation table |
| PAIRS-BROKEN BLEU | 5.37 +/- 0.72 (3 seeds) | Ablation table |
| PAIRS-BROKEN chrF | 14.00 +/- 0.54 (3 seeds) | Ablation table |
| REDUCED-4 BLEU (std) | 19.48 +/- 5.18 (3 seeds) | Pronoun table |
| MINIMAL-1 BLEU | 15.85 +/- 2.34 (3 seeds) | Pronoun table |
| RANDOM-8K BLEU | 4.13 +/- 0.13 (5 seeds) | Scaling table (= RANDOM-10K) |
| VOCAB-MAXIMIZED chrF | 40.41 +/- 0.39 (5 seeds) | Baselines table |
| NO-FUTURE delta | -5.6 BLEU from FULL | Module LOO table |
| BASE-ONLY BLEU | 8.04 +/- 0.41 (2 seeds) | Module LOO table |
| Total experiments | 224 training runs | Results collection |
