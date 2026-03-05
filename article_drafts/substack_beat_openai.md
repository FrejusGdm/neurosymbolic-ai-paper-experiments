# How 1,800 Sentences Outperformed GPT-4 on My Grandmother's Language

My grandmother speaks Adja. So does about a million other people across Benin and Togo. If you've never heard of it, that's kind of the point.

Adja is a Gbe language — part of the same family as Fon and Ewe — with a rich tonal system, complex verb morphology, and essentially zero presence on the internet. Google Translate doesn't know it exists. ChatGPT will politely hallucinate if you ask it to translate French into Adja. There is no Wikipedia in Adja, no parallel corpus sitting on a university server, no dataset on Hugging Face.

When I decided to build a French-to-Adja translator, I knew I was starting from nothing. No training data. No benchmarks. No pretrained models that had ever seen a single Adja sentence.

What I didn't know was that the solution would have almost nothing to do with which AI model I used — and everything to do with how I designed 1,800 sentences.

---

## The Experiment: David vs Goliath

Here's the setup. I fine-tuned two models on the exact same 1,800 French-Adja sentence pairs:

- **NLLB-200** — Meta's open-source translation model. 600 million parameters. Runs on a free Google Colab GPU. Designed for low-resource languages, but has never seen Adja.
- **GPT-4.1** — OpenAI's frontier model. Fine-tuned through their API. The kind of model companies spend millions training.

Same data. Same test set of 1,455 sentence pairs. Same BLEU score evaluation.

The results:

| Model | Parameters | Training Data | BLEU Score |
|-------|-----------|---------------|------------|
| NLLB-200 (fine-tuned) | 600M | 1,800 structured sentences | **19.9** |
| GPT-4.1 (fine-tuned) | undisclosed (massive) | 1,800 structured sentences | **15.2** |

The small, free, open-source model beat OpenAI's flagship by nearly 5 BLEU points.

**[GRAPH 1: Model Comparison Bar Chart — NLLB-200 vs GPT-4.1, both fine-tuned on STRUCTURED-2K. Side-by-side bars, BLEU on y-axis. Colors: blue for NLLB, red/orange for GPT-4.1.]**

Now, I want to be careful here. This is one language pair, one seed for GPT-4.1 (vs. five seeds averaged for NLLB), and a proof-of-concept scale. I'll get to the caveats. But these numbers are real, and they tell an important story — just not the one you might expect.

The story isn't "open-source beats proprietary." The story is that for a language the internet has never seen, the model matters far less than the data you feed it.

---

## The Real Story: It's Not the Model, It's the Data

Here's the finding that actually keeps me up at night.

I also trained NLLB-200 on 10,000 randomly selected sentences from Tatoeba — a community-sourced translation database. Ten thousand sentences. More than five times the data.

BLEU score: **4.1**.

Let me say that again. 1,800 *structured* sentences scored 19.9. 10,000 *random* sentences scored 4.1. Five times less data, five times better performance. The gap isn't marginal — it's the difference between a usable translator and gibberish.

So what makes "structured" different from "random"?

### The Five-Module Curriculum

Instead of scraping whatever French-Adja sentence pairs I could find, I *designed* the training data like a language textbook. Five modules, each building on the last:

**Module 1 — Present tense basics.** Simple subject-verb-object sentences across 8 pronouns and 10 common verbs:
> *Je mange du riz.* (I eat rice.)
> *Elle achete des vetements.* (She buys clothes.)

**Module 2 — Negation.** Take every Module 1 sentence and negate it:
> *Je ne mange pas de riz.* (I don't eat rice.)

**Module 3 — Past tense.** Transform to passe compose:
> *J'ai mange du riz.* (I ate rice.)

**Module 4 — Future tense.** Transform to near future:
> *Je vais manger du riz.* (I'm going to eat rice.)

**Module 5 — Questions.** Both yes/no and wh-questions:
> *Est-ce que je mange du riz ?* (Do I eat rice?)
> *Qu'est-ce que je mange ?* (What do I eat?)

The key insight: every sentence in Modules 2-5 links back to exactly one sentence in Module 1. Only one grammatical feature changes at a time. The model doesn't just see translations — it sees *how French grammar transforms*, and can learn the corresponding Adja patterns.

This is the opposite of how most NMT datasets are built. The standard approach is: get as many sentence pairs as you can, from wherever you can, and hope the model figures it out. For high-resource languages with millions of examples, that works. For Adja, it's a disaster.

**[GRAPH 2: Data Efficiency — Structured vs Random scaling curves. X-axis: training set size (200, 500, 1K, 2K, 4K, 10K). Y-axis: BLEU. Two lines: structured (steep rise, plateaus ~20) vs random (flat near 2-4 regardless of size). The structured line at 200 sentences already beats random at 10,000.]**

Look at the scaling curves. Random data barely improves as you add more — 200 random sentences score 0.5 BLEU, 10,000 random sentences score 4.1. The model is drowning in noise. Structured data, on the other hand, shows clear gains with every increment: 200 structured sentences already hit 9.4 BLEU. By 1,000, it's at 18.8. The model is *learning grammar*, not memorizing phrases.

---

## The Kill Shot: Same Data, Different Structure

You might be thinking: "Maybe the structured sentences are just better sentences. Cleaner, simpler, easier to translate. It's not the structure — it's the quality."

Fair objection. So I ran the ablation.

I took the full set of 3,823 structured sentences — all five modules — and broke the minimal pair links. Same sentences, same translations, but shuffled so that "je mange du riz" is no longer linked to "je ne mange pas de riz." The model sees the same data. It just can't see how the pieces connect.

| Condition | Sentences | BLEU Score |
|-----------|-----------|------------|
| Minimal pairs **intact** | 3,823 | **22.9** |
| Minimal pairs **broken** | 3,823 | **5.4** |

Same sentences. Same translations. The only difference is whether the model can see the grammatical relationships between them. BLEU drops from 22.9 to 5.4 — a fourfold collapse.

**[GRAPH 3: Pairs Intact vs Broken — Two bars, same data size (3,823). Intact: 22.9 (green). Broken: 5.4 (red). Dramatic visual contrast. Maybe add a dotted line at random-10K level (4.1) to show that broken pairs perform barely better than random noise.]**

This is the result that makes the case. It's not about having "good" sentences. It's about organizing them so the model can extract grammatical patterns. Structure is the signal. Remove it, and you're back to noise.

---

## What This Means for Endangered Languages

There are roughly 7,000 languages spoken on Earth. Most of them will never have a million-sentence parallel corpus. Most of them will never be added to Google Translate. The standard playbook — "just get more data" — doesn't work when the data doesn't exist and the speakers number in the thousands or millions rather than billions.

But here's what our results suggest: you might not need more data. You need *better* data.

A linguist who understands the target language's grammar, a set of sentence templates, and 1,800 carefully designed examples can produce a translation system that outperforms a model trained on 5x more random data — and outperforms GPT-4.1 fine-tuned on the same examples.

This isn't about AI replacing linguists. It's about linguists and AI working together. The linguistic structure is the ingredient that makes small data powerful. Without it, you can throw all the compute and all the parameters in the world at the problem, and you'll get BLEU 4.

The tools to do this are free. NLLB-200 is open-source. Google Colab gives you a free GPU. The sentence generation pipeline is a few hundred lines of Python. The bottleneck isn't technology — it's the linguistic expertise to design the curriculum, and a native speaker to provide the translations.

---

## The Honest Part

I want to be transparent about what this is and isn't.

**What it is:** A proof-of-concept on one language pair (French to Adja) showing that data structure dramatically outperforms data quantity in extremely low-resource NMT, and that a small open-source model outperforms GPT-4.1 fine-tuning on the same structured data.

**What it isn't (yet):**
- The GPT-4.1 result is from a single training run (one seed), while the NLLB results are averaged across five seeds. More GPT-4.1 runs would give a fairer comparison.
- We tested on 10 common verbs across 8 pronouns. Real-world Adja has far more complexity — tone marking, serial verb constructions, aspect distinctions our curriculum doesn't yet cover.
- BLEU 19.9 is a promising start, not a production-ready translator. For context, high-resource language pairs like French-English score in the 40s and 50s.

But none of these caveats change the core finding: structure beats quantity, and a 600M-parameter model beats a frontier model when the data is designed right.

---

## What's Next

A few things I'm working on:

**Zero-shot GPT-4 baseline.** We fine-tuned GPT-4.1, but what does raw GPT-4 produce for Adja without *any* training data? My guess: near-zero BLEU, since it's never seen the language. This would make the fine-tuning comparison even sharper.

**Gemini fine-tuning.** We're running the same experiment with Google's Gemini to see if the pattern holds across API providers. Results coming soon.

**Cross-lingual transfer.** Adja is closely related to Fon (~53K French-Fon pairs exist) and Ewe (~23K French-Ewe pairs). Can we bootstrap Adja translation by pre-training on related languages, then fine-tuning on our structured curriculum?

**Scaling the curriculum.** Our current system covers 10 verbs. What happens with 50? 100? There's a sweet spot between curriculum coverage and diminishing returns, and we haven't found it yet.

If you're working on low-resource NMT, or if you speak a language that the internet has forgotten, I'd love to hear from you. The code for generating the structured corpus is open. The approach generalizes to any language pair where you have a linguist and a translator.

My grandmother's language deserves better than "Sorry, this language is not supported." And now I know that building something better doesn't require OpenAI's budget — just the right 1,800 sentences.

---

*Josue Godeme is a researcher working on neural machine translation for West African languages. This work is part of an ongoing research project on linguistically-informed corpus design for low-resource NMT.*

*All experimental code is available on GitHub. The Adja translation data remains private out of respect for the language community.*
