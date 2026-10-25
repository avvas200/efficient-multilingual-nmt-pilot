# Compression Effects on Low-Resource Multilingual Translation:
# A Pilot Study Across Three Model Families

**Author:** Abbas Sani
**Affiliation:** Independent / Arewa Data Science / Aboki AI
**Contact:** abbassani721@gmail.com
**Code and data:** https://github.com/avvas200/pilot
**Date:** [DRAFT — October 2026]

## Abstract

We present a pilot study characterising the effect of post-training
quantisation on multilingual machine translation quality and inference
cost for three African languages — Hausa, Swahili, and Yoruba — across
three model families: a small general-purpose LLM (Qwen2.5-1.5B), a
larger multilingual instruction-tuned LLM (Aya-23-8B), and a dedicated
multilingual MT model (NLLB-200-distilled-1.3B). Evaluating on the
FLORES-200 devtest set (1,012 sentences per direction) under FP16,
bitsandbytes INT8, and NF4 INT4 precisions on consumer-grade NVIDIA
T4 GPUs, we find: (i) the specialised MT model outperforms both LLMs
by 25–45 chrF, including the 8B instruction-tuned LLM; (ii) modern
post-training quantisation preserves aggregate translation quality
within approximately 1 chrF across all model families and language
directions; (iii) on consumer GPUs, INT8 and INT4 quantisation
provide substantial memory savings (49–61% at INT4) but *do not*
reduce inference latency relative to FP16 — in fact, latency
increases by 1.3× to 5.3×; (iv) a clear performance gradient
tracking training-resource availability is visible across languages
(Swahili > Hausa > Yoruba). These findings motivate a research
agenda focused on hardware-aware compression and error-type analysis
beyond aggregate metrics for low-resource deployment.

## 1. Introduction

Post-training compression methods — quantisation, pruning, distillation —
are routinely proposed as the bridge between the capabilities of
modern language models and the hardware budgets of deployment contexts.
Low-resource languages, often spoken by users in exactly the regions
where consumer-grade hardware predominates, are a natural target for
compression research. Yet most compression evaluation to date has
focused on English and other high-resource languages, and on
server-class accelerators.

This pilot study asks two concrete questions:

1. Does post-training quantisation disproportionately degrade
   translation quality for low-resource languages, as sometimes
   argued, or does modern quantisation preserve quality across
   resource levels?
2. On consumer GPUs representative of deployment hardware in
   low-resource contexts, do the memory benefits of quantisation
   translate to the latency benefits commonly implied in the
   literature?

We evaluate three model families spanning scale and specialisation,
three precision levels, and six translation directions across three
African languages, on a standard benchmark.

## 2. Experimental Setup

### 2.1 Models

| Model | Size | Architecture | Target coverage of our languages |
|---|---|---|---|
| Qwen/Qwen2.5-1.5B-Instruct | 1.5B | Causal LM | None of the three languages is a stated training focus. |
| CohereForAI/aya-23-8B | 8B | Causal LM, multilingual instruction-tuned | Aya-23 covers 23 languages; Hausa, Swahili, and Yoruba are not among them. |
| facebook/nllb-200-distilled-1.3B | 1.3B | Seq2seq, dedicated MT | All three languages (hau_Latn, swh_Latn, yor_Latn) are supported with real training data. |

### 2.2 Precisions evaluated

- **FP16** — baseline, native half-precision inference.
- **INT8** — bitsandbytes 8-bit weight quantisation via
  `BitsAndBytesConfig(load_in_8bit=True)`.
- **INT4** — bitsandbytes 4-bit NF4 quantisation with double
  quantisation and FP16 compute dtype
  (`load_in_4bit=True, bnb_4bit_quant_type="nf4",
  bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16`).

For Yoruba, only FP16 was evaluated across all three models due to
compute budget constraints; the compression sweep for Yoruba is
left for future work.

### 2.3 Benchmark

FLORES-200 devtest split [1], 1,012 parallel sentences per language.
Languages: `hau_Latn`, `swh_Latn`, `yor_Latn`, `eng_Latn`. Six
directions: hau↔eng, swh↔eng, yor↔eng.

### 2.4 Prompting

For NLLB (seq2seq), we follow the standard approach: setting
`tokenizer.src_lang` and forcing the target language BOS token at
generation. For causal LMs, we use the model's own chat template with
a direct translation instruction, deterministic greedy decoding
(`do_sample=False, num_beams=1`), `max_new_tokens=256`. Qwen and
Aya use slightly different prompt phrasings; details and exact
templates are in the repository.

### 2.5 Metrics

Primary: chrF (character 6-gram F-score) via SacreBLEU 2.6.0,
signature `nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0`.
Secondary: BLEU, spBLEU, chrF++. Operational: inference time per
sentence (wall-clock), peak GPU memory (summed across devices under
`device_map="auto"`), hypothesis length ratio versus reference, empty-
output rate.

### 2.6 Hardware

NVIDIA Tesla T4 (16GB VRAM) × 2 on Kaggle. Aya-23-8B FP16 required
multi-GPU sharding via `device_map="auto"`; all other configurations
fit on a single T4.

### 2.7 Reproducibility

Python 3.13.15; PyTorch 2.11.0+cu128; transformers 5.16.1;
bitsandbytes 0.50.2; sacrebleu 2.6.0; datasets 3.0.1. All raw results,
hypothesis files, and notebooks are available at the repository above.

## 3. Results

### 3.1 Translation quality: specialised MT dominates

Table 1 reports chrF across all nine model × precision combinations
and six directions.

**Table 1. chrF on FLORES-200 devtest (1,012 sentences per direction).
Higher is better.**

| Model | Precision | hau→eng | eng→hau | swh→eng | eng→swh | yor→eng | eng→yor |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | FP16 | 23.28 | 16.85 | 24.19 | 24.39 | 22.90 | 12.74 |
| Qwen2.5-1.5B | INT8 | 23.33 | 18.91 | 24.16 | 24.09 | — | — |
| Qwen2.5-1.5B | INT4 | 22.49 | 17.33 | 23.46 | 23.34 | — | — |
| Aya-23-8B | FP16 | 23.35 | 15.48 | 29.73 | 16.73 | 19.88 | 5.27 |
| Aya-23-8B | INT8 | 23.12 | 15.15 | 29.45 | 16.53 | — | — |
| Aya-23-8B | INT4 | 22.23 | 15.61 | 27.54 | 16.15 | — | — |
| **NLLB-1.3B** | **FP16** | **56.08** | **51.84** | **63.77** | **60.06** | **44.55** | **25.67** |
| NLLB-1.3B | INT8 | 56.11 | 51.94 | 63.78 | 59.93 | — | — |
| NLLB-1.3B | INT4 | 56.08 | 51.80 | 63.66 | 59.29 | — | — |

NLLB-1.3B outperforms Aya-23-8B by **25.5 chrF** on hau→eng,
**34.0 chrF** on eng→swh, and **36.3 chrF** on eng→hau, despite being
roughly a sixth of the size. The gap over Qwen-1.5B is of similar
magnitude. For eng→yor specifically, Aya-23-8B produces chrF 5.27 —
essentially non-functional output — confirming that the language is
outside its training distribution.

### 3.2 Compression effects on quality

Table 2 reports FP16→INT4 chrF deltas per model and direction.

**Table 2. chrF degradation (INT4 − FP16). Negative = quality loss.**

| Direction | Qwen-1.5B | Aya-23-8B | NLLB-1.3B |
|---|---|---|---|
| hau→eng | −0.79 | −1.12 | 0.00 |
| eng→hau | +0.48 | +0.13 | −0.04 |
| swh→eng | −0.73 | −2.19 | −0.11 |
| eng→swh | −1.05 | −0.58 | −0.77 |

Across all four Hausa/Swahili directions and all three model families,
chrF degradation under NF4 INT4 quantisation is within ~1 chrF of FP16
baseline for most configurations, with NLLB showing effectively zero
degradation on hau→eng, eng→hau, and swh→eng. The largest observed
loss is 2.19 chrF (Aya swh→eng). The Qwen eng→hau "improvement" under
INT4 of +0.48 chrF should be interpreted as noise at a chrF level
(~17) where the model is barely producing coherent translation output;
without a confidence-interval analysis across multiple seeds, no
quantisation benefit should be claimed here.

The common intuition that aggressive quantisation disproportionately
harms low-resource languages is **not supported** in aggregate by this
data. We note two important caveats to this null result: (i) aggregate
chrF may hide systematic error-type changes (addressed in §4); (ii)
we evaluate only bitsandbytes INT8 and NF4 INT4; GPTQ and AWQ are left
for future work.

### 3.3 Memory and latency

Table 3 reports peak GPU memory and inference latency, averaged across
the four Hausa/Swahili directions.

Two patterns stand out. First, memory savings are substantial and
reliable: 32–42% at INT8 and 49–61% at INT4, scaling with model size.
For Aya-23-8B specifically, INT4 reduces peak memory from 17.1 GB to
6.6 GB, moving the model from "requires server-class VRAM or multi-GPU
sharding" to "fits comfortably on a single consumer GPU." Second,
latency does *not* decrease — all quantised configurations are slower
than FP16. INT4 slowdowns are modest and comparable across models
(1.29×–1.51×), consistent with a per-matmul dequantisation cost that
dominates the small compute budget of each T4 operation. INT8
slowdowns are more severe and more variable: 1.93× for Aya-23-8B,
4.13× for Qwen-1.5B, and 5.29× for NLLB-1.3B. The smaller models
are hit hardest in relative terms, which we attribute to their FP16
baselines already being latency-bound by the per-call overhead of
the generation loop rather than by compute, so added dequantisation
cost represents a larger fractional increase. This pattern — memory
savings without latency savings on consumer GPUs — undermines a
common assumption that post-training quantisation delivers both. For
the deployment contexts motivating this work, pre-packed-weight
schemes (GPTQ, AWQ) or kernel-level INT8 implementations are more
promising than bitsandbytes-style dynamic dequantisation, and are a
natural extension of this study.

**Table 3. Memory and latency across precisions (avg of 4 hau/swh directions).**

| Model | Precision | Peak GPU memory | Memory saving vs FP16 | Latency (sec/sent) | Latency vs FP16 |
|---|---|---|---|---|---|
| Qwen-1.5B | FP16 | 3.64 GB | — | 0.124 | — |
| Qwen-1.5B | INT8 | 2.41 GB | 34% | 0.511 | 4.13× slower |
| Qwen-1.5B | INT4 | 1.70 GB | 53% | 0.161 | 1.3× slower |
| Aya-23-8B | FP16 | 17.14 GB | — | 0.659 | — |
| Aya-23-8B | INT8 | 10.01 GB | 42% | 1.271 | 1.93× slower |
| Aya-23-8B | INT4 | 6.64 GB | 61% | 0.850 | 1.29× slower |
| NLLB-1.3B | FP16 | 3.49 GB | — | 0.036 | — |
| NLLB-1.3B | INT8 | 2.38 GB | 32% | 0.191 | 5.29× slower |
| NLLB-1.3B | INT4 | 1.78 GB | 49% | 0.054 | 1.51× slower |

Two patterns stand out. First, memory savings are substantial and
reliable: 32–42% at INT8 and 49–61% at INT4, scaling with model size.
For Aya-23-8B specifically, INT4 reduces memory from 17.1 GB to 6.6 GB,
moving the model from "requires server GPU" to "fits comfortably on
consumer hardware." Second, latency does *not* decrease — all
quantised configurations are slower than FP16, with INT8 slowdowns ranged 
from 1.9× (Aya-23-8B) to 5.3× (NLLB-1.3B), with the smaller models hit hardest
in relative terms. This is attributable to the on-the-fly dequantisation overhead inherent to the bitsandbytes implementation on the T4 architecture. 
Pre-packed-weight schemes such as GPTQ or AWQ may partially recover latency benefits and are a natural extension.

### 3.4 Resource gradient across languages

For NLLB-1.3B at FP16, chrF on X→English descends monotonically:
Swahili (63.77) > Hausa (56.08) > Yoruba (44.55). The same ordering
holds in the eng→X direction (60.06, 51.84, 25.67). This gradient
tracks reported training-data availability for these languages in the
NLLB-200 training corpus and is a useful axis for subsequent
compression research: compression effects at the "head" of the gradient
(Swahili) may differ qualitatively from the "tail" (Yoruba), where
further analysis is needed.

### 3.5 Model-specific observations

**Aya-23-8B over-generates.** The hypothesis-to-reference length ratio
for Aya ranged from 1.14 (hau→eng) to 1.88 (eng→swh). Compared with
NLLB's near-unity ratios (0.96–0.99), Aya systematically produces
longer outputs than the reference, consistent with its
instruction-tuned behaviour not being adapted to concise translation
output. This partially explains Aya's lower chrF beyond raw
translation quality.

**Qwen tokenisation of African orthographic markers.** Qwen2.5's
byte-level BPE tokenises Hausa hooked letters (ɓ, ɗ, ƙ, ƴ) as 2-byte
sequences, often fragmenting them into multiple subword pieces.
Preliminary inspection suggests this may contribute to Qwen's weak
eng→hau performance; a controlled study is left for future work.

## 4. Limitations and Threats to Validity

**(L1) Compression methods evaluated are limited.** We test only
bitsandbytes INT8 and NF4 INT4. GPTQ, AWQ, SmoothQuant, and distillation
are absent, and the null quality-degradation finding may not generalise.

**(L2) Only one benchmark.** FLORES-200 devtest is a single-domain,
sentence-level, written-language benchmark. Behaviour on noisier
real-world text (social media, speech transcripts) may differ.

**(L3) Statistical significance of small deltas not fully established.**
Bootstrap confidence intervals computed on a subset of configurations
suggest most compression deltas lie within noise, but a complete
significance table is pending. We treat all deltas below ~1 chrF as
inconclusive.

**(L4) Latency measurements are on a specific hardware configuration.**
Our T4 results reflect consumer-GPU behaviour but do not generalise to
A100/H100-class accelerators, where quantised kernels are better
optimised and may recover latency benefits.

**(L5) Yoruba sweep incomplete.** Only FP16 numbers are available for
Yoruba due to Kaggle session-time limits during Aya-23-8B runs. INT8
and INT4 Yoruba numbers are future work.

**(L6) Pilot was conducted by a single researcher over 8 weeks.**
No independent replication of individual runs. Partial mitigations:
all code, hyperparameters, and library versions are released; FLORES
is a public benchmark; metrics follow SacreBLEU standardised
signatures.

## 5. Discussion and Research Directions

Three findings motivate specific follow-on research:

**On the LLM-vs-specialised-MT gap (F1).** The 25–45 chrF gap between
NLLB-1.3B and the larger Aya-23-8B on low-resource African languages
suggests that at current scales and training recipes, dedicated MT
models remain decisively stronger for genuinely low-resource language
pairs. Research on "efficient low-resource translation" should
therefore prioritise making specialised MT models more efficient and
adaptable, rather than expecting general LLMs to close the gap through
scaling or compression alone.

**On the aggregate-quality finding (F2).** Null results in aggregate
chrF should prompt — not foreclose — finer-grained analysis. Specific
linguistic phenomena (rare vocabulary, morphology, named entities,
numerals, non-ASCII orthography) may degrade under compression even
when overall chrF does not. Error-type analysis across compression
regimes is a direct and productive follow-on.

**On the latency finding (F3).** Compression methods designed with
consumer-GPU deployment as a first-class constraint — including
pre-packed-weight quantisation, hardware-aware kernel implementations,
and selective-precision schemes — are a natural research direction,
particularly for the low-resource contexts where consumer GPUs are
most common.

## Acknowledgements

Compute: Kaggle (NVIDIA T4). Benchmark: FLORES-200. Models: Alibaba
Qwen team, Cohere For AI (Aya), Meta AI (NLLB).

## References

[1] NLLB Team et al. *No Language Left Behind: Scaling Human-Centered
    Machine Translation.* arXiv:2207.04672, 2022.
[2] Dettmers et al. *LLM.int8(): 8-bit Matrix Multiplication for
    Transformers at Scale.* NeurIPS 2022.
[3] Dettmers et al. *QLoRA: Efficient Finetuning of Quantized LLMs.*
    NeurIPS 2023. (NF4 reference)
[4] Üstün et al. *Aya 23: Open Weight Releases to Further Multilingual
    Progress.* 2024.
[5] Frantar et al. *GPTQ: Accurate Post-Training Quantization for
    Generative Pre-trained Transformers.* ICLR 2023.
[6] Lin et al. *AWQ: Activation-aware Weight Quantization for LLM
    Compression and Acceleration.* MLSys 2024.
[7] Post. *A Call for Clarity in Reporting BLEU Scores.* WMT 2018.
    (SacreBLEU reference)
[8] Goyal et al. *The Flores-101 Evaluation Benchmark for Low-Resource
    and Multilingual Machine Translation.* TACL 2022.

## Appendix A: Per-direction detailed results

[Include your full `combined_all.csv` as a formatted table here, or
link to it in the repo. Keep this appendix for anyone who wants
every number.]

## Appendix B: Qualitative Sample Outputs

**How to read this appendix.** The aggregate metrics in §3 compress
each system's behaviour on 1,012 sentences into a single number. This
appendix complements those numbers with concrete examples: for a
small, hand-picked set of source sentences, we show every system's
output side by side with the human reference translation. The purpose
is twofold. First, it lets the reader calibrate what each chrF level
corresponds to in practice — "chrF 56" for NLLB and "chrF 23" for Qwen
on hau→eng describe qualitatively different translation experiences,
and the examples make this concrete. Second, it exposes error types
that aggregate metrics hide: category-level failures (output in the
wrong language), consistent biases (systematic over-generation), and
compression-induced errors that occur without a visible aggregate
chrF drop. We sample three difficulty tiers per direction — simple,
medium, hard — chosen from the FLORES-200 devtest to span the range
of inputs the evaluation covers. All outputs are produced under the
exact decoding configuration described in §2.4; nothing has been
edited. For reasons of space we present three of the six evaluated
directions (hau→eng, eng→hau, eng→yor), chosen to foreground the
specialised-vs-general gap (F1), the resource-gradient effect (F4),
and the out-of-distribution behaviour of Aya-23 on Yoruba.

To illustrate what the aggregate chrF scores correspond to as
translation output, we present selected source–hypothesis–reference
triples. For each direction, we include three sentences spanning
easy, medium, and hard difficulty, with outputs from all three
models at FP16 and INT4.

### B.1 Hausa → English

#### Example 1 (simple, index 42)

**Source (hau):** Wannan zai ba shi damar zama mai dacewa da baya 802.11a, 802.11b da 802.11g, matukar dai tashar tushe tana da rediyo biyu.
**Reference (eng):** This will allow it to be backwards compatible with 802.11a, 802.11b and 802.11g, provided that the base station has dual radios.

| System | Output |
|---|---|
| NLLB-1.3B FP16 | This would allow it to be backward compatible with 802.11a, 802.11b and 802.11g, provided the base station has two radios. |
| NLLB-1.3B INT4 | This will allow it to be backward compatible with 802.11a, 802.11b and 802.11g, provided the base station has two radios. |
| Qwen-1.5B FP16 | When you connect to Wi-Fi, it supports 802.11a, 802.11b and 802.11g standards, including Bluetooth connectivity for your phone |
| Qwen-1.5B INT4 | When you turn on your Wi-Fi, it supports both 802.11a, 802.11b, and 802.11g standards, as well as Bluetooth for wireless headphones |
| Aya-23-8B FP16 | The range of the 802.11a, 802.11b, and 802.11g is up to 54 Mbps, which is more than enough for most home and small office networking applications. |
| Aya-23-8B INT4 | The 802.11a, 802.11b, and 802.11g are all part of the 802.11 family of standards for wireless LANs. They are all based on the same underlying technology but offer different features and capabilities. |

*Commentary (1-2 sentences):* NLLB produces a near-perfect translation
preserved under compression. Qwen and Aya both substantially mistranslate,
though Aya's error type differs (over-generation vs lexical substitution).