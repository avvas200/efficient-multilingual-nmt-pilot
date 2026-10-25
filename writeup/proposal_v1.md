# Deployment-Realistic Compression for Low-Resource Neural Machine
# Translation: Beyond Aggregate Quality

**Applicant:** Abbas Sani
**Target programme:** PhD, [primary university — e.g., The Chinese University of Hong Kong]
**Target supervisor(s):** [top 2–3 names from your shortlist]
**Fellowship:** Hong Kong PhD Fellowship Scheme (HKPFS), 2027 intake

---

## 1. Research Question and Motivation

Over four billion people speak languages that are poorly served by modern
large language models. For African languages such as Hausa, Yoruba, and
Swahili, the gap between the performance that cloud-hosted multilingual
systems advertise and what users actually receive on the devices they own
is substantial. Closing this gap is both a scientific and a deployment
problem: it requires models that are competent on these languages *and*
that run within the memory, latency, and energy budgets of consumer
hardware common in the regions where the languages are spoken.

Model compression — post-training quantisation, pruning, distillation —
is widely proposed as the bridge. Yet the compression literature evaluates
almost exclusively on English and other high-resource languages, and on
server-class accelerators. Two claims commonly made in that literature —
that compression preserves quality adequately, and that it reduces
inference latency — have not been systematically examined in the regime
where they matter most: low-resource languages on consumer GPUs.

**This proposal addresses the following research question:**
*Under what conditions do post-training compression methods preserve
multilingual translation quality for low-resource languages, and when
do their deployment-time costs undermine the practical case for
compression?*

The question has three components the proposed PhD will address in turn:
(a) **what** is lost under compression beyond what aggregate metrics
reveal; (b) **where** compression fails — which model families, language
pairs, and compression methods; and (c) **how** methods can be designed
to preserve both quality and realistic latency on consumer hardware.

## 2. Preliminary Findings

To establish feasibility and sharpen the research direction, I conducted
an 8-week pilot study [GitHub: TODO-link]. The pilot evaluated three
model families — a small general LLM (Qwen2.5-1.5B), a larger instruction-
tuned multilingual LLM (Aya-23-8B), and a dedicated multilingual MT model
(NLLB-200-distilled-1.3B) — under three post-training compression regimes
(FP16, bitsandbytes INT8, NF4 INT4) across six translation directions
spanning English and three African languages (Hausa, Swahili, Yoruba) on
the FLORES-200 benchmark [1]. Experiments ran on consumer GPUs (NVIDIA
T4) matching the hardware common in deployment contexts of interest.

**Four preliminary findings emerged:**

**(F1) Specialised MT models dominate general LLMs on low-resource
translation.** NLLB-1.3B outperformed Aya-23-8B by **25.5 chrF** on
hau→eng and **34.0 chrF** on eng→swh, despite being an order of
magnitude smaller. The widely-reported trend of general-purpose LLMs
closing on specialised MT does not hold in this low-resource regime.

**(F2) Modern post-training quantisation preserves aggregate quality
remarkably well.** Across all three models and six directions, chrF
degradation from FP16 to INT4 was within approximately 1 point in most
cells and effectively zero for NLLB. The intuition that aggressive
compression disproportionately harms low-resource languages was *not*
supported by aggregate metrics.

**(F3) On consumer GPUs, quantisation does not deliver its advertised
latency benefit.** All INT8 and INT4 configurations were slower than
FP16 by factors of 1.3× to 5.3×, due to on-the-fly dequantisation
overhead in current implementations of bitsandbytes. Memory savings
were substantial (49–61% at INT4), but memory alone is rarely the
deployment bottleneck in this regime.

**(F4) A clear resource gradient is visible across languages.** For
NLLB, chrF on X→English descends monotonically from Swahili (63.8) to
Hausa (56.1) to Yoruba (44.6), tracking training-data availability.
This gradient will be a key lever for the proposed research.

Findings (F2) and (F3) together motivate the proposed direction:
aggregate quality is not where the compression story breaks for low-
resource languages — latency and *what gets lost at the error level* are.

## 3. Proposed Research Programme

**Year 1: Characterisation.** Build a comprehensive, honest map of
compression effects across (model family × compression method ×
language resource level × error type). Extend the pilot to include
GPTQ, AWQ, and SmoothQuant [2,3]; add additional African, South Asian,
and Southeast Asian languages; and conduct fine-grained error analysis
that goes beyond chrF/BLEU — identifying specific linguistic phenomena
(morphology, named entities, numerals, hooked characters) that degrade
under compression even when aggregate metrics do not. **Target output:**
one workshop paper (AfricaNLP, WiNLP, or ENLSP@NeurIPS) by month 10.

**Year 2: Intervention.** Design and evaluate compression methods
intentionally aware of low-resource language structure. Candidate
directions include (a) **calibration-data-aware quantisation** —
using held-out low-resource text as the calibration corpus for GPTQ /
AWQ, testing whether this preserves rare-token embeddings more
effectively; (b) **selective-precision schemes** keeping embedding and
output projection layers in higher precision while quantising the
backbone, hypothesised to preserve low-frequency vocabulary; and (c)
**post-compression adapter recovery** — small, parameter-efficient
adapters trained to recover language-specific capability lost under
aggressive compression. **Target output:** one full paper submission
(ACL, EMNLP, or NAACL) by month 24.

**Year 3: Deployment case study and thesis.** Demonstrate the methods
on a deployment-realistic task: machine translation on consumer-grade
hardware for an African-language use case developed with real users
[reference your Aboki AI initiative here if the supervisor cares about
applied work]. Produce an open-source release and thesis.

## 4. Why This Research, Why Here

My background combines a Master's thesis on transformer optimisation
for low-resource language generation (Mewar University, 2025, CGPA
8.96/10) with production software engineering experience building
systems used across Nigerian civic contexts. I lead Aboki AI, an
ongoing initiative for Hausa-language AI access. This combination —
model-level research grounding plus production deployment experience
plus direct engagement with a low-resource language community —
positions me unusually well for research that spans ML methods and
deployment realism.

[Target supervisor] has produced work on [cite one specific paper of
theirs, e.g. parameter-efficient fine-tuning, model routing, efficient
inference]. The proposed programme extends that line into the low-
resource multilingual setting, where the trade-offs identified in the
pilot are most consequential and least understood.

## References

[1] NLLB Team et al. *No Language Left Behind: Scaling Human-Centered
    Machine Translation.* arXiv:2207.04672, 2022.
[2] Frantar et al. *GPTQ: Accurate Post-Training Quantization for
    Generative Pre-trained Transformers.* ICLR 2023.
[3] Xiao et al. *SmoothQuant: Accurate and Efficient Post-Training
    Quantization for Large Language Models.* ICML 2023.
[4] Costa-jussà et al. *Scaling Neural Machine Translation to 200
    Languages.* Nature 630, 2024.
[5] Üstün et al. *Aya Model: An Instruction Finetuned Open-Access
    Multilingual Language Model.* ACL 2024.

[Add 3-5 more references specific to the supervisor's recent work —
cite at least one of theirs by name.]