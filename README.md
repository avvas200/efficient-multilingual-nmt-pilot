# Deployment-Realistic Compression for Low-Resource NMT

**Pilot study:** How does post-training quantisation affect multilingual translation quality and inference latency for low-resource African languages on consumer GPUs?

This repository accompanies a pilot study and research proposal prepared for PhD applications (2027 intake). Full technical report and proposal are in [`writeup/`](writeup/).

## Summary of findings

Evaluated three model families across three precisions on FLORES-200 translation covering Hausa, Swahili, and Yoruba:

- **NLLB-200-distilled-1.3B** dominates both LLMs by **32.7–43.3 chrF** on the Hausa and Swahili directions, and by **12.9–24.7 chrF** on Yoruba — despite being roughly a sixth the size of Aya-23-8B.
- **Modern post-training quantisation** (bitsandbytes INT8, NF4 INT4) preserves aggregate chrF within ~1 point of FP16 in most cells across all model families and language directions; the single worst case is −2.19 chrF.
- **On consumer GPUs (T4)**, quantisation delivers substantial memory savings (49–61% at INT4) but **latency increases** by 1.3× to 5.3× relative to FP16 — undermining a common assumption about compression and deployment.
- A clear **resource gradient** is visible across languages: Swahili > Hausa > Yoruba, tracking training-data availability.

Full numbers and discussion: [`writeup/technical_report_v1.md`](writeup/technical_report_v1.md). The arXiv preprint source (corrected and self-contained) is in [`writeup/arxiv/`](writeup/arxiv/).

## Repository structure

```
efficient-multilingual-nmt-pilot/
├── notebooks/                      # Kaggle notebooks for each experiment stage
│   ├── 01_week1_baseline.ipynb     # Qwen + Aya + NLLB sweep across FP16/INT8/INT4
│   ├── 02_yoruba_extension.ipynb   # Yoruba FP16 across all three models
│   └── 03-bootstrap-cis.ipynb      # final-output collection run (see note below)
├── results/
│   ├── combined_all.csv            # All model × precision × direction results
│   ├── table_main_chrf.csv         # Publication-ready chrF pivot
│   └── yoruba_full_*.csv           # Raw output of the Yoruba extension run
├── writeup/
│   ├── proposal_v1.md              # 2-page research proposal (PhD applications)
│   ├── technical_report_v1.md      # Original technical report draft (superseded)
│   └── arxiv/                      # arXiv preprint source (LaTeX) + SUBMISSION.md
├── scripts/
│   ├── gen_appendix.py             # Regenerates the appendix table from the CSV
│   └── verify_paper.py             # Re-derives every paper number from the CSV
├── requirements.txt
└── README.md
```

> **Note on `03-bootstrap-cis.ipynb`.** Despite its filename, this notebook
> does not compute bootstrap confidence intervals — it is a final
> output-collection run of the main sweep. No significance analysis was
> performed for this pilot; see limitation L3 in the preprint. Per-sentence
> hypothesis files were not retained, which is why no `results/hyps/`
> directory is shipped.


## Reproducing the main results

**Setup.** Experiments were run on Kaggle (NVIDIA Tesla T4 ×2) with Python 3.13.15. Full dependency pins in [`requirements.txt`](requirements.txt).

**Notebooks.** Each notebook under `notebooks/` is standalone and can be opened directly in Kaggle:
1. `01_week1_baseline.ipynb` — the main sweep. Expected runtime: ~10 hours on T4×2 (3 models × 3 precisions × 4 directions × 1012 sentences).
2. `02_yoruba_extension.ipynb` — Yoruba FP16 across all three models.
3. `03-bootstrap-cis.ipynb` — final output-collection run (GPU). The filename is a leftover from a planned significance analysis that was not carried out.

**Data.** FLORES-200 devtest is loaded on-the-fly from HuggingFace (`facebook/flores`). All model checkpoints are also loaded from the Hub (`Qwen/Qwen2.5-1.5B-Instruct`, `CohereForAI/aya-23-8B`, `facebook/nllb-200-distilled-1.3B`). Aya-23-8B is a gated model; an HF access token is required.

**Results reproduction.** Every table in the preprint — including the full per-direction appendix — is derived from `results/combined_all.csv` directly, with no compute required; `results/table_main_chrf.csv` is the chrF pivot. `python scripts/verify_paper.py .` re-derives all 115 numbers asserted in the paper from that CSV and fails on any mismatch. No significance analysis was performed (see L3).

## Limitations

Limitations are documented in §4 of the preprint. Briefly: only bitsandbytes compression methods were evaluated (GPTQ/AWQ left for future work); only FLORES-200 as benchmark; Yoruba compression sweep is incomplete; **no statistical significance testing was performed**, so no delta below ~1 chrF should be read as a real effect; **per-sentence outputs were not retained**, which blocks independent re-scoring and error analysis; latency measurements are specific to T4 consumer hardware; and no run was independently replicated.

## Citation

If this work is useful to you, please reach out — happy to discuss. Formal citation forthcoming.

## Contact

Abbas Sani — abbassani721@gmail.com
Based in Kano, Nigeria. Preparing PhD applications for 2027 intake.

## Acknowledgements

Compute: Kaggle. Benchmark: FLORES-200 (Meta AI). Models: Alibaba Qwen team, Cohere For AI, Meta AI (NLLB).