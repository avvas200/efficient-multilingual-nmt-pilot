# Deployment-Realistic Compression for Low-Resource NMT

**Pilot study:** How does post-training quantisation affect multilingual translation quality and inference latency for low-resource African languages on consumer GPUs?

This repository accompanies a pilot study and research proposal prepared for PhD applications (2027 intake). Full technical report and proposal are in [`writeup/`](writeup/).

## Summary of findings

Evaluated three model families across three precisions on FLORES-200 translation covering Hausa, Swahili, and Yoruba:

- **NLLB-200-distilled-1.3B** dominates both LLMs by **25–45 chrF** on low-resource African translation, despite being smaller than Aya-23-8B.
- **Modern post-training quantisation** (bitsandbytes INT8, NF4 INT4) preserves aggregate chrF within ~1 point of FP16 across all model families and language directions.
- **On consumer GPUs (T4)**, quantisation delivers substantial memory savings (49–61% at INT4) but **latency increases** by 1.3× to 5.3× relative to FP16 — undermining a common assumption about compression and deployment.
- A clear **resource gradient** is visible across languages: Swahili > Hausa > Yoruba, tracking training-data availability.

Full numbers and discussion: [`writeup/technical_report.md`](writeup/technical_report.md).

## Repository structure

pilot/
├── notebooks/ # Kaggle/Colab notebooks for each experiment stage
│ ├── 01_week1_baseline.ipynb # Qwen + Aya + NLLB sweep across FP16/INT8/INT4
│ ├── 02_yoruba_extension.ipynb # Yoruba FP16 across all three models
│ └── 03_bootstrap_cis.ipynb # chrF confidence intervals (Colab, CPU)
├── results/
│ ├── combined_all.csv # All model × precision × direction results
│ ├── table_main_chrf.csv # Publication-ready chrF table
│ └── hyps/ # Per-sentence SRC/HYP/REF triples
├── writeup/
│ ├── proposal.md # 2-page research proposal
│ └── technical_report.md # Full technical report with tables
├── requirements.txt
└── README.md


## Reproducing the main results

**Setup.** Experiments were run on Kaggle (NVIDIA Tesla T4 ×2) with Python 3.13.15. Full dependency pins in [`requirements.txt`](requirements.txt).

**Notebooks.** Each notebook under `notebooks/` is standalone and can be opened directly in Kaggle:
1. `01_week1_baseline.ipynb` — the main sweep. Expected runtime: ~10 hours on T4×2 (3 models × 3 precisions × 4 directions × 1012 sentences).
2. `02_yoruba_extension.ipynb` — Yoruba FP16 across all three models.
3. `03_bootstrap_cis.ipynb` — statistical significance analysis (runs on CPU in Colab, ~30 min).

**Data.** FLORES-200 devtest is loaded on-the-fly from HuggingFace (`facebook/flores`). All model checkpoints are also loaded from the Hub (`Qwen/Qwen2.5-1.5B-Instruct`, `CohereForAI/aya-23-8B`, `facebook/nllb-200-distilled-1.3B`). Aya-23-8B is a gated model; an HF access token is required.

**Results reproduction.** The headline table (Table 1 in the technical report) can be reproduced from `results/combined_all.csv` directly — no compute required — with `results/table_main_chrf.csv` as the pivot output. See `03_bootstrap_cis.ipynb` for the significance analysis.

## Limitations

Honest limitations are documented in §4 of the technical report. Briefly: only bitsandbytes compression methods were evaluated (GPTQ/AWQ left for future work); only FLORES-200 as benchmark; Yoruba compression sweep is incomplete; latency measurements are specific to T4 consumer hardware.

## Citation

If this work is useful to you, please reach out — happy to discuss. Formal citation forthcoming.

## Contact

Abbas Sani — abbassani721@gmail.com
Based in Kano, Nigeria. Preparing PhD applications for 2027 intake.

## Acknowledgements

Compute: Kaggle. Benchmark: FLORES-200 (Meta AI). Models: Alibaba Qwen team, Cohere For AI, Meta AI (NLLB).