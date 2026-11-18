# arXiv submission checklist

Source for the preprint *Compression Effects on Low-Resource Multilingual
Translation: A Pilot Study Across Three Model Families*.

## Files to upload

Upload these three files (or a `.tar.gz` containing them). Nothing else.

| File | Role |
|---|---|
| `main.tex` | The paper. Entry point. |
| `appendix_a.tex` | Appendix A longtable, `\input` by `main.tex`. |
| `refs.bib` | **Not used by the build.** Optional; safe to omit. |

`refs.bib` is kept in sync for reuse if you retarget the paper to an
ACL/NeurIPS template. `main.tex` uses an embedded `thebibliography`
instead, so arXiv needs no BibTeX pass and there is no `.bbl` to upload.

## Build

```bash
pdflatex main && pdflatex main
```

Two passes — the second resolves `\ref` and the longtable column widths.
Only standard TeX Live packages are used (`geometry`, `booktabs`,
`longtable`, `array`, `caption`, `microtype`, `hyperref`, `amsmath`,
`amssymb`, `url`). **This has not been compiled locally** — no TeX
distribution is installed on this machine. Build it once before
uploading, either locally or on Overleaf, and check:

- the Appendix A longtable breaks cleanly across pages;
- Table 2's `\cmidrule` spans line up over the two three-column groups;
- the Section 2.1 model table doesn't overrun the margin.

## Regenerating and checking the tables

Both run on the CSV alone — no GPU, no model downloads.

```bash
python scripts/gen_appendix.py results/combined_all.csv > writeup/arxiv/appendix_a.tex
```

```bash
python scripts/verify_paper.py .
```

`verify_paper.py` re-derives all 115 numbers asserted in `main.tex`
(every Table 1/2/3 cell, every prose figure and range) from
`results/combined_all.csv` and fails loudly on any mismatch. Run it after
any edit to the numbers, and after extending the sweep. It currently
reports:

```
checks run: 115
All numbers in main.tex match results/combined_all.csv.
```

Note that five Table 3 memory averages land on exact `.xx5` ties
(e.g. Qwen INT8 = 2.405 GB); the paper rounds these half-up, and the
verifier uses `Decimal` arithmetic to match rather than binary floats.

## Form fields

**Primary category:** `cs.CL` (Computation and Language)
**Cross-list:** `cs.LG`

**Licence:** CC BY 4.0 is the usual choice for work you want reused and
cited. arXiv's default (`arXiv.org perpetual, non-exclusive license`) is
more restrictive. You cannot change this after announcement.

**Abstract** — paste as plain text (arXiv strips LaTeX; 1,920 char limit,
this is ~1,500):

> We present a pilot study characterising the effect of post-training
> quantisation on multilingual machine translation quality and inference
> cost for three African languages - Hausa, Swahili, and Yoruba - across
> three model families: a small general-purpose LLM (Qwen2.5-1.5B), a
> larger multilingual instruction-tuned LLM (Aya-23-8B), and a dedicated
> multilingual MT model (NLLB-200-distilled-1.3B). Evaluating on the
> FLORES-200 devtest set (1,012 sentences per direction) under FP16,
> bitsandbytes INT8, and NF4 INT4 precisions on consumer-grade NVIDIA T4
> GPUs, we find: (i) the specialised MT model outperforms both LLMs by
> 32.7-43.3 chrF on the Hausa and Swahili directions, and by 12.9-24.7
> chrF on Yoruba, despite being roughly a sixth the size of the 8B
> instruction-tuned LLM; (ii) post-training quantisation preserves
> aggregate translation quality within approximately 1 chrF across all
> model families and language directions, with a single worst case of
> 2.19 chrF; (iii) on consumer GPUs, INT8 and INT4 quantisation provide
> substantial memory savings (49-61% at INT4) but do not reduce inference
> latency relative to FP16 - in fact, latency increases by 1.3x to 5.3x;
> (iv) a clear performance gradient tracking training-resource
> availability is visible across languages (Swahili > Hausa > Yoruba).
> These findings motivate a research agenda focused on hardware-aware
> compression and error-type analysis beyond aggregate metrics for
> low-resource deployment.

**Comments field** (optional, shows on the abstract page):
`Pilot study. 3 model families x 3 precisions x 6 directions on FLORES-200. Code and raw results: https://github.com/avvas200/efficient-multilingual-nmt-pilot`

**ACM/MSC classes:** leave blank.

## Before you click submit

- [ ] **Endorsement.** A first submission to `cs.CL` needs an endorser
      unless your account is already auto-endorsed. Check this *first* —
      it can take days to arrange. Your Mewar supervisor, or an
      AfricaNLP/Masakhane contact who has published to cs.CL, can endorse.
- [ ] **Push the repo public.** The paper cites
      `github.com/avvas200/efficient-multilingual-nmt-pilot` in the
      introduction and appendix. Confirm it resolves, and that the
      corrected `requirements.txt` and `README.md` are pushed.
- [ ] **ORCID.** Link one to your arXiv account if you have it; it makes
      the PhD-application paper trail cleaner.
- [ ] **Author name.** Submitted as "Abbas Sani". Use the same form on
      every future paper so arXiv groups them.
- [ ] Compile once and read the PDF end to end.

## Known gaps carried into v1

These are disclosed in the paper's Limitations section rather than
hidden, which is the right call for a preprint — but they are the first
things a reader will ask about, and the obvious content for a v2.

1. **No significance testing** (L3). Decoding is greedy and
   deterministic, so the uncertainty that matters is sampling variance
   over the 1,012-sentence test set. A paired bootstrap over sentences
   would settle whether any delta in Table 2 is real. This needs the
   per-sentence hypotheses, which were not retained.
2. **Per-sentence outputs not released** (L6). The original draft's
   Appendix B showed source/hypothesis/reference triples, but no
   hypothesis files exist in the repo and no notebook writes them, so
   that appendix was **omitted from the preprint**. If you still have the
   outputs in a Kaggle output archive or on Drive, restoring Appendix B
   plus bootstrap CIs would be the single highest-value addition to v2 —
   and would close L3 and L6 together.
3. **Yoruba compression sweep incomplete** (L5). FP16 only.

## Corrections applied relative to `../technical_report_v1.md`

| Item | Was | Now |
|---|---|---|
| NLLB vs Aya, hau→eng | 25.5 chrF | **32.73** chrF |
| NLLB vs Aya, eng→swh | 34.0 chrF | **43.33** chrF (34.04 is swh→eng) |
| Abstract gap range | "25–45 chrF" | **32.7–43.3** (hau/swh), **12.9–24.7** (yor) |
| Aya length ratio, low end | 1.14 (hau→eng) | **1.18 (swh→eng)** |
| Size comparison in proposal | "order of magnitude smaller" | "roughly a sixth the size" |
| §3.3 | "Two patterns stand out" paragraph duplicated | merged |
| L3 | claimed bootstrap CIs "suggest deltas lie within noise" | states plainly that no significance testing was done |
| Appendix A | placeholder | full 42-row table generated from the CSV |
| Table 2 | INT4 deltas only | INT8 and INT4 deltas |
| Date / repo URL | "[DRAFT — October 2026]", `avvas200/pilot` | November 2026, correct URL |

Tables 1 and 3 were checked cell by cell against
`results/combined_all.csv` and were already correct, including every
memory saving and latency multiplier.
