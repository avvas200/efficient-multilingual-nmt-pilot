"""Generate the Appendix A LaTeX longtable from results/combined_all.csv."""
import csv
import sys

CSV = sys.argv[1]
rows = list(csv.DictReader(open(CSV, encoding="utf-8")))

NAME = {"qwen-1.5b": "Qwen2.5-1.5B", "aya-8b": "Aya-23-8B", "nllb-1.3b": "NLLB-1.3B"}
DIRS = ["hau-eng", "eng-hau", "swh-eng", "eng-swh", "yor-eng", "eng-yor"]
NL = " \\\\"


def arrow(d):
    s, t = d.split("-")
    return s + "$\\rightarrow$" + t


def g(r, k, fmt="%.2f"):
    v = r.get(k, "")
    if v is None or str(v).strip() == "":
        return "--"
    try:
        return fmt % float(v)
    except ValueError:
        return str(v)


HEAD = "Model & Prec. & Direction & BLEU & spBLEU & chrF & chrF++ & Len. ratio & s/sent"

out = []
out.append("% Generated from results/combined_all.csv -- do not edit by hand.")
out.append("\\begin{longtable}{llcrrrrrr}")
out.append("\\caption{Full per-direction results for all evaluated "
           "configurations on FLORES-200 devtest (1{,}012 sentences per "
           "direction). ``Len.\\ ratio'' is hypothesis-to-reference length "
           "ratio; ``s/sent'' is wall-clock inference time per sentence. "
           "Dashes mark metrics not recorded by the Yoruba extension run.}"
           "\\label{tab:appendix}" + NL)
out.append("\\toprule")
out.append(HEAD + NL)
out.append("\\midrule")
out.append("\\endfirsthead")
out.append("\\multicolumn{9}{l}{\\footnotesize\\itshape "
           "Table \\thetable\\ continued from previous page.}" + NL)
out.append("\\toprule")
out.append(HEAD + NL)
out.append("\\midrule")
out.append("\\endhead")
out.append("\\midrule")
out.append("\\multicolumn{9}{r}{\\footnotesize\\itshape Continued on next "
           "page.}" + NL)
out.append("\\endfoot")
out.append("\\bottomrule")
out.append("\\endlastfoot")
for m in ["qwen-1.5b", "aya-8b", "nllb-1.3b"]:
    first = True
    for p in ["fp16", "int8", "int4"]:
        for d in DIRS:
            sel = [
                r
                for r in rows
                if r["model_key"] == m and r["precision"] == p and r["direction"] == d
            ]
            if not sel:
                continue
            r = sel[0]
            label = NAME[m] if first else ""
            first = False
            cells = [
                label,
                p.upper(),
                arrow(d),
                g(r, "bleu"),
                g(r, "spbleu"),
                g(r, "chrf"),
                g(r, "chrfpp"),
                g(r, "len_ratio", "%.3f"),
                g(r, "sec_per_sentence", "%.3f"),
            ]
            out.append(" & ".join(cells) + NL)
    if m != "nllb-1.3b":
        out.append("\\midrule")
out.append("\\end{longtable}")

sys.stdout.write("\n".join(out) + "\n")
