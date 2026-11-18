"""Re-derive every number asserted in main.tex from results/combined_all.csv."""
import csv
import io
import re
import sys

repo = sys.argv[1]
rows = list(csv.DictReader(io.open(repo + "/results/combined_all.csv", encoding="utf-8")))
tex = io.open(repo + "/writeup/arxiv/main.tex", encoding="utf-8").read()
# Paper text is hard-wrapped; collapse whitespace so prose literals match.
tex = re.sub(r"\s+", " ", tex)

from decimal import Decimal, ROUND_HALF_UP


def dmean(vals):
    """Exact-decimal mean, so .xx5 ties round half-up as a typesetter would."""
    tot = sum(Decimal(v) for v in vals)
    return tot / Decimal(len(vals))


def d2(x):
    return str(Decimal(x).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def d3(x):
    return str(Decimal(x).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

DIRS4 = ["hau-eng", "eng-hau", "swh-eng", "eng-swh"]
ALL6 = DIRS4 + ["yor-eng", "eng-yor"]
fails, checks = [], 0


def cell(m, p, d, k="chrf"):
    sel = [r for r in rows if r["model_key"] == m and r["precision"] == p and r["direction"] == d]
    return float(sel[0][k]) if sel and sel[0][k] else None


def rawcell(m, p, d, k):
    """The CSV string, so Decimal sees the exact decimal the run recorded."""
    sel = [r for r in rows if r["model_key"] == m and r["precision"] == p and r["direction"] == d]
    return sel[0][k]


def want(label, expected, tol=0.005):
    """Assert the literal `expected` (as written in the paper) appears in main.tex."""
    global checks
    checks += 1
    if expected not in tex:
        fails.append("%s: expected literal %r not found in main.tex" % (label, expected))


# --- Table 1: every chrF cell ------------------------------------------------
for m in ["qwen-1.5b", "aya-8b", "nllb-1.3b"]:
    for p in ["fp16", "int8", "int4"]:
        for d in ALL6:
            v = cell(m, p, d)
            if v is None:
                continue
            want("T1 %s/%s/%s" % (m, p, d), "%.2f" % v)

# --- Table 2: deltas ---------------------------------------------------------
for p in ["int8", "int4"]:
    for m in ["qwen-1.5b", "aya-8b", "nllb-1.3b"]:
        for d in DIRS4:
            delta = cell(m, p, d) - cell(m, "fp16", d)
            s = "$0.00$" if abs(delta) < 0.005 else "$%+.2f$" % delta
            s = s.replace("+", "+").replace("-", "-")
            lit = "$0.00$" if abs(delta) < 0.005 else ("$%s%.2f$" % ("+" if delta > 0 else "-", abs(delta)))
            checks += 1
            if lit not in tex:
                fails.append("T2 %s/%s/%s: expected %s" % (m, p, d, lit))

# --- Table 3: memory / latency ----------------------------------------------
for m, nm in [("qwen-1.5b", "Qwen"), ("aya-8b", "Aya"), ("nllb-1.3b", "NLLB")]:
    base_mem = dmean([rawcell(m, "fp16", d, "peak_gpu_gb") for d in DIRS4])
    base_lat = dmean([rawcell(m, "fp16", d, "sec_per_sentence") for d in DIRS4])
    for p in ["fp16", "int8", "int4"]:
        mem = dmean([rawcell(m, p, d, "peak_gpu_gb") for d in DIRS4])
        lat = dmean([rawcell(m, p, d, "sec_per_sentence") for d in DIRS4])
        want("T3 mem %s/%s" % (nm, p), "%s\\,GB" % d2(mem))
        want("T3 lat %s/%s" % (nm, p), d3(lat))
        if p != "fp16":
            save = (Decimal(1) - mem / base_mem) * 100
            want("T3 save %s/%s" % (nm, p),
                 "%s\\%%" % save.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
            want("T3 slow %s/%s" % (nm, p), "%s$\\times$ slower" % d2(lat / base_lat))

# --- Prose: NLLB-vs-LLM gaps -------------------------------------------------
gaps = {}
for d in ALL6:
    for m in ["aya-8b", "qwen-1.5b"]:
        gaps[(m, d)] = cell("nllb-1.3b", "fp16", d) - cell(m, "fp16", d)

for d in DIRS4:
    want("gap aya %s" % d, "%.2f chrF on %s" % (gaps[("aya-8b", d)], d.replace("-", "$\\rightarrow$")))

hs_all = [gaps[(m, d)] for m in ["aya-8b", "qwen-1.5b"] for d in DIRS4]
yo_all = [gaps[(m, d)] for m in ["aya-8b", "qwen-1.5b"] for d in ["yor-eng", "eng-yor"]]
want("abstract hau/swh range", "%.1f--%.1f chrF" % (min(hs_all), max(hs_all)))
want("abstract yoruba range", "%.1f--%.1f chrF on Yoruba" % (min(yo_all), max(yo_all)))

qwen4 = [gaps[("qwen-1.5b", d)] for d in DIRS4]
want("qwen margin range", "(%.2f--%.2f chrF" % (min(qwen4), max(qwen4)))
want("yor-eng aya gap", "%.2f chrF over Aya" % gaps[("aya-8b", "yor-eng")])
want("yor-eng qwen gap", "%.2f over Qwen" % gaps[("qwen-1.5b", "yor-eng")])

# --- Prose: worst INT4 loss, Aya length ratios, NLLB ratios ------------------
worst = min(cell(m, "int4", d) - cell(m, "fp16", d)
            for m in ["qwen-1.5b", "aya-8b", "nllb-1.3b"] for d in DIRS4)
want("worst int4 loss", "%.2f chrF" % abs(worst))

aya_lr = [cell("aya-8b", "fp16", d, "len_ratio") for d in DIRS4]
want("aya len ratio lo", "%.2f (swh$\\rightarrow$eng)" % min(aya_lr))
want("aya len ratio hi", "%.2f (eng$\\rightarrow$swh)" % max(aya_lr))

nllb_lr = [cell("nllb-1.3b", "fp16", d, "len_ratio") for d in DIRS4]
want("nllb len ratios", "(%.2f--%.2f)" % (min(nllb_lr), max(nllb_lr)))

# --- Prose: INT4 memory saving range, INT8 saving range, slowdown ranges -----
def saverange(p):
    out = []
    for m in ["qwen-1.5b", "aya-8b", "nllb-1.3b"]:
        bm = dmean([rawcell(m, "fp16", d, "peak_gpu_gb") for d in DIRS4])
        mm = dmean([rawcell(m, p, d, "peak_gpu_gb") for d in DIRS4])
        pct = (Decimal(1) - mm / bm) * 100
        out.append(int(pct.quantize(Decimal("1"), rounding=ROUND_HALF_UP)))
    return min(out), max(out)


lo8, hi8 = saverange("int8")
lo4, hi4 = saverange("int4")
want("int8 saving range", "%d--%d\\%% at INT8" % (lo8, hi8))
want("int4 saving range", "%d--%d\\%% at INT4" % (lo4, hi4))

int4x = []
for m in ["qwen-1.5b", "aya-8b", "nllb-1.3b"]:
    bl = dmean([rawcell(m, "fp16", d, "sec_per_sentence") for d in DIRS4])
    q4 = dmean([rawcell(m, "int4", d, "sec_per_sentence") for d in DIRS4])
    int4x.append(q4 / bl)
want("int4 slowdown range",
     "(%s$\\times$--%s$\\times$)" % (d2(min(int4x)), d2(max(int4x))))

# --- Resource gradient -------------------------------------------------------
for d, lang in [("swh-eng", "Swahili"), ("hau-eng", "Hausa"), ("yor-eng", "Yoruba")]:
    want("gradient %s" % lang, "%s (%.2f)" % (lang, cell("nllb-1.3b", "fp16", d)))

print("checks run: %d" % checks)
if fails:
    print("FAILURES (%d):" % len(fails))
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("All numbers in main.tex match results/combined_all.csv.")
