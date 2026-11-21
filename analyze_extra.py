"""Extra analyses that need only the CAD test texts + saved predictions (no IMDB texts).

    python analyze_extra.py --runs runs/final --cad_root data/cad --out report

1. Paired-bootstrap tests for the CAD-family ablations.
2. Margin antisymmetry on test pairs: |m(x) + m(x~)|, the quantity the CF term targets.
3. Pair accuracy by edit type (does the human edit add/remove a negation cue?).
4. Qualitative pairs where methods disagree.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

from analyze import NAME, PLAIN, boot
from data import load_cad_split

NEG = re.compile(r"\b(not|no|never|nothing|nobody|none|neither|nor|without|hardly|barely)\b|n't\b", re.I)
CONDS = ["erm", "erm2x", "wd", "ls", "rdrop", "mixout", "cad", "cad_rdrop", "cad_cf", "cfrdrop", "cad_cfm"]


def seeds(runs, c, key):
    return [dict(np.load(f))[key] for f in sorted(glob.glob(os.path.join(runs, f"{c}_s*", "preds.npz")))]


def margin(p):
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def edit_type(o, c):
    d = len(NEG.findall(c)) - len(NEG.findall(o))
    return "negation added" if d > 0 else ("negation removed" if d < 0 else "no negation change")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/final")
    ap.add_argument("--cad_root", default="data/cad")
    ap.add_argument("--out", default="report")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    te = load_cad_split(a.cad_root, "test")
    yo = np.array([p[1] for p in te]); yc = np.array([p[3] for p in te])
    res = {}

    # 1. extra significance tests on seed-averaged predictions
    def correct(c):
        po = np.mean(seeds(a.runs, c, "cad_orig"), 0); pc = np.mean(seeds(a.runs, c, "cad_cf"), 0)
        ko, kc = (po >= .5) == yo, (pc >= .5) == yc
        return {"orig": ko, "cf": kc, "pair": ko & kc}
    comps = [("cad_cf", "cad"), ("cad_rdrop", "cad"), ("cfrdrop", "cad_rdrop"), ("mixout", "erm"), ("ls", "erm"),
             ("cad_cfm", "cad"), ("cad_cfm", "cad_cf")]
    comps = [(x, y) for x, y in comps if glob.glob(os.path.join(a.runs, f"{x}_s*")) and glob.glob(os.path.join(a.runs, f"{y}_s*"))]
    present = [c for c in CONDS if glob.glob(os.path.join(a.runs, f"{c}_s*", "preds.npz"))]
    res["signif_extra"] = {}
    for x, y in comps:
        cx, cy = correct(x), correct(y)
        for k in ("orig", "cf", "pair"):
            d, lo, hi, p = boot(cx[k], cy[k])
            res["signif_extra"][f"{x}-vs-{y}:{k}"] = dict(delta=d, lo=lo, hi=hi, p=p)

    # 2. margin antisymmetry |m(x)+m(x~)| (0 = perfectly antisymmetric), per seed, then mean/sd
    res["antisym"] = {}
    for c in present:
        vals, meds = [], []
        for po, pc in zip(seeds(a.runs, c, "cad_orig"), seeds(a.runs, c, "cad_cf")):
            ma, mb = margin(po), margin(pc)
            # scale-free: 0 = perfectly antisymmetric margins, 1 = same sign (pair not flipped)
            v = np.abs(ma + mb) / (np.abs(ma) + np.abs(mb)); vals.append(v.mean()); meds.append(np.median(np.abs(ma)))
        res["antisym"][c] = dict(mean=float(np.mean(vals)), sd=float(np.std(vals)), median_abs_margin=float(np.mean(meds)))

    # 3. pair accuracy by edit type (per seed -> mean/sd)
    types = np.array([edit_type(o, c) for o, _, c, _, _ in te])
    res["edit_share"] = {t: float((types == t).mean()) for t in np.unique(types)}
    res["edit_pair_acc"] = {}
    for c in present:
        out = {}
        for t in np.unique(types):
            m = types == t
            accs = [(((po >= .5) == yo) & ((pc >= .5) == yc))[m].mean()
                    for po, pc in zip(seeds(a.runs, c, "cad_orig"), seeds(a.runs, c, "cad_cf"))]
            out[t] = (float(np.mean(accs)), float(np.std(accs)))
        res["edit_pair_acc"][c] = out

    show = [c for c in ["erm", "erm2x", "rdrop", "mixout", "cad", "cad_rdrop", "cad_cf", "cfrdrop", "cad_cfm"] if c in present]
    tt = ["negation added", "negation removed", "no negation change"]
    L = [r"\begin{tabular}{l" + "c" * (len(tt) + 1) + "}", r"\toprule",
         "Method & " + " & ".join({"negation added": "+neg.", "negation removed": "$-$neg.", "no negation change": "other"}[t] for t in tt)
         + r" & Asym. $\downarrow$ \\", r"\midrule"]
    for c in show:
        cells = [f"{100*res['edit_pair_acc'][c][t][0]:.1f}" for t in tt]
        cells.append(f"{res['antisym'][c]['mean']:.3f}")
        L.append(("\\midrule\n" if c == "cad" else "") + NAME[c] + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(a.out, "tables", "edits.tex"), "w").write("\n".join(L))

    # 4. qualitative: pairs ERM gets wrong but CAD-family gets right, and pairs everyone misses
    po = {c: np.mean(seeds(a.runs, c, "cad_orig"), 0) for c in ("erm", "cad", "cfrdrop")}
    pc = {c: np.mean(seeds(a.runs, c, "cad_cf"), 0) for c in ("erm", "cad", "cfrdrop")}
    ok = {c: ((po[c] >= .5) == yo) & ((pc[c] >= .5) == yc) for c in po}
    ex = []
    for i in np.where(~ok["erm"] & ok["cfrdrop"])[0][:6].tolist() + np.where(~ok["cfrdrop"] & ~ok["cad"])[0][:6].tolist():
        o, lo_, c_, lc, _ = te[i]
        ex.append(dict(i=int(i), orig=o, cf=c_, y=int(lo_), edit=types[i],
                       p_pos={k: [round(float(po[k][i]), 3), round(float(pc[k][i]), 3)] for k in po}))
    res["examples"] = ex
    json.dump(res, open(os.path.join(a.out, "summary_extra.json"), "w"), indent=1)
    if os.path.exists(os.path.join(a.out, "summary.json")):
        combined_signif(a.out)
    print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=1))


def combined_signif(out):
    """Table 2: merges the bootstrap tests from analyze.py (summary.json) and this script."""
    S = json.load(open(os.path.join(out, "summary.json")))["signif"]
    E = json.load(open(os.path.join(out, "summary_extra.json")))["signif_extra"]
    key = {"cf": "cad_cf_acc", "pair": "cad_pair_acc", "imdb": "imdb_acc"}
    N = {"erm": "ERM", "erm2x": "ERM-2$\\times$", "cad": "CAD", "cad_rdrop": "CAD+R-Drop", "cad_cf": "CAD+CF",
         "cfrdrop": "CF-R-Drop", "rdrop": "R-Drop", "mixout": "Mixout", "cad_cfm": "CAD+CF-margin"}

    def cell(a, b, k):
        d = S.get(f"{a}-vs-{b}:{key[k]}") or E.get(f"{a}-vs-{b}:{k}")
        if d is None:
            return "--"
        star = "$^{**}$" if d["p"] < .01 else ("$^{*}$" if d["p"] < .05 else "")
        v = 100 * d["delta"]
        return ("$+$" if v >= 0 else "$-$") + f"{abs(v):.1f}" + star

    rows = [("rdrop", "erm"), ("mixout", "erm"), ("cad", "erm2x"), ("cad_cf", "cad"), ("cad_rdrop", "cad"),
            ("cfrdrop", "cad_rdrop"), ("cfrdrop", "cad"), ("cad_cfm", "cad"), ("cad_cfm", "cad_cf")]
    rows = [(x, y) for x, y in rows if cell(x, y, "pair") != "--"]
    L = [r"\begin{tabular}{lccc}", r"\toprule", r"Comparison & CF & Pair & IMDB \\", r"\midrule"]
    L += [f"{N[a]} vs.\\ {N[b]} & {cell(a, b, 'cf')} & {cell(a, b, 'pair')} & {cell(a, b, 'imdb')} \\\\" for a, b in rows]
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "tables", "signif.tex"), "w").write("\n".join(L))


if __name__ == "__main__":
    main()
