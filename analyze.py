"""Aggregate final runs into LaTeX tables, figures and a JSON summary.

    python analyze.py --runs runs/final --cache data/cache.pkl --out report

Writes report/tables/{main,signif,negation,sweep}.tex, report/figs/*.pdf, report/summary.json
"""
import argparse
import glob
import json
import os
import pickle
import re
from collections import defaultdict

import numpy as np

ORDER = ["erm", "erm2x", "wd", "ls", "rdrop", "mixout", "cad", "cad_rdrop", "cad_cf", "cfrdrop", "cad_cfm"]
NAME = {"erm": "ERM", "erm2x": "ERM, 2$\\times$ orig.", "wd": "Weight decay", "ls": "Label smoothing",
        "rdrop": "R-Drop", "mixout": "Mixout", "cad": "CAD", "cad_rdrop": "CAD + R-Drop",
        "cad_cf": "CAD + CF", "cfrdrop": "CF-R-Drop (ours)", "cad_cfm": "CAD + CF-margin (ours)"}
PLAIN = {k: v.replace("$\\times$", "x") for k, v in NAME.items()}
METRICS = [("imdb_acc", "IMDB"), ("cad_orig_acc", "Orig."), ("cad_cf_acc", "CF"),
           ("cad_pair_acc", "Pair"), ("sst2_acc", "SST-2"), ("imdb_ece", "ECE")]
LOWER_BETTER = {"imdb_ece"}

# ink + validated categorical palette (dataviz reference instance, light mode)
INK, INK2, GRID_C, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"

NEG = re.compile(r"\b(not|no|never|nothing|nobody|none|neither|nor|without|hardly|barely)\b|n't\b", re.I)
CONTRAST = re.compile(r"\b(but|however|although|though|despite|yet|nevertheless|whereas)\b", re.I)


def load(runs):
    R = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(runs, "*", "results.json"))):
        r = json.load(open(f))
        p = os.path.join(os.path.dirname(f), "preds.npz")
        r["preds"] = dict(np.load(p)) if os.path.exists(p) else None
        R[r["condition"]].append(r)
    return R


def fmt(m, s, pct=True):
    return f"{100*m:.1f}$\\pm${100*s:.1f}" if pct else f"{100*m:.1f}$\\pm${100*s:.1f}"


def main_table(R, out):
    stats = {}
    for c in ORDER:
        if c not in R:
            continue
        stats[c] = {k: (float(np.mean([r[k] for r in R[c]])), float(np.std([r[k] for r in R[c]])))
                    for k, _ in METRICS}
        stats[c]["n_seeds"] = len(R[c])
    best = {}
    for k, _ in METRICS:
        vals = {c: stats[c][k][0] for c in stats}
        best[k] = (min if k in LOWER_BETTER else max)(vals, key=vals.get)
    lines = [r"\begin{tabular}{l" + "c" * len(METRICS) + "}", r"\toprule",
             r" & \multicolumn{1}{c}{In-dist.} & \multicolumn{3}{c}{CAD test (Kaushik et al.)} & OOD & Calib. \\",
             r"\cmidrule(lr){2-2}\cmidrule(lr){3-5}\cmidrule(lr){6-6}\cmidrule(lr){7-7}",
             "Method & " + " & ".join(h for _, h in METRICS) + r" \\", r"\midrule"]
    for c in stats:
        if c == "cad":
            lines.append(r"\midrule")
        cells = []
        for k, _ in METRICS:
            m, s = stats[c][k]
            cell = f"{100*m:.1f}" + r"{\scriptsize$\pm$" + f"{100*s:.1f}" + "}"
            cells.append(r"\textbf{" + cell + "}" if best[k] == c else cell)
        lines.append(NAME[c] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "tables", "main.tex"), "w").write("\n".join(lines))
    return stats


def avg_probs(R, c, key):
    return np.mean([r["preds"][key] for r in R[c]], axis=0)


def boot(correct_a, correct_b, n=10000, seed=0):
    """Paired bootstrap over test examples on seed-averaged predictions.
    Returns (delta, lo, hi, p_two_sided)."""
    rng = np.random.default_rng(seed)
    d = correct_a.astype(float) - correct_b.astype(float)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(1)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return float(d.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975)), float(min(p, 1.0))


def signif(R, data, out):
    yo = np.array([p[1] for p in data["cad_test_pairs"]]); yc = np.array([p[3] for p in data["cad_test_pairs"]])
    yi = np.array(data["imdb_test"][1]); ys = np.array(data["sst2_dev"][1])

    def correct(c):
        po, pc = avg_probs(R, c, "cad_orig"), avg_probs(R, c, "cad_cf")
        ko, kc = (po >= .5) == yo, (pc >= .5) == yc
        return {"cad_cf_acc": kc, "cad_pair_acc": ko & kc,
                "imdb_acc": (avg_probs(R, c, "imdb") >= .5) == yi,
                "sst2_acc": (avg_probs(R, c, "sst2") >= .5) == ys}

    comps = [("cfrdrop", "erm"), ("cfrdrop", "erm2x"), ("cfrdrop", "cad"), ("cad", "erm2x"), ("rdrop", "erm"),
             ("cad_cfm", "cad"), ("cad_cfm", "cad_cf")]
    comps = [(a, b) for a, b in comps if a in R and b in R]
    res, lines = {}, [r"\begin{tabular}{lcccc}", r"\toprule",
                      r"Comparison & CF & Pair & IMDB & SST-2 \\", r"\midrule"]
    for a, b in comps:
        ca, cb = correct(a), correct(b)
        row = []
        for k in ("cad_cf_acc", "cad_pair_acc", "imdb_acc", "sst2_acc"):
            d, lo, hi, p = boot(ca[k], cb[k])
            res[f"{a}-vs-{b}:{k}"] = dict(delta=d, lo=lo, hi=hi, p=p)
            star = "$^{**}$" if p < .01 else ("$^{*}$" if p < .05 else "")
            row.append(f"{100*d:+.1f}{star}")
        lines.append(f"{PLAIN[a]} vs. {PLAIN[b]} & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "tables", "signif.tex"), "w").write("\n".join(lines))
    return res


def negation(R, data, out, conds=("erm", "erm2x", "cad", "cfrdrop", "cad_cfm")):
    """Error rates conditioned on linguistic phenomena, WITH base rates.
    The old report's '77% of errors are negation' ignored that most long reviews
    contain a negation token at all; here we compare against prevalence."""
    x, y = data["imdb_test"]; y = np.array(y)
    n_neg = np.array([len(NEG.findall(t)) for t in x])
    feats = {"no negation": n_neg == 0, "1-2 negations": (n_neg >= 1) & (n_neg <= 2),
             "3+ negations": n_neg >= 3,
             "contrast connective": np.array([bool(CONTRAST.search(t)) for t in x])}
    res = {"prevalence": {k: float(v.mean()) for k, v in feats.items()}}
    lines = [r"\begin{tabular}{l" + "c" * (1 + len(conds)) + "}", r"\toprule",
             "Subset & Share & " + " & ".join(PLAIN[c] for c in conds if c in R) + r" \\", r"\midrule"]
    for k, m in feats.items():
        row = [f"{100*m.mean():.0f}\\%"]
        for c in conds:
            if c not in R:
                continue
            err = (avg_probs(R, c, "imdb") >= .5) != y
            res.setdefault(c, {})[k] = float(err[m].mean())
            res[c]["overall"] = float(err.mean())
            row.append(f"{100*err[m].mean():.1f}")
        lines.append(k + " & " + " & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "tables", "negation.tex"), "w").write("\n".join(lines))
    return res, feats


def sweep_table(sweep_dir, out):
    rows = []
    for f in sorted(glob.glob(os.path.join(sweep_dir, "*", "results.json"))):
        r = json.load(open(f))
        rows.append((r["name"].rsplit("_s", 1)[0], r["lr"], r["dev_acc"], r["best_epoch"]))
    lines = [r"\begin{tabular}{lccc}", r"\toprule", r"Run & LR & Dev acc. & Best ep. \\", r"\midrule"]
    lines += [f"{n.replace('_', ' ')} & {lr:g} & {100*a:.1f} & {e} \\\\" for n, lr, a, e in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(out, "tables", "sweep.tex"), "w").write("\n".join(lines))
    best = os.path.join(sweep_dir, "best.json")
    return {"runs": rows, "best": json.load(open(best)) if os.path.exists(best) else None}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2); ax.spines[s].set_linewidth(.6)
    ax.tick_params(colors=INK2, labelsize=7, width=.6)
    ax.yaxis.grid(True, color=GRID_C, linewidth=.6); ax.set_axisbelow(True)
    ax.set_facecolor(SURF)


def figures(R, stats, neg_res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "text.color": INK,
                         "axes.labelcolor": INK, "figure.facecolor": "white"})
    conds = [c for c in ORDER if c in stats]

    # Fig. 2: CAD test accuracies (orig / counterfactual / pair) per condition
    fig, ax = plt.subplots(figsize=(7.0, 2.2))
    w, gap = 0.26, 0.02
    xs = np.arange(len(conds))
    for j, (k, lab, col) in enumerate([("cad_orig_acc", "Original", C1), ("cad_cf_acc", "Counterfactual", C2),
                                       ("cad_pair_acc", "Pair (both correct)", C3)]):
        m = [100 * stats[c][k][0] for c in conds]; s = [100 * stats[c][k][1] for c in conds]
        ax.bar(xs + (j - 1) * (w + gap), m, w, color=col, label=lab, yerr=s,
               error_kw=dict(elinewidth=.6, capsize=1.5, ecolor=INK2))
    ax.set_xticks(xs); ax.set_xticklabels([PLAIN[c] for c in conds], rotation=20, ha="right")
    lo = min(100 * stats[c]["cad_pair_acc"][0] for c in conds)
    ax.set_ylim(max(0, 5 * np.floor((lo - 8) / 5)), 100); ax.set_ylabel("Accuracy (%)")
    ax.axvline(5.5, color=INK2, lw=.6, ls=":")
    style(ax); ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout(); fig.savefig(os.path.join(out, "figs", "cad_acc.pdf")); plt.close(fig)

    # Fig. 3: in-distribution vs counterfactual robustness trade-off
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    grp = {c: (C3 if c == "cfrdrop" else C2 if c.startswith("cad") else C1) for c in conds}
    for c in conds:
        xm, xsd = (100 * v for v in stats[c]["imdb_acc"]); ym, ysd = (100 * v for v in stats[c]["cad_pair_acc"])
        ax.errorbar(xm, ym, xerr=xsd, yerr=ysd, fmt="o", ms=5, color=grp[c], mec=SURF, mew=1,
                    elinewidth=.6, ecolor=INK2, capsize=1.5)
        ax.annotate(PLAIN[c], (xm, ym), xytext=(4, 3), textcoords="offset points", fontsize=6, color=INK2)
    ax.set_xlabel("IMDB test accuracy (%)"); ax.set_ylabel("CAD pair accuracy (%)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=C1, label="Original data only"),
         Line2D([], [], marker="o", ls="", color=C2, label="With CAD"),
         Line2D([], [], marker="o", ls="", color=C3, label="CF-R-Drop (ours)")]
    style(ax); ax.xaxis.grid(True, color=GRID_C, lw=.6)
    ax.legend(handles=h, frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(out, "figs", "tradeoff.pdf")); plt.close(fig)

    # Fig. 4: error rate by negation density (with prevalence in the tick labels)
    nc = [c for c in ("erm", "erm2x", "cad", "cfrdrop") if c in neg_res]
    keys = ["no negation", "1-2 negations", "3+ negations", "contrast connective"]
    cols = {"erm": C1, "erm2x": "#7aaee8", "cad": C2, "cfrdrop": C3}
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    w = 0.8 / len(nc)
    for j, c in enumerate(nc):
        ax.bar(np.arange(len(keys)) + (j - (len(nc) - 1) / 2) * w, [100 * neg_res[c][k] for k in keys],
               w * 0.92, color=cols[c], label=PLAIN[c])
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{k}\n({100*neg_res['prevalence'][k]:.0f}% of test)" for k in keys], fontsize=6)
    ax.set_ylabel("Error rate (%)"); style(ax); ax.legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out, "figs", "negation.pdf")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/final")
    ap.add_argument("--sweep", default="runs/sweep")
    ap.add_argument("--cache", default="data/cache.pkl")
    ap.add_argument("--out", default="report")
    a = ap.parse_args()
    for d in ("tables", "figs"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    data = pickle.load(open(a.cache, "rb"))
    R = load(a.runs)
    stats = main_table(R, a.out)
    sig = signif(R, data, a.out)
    neg, _ = negation(R, data, a.out)
    sw = sweep_table(a.sweep, a.out)
    figures(R, stats, neg, a.out)
    curves = {c: [r["history"] for r in R[c]] for c in R}
    json.dump({"stats": stats, "signif": sig, "negation": neg, "sweep": sw, "meta": data["meta"],
               "best_epochs": {c: [r["best_epoch"] for r in R[c]] for c in R},
               "minutes": {c: [r["train_minutes"] for r in R[c]] for c in R}, "curves": curves},
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
