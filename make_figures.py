"""Paper figures from report/summary.json (+ summary_extra.json).

    python make_figures.py --report report
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from analyze import C1, C2, C3, GRID_C, INK, INK2, ORDER, PLAIN, SURF, style

SHORT = {"erm": "ERM", "erm2x": "ERM-2x", "wd": "WD", "ls": "LS", "rdrop": "R-Drop", "mixout": "Mixout",
         "cad": "CAD", "cad_rdrop": "CAD+R-Drop", "cad_cf": "CAD+CF", "cfrdrop": "CF-R-Drop",
         "cad_cfm": "CF-margin"}
# label offsets (points) chosen by hand to avoid collisions in the scatter
OFF = {"erm": (-6, -9), "erm2x": (5, -3), "wd": (5, -3), "ls": (-8, 5), "rdrop": (5, -7), "mixout": (5, 2),
       "cad": (5, 1), "cad_rdrop": (-12, -10), "cad_cf": (-34, 3), "cfrdrop": (5, -7),
       "cad_cfm": (5, 3)}


def setup():
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "text.color": INK,
                         "axes.labelcolor": INK, "figure.facecolor": "white"})


def fig_cad(stats, out):
    conds = [c for c in ORDER if c in stats]
    fig, ax = plt.subplots(figsize=(7.0, 2.1))
    w, gap = 0.26, 0.02
    xs = np.arange(len(conds))
    for j, (k, lab, col) in enumerate([("cad_orig_acc", "Original reviews", C1), ("cad_cf_acc", "Counterfactuals", C2),
                                       ("cad_pair_acc", "Pairs (both correct)", C3)]):
        m = [100 * stats[c][k][0] for c in conds]; s = [100 * stats[c][k][1] for c in conds]
        ax.bar(xs + (j - 1) * (w + gap), m, w, color=col, label=lab, yerr=s,
               error_kw=dict(elinewidth=.6, capsize=1.5, ecolor=INK2))
    tick = dict(SHORT, cfrdrop="CF-R-Drop*", cad_cfm="CF-margin*", ls="Label smooth.", wd="Weight decay")
    ax.set_xticks(xs); ax.set_xticklabels([tick[c] for c in conds], rotation=15, ha="right")
    ax.set_ylim(80, 100); ax.set_ylabel("CAD test accuracy (%)")
    ax.axvline(5.5, color=INK2, lw=.6, ls=":")
    ax.text(2.5, 99.3, "original reviews only", ha="center", va="top", fontsize=7, color=INK2)
    ax.text((6 + len(conds) - 1) / 2, 99.3, "with counterfactual pairs (*ours)", ha="center", va="top", fontsize=7, color=INK2)
    style(ax)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout(); fig.savefig(os.path.join(out, "cad_acc.pdf")); plt.close(fig)


def fig_tradeoff(stats, out):
    conds = [c for c in ORDER if c in stats]
    fig, ax = plt.subplots(figsize=(3.3, 2.45))
    grp = {c: (C3 if c in ("cfrdrop", "cad_cfm") else C2 if c.startswith("cad") else C1) for c in conds}
    for c in conds:
        xm, xsd = (100 * v for v in stats[c]["imdb_acc"]); ym, ysd = (100 * v for v in stats[c]["cad_pair_acc"])
        ax.errorbar(xm, ym, xerr=xsd, yerr=ysd, fmt="o", ms=5, color=grp[c], mec=SURF, mew=1,
                    elinewidth=.5, ecolor=GRID_C, capsize=0, zorder=3)
        ax.annotate(SHORT[c], (xm, ym), xytext=OFF[c], textcoords="offset points", fontsize=6, color=INK2, zorder=4)
    ax.set_xlabel("IMDB test accuracy (%)"); ax.set_ylabel("CAD pair accuracy (%)")
    h = [Line2D([], [], marker="o", ls="", color=C1, label="Original data only"),
         Line2D([], [], marker="o", ls="", color=C2, label="With CAD pairs"),
         Line2D([], [], marker="o", ls="", color=C3, label="Ours (CF-R-Drop, CF-margin)")]
    style(ax); ax.xaxis.grid(True, color=GRID_C, lw=.6)
    ax.legend(handles=h, frameon=False, fontsize=6, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(out, "tradeoff.pdf")); plt.close(fig)


def fig_negation(neg, extra, out):
    """(a) IMDB error rate by negation density; (b) CAD pair accuracy by edit type."""
    conds = [c for c in ("erm", "erm2x", "cad", "cfrdrop") if c in neg]
    cols = {"erm": C1, "erm2x": "#86b6ec", "cad": C2, "cfrdrop": C3}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw=dict(width_ratios=[1.15, 1]))
    keys = ["no negation", "1-2 negations", "3+ negations"]
    labels = ["none", "1–2", "3+"]
    ax = axes[0]; w = 0.8 / len(conds)
    for j, c in enumerate(conds):
        ax.bar(np.arange(len(keys)) + (j - (len(conds) - 1) / 2) * w, [100 * neg[c][k] for k in keys],
               w * 0.9, color=cols[c], label=PLAIN[c])
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{l}\n({100*neg['prevalence'][k]:.0f}% of test)" for l, k in zip(labels, keys)], fontsize=7)
    ax.set_xlabel("Negation cues in the review"); ax.set_ylabel("IMDB error rate (%)")
    ax.set_title("(a) IMDB errors vs. negation density", fontsize=8, loc="left")
    ax.set_ylim(0, 10); style(ax)
    ax.legend(frameon=False, fontsize=6.5, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.0))

    ax = axes[1]
    tt = ["negation added", "negation removed", "no negation change"]
    tl = ["adds\nnegation", "removes\nnegation", "other\nedit"]
    for j, c in enumerate(conds):
        m = [100 * extra["edit_pair_acc"][c][t][0] for t in tt]; s = [100 * extra["edit_pair_acc"][c][t][1] for t in tt]
        ax.bar(np.arange(len(tt)) + (j - (len(conds) - 1) / 2) * w, m, w * 0.9, color=cols[c], yerr=s,
               error_kw=dict(elinewidth=.5, capsize=1, ecolor=INK2))
    ax.set_xticks(range(len(tt)))
    ax.set_xticklabels([f"{l} ({100*extra['edit_share'][t]:.0f}%)" for l, t in zip(tl, tt)], fontsize=7)
    ax.set_xlabel("Type of human counterfactual edit"); ax.set_ylabel("CAD pair accuracy (%)")
    ax.set_title("(b) CAD pairs by edit type", fontsize=8, loc="left")
    ax.set_ylim(70, 100); style(ax)
    fig.tight_layout(w_pad=2.0); fig.savefig(os.path.join(out, "negation.pdf")); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--report", default="report"); a = ap.parse_args()
    setup()
    S = json.load(open(os.path.join(a.report, "summary.json")))
    E = json.load(open(os.path.join(a.report, "summary_extra.json")))
    out = os.path.join(a.report, "figs"); os.makedirs(out, exist_ok=True)
    stats = {c: {k: tuple(v) for k, v in d.items() if k != "n_seeds"} for c, d in S["stats"].items()}
    fig_cad(stats, out); fig_tradeoff(stats, out); fig_negation(S["negation"], E, out)
    print("figures ->", out)
