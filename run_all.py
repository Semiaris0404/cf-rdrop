"""Run the whole study: (1) hyper-parameter sweep on dev, (2) final runs with 3 seeds.

    python run_all.py --stage sweep   # ~13 dev-only runs, seed 1
    python run_all.py --stage final   # 10 conditions x 3 seeds, uses sweep/best.json
    python run_all.py --stage all
    python run_all.py --stage extra   # follow-up: margin-space CF term (cad_cfm), sweep + 3 seeds

Every run writes runs/<stage>/<name>/results.json and is skipped if that file exists,
so the script can simply be re-launched after a Colab disconnect.

Tuning protocol (identical budget for every regulariser, selection on ORIGINAL dev only):
  * learning rate: swept once on the ERM baseline, then shared by all conditions;
  * each regulariser gets the same 2-value grid for its own strength;
  * combined conditions (cad_rdrop, cfrdrop) reuse the strengths chosen for their parts
    and receive no extra tuning.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

LRS = [1e-5, 2e-5, 3e-5]
GRID = {                        # condition -> (flag, values)
    "wd":     ("wd", [0.01, 0.1]),
    "ls":     ("ls", [0.1, 0.2]),
    "rdrop":  ("rdrop", [0.5, 1.0]),
    "mixout": ("mixout", [0.3, 0.7]),
    "cad_cf": ("cf", [0.5, 1.0]),
}
FINAL = ["erm", "erm2x", "wd", "ls", "rdrop", "mixout", "cad", "cad_rdrop", "cad_cf", "cfrdrop"]
SEEDS = [1, 2, 3]


def run(extra, common):
    cmd = [sys.executable, "train.py"] + common + extra
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def best_of(pattern):
    """Return the results.json (dict) with the highest dev accuracy; ties keep the first."""
    best = None
    for f in sorted(glob.glob(pattern)):
        r = json.load(open(f))
        if best is None or r["dev_acc"] > best["dev_acc"]:
            best = r
    return best


def sweep(common, root):
    out = ["--out", os.path.join(root, "sweep"), "--dev_only", "--seed", "1"]
    for lr in LRS:
        run(["--condition", "erm", "--lr", str(lr), "--tag", f"_lr{lr:g}"] + out, common)
    lr = best_of(os.path.join(root, "sweep", "erm_lr*_s1", "results.json"))["lr"]
    best = {"lr": lr}
    for cond, (flag, vals) in GRID.items():
        for v in vals:
            run(["--condition", cond, "--lr", str(lr), f"--{flag}", str(v), "--tag", f"_{flag}{v:g}"] + out, common)
        r = best_of(os.path.join(root, "sweep", f"{cond}_{flag}*_s1", "results.json"))
        best[flag] = r["cfg"][flag]
    with open(os.path.join(root, "sweep", "best.json"), "w") as f:
        json.dump(best, f, indent=1)
    print("selected:", best)
    return best


def extra(common, root):
    """Follow-up: margin-space CF term. Same protocol: shared LR, 2-value grid on seed 1, 3 seeds."""
    best_path = os.path.join(root, "sweep", "best.json")
    best = json.load(open(best_path))
    lr = best["lr"]
    out = ["--out", os.path.join(root, "sweep"), "--dev_only", "--seed", "1"]
    for v in [0.01, 0.1]:
        run(["--condition", "cad_cfm", "--lr", str(lr), "--cfm", str(v), "--tag", f"_cfm{v:g}"] + out, common)
    r = best_of(os.path.join(root, "sweep", "cad_cfm_cfm*_s1", "results.json"))
    best["cfm"] = r["cfg"]["cfm"]
    json.dump(best, open(best_path, "w"), indent=1)
    print("selected cfm:", best["cfm"])
    for seed in SEEDS:
        run(["--condition", "cad_cfm", "--seed", str(seed), "--lr", str(lr), "--cfm", str(best["cfm"]),
             "--out", os.path.join(root, "final")], common)


def final(common, root):
    best = json.load(open(os.path.join(root, "sweep", "best.json")))
    for seed in SEEDS:                 # seed-major order: a full replicate finishes first
        for cond in FINAL:
            extra = ["--condition", cond, "--seed", str(seed), "--lr", str(best["lr"]),
                     "--out", os.path.join(root, "final")]
            for flag in ("wd", "ls", "rdrop", "mixout"):
                if cond in (flag, ) or (flag == "rdrop" and cond in ("cad_rdrop", "cfrdrop")):
                    extra += [f"--{flag}", str(best[flag])]
            if cond in ("cad_cf", "cfrdrop"):
                extra += ["--cf", str(best["cf"])]
            run(extra, common)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sweep", "final", "all", "extra"], default="all")
    ap.add_argument("--root", default="runs")
    ap.add_argument("--cad_root", default="data/cad")
    ap.add_argument("--cache", default="data/cache.pkl")
    ap.add_argument("--epochs", default="10")
    ap.add_argument("--extra", default="", help="extra args passed to train.py, e.g. '--tiny'")
    a = ap.parse_args()
    common = ["--cad_root", a.cad_root, "--cache", a.cache, "--epochs", a.epochs] + a.extra.split()
    if a.stage in ("sweep", "all"):
        sweep(common, a.root)
    if a.stage in ("final", "all"):
        final(common, a.root)
    if a.stage == "extra":
        extra(common, a.root)
