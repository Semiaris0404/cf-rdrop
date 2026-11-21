"""Data loading for the AI6103 project.

Sources
-------
* CAD  : Kaushik, Hovy & Lipton (ICLR 2020), counterfactually-augmented IMDB.
         Cloned from https://github.com/acmi-lab/counterfactually-augmented-data
         Paired files store each original review immediately followed by its
         human-revised, label-flipped counterfactual (same batch_id).
* IMDB : Maas et al. (ACL 2011), 25k test reviews (HuggingFace `stanfordnlp/imdb`).
* SST-2: Socher et al. (EMNLP 2013), 872 validation sentences (`stanfordnlp/sst2`),
         used as an out-of-domain test set.

Labels everywhere: 0 = negative, 1 = positive.
"""
import csv
import os
import random
import re

import pandas as pd

LABEL = {"Negative": 0, "Positive": 1}


def norm_text(t: str) -> str:
    """Aggressive normalisation used only for de-duplication / leakage checks."""
    t = t.replace("<br />", " ").lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())[:300]


def load_cad_split(cad_root: str, split: str):
    """Return list of pairs [(orig_text, orig_label, cf_text, cf_label, pair_id)]."""
    path = os.path.join(cad_root, "sentiment", "combined", "paired", f"{split}_paired.tsv")
    df = pd.read_csv(path, sep="\t", quoting=csv.QUOTE_MINIMAL)
    pairs = []
    for bid, g in df.groupby("batch_id", sort=False):
        assert len(g) == 2, f"batch {bid} has {len(g)} rows"
        a, b = g.iloc[0], g.iloc[1]          # first row = original, second = revision
        la, lb = LABEL[a.Sentiment], LABEL[b.Sentiment]
        assert la != lb, f"batch {bid}: labels not flipped"
        pairs.append((str(a.Text), la, str(b.Text), lb, int(bid)))
    return pairs


def load_hf(name, split, text_key, label_key):
    from datasets import load_dataset
    ds = load_dataset(name, split=split)
    return list(ds[text_key]), list(ds[label_key])


def build_data(cad_root: str, seed_extra: int = 1234):
    """Everything the experiments need, built once and cached by the caller."""
    tr = load_cad_split(cad_root, "train")
    dv = load_cad_split(cad_root, "dev")
    te = load_cad_split(cad_root, "test")

    imdb_tr_x, imdb_tr_y = load_hf("stanfordnlp/imdb", "train", "text", "label")
    imdb_te_x, imdb_te_y = load_hf("stanfordnlp/imdb", "test", "text", "label")
    sst_x, sst_y = load_hf("stanfordnlp/sst2", "validation", "sentence", "label")

    cad_keys = set()
    for split in (tr, dv, te):
        for o, _, c, _, _ in split:
            cad_keys.add(norm_text(o))
            cad_keys.add(norm_text(c))
    cad_train_orig_keys = {norm_text(o) for o, *_ in tr}

    # Leakage guard: drop IMDB test reviews that are CAD *training* originals.
    keep = [i for i, x in enumerate(imdb_te_x) if norm_text(x) not in cad_train_orig_keys]
    n_leak = len(imdb_te_x) - len(keep)
    imdb_te_x = [imdb_te_x[i] for i in keep]
    imdb_te_y = [imdb_te_y[i] for i in keep]

    # Size-matched control: 1,707 extra *original* IMDB train reviews, label-balanced,
    # disjoint from every CAD review, and of comparable length (<= 400 words, since
    # Kaushik et al. restricted CAD to shorter reviews).
    rng = random.Random(seed_extra)
    cand = {0: [], 1: []}
    for x, y in zip(imdb_tr_x, imdb_tr_y):
        if norm_text(x) in cad_keys or len(x.split()) > 400:
            continue
        cand[y].append(x)
    n_neg = sum(1 for p in tr if p[1] == 0)
    n_pos = len(tr) - n_neg
    extra = [(x, 0) for x in rng.sample(cand[0], n_neg)] + [(x, 1) for x in rng.sample(cand[1], n_pos)]
    rng.shuffle(extra)

    return {
        "cad_train_pairs": tr,
        "cad_dev_pairs": dv,
        "cad_test_pairs": te,
        "extra_orig": extra,
        "imdb_test": (imdb_te_x, imdb_te_y),
        "sst2_dev": (sst_x, sst_y),
        "meta": {"imdb_test_removed_for_leakage": n_leak, "imdb_test_size": len(imdb_te_x)},
    }


def train_examples(data, mode: str):
    """Return (examples, paired) where examples is a list of (text, label).

    mode = 'orig'      : 1,707 CAD originals
           'orig2x'    : 1,707 originals + 1,707 extra IMDB originals (size-matched control)
           'cad'       : 1,707 originals + their 1,707 counterfactuals, kept as pairs
    For 'cad', examples are laid out [o_1, c_1, o_2, c_2, ...] and paired=True.
    """
    tr = data["cad_train_pairs"]
    if mode == "orig":
        return [(o, lo) for o, lo, *_ in tr], False
    if mode == "orig2x":
        return [(o, lo) for o, lo, *_ in tr] + list(data["extra_orig"]), False
    if mode == "cad":
        ex = []
        for o, lo, c, lc, _ in tr:
            ex += [(o, lo), (c, lc)]
        return ex, True
    raise ValueError(mode)
