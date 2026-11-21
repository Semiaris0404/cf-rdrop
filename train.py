"""Fine-tune RoBERTa-base on (counterfactually-augmented) IMDB with a chosen regularizer.

Example
-------
python train.py --condition cfrdrop --seed 1 --lr 2e-5 --rdrop 1.0 --cf 1.0 \
    --cad_root data/cad --cache data/cache.pkl --out runs/final

Conditions (see PRESETS): erm, erm2x, wd, ls, rdrop, mixout, cad, cad_rdrop, cad_cf, cfrdrop.
CLI flags override the preset's hyper-parameters.
"""
import argparse
import copy
import json
import math
import os
import pickle
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data import build_data, train_examples
from regularizers import apply_mixout, cf_consistency, cf_margin, ece_score, sym_kl

PRESETS = {
    "erm":       dict(data="orig"),
    "erm2x":     dict(data="orig2x"),
    "wd":        dict(data="orig", wd=0.1),
    "ls":        dict(data="orig", ls=0.1),
    "rdrop":     dict(data="orig", rdrop=1.0),
    "mixout":    dict(data="orig", mixout=0.7),
    "cad":       dict(data="cad"),
    "cad_rdrop": dict(data="cad", rdrop=1.0),
    "cad_cf":    dict(data="cad", cf=1.0),
    "cfrdrop":   dict(data="cad", rdrop=1.0, cf=1.0),
    "cad_cfm":   dict(data="cad", cfm=0.1),          # margin-space CF term (follow-up)
}
DEFAULTS = dict(data="orig", wd=0.0, ls=0.0, rdrop=0.0, mixout=0.0, cf=0.0, cfm=0.0)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=PRESETS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=16, help="examples per step (pairs count as 2)")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--eval_bs", type=int, default=128)
    ap.add_argument("--warmup", type=float, default=0.1)
    for k in ("wd", "ls", "rdrop", "mixout", "cf", "cfm"):
        ap.add_argument(f"--{k}", type=float, default=None)
    ap.add_argument("--cad_root", default="data/cad")
    ap.add_argument("--cache", default="data/cache.pkl")
    ap.add_argument("--out", default="runs/final")
    ap.add_argument("--tag", default="", help="suffix for the run name (used by the sweep)")
    ap.add_argument("--dev_only", action="store_true", help="skip test sets (hyper-parameter sweep)")
    ap.add_argument("--tiny", action="store_true", help="CPU smoke test with a tiny random model")
    args = ap.parse_args()
    cfg = dict(DEFAULTS, **PRESETS[args.condition])
    for k in ("wd", "ls", "rdrop", "mixout", "cf", "cfm"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    if (cfg["cf"] > 0 or cfg["cfm"] > 0) and cfg["data"] != "cad":
        ap.error("--cf needs paired counterfactual data")
    args.cfg = cfg
    return args


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def get_data(args):
    if os.path.exists(args.cache):
        with open(args.cache, "rb") as f:
            return pickle.load(f)
    data = build_data(args.cad_root)
    os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
    with open(args.cache, "wb") as f:
        pickle.dump(data, f)
    return data


class Encoded:
    """Pre-tokenised texts; batches are padded dynamically."""

    def __init__(self, tok, texts, labels, max_len):
        enc = tok(list(texts), truncation=True, max_length=max_len)
        self.ids = enc["input_ids"]
        self.y = list(labels)
        self.pad = tok.pad_token_id

    def __len__(self):
        return len(self.y)

    def batch(self, idx, device):
        L = max(len(self.ids[i]) for i in idx)
        ids = torch.full((len(idx), L), self.pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, : len(self.ids[i])] = torch.tensor(self.ids[i])
            att[r, : len(self.ids[i])] = 1
        y = torch.tensor([self.y[i] for i in idx])
        return ids.to(device), att.to(device), y.to(device)


def train_batches(n, bs, paired, rng):
    """Index batches. Paired data is [o1,c1,o2,c2,...]; pairs are shuffled as units so
    every batch keeps each original next to its counterfactual."""
    if paired:
        units = list(range(n // 2)); rng.shuffle(units)
        per = bs // 2
        for s in range(0, len(units), per):
            yield [j for u in units[s : s + per] for j in (2 * u, 2 * u + 1)]
    else:
        idx = list(range(n)); rng.shuffle(idx)
        for s in range(0, n, bs):
            yield idx[s : s + bs]


@torch.no_grad()
def predict(model, enc, bs, device, amp_dtype):
    """Return P(positive) for every example, processed in length-sorted order (fast)."""
    model.eval()
    order = sorted(range(len(enc)), key=lambda i: len(enc.ids[i]))
    out = np.zeros(len(enc), dtype=np.float32)
    for s in range(0, len(order), bs):
        idx = order[s : s + bs]
        ids, att, _ = enc.batch(idx, device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(input_ids=ids, attention_mask=att).logits
        out[idx] = torch.softmax(logits.float(), -1)[:, 1].cpu().numpy()
    return out


def acc(p, y):
    return float(((np.asarray(p) >= 0.5).astype(int) == np.asarray(y)).mean())


def macro_f1(p, y):
    from sklearn.metrics import f1_score
    return float(f1_score(y, (np.asarray(p) >= 0.5).astype(int), average="macro"))


def main():
    args = parse()
    cfg = args.cfg
    name = f"{args.condition}{args.tag}_s{args.seed}"
    out_dir = os.path.join(args.out, name)
    if os.path.exists(os.path.join(out_dir, "results.json")):
        print(f"[skip] {name} already finished"); return
    os.makedirs(out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = None
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    if args.tiny:  # offline smoke test: tiny random RoBERTa + local tokenizer stand-in
        from tiny import tiny_model_and_tok
        model, tok = tiny_model_and_tok()
    else:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    if cfg["mixout"] > 0:
        n = apply_mixout(model, cfg["mixout"])
        print(f"mixout p={cfg['mixout']} applied to {n} linear layers; dropout disabled")
    model.to(device)

    data = get_data(args)
    ex, paired = train_examples(data, cfg["data"])
    tr = Encoded(tok, [t for t, _ in ex], [y for _, y in ex], args.max_len)

    def enc_pairs(pairs):
        return (Encoded(tok, [p[0] for p in pairs], [p[1] for p in pairs], args.max_len),
                Encoded(tok, [p[2] for p in pairs], [p[3] for p in pairs], args.max_len))

    dev_o, dev_c = enc_pairs(data["cad_dev_pairs"])

    # AdamW; no decay on biases / LayerNorm (standard BERT recipe).
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": cfg["wd"]},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=args.lr)
    steps_per_epoch = math.ceil(len(tr) / args.bs)
    total = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total), total)

    rng = random.Random(args.seed)
    history, best = [], {"dev_acc": -1}
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        sums = {"loss": 0.0, "ce": 0.0, "rdrop": 0.0, "cf": 0.0, "cfm": 0.0}; nb = 0
        for idx in train_batches(len(tr), args.bs, paired, rng):
            ids, att, y = tr.batch(idx, device)
            k = 2 if cfg["rdrop"] > 0 else 1                     # R-Drop: two dropout passes
            if k == 2:
                ids, att = ids.repeat(2, 1), att.repeat(2, 1)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(input_ids=ids, attention_mask=att).logits.float()
            passes = logits.chunk(k, dim=0)
            ce = sum(F.cross_entropy(z, y, label_smoothing=cfg["ls"]) for z in passes) / k
            loss = ce
            rd = cf = cfm = torch.zeros((), device=device)
            if k == 2:
                rd = sym_kl(passes[0], passes[1]); loss = loss + cfg["rdrop"] * rd
            if cfg["cf"] > 0:
                cf = sum(cf_consistency(z) for z in passes) / k; loss = loss + cfg["cf"] * cf
            if cfg["cfm"] > 0:
                cfm = sum(cf_margin(z) for z in passes) / k; loss = loss + cfg["cfm"] * cfm
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            for key, v in (("loss", loss), ("ce", ce), ("rdrop", rd), ("cf", cf), ("cfm", cfm)):
                sums[key] += float(v.detach())
            nb += 1
        p_o = predict(model, dev_o, args.eval_bs, device, amp_dtype)
        p_c = predict(model, dev_c, args.eval_bs, device, amp_dtype)
        rec = {"epoch": ep, **{k: v / nb for k, v in sums.items()},
               "dev_acc": acc(p_o, dev_o.y), "dev_cf_acc": acc(p_c, dev_c.y),
               "min": round((time.time() - t0) / 60, 2)}
        history.append(rec); print(json.dumps(rec))
        # Model selection uses ORIGINAL dev reviews only, for every condition, so that
        # baselines are never selected with counterfactual information they did not train on.
        if rec["dev_acc"] > best["dev_acc"]:
            best = {"dev_acc": rec["dev_acc"], "epoch": ep,
                    "state": {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}}

    model.load_state_dict(best["state"])
    res = {"name": name, "condition": args.condition, "seed": args.seed, "cfg": cfg,
           "lr": args.lr, "epochs": args.epochs, "bs": args.bs, "max_len": args.max_len,
           "best_epoch": best["epoch"], "dev_acc": best["dev_acc"], "history": history,
           "train_minutes": round((time.time() - t0) / 60, 2), "meta": data["meta"],
           "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
           "torch": torch.__version__, "amp": str(amp_dtype)}
    preds = {}
    if not args.dev_only:
        te_o, te_c = enc_pairs(data["cad_test_pairs"])
        po, pc = predict(model, te_o, args.eval_bs, device, amp_dtype), predict(model, te_c, args.eval_bs, device, amp_dtype)
        ok_o = (po >= 0.5).astype(int) == np.array(te_o.y)
        ok_c = (pc >= 0.5).astype(int) == np.array(te_c.y)
        res.update(cad_orig_acc=float(ok_o.mean()), cad_cf_acc=float(ok_c.mean()),
                   cad_pair_acc=float((ok_o & ok_c).mean()))
        ix, iy = data["imdb_test"]; im = Encoded(tok, ix, iy, args.max_len)
        pi = predict(model, im, args.eval_bs, device, amp_dtype)
        res.update(imdb_acc=acc(pi, iy), imdb_f1=macro_f1(pi, iy), imdb_ece=ece_score(pi, iy))
        sx, sy = data["sst2_dev"]; ss = Encoded(tok, sx, sy, args.max_len)
        ps = predict(model, ss, args.eval_bs, device, amp_dtype)
        res.update(sst2_acc=acc(ps, sy), sst2_ece=ece_score(ps, sy))
        preds = dict(cad_orig=po, cad_cf=pc, imdb=pi, sst2=ps)
        print({k: round(v, 4) for k, v in res.items() if isinstance(v, float)})
    if preds:
        np.savez_compressed(os.path.join(out_dir, "preds.npz"), **preds)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
