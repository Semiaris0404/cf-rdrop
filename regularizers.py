"""Regularizers compared in the project.

* R-Drop         (Liang et al., NeurIPS 2021): symmetric KL between two dropout passes.
* Mixout         (Lee, Cho & Kang, ICLR 2020): stochastically mixes fine-tuned weights
                 with their pretrained values instead of with zero.
* CF consistency (this project): symmetric KL between p(y|x) and the class-swapped
                 p(y|x~), where x~ is the label-flipped counterfactual of x.
Weight decay and label smoothing are handled by the optimiser / loss in train.py.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sym_kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """0.5 * [KL(p||q) + KL(q||p)], averaged over the batch (R-Drop's regulariser)."""
    lp = F.log_softmax(logits_p.float(), dim=-1)
    lq = F.log_softmax(logits_q.float(), dim=-1)
    kl_pq = F.kl_div(lq, lp, log_target=True, reduction="batchmean")  # KL(p||q)
    kl_qp = F.kl_div(lp, lq, log_target=True, reduction="batchmean")  # KL(q||p)
    return 0.5 * (kl_pq + kl_qp)


def cf_consistency(logits: torch.Tensor) -> torch.Tensor:
    """Counterfactual consistency for a batch laid out [o_1, c_1, o_2, c_2, ...].

    For binary sentiment, the counterfactual flips the label, so the model's belief that
    x is positive should equal its belief that x~ is negative. Swapping the two class
    logits of x~ (the permutation Pi) and applying the symmetric KL enforces
    p(y|x) = Pi p(y|x~).
    """
    z_o = logits[0::2]
    z_c = logits[1::2].flip(-1)          # Pi: swap negative/positive logits
    return sym_kl(z_o, z_c)


def cf_margin(logits: torch.Tensor) -> torch.Tensor:
    """Margin-space counterfactual consistency (non-saturating variant).

    m = z_pos - z_neg. Antisymmetry m(x) + m(x~) = 0 is penalised directly, so the
    gradient 2(m(x)+m(x~)) does not vanish when both predictions are confident,
    unlike the probability-space symmetric KL of cf_consistency.
    """
    m = (logits[:, 1] - logits[:, 0]).float()
    return ((m[0::2] + m[1::2]) ** 2).mean()


# --------------------------------------------------------------------------- Mixout
class MixLinear(nn.Module):
    """Linear layer with mixout(u, p): W' = u + M*(W-u)/(1-p), M ~ Bernoulli(1-p).

    E[W'] = W, so no rescaling is needed at evaluation time (W' = W in eval mode).
    """

    def __init__(self, linear: nn.Linear, p: float):
        super().__init__()
        self.p = p
        self.weight = linear.weight
        self.bias = linear.bias
        self.register_buffer("target", linear.weight.detach().clone(), persistent=False)

    def forward(self, x):
        if self.training and self.p > 0:
            mask = torch.empty_like(self.weight).bernoulli_(1.0 - self.p)
            w = self.target + mask * (self.weight - self.target) / (1.0 - self.p)
        else:
            w = self.weight
        return F.linear(x, w, self.bias)


def apply_mixout(model: nn.Module, p: float) -> int:
    """Replace every nn.Linear inside the encoder with MixLinear and zero all dropout,
    following Lee et al. (mixout replaces dropout). The randomly initialised classifier
    head has no meaningful pretrained target, so it is left untouched."""
    n = 0
    encoder = getattr(model, model.base_model_prefix)
    for name, module in list(encoder.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, child_name, MixLinear(child, p))
                n += 1
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
    return n


# --------------------------------------------------------------------------- metrics
def ece_score(probs_pos, labels, n_bins: int = 15) -> float:
    """Expected calibration error (top-label, equal-width bins)."""
    import numpy as np
    probs_pos = np.asarray(probs_pos, dtype=float)
    labels = np.asarray(labels)
    conf = np.maximum(probs_pos, 1 - probs_pos)
    pred = (probs_pos >= 0.5).astype(int)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)
