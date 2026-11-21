# Counterfactual R-Drop: regularising low-resource RoBERTa fine-tuning for sentiment

AI6103 Deep Learning, final project, Zhichang Wang (NTU).

This code builds on **R-Drop** (Liang et al., NeurIPS 2021), which makes the
predictions of two dropout passes agree, and on the **counterfactually-augmented IMDB data**
of Kaushik, Hovy & Lipton (ICLR 2020). It adds a counterfactual-consistency term. For an
original review $x$ and its human-edited, label-flipped counterpart $\tilde x$, the term
pushes $p(y\mid x)$ towards the class-swapped $\Pi\,p(y\mid\tilde x)$:

$$\mathcal L = \tfrac1K\sum_{k}\mathrm{CE}^{(k)} + \alpha\,\mathcal D_{\rm sym}\big(p^{(1)}, p^{(2)}\big)
 + \beta\,\tfrac1K\sum_k \mathcal D_{\rm sym}\big(p^{(k)}(\cdot\mid x),\ \Pi\, p^{(k)}(\cdot\mid\tilde x)\big)$$

$K=2$ dropout passes, $\mathcal D_{\rm sym}(p,q)=\tfrac12[\mathrm{KL}(p\Vert q)+\mathrm{KL}(q\Vert p)]$, and $\Pi$ swaps
the two class probabilities. The study compares it with five regularisers: weight decay, label smoothing, R-Drop,
Mixout (Lee et al., ICLR 2020) and plain CAD augmentation. A size-matched ERM baseline is
also included.

## Files
| File | Purpose |
|---|---|
| `data.py` | Loads CAD pairs, IMDB test (with a leakage filter), SST-2 dev, and the size-matched control set |
| `regularizers.py` | R-Drop KL, counterfactual consistency, Mixout (`MixLinear`), ECE |
| `train.py` | One run: `--condition {erm,erm2x,wd,ls,rdrop,mixout,cad,cad_rdrop,cad_cf,cfrdrop,cad_cfm}` |
| `run_all.py` | Dev sweep (equal budget per regulariser) and then 10 conditions × 3 seeds. Finished runs are skipped |
| `analyze.py` | Table 1, IMDB bootstrap tests, negation base-rate analysis (needs the IMDB texts in `data/cache.pkl`) |
| `analyze_extra.py` | CAD-only analyses: ablation bootstrap tests, margin asymmetry, edit-type accuracy, qualitative pairs, Table 2 |
| `make_figures.py` | Final paper figures from `report/summary*.json` |
| `tiny.py` | Offline CPU smoke test (`--tiny`). Never used for reported numbers |
| `AI6103_project.ipynb` | Colab driver |

## Reproduce
```bash
pip install -r requirements.txt
git clone https://github.com/acmi-lab/counterfactually-augmented-data data/cad
python run_all.py --stage sweep     # 13 dev-only runs
python run_all.py --stage final     # 30 runs
python run_all.py --stage extra     # follow-up: margin-space CF term, 2 sweep + 3 final runs
python analyze.py                   # -> report/summary.json, tables/main.tex, tables/negation.tex
python analyze_extra.py             # -> report/summary_extra.json, tables/edits.tex, tables/signif.tex
python make_figures.py              # -> report/figs/*.pdf
```
Hardware: one Colab NVIDIA L4 GPU (A100 also works) with bf16 autocast. Hyper-parameters: RoBERTa-base, AdamW, batch
size 16, 10 epochs, 10% linear warm-up, max length 512 tokens, gradient clipping at 1.0. Model selection
uses accuracy on the *original* CAD dev reviews only.

## Results (3 seeds, mean ± s.d.)
| Method | IMDB acc. | CAD pair acc. | SST-2 acc. |
|---|---|---|---|
| ERM | 92.6 ± 0.2 | 86.7 ± 2.0 | 85.9 ± 1.1 |
| ERM, 2× original data | 93.2 ± 0.7 | 85.4 ± 0.5 | 86.2 ± 1.4 |
| Mixout | 92.7 ± 0.2 | 88.0 ± 1.1 | 85.6 ± 1.9 |
| CAD | 92.1 ± 0.2 | **91.4 ± 0.4** | 86.8 ± 0.3 |
| CAD + R-Drop | 91.2 ± 0.5 | 88.3 ± 0.5 | **89.7 ± 0.3** |
| CF-R-Drop (ours) | 91.5 ± 0.7 | 89.5 ± 1.6 | 86.7 ± 1.2 |
| CAD + CF-margin (ours, follow-up) | 91.5 ± 0.4 | 90.0 ± 0.3 | 84.7 ± 1.2 |

See the report for all ten conditions, the significance tests, and the analysis of why the
consistency term reduces margin asymmetry without raising accuracy.
