"""Step 2 - Can the wasp assemblage identify the landscape? Benchmark of classifiers (objective O1).

    python run/02_classifier_benchmark.py          # ~15 min on 2 cores
    WASPS_QUICK=1 python run/02_classifier_benchmark.py   # smoke test, ~1 min

Every model, from the null reference to the forests, is evaluated with the same 5-fold stratified
cross-validation repeated 50 times; metrics are computed on the out-of-fold predictions. The main
models are also tested against label permutations.

Outputs
  paper/figures/fig2_benchmark.pdf    balanced accuracy per model + confusion matrices
  paper/generated/tab_benchmark.tex   metrics per model (mean and 95% interval over repetitions)
  paper/generated/res_02.tex          numbers quoted in the text;  data/results/02.json
  data/results/02_cv_metrics.csv      metrics of every repetition and model
  data/results/02_oof_probabilities.csv   mean out-of-fold probabilities per point (main model)
"""

# --- repository layout: src/ and run/ go on the import path, and relative paths
# --- such as "data/..." are resolved from the repository root. Keep this block in every script.
import os as _os
import sys as _sys
from pathlib import Path as _Path

_HERE = _Path(__file__).resolve().parent if "__file__" in globals() else _Path.cwd()
_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "src").is_dir() and (p / "run").is_dir())
for _p in (_ROOT / "run", _ROOT / "src", _HERE):
    if str(_p) not in _sys.path:
        _sys.path.insert(0, str(_p))
_os.chdir(_ROOT)
# ---------------------------------------------------------------------------------

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml import (SEED, SETTINGS, interval, make_models, mean_confusion, metrics_table,
                permutation_test, repeated_oof)
from paths import save_figure
from report import p_txt, ci_fmt, f2, p_fmt, panel_label, pct, save_results, save_table, set_style
from wasps import COLOR, LABEL, LANDSCAPES, load_records, site_matrix

set_style()
rec = load_records()
X, meta = site_matrix(rec)
y = meta["landscape"].to_numpy()
Xv = X.to_numpy(float)
specs = make_models()
R = SETTINGS["n_repeats"]

# ---------------------------------------------------------------- replicate the first draft (OOB)
# The draft used ranger in R: 1000 trees, mtry = 4, out-of-bag accuracy. Same model in scikit-learn:
rf_oob = RandomForestClassifier(n_estimators=1000, max_features=4, oob_score=True,
                                random_state=SEED).fit(Xv, y)
print(f"RF out-of-bag accuracy (draft replication): {rf_oob.oob_score_:.3f}")

# ---------------------------------------------------------------- repeated CV of every model
tables, conf, oof_mean = [], {}, {}
for key, s in specs.items():
    t0 = time.time()
    P, Yhat, classes = repeated_oof(s.estimator, Xv, y, n_repeats=R)
    tab = metrics_table(y, P, Yhat, classes).assign(model=key)
    tables.append(tab)
    conf[key] = mean_confusion(y, Yhat, classes)
    oof_mean[key] = P.mean(0)
    print(f"{s.label:34s} balanced acc = {tab.balanced_accuracy.mean():.3f} "
          f"(Pantanal recall {tab.recall_pantanal.mean():.2f}, Pantanal->Cerrado "
          f"{tab.pantanal_to_cerrado.mean():.2f})  [{time.time() - t0:.0f}s]")
cv = pd.concat(tables, ignore_index=True)
cv.to_csv("data/results/02_cv_metrics.csv", index=False)

# ---------------------------------------------------------------- permutation tests
perm = {}
for key in ("logistic", "knn", "rf_weighted", "brf"):
    if key not in specs:
        continue
    est = specs[key].estimator
    if hasattr(est, "n_estimators"):
        est = est.set_params(n_estimators=SETTINGS["n_trees_perm"])
        n_perm = SETTINGS["n_perm_forest"]
    else:
        n_perm = SETTINGS["n_perm_cheap"]
    t0 = time.time()
    perm[key] = permutation_test(est, Xv, y, n_perm)
    print(f"permutation test {key:12s} observed = {perm[key]['observed']:.3f}  "
          f"null mean = {perm[key]['null'].mean():.3f}  p = {perm[key]['p']:.4f}  "
          f"[{time.time() - t0:.0f}s]")
specs = make_models()          # restore the original number of trees

# ---------------------------------------------------------------- table
metric_cols = [("balanced_accuracy", f2), ("f1_macro", f2), ("kappa", f2), ("log_loss", f2),
               ("recall_pantanal", f2), ("pantanal_to_cerrado", f2)]
tex = ["\\begin{tabular}{lcccccccc}", "\\toprule",
       "Model & Balanced acc. & Macro-$F_1$ & $\\kappa$ & Log-loss & Pantanal recall & "
       "Pan.$\\to$Cer. & $p_{\\mathrm{perm}}$\\\\", "\\midrule"]
summary = {}
for key, s in specs.items():
    t = cv[cv.model == key]
    cells = []
    for col, fmt in metric_cols:
        m, lo, hi = interval(t[col])
        summary[(key, col)] = (m, lo, hi)
        cells.append(f"{fmt(m)} [{fmt(lo)}, {fmt(hi)}]" if col == "balanced_accuracy"
                     else fmt(m))
    pp = p_fmt(perm[key]["p"], perm[key]["n_perm"]) if key in perm else "--"
    tex.append(f"{s.label} & " + " & ".join(cells) + f" & {pp}\\\\")
    if key == "null_prior":
        tex.append("\\midrule")
tex += ["\\bottomrule", "\\end{tabular}"]
save_table("tab_benchmark", "\n".join(tex))

# ---------------------------------------------------------------- out-of-fold probabilities
main = "brf" if "brf" in specs else "rf_weighted"
probs = pd.DataFrame(oof_mean[main], columns=[f"P_{c}" for c in LANDSCAPES])
probs.insert(0, "point_id", meta["point_id"].to_numpy())
probs.insert(1, "landscape", y)
probs.insert(2, "campaign", meta["campaign"].to_numpy())
probs["predicted"] = np.array(LANDSCAPES)[oof_mean[main].argmax(1)]
probs.to_csv("data/results/02_oof_probabilities.csv", index=False)

# ---------------------------------------------------------------- figure 2
order = list(specs)
fig = plt.figure(figsize=(7.0, 4.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.55, wspace=0.35)
ax = fig.add_subplot(gs[0, :])
data = [cv.loc[cv.model == k, "balanced_accuracy"].to_numpy() for k in order]
rng = np.random.default_rng(SEED)
for i, (k, d) in enumerate(zip(order, data)):
    col = {"null": "#9AA0A6", "baseline": "#5F6368", "main": "#1A73E8",
           "robustness": "#5F6368"}[specs[k].role]
    ax.scatter(d, i + rng.uniform(-0.18, 0.18, len(d)), s=5, color=col, alpha=0.35, lw=0)
    m, lo, hi = interval(d)
    ax.plot([lo, hi], [i, i], color="black", lw=1.2)
    ax.plot(m, i, "o", color="black", ms=3.5)
    if k in perm:
        ax.text(1.005, i, f"p = {p_fmt(perm[k]['p'], perm[k]['n_perm']).replace('$', '')}",
                va="center", fontsize=6.5, transform=ax.get_yaxis_transform())
ax.axvline(1 / 3, ls="--", lw=0.7, color="gray")
ax.text(1 / 3, len(order) - 0.4, " chance (1/3)", fontsize=6.5, color="gray", va="bottom")
ax.set_yticks(range(len(order)), [specs[k].label for k in order])
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel(f"Balanced accuracy (out-of-fold, {R} repetitions of 5-fold CV)")
panel_label(ax, "a")

for j, (key, title) in enumerate((("rf_default", "Random forest, unweighted"),
                                  (main, specs[main].label))):
    ax = fig.add_subplot(gs[1, j])
    C = conf[key]
    Cn = C / C.sum(1, keepdims=True)
    ax.imshow(Cn, cmap="Blues", vmin=0, vmax=1)
    for a in range(3):
        for b in range(3):
            ax.text(b, a, f"{C[a, b]:.1f}\n({100 * Cn[a, b]:.0f}%)", ha="center", va="center",
                    fontsize=6.5, color="white" if Cn[a, b] > 0.55 else "black")
    ax.set_xticks(range(3), [LABEL[c] for c in LANDSCAPES])
    ax.set_yticks(range(3), [LABEL[c] for c in LANDSCAPES])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title(title, loc="left")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    panel_label(ax, "bc"[j])
save_figure(fig, "fig2_benchmark")

# ---------------------------------------------------------------- numbers for the text
def ci(key, col, fmt=f2):
    m, lo, hi = summary[(key, col)]
    return ci_fmt(m, lo, hi, fmt)


models_nn = [k for k in specs if specs[k].role != "null"]
best = max(models_nn, key=lambda k: summary[(k, "balanced_accuracy")][0])
worst = min(models_nn, key=lambda k: summary[(k, "balanced_accuracy")][0])
p2c = [summary[(k, "pantanal_to_cerrado")][0] for k in models_nn]
p2f = [cv.loc[cv.model == k, "pantanal_to_floodplain"].mean() for k in models_nn]
vals = {
    "nrep": R, "nsplits": SETTINGS["n_splits"], "ntrees": SETTINGS["n_trees"],
    "nperm-forest": SETTINGS["n_perm_forest"], "nperm-cheap": SETTINGS["n_perm_cheap"],
    "oob-acc-draft-replication": pct(rf_oob.oob_score_),
    "best-model": specs[best].label, "best-balacc": ci(best, "balanced_accuracy"),
    "worst-model": specs[worst].label, "worst-balacc": ci(worst, "balanced_accuracy"),
    "pan-to-cer-min": pct(min(p2c)), "pan-to-cer-max": pct(max(p2c)),
    "pan-to-fpl-max": pct(max(p2f)),
    "null-balacc": ci("null_prior", "balanced_accuracy"),
    "nir": pct(max(np.mean(y == c) for c in LANDSCAPES)),
}
for key in specs:
    vals[f"{key}-balacc"] = ci(key, "balanced_accuracy")
    vals[f"{key}-kappa"] = f2(summary[(key, "kappa")][0])
    vals[f"{key}-f1"] = f2(summary[(key, "f1_macro")][0])
    vals[f"{key}-recall-pan"] = pct(summary[(key, "recall_pantanal")][0])
    vals[f"{key}-pan-to-cer"] = pct(summary[(key, "pantanal_to_cerrado")][0])
    t = cv[cv.model == key]
    vals[f"{key}-recall-fpl"] = pct(t.recall_floodplain.mean())
    vals[f"{key}-recall-cer"] = pct(t.recall_cerrado.mean())
    vals[f"{key}-pan-to-fpl"] = pct(t.pantanal_to_floodplain.mean())
    C = conf[key]                      # mean confusion: rows observed, columns predicted
    iC, iP = LANDSCAPES.index("cerrado"), LANDSCAPES.index("pantanal")
    vals[f"{key}-cer-to-pan"] = pct(C[iC, iP] / C[iC].sum())
    if key in perm:
        vals[f"{key}-pperm"] = p_txt(perm[key]["p"], perm[key]["n_perm"])
        vals[f"{key}-perm-null"] = f2(perm[key]["null"].mean())
vals["main-model"] = specs[main].label
save_results("02", vals, raw={
    "summary": {f"{k}|{c}": v for (k, c), v in summary.items()},
    "confusion_mean": {k: v for k, v in conf.items()},
    "permutation": {k: {"observed": v["observed"], "p": v["p"], "n_perm": v["n_perm"],
                        "null_mean": float(v["null"].mean())} for k, v in perm.items()},
    "oob_draft_replication": rf_oob.oob_score_})
print("done: fig2_benchmark, tab_benchmark, res_02")
