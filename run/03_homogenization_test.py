"""Step 3 - Direct test of the homogenisation hypothesis H2 (objective O2).

    python run/03_homogenization_test.py           # ~3 min on 2 cores

H2 predicts that the anthropised wetland (Pantanal) is HARDER to tell apart from the dry savanna
(Cerrado) than from the preserved wetland (floodplain). Instead of reading this from the errors of
one 3-class confusion matrix, we fit one binary classifier per pair of landscapes and compare their
separability (AUC = probability that the model ranks a Pantanal point above a point of the other
landscape; 0.5 = indistinguishable, 1 = perfectly separable):

    H2:  AUC(Pantanal x Cerrado)  <  AUC(Pantanal x Floodplain)

Uncertainty: CV repetitions (model variability) and a paired bootstrap over points (sampling
variability; the Pantanal points are shared by both AUCs, so they are resampled once).
A model-free counterpart uses the Jaccard dissimilarity of each Pantanal point to the other areas.

Outputs
  paper/figures/fig3_homogenization.pdf   pairwise AUCs + simplex of predicted probabilities
  paper/generated/tab_pairwise.tex        AUC per pair and model, difference test
  paper/generated/res_03.tex              numbers quoted in the text;  data/results/03.json
  data/results/03_point_scores.csv        per-point scores used in the test
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from community import jaccard
from ml import (SEED, SETTINGS, bootstrap_auc, interval, main_model, make_models,
                paired_auc_bootstrap, pairwise_scores, repeated_oof)
from paths import save_figure
from report import p_txt, f2, p_fmt, panel_label, pct, save_results, save_table, set_style
from wasps import COLOR, LABEL, LANDSCAPES, load_records, site_matrix

set_style()
rec = load_records()
X, meta = site_matrix(rec)
y = meta["landscape"].to_numpy()
Xv = X.to_numpy(float)

PAIRS = [("pantanal", "cerrado"), ("pantanal", "floodplain"), ("cerrado", "floodplain")]
MODELS = {"brf": ("Balanced random forest", main_model()),
          "logistic": ("Logistic regression (L2)", make_models()["logistic"].estimator)}

# ---------------------------------------------------------------- pairwise AUCs
res, boot, rows = {}, {}, []
for mk, (mlabel, est) in MODELS.items():
    for a, b in PAIRS:
        r = pairwise_scores(est, Xv, y, a, b)
        bs = bootstrap_auc(r, seed=SEED)
        res[(mk, a, b)], boot[(mk, a, b)] = r, bs
        m_rep, lo_rep, hi_rep = interval(r["auc_reps"])
        _, lo, hi = interval(bs)
        rows.append({"model": mk, "pair": f"{a}-{b}", "auc": r["auc"], "boot_lo": lo,
                     "boot_hi": hi, "rep_mean": m_rep, "rep_lo": lo_rep, "rep_hi": hi_rep})
        print(f"{mk:9s} AUC {LABEL[a]} x {LABEL[b]:10s} = {r['auc']:.3f} "
              f"[bootstrap {lo:.3f}, {hi:.3f}]  (CV repetitions {lo_rep:.3f}-{hi_rep:.3f})")
pair_tab = pd.DataFrame(rows)

# paired difference: AUC(Pan x Fpl) - AUC(Pan x Cer)  (> 0 supports H2)
diff = {}
for mk in MODELS:
    pb = paired_auc_bootstrap(res[(mk, "pantanal", "floodplain")],
                              res[(mk, "pantanal", "cerrado")], seed=SEED)
    d = pb["diff"]
    diff[mk] = {"mean": float(d.mean()), "lo": float(np.quantile(d, 0.025)),
                "hi": float(np.quantile(d, 0.975)), "p_le0": float((d <= 0).mean()),
                "point": res[(mk, "pantanal", "floodplain")]["auc"]
                - res[(mk, "pantanal", "cerrado")]["auc"]}
    print(f"{mk:9s} AUC(Pan x Fpl) - AUC(Pan x Cer) = {diff[mk]['point']:.3f} "
          f"[{diff[mk]['lo']:.3f}, {diff[mk]['hi']:.3f}], bootstrap P(diff <= 0) = "
          f"{diff[mk]['p_le0']:.4f}")

# ---------------------------------------------------------------- 3-class probabilities
P, _, classes = repeated_oof(main_model(), Xv, y, n_repeats=SETTINGS["n_rep_pairwise"])
Pm = P.mean(0)
iF, iP, iC = (classes.index(k) for k in ("floodplain", "pantanal", "cerrado"))
pan = y == "pantanal"
p_cer, p_fpl, p_pan = Pm[pan, iC], Pm[pan, iF], Pm[pan, iP]
w = wilcoxon(p_cer, p_fpl, alternative="greater")
pull = p_cer / (p_cer + p_fpl)          # 0.5 = equally Cerrado-like and floodplain-like
print(f"Pantanal points: mean P(Cerrado) = {p_cer.mean():.2f}, P(floodplain) = {p_fpl.mean():.2f},"
      f" P(Pantanal) = {p_pan.mean():.2f}; Wilcoxon p = {w.pvalue:.4f}")

# ---------------------------------------------------------------- model-free counterpart
D = jaccard(X)
d_cer = D[np.ix_(pan, y == "cerrado")].mean(1)
d_fpl = D[np.ix_(pan, y == "floodplain")].mean(1)
wd = wilcoxon(d_fpl, d_cer, alternative="greater")
n_closer_cer = int((d_cer < d_fpl).sum())
print(f"Jaccard: Pantanal points closer to Cerrado than to floodplain: {n_closer_cer}/{pan.sum()} "
      f"(Wilcoxon p = {wd.pvalue:.4f})")

# ---------------------------------------------------------------- per-point table
pts = pd.DataFrame({"point_id": meta["point_id"].to_numpy(), "landscape": y,
                    "campaign": meta["campaign"].to_numpy(),
                    "P_floodplain": Pm[:, iF], "P_pantanal": Pm[:, iP], "P_cerrado": Pm[:, iC]})
for (mk, a, b), r in res.items():
    col = f"score_{mk}_{a}_vs_{b}"
    pts[col] = np.nan
    pts.loc[r["idx"], col] = r["score"]
pts.loc[pan, "jaccard_to_cerrado"] = d_cer
pts.loc[pan, "jaccard_to_floodplain"] = d_fpl
pts.to_csv("data/results/03_point_scores.csv", index=False)

# ---------------------------------------------------------------- table
tex = ["\\begin{tabular}{llccc}", "\\toprule",
       "Model & Pair & AUC & 95\\% CI (bootstrap) & Range over CV repetitions\\\\", "\\midrule"]
for mk, (mlabel, _) in MODELS.items():
    for i, (a, b) in enumerate(PAIRS):
        r = pair_tab[(pair_tab.model == mk) & (pair_tab.pair == f"{a}-{b}")].iloc[0]
        tex.append(f"{mlabel if i == 0 else ''} & {LABEL[a]} $\\times$ {LABEL[b]} & {r.auc:.2f} & "
                   f"[{r.boot_lo:.2f}, {r.boot_hi:.2f}] & {r.rep_lo:.2f}--{r.rep_hi:.2f}\\\\")
    d = diff[mk]
    tex.append(f" & $\\Delta$ = AUC$_{{\\mathrm{{Pan}}\\times\\mathrm{{Fpl}}}}$ $-$ "
               f"AUC$_{{\\mathrm{{Pan}}\\times\\mathrm{{Cer}}}}$ & {d['point']:.2f} & "
               f"[{d['lo']:.2f}, {d['hi']:.2f}] & $P(\\Delta\\leq0)$ = {d['p_le0']:.3f}\\\\")
    tex.append("\\midrule" if mk != list(MODELS)[-1] else "\\bottomrule")
tex.append("\\end{tabular}")
save_table("tab_pairwise", "\n".join(tex))

# ---------------------------------------------------------------- figure 3
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"width_ratios": [1.05, 1]})
ax = axs[0]
names = [f"{LABEL[a]}\n$\\times$ {LABEL[b]}" for a, b in PAIRS]
for k, (mk, mark, dx) in enumerate((("brf", "o", -0.12), ("logistic", "s", 0.12))):
    for i, (a, b) in enumerate(PAIRS):
        bs = boot[(mk, a, b)]
        parts = ax.violinplot(bs, positions=[i + dx], widths=0.2, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor("#1A73E8" if mk == "brf" else "#9AA0A6")
            pc.set_alpha(0.35)
        r = res[(mk, a, b)]
        _, lo, hi = interval(bs)
        ax.errorbar(i + dx, r["auc"], yerr=[[r["auc"] - lo], [hi - r["auc"]]], fmt=mark,
                    color="black", ms=4, lw=0.9, capsize=2,
                    label=MODELS[mk][0] if i == 0 else None)
ax.axhline(0.5, ls="--", lw=0.7, color="gray")
ax.text(2.45, 0.505, "indistinguishable", fontsize=6.5, color="gray", ha="right", va="bottom")
ax.set_xticks(range(3), names)
ax.set_ylim(0.2, 1.05)
ax.set_ylabel("Separability (AUC, out-of-fold)")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.62, 0.02), fontsize=6.5)
d = diff["brf"]
ax.set_title(f"$\\Delta$AUC (Pan$\\times$Fpl $-$ Pan$\\times$Cer) = {d['point']:.2f} "
             f"[{d['lo']:.2f}, {d['hi']:.2f}]", loc="left", fontsize=7)
panel_label(ax, "a")

ax = axs[1]
V = {"floodplain": np.array([0.0, 0.0]), "cerrado": np.array([1.0, 0.0]),
     "pantanal": np.array([0.5, np.sqrt(3) / 2])}
tri = np.vstack([V["floodplain"], V["cerrado"], V["pantanal"], V["floodplain"]])
ax.plot(tri[:, 0], tri[:, 1], color="black", lw=0.7)
cen = (V["floodplain"] + V["cerrado"] + V["pantanal"]) / 3
for a, b in (("floodplain", "cerrado"), ("cerrado", "pantanal"), ("pantanal", "floodplain")):
    mid = (V[a] + V[b]) / 2
    ax.plot([cen[0], mid[0]], [cen[1], mid[1]], color="gray", lw=0.5, ls=":")
xy = Pm[:, iF, None] * V["floodplain"] + Pm[:, iC, None] * V["cerrado"] \
    + Pm[:, iP, None] * V["pantanal"]
for g in LANDSCAPES:
    m = y == g
    ax.scatter(xy[m, 0], xy[m, 1], s=24 if g == "pantanal" else 14, color=COLOR[g],
               alpha=0.9 if g == "pantanal" else 0.55, lw=0.4, ec="white", label=LABEL[g],
               zorder=3 if g == "pantanal" else 2)
for g, (dx, dy, ha) in {"floodplain": (-0.03, -0.05, "left"), "cerrado": (0.03, -0.05, "right"),
                        "pantanal": (0, 0.03, "center")}.items():
    ax.text(V[g][0] + dx, V[g][1] + dy, LABEL[g], ha=ha, va="top" if g != "pantanal" else "bottom",
            fontsize=7, color=COLOR[g], fontweight="bold")
ax.set_xlim(-0.08, 1.08)
ax.set_ylim(-0.1, 0.98)
ax.set_aspect("equal")
ax.axis("off")
ax.legend(frameon=False, loc="upper left", fontsize=6.5, markerscale=0.9)
ax.set_title("Out-of-fold class probabilities", loc="left")
panel_label(ax, "b")
save_figure(fig, "fig3_homogenization")

# ---------------------------------------------------------------- numbers for the text
def auc_txt(mk, a, b):
    r = pair_tab[(pair_tab.model == mk) & (pair_tab.pair == f"{a}-{b}")].iloc[0]
    return f"{r.auc:.2f} [{r.boot_lo:.2f}, {r.boot_hi:.2f}]"


vals = {"nrep-pairwise": SETTINGS["n_rep_pairwise"], "nboot": SETTINGS["n_boot"]}
for mk in MODELS:
    vals[f"{mk}-auc-pancer"] = auc_txt(mk, "pantanal", "cerrado")
    vals[f"{mk}-auc-panfpl"] = auc_txt(mk, "pantanal", "floodplain")
    vals[f"{mk}-auc-cerfpl"] = auc_txt(mk, "cerrado", "floodplain")
    vals[f"{mk}-dauc"] = f"{diff[mk]['point']:.2f} [{diff[mk]['lo']:.2f}, {diff[mk]['hi']:.2f}]"
    vals[f"{mk}-dauc-p"] = p_txt(diff[mk]["p_le0"]) if diff[mk]["p_le0"] > 0 else \
        f"$<$ {1 / SETTINGS['n_boot']:.1g}"
vals.update({
    "pan-pcer": f2(p_cer.mean()), "pan-pfpl": f2(p_fpl.mean()), "pan-ppan": f2(p_pan.mean()),
    "pan-pull": f2(pull.mean()), "pan-wilcoxon-p": p_txt(w.pvalue),
    "pan-closer-cer": f"{n_closer_cer} of {int(pan.sum())}",
    "pan-jac-cer": f2(d_cer.mean()), "pan-jac-fpl": f2(d_fpl.mean()),
    "pan-jac-wilcoxon-p": p_txt(wd.pvalue),
})
save_results("03", vals, raw={"pairwise": rows, "difference": diff,
                              "pantanal_probabilities": {"cerrado": p_cer, "floodplain": p_fpl,
                                                         "pantanal": p_pan}})
print("done: fig3_homogenization, tab_pairwise, res_03")
