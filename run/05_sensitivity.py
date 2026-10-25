"""Step 5 - How robust are the conclusions? (objective O4)

    python run/05_sensitivity.py                   # ~12 min on 2 cores

Each scenario changes ONE analytical decision and re-estimates the quantities that carry the
conclusions: balanced accuracy (O1), Pantanal recall and Pantanal->Cerrado error rate, and the two
pairwise AUCs of the homogenisation test (O2).

  reference          Balanced RF, 5-fold stratified CV, record frequencies
  spatial blocks     CV with spatial blocks inside each landscape (neighbouring points held out)
  presence/absence   0/1 instead of record frequencies
  no singletons      species recorded in a single point removed
  equal n (11)       30 random subsamples with 11 points per landscape
  no A. bequaerti    the dominant species shared by Pantanal and Cerrado removed
  mtry = 2 / 8       number of species tried at each split
  leaf >= 3          minimum of 3 points per terminal node

Sampling-campaign checks (the landscapes were sampled in different months and years):
  - within the floodplain, how much does composition change between campaigns (PERMANOVA)?
  - campaign transfer: are floodplain points of a held-out campaign still recognised?
  - Pantanal points of the two campaigns: predicted probabilities.

Outputs
  paper/figures/figS2_sensitivity.pdf   forest plot of the key quantities per scenario
  paper/generated/tab_sensitivity.tex   table of the scenarios
  paper/generated/res_05.tex            numbers quoted in the text;  data/results/05.json
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
from sklearn.base import clone

from community import jaccard, permanova
from ml import (SEED, SETTINGS, interval, main_model, metrics_table, pairwise_scores,
                repeated_oof, spatial_block_splits)
from paths import save_figure
from report import p_txt, f2, p_fmt, pct, save_results, save_table, set_style
from wasps import LABEL, LANDSCAPES, load_records, site_matrix

set_style()
rec = load_records()
X, meta = site_matrix(rec)
y = meta["landscape"].to_numpy()
NREP = max(2, SETTINGS["n_rep_pairwise"] // 2)          # repetitions per scenario
N_SUB = 30 if not _os.environ.get("WASPS_QUICK") else 3   # subsamples for the equal-n scenario


def evaluate(Xs, ys, model=None, splits3=None, split_fn=None, n_rep=NREP, seed=SEED):
    """Key quantities for one scenario."""
    model = model if model is not None else main_model()
    Xv = Xs.to_numpy(float) if hasattr(Xs, "to_numpy") else np.asarray(Xs, float)
    P, Yhat, classes = repeated_oof(model, Xv, ys, n_repeats=n_rep, splits=splits3, seed=seed)
    m = metrics_table(ys, P, Yhat, classes)
    pc = pairwise_scores(model, Xv, ys, "pantanal", "cerrado", n_rep, seed, split_fn)
    pf = pairwise_scores(model, Xv, ys, "pantanal", "floodplain", n_rep, seed, split_fn)
    return {"balanced_accuracy": m.balanced_accuracy.mean(),
            "recall_pantanal": m.recall_pantanal.mean(),
            "pantanal_to_cerrado": m.pantanal_to_cerrado.mean(),
            "auc_pan_cer": pc["auc"], "auc_pan_fpl": pf["auc"],
            "auc_pan_cer_reps": pc["auc_reps"], "P": P.mean(0), "classes": classes}


scen = {}
t0 = time.time()
scen["Reference"] = evaluate(X, y)
print(f"reference done [{time.time() - t0:.0f}s]")

# spatial blocks inside each landscape
blocks3 = spatial_block_splits(meta, n_blocks=3, n_repeats=NREP)
scen["Spatial blocks (3 per landscape)"] = evaluate(
    X, y, splits3=blocks3,
    split_fn=lambda idx: spatial_block_splits(meta.iloc[idx], n_blocks=3, n_repeats=NREP))

scen["Presence/absence"] = evaluate((X > 0).astype(int), y)
keep = (X > 0).sum(axis=0) >= 2
scen[f"Without singletons ({int((~keep).sum())} species)"] = evaluate(X.loc[:, keep], y)
if "Agelaia bequaerti" in X.columns:
    scen["Without $A.~bequaerti$"] = evaluate(X.drop(columns="Agelaia bequaerti"), y)
for label, params in (("mtry = 2", {"max_features": 2}), ("mtry = 8", {"max_features": 8}),
                      ("Minimum leaf size 3", {"min_samples_leaf": 3})):
    scen[label] = evaluate(X, y, model=clone(main_model()).set_params(**params))
print(f"scenarios done [{time.time() - t0:.0f}s]")

# equal sample size: 11 points per landscape
rng = np.random.default_rng(SEED)
n_min = int(pd.Series(y).value_counts().min())
subs = []
for k in range(N_SUB):
    idx = np.concatenate([rng.choice(np.where(y == g)[0], n_min, replace=False)
                          for g in LANDSCAPES])
    r = evaluate(X.iloc[idx], y[idx], n_rep=2, seed=SEED + k)
    subs.append(r)
scen[f"Equal n ({n_min} per landscape)"] = {
    k: np.mean([s[k] for s in subs]) for k in ("balanced_accuracy", "recall_pantanal",
                                               "pantanal_to_cerrado", "auc_pan_cer", "auc_pan_fpl")}
scen[f"Equal n ({n_min} per landscape)"]["auc_pan_cer_reps"] = np.array([s["auc_pan_cer"] for s in subs])
print(f"equal-n done [{time.time() - t0:.0f}s]")

# ---------------------------------------------------------------- sampling-campaign checks
fpl = meta.landscape.eq("floodplain").to_numpy()
D_f = jaccard(X[fpl])
camp_f = meta.loc[fpl, "campaign"].to_numpy()
perm_camp = permanova(D_f, camp_f, SETTINGS["n_perm_community"], SEED)
perm_land = permanova(jaccard(X), y, SETTINGS["n_perm_community"], SEED)
print(f"floodplain campaigns PERMANOVA R2 = {perm_camp['R2']:.3f} (p = {perm_camp['p']:.3f}); "
      f"landscapes R2 = {perm_land['R2']:.3f}")

transfer = {}
Xv = X.to_numpy(float)
for c in sorted(set(camp_f)):
    test = np.where(fpl & meta.campaign.eq(c).to_numpy())[0]
    train = np.setdiff1d(np.arange(len(y)), test)
    hits = []
    for k in range(NREP):
        m = clone(main_model()).set_params(random_state=SEED + k).fit(Xv[train], y[train])
        hits.append(np.mean(m.predict(Xv[test]) == "floodplain"))
    transfer[c] = (len(test), float(np.mean(hits)))
    print(f"floodplain campaign {c} held out (n = {len(test)}): {100 * np.mean(hits):.0f}% "
          "classified as floodplain")

Pref, cls = scen["Reference"]["P"], scen["Reference"]["classes"]
pan_camp = {}
for c in sorted(meta.loc[~fpl & meta.landscape.eq("pantanal").to_numpy(), "campaign"].unique()):
    m = (meta.landscape.eq("pantanal") & meta.campaign.eq(c)).to_numpy()
    pan_camp[c] = {g: float(Pref[m, cls.index(g)].mean()) for g in LANDSCAPES} | {"n": int(m.sum())}
    print(f"Pantanal campaign {c} (n = {m.sum()}):",
          {g: round(v, 2) for g, v in pan_camp[c].items() if g != "n"})

# ---------------------------------------------------------------- table + figure
cols = [("balanced_accuracy", "Balanced acc."), ("recall_pantanal", "Pantanal recall"),
        ("pantanal_to_cerrado", "Pan.$\\to$Cer."), ("auc_pan_cer", "AUC Pan$\\times$Cer"),
        ("auc_pan_fpl", "AUC Pan$\\times$Fpl")]
tex = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
       "Scenario & " + " & ".join(c[1] for c in cols) + "\\\\", "\\midrule"]
for name, r in scen.items():
    tex.append(name + " & " + " & ".join(f2(r[c]) for c, _ in cols) + "\\\\")
tex += ["\\bottomrule", "\\end{tabular}"]
save_table("tab_sensitivity", "\n".join(tex))

names = list(scen)
fig, axs = plt.subplots(1, 3, figsize=(7.0, 3.0), sharey=True)
for ax, (col, title) in zip(axs, (("balanced_accuracy", "Balanced accuracy"),
                                  ("pantanal_to_cerrado", "Pantanal points\npredicted as Cerrado"),
                                  ("auc_pan_cer", "AUC Pantanal $\\times$ Cerrado"))):
    v = [scen[n][col] for n in names]
    ax.plot(v, range(len(names)), "o", color="#1A73E8", ms=4)
    if col == "auc_pan_cer":
        for i, n in enumerate(names):
            reps = scen[n].get("auc_pan_cer_reps")
            if reps is not None and len(reps) > 1:
                _, lo, hi = interval(reps)
                ax.plot([lo, hi], [i, i], color="#1A73E8", lw=1)
        ax.axvline(0.5, ls="--", lw=0.7, color="gray")
    ax.axvline(scen["Reference"][col], lw=0.6, color="black", alpha=0.4)
    ax.set_title(title, loc="left")
    ax.set_xlim(0, 1)
axs[0].set_yticks(range(len(names)), names)
axs[0].invert_yaxis()
save_figure(fig, "figS2_sensitivity")

# ---------------------------------------------------------------- numbers for the text
ref = scen["Reference"]
sp = scen["Spatial blocks (3 per landscape)"]
eq = scen[f"Equal n ({n_min} per landscape)"]
auc_pc = [r["auc_pan_cer"] for r in scen.values()]
vals = {
    "sens-n-scenarios": len(scen), "sens-nrep": NREP, "sens-nsub": N_SUB,
    "sens-auc-pancer-min": f2(min(auc_pc)), "sens-auc-pancer-max": f2(max(auc_pc)),
    "sens-auc-panfpl-min": f2(min(r["auc_pan_fpl"] for r in scen.values())),
    "sens-balacc-min": f2(min(r["balanced_accuracy"] for r in scen.values())),
    "sens-balacc-max": f2(max(r["balanced_accuracy"] for r in scen.values())),
    "sens-spatial-balacc": f2(sp["balanced_accuracy"]), "sens-ref-balacc": f2(ref["balanced_accuracy"]),
    "sens-spatial-auc-pancer": f2(sp["auc_pan_cer"]), "sens-ref-auc-pancer": f2(ref["auc_pan_cer"]),
    "sens-equal-pan-to-cer": pct(eq["pantanal_to_cerrado"]),
    "sens-equal-auc-pancer": f2(eq["auc_pan_cer"]),
    "camp-fpl-R2": f"{perm_camp['R2']:.3f}", "camp-fpl-p": p_txt(perm_camp["p"], SETTINGS["n_perm_community"]),
    "camp-land-R2": f"{perm_land['R2']:.3f}",
    "transfer-fpl": "; ".join(f"{c}: {100 * h:.0f}\\% of {n}" for c, (n, h) in transfer.items()),
    "transfer-fpl-min": pct(min(h for _, h in transfer.values()), 0),
}
if "Without $A.~bequaerti$" in scen:
    nb = scen["Without $A.~bequaerti$"]
    vals.update({"sens-nobeq-auc-pancer": f2(nb["auc_pan_cer"]),
                 "sens-nobeq-pan-to-cer": pct(nb["pantanal_to_cerrado"]),
                 "sens-nobeq-balacc": f2(nb["balanced_accuracy"])})
for c, d in pan_camp.items():
    k = c.replace("-", "")
    vals[f"pan-camp-{k}"] = (f"{d['n']} points: P(Cerrado) = {d['cerrado']:.2f}, "
                             f"P(Pantanal) = {d['pantanal']:.2f}, P(floodplain) = {d['floodplain']:.2f}")
save_results("05", vals, raw={
    "scenarios": {n: {k: v for k, v in r.items() if k in dict(cols)} for n, r in scen.items()},
    "floodplain_campaign_permanova": perm_camp, "campaign_transfer": transfer,
    "pantanal_campaigns": pan_camp})
print("done: figS2_sensitivity, tab_sensitivity, res_05")
