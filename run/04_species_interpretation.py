"""Step 4 - Which species draw (or blur) the borders between landscapes? (objective O3)

    python run/04_species_interpretation.py        # ~10 min on 2 cores

Three complementary lenses on the same question:
  1. permutation importance computed on held-out points (how much worse the model gets when a
     species is shuffled), with p-values from label permutations (Altmann et al. 2010);
  2. partial dependence (how the predicted probability of each landscape changes when a species is
     recorded) and SHAP values (contribution of each species to each prediction);
  3. the indicator value IndVal.g of community ecology, so that the ML lens can be compared with
     the indicator-species analysis of the companion manuscript.

Outputs
  paper/figures/fig4_importance.pdf          importance + occurrence of each species per landscape
  paper/figures/fig5_partial_dependence.pdf  response of the class probabilities to key species
  paper/figures/figS1_shap.pdf               SHAP values (class-weighted random forest)
  paper/generated/tab_species.tex            Table S1 (one row per species)
  paper/generated/res_04.tex                 numbers quoted in the text;  data/results/04.json
  data/results/04_species.csv                the same table in CSV
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
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone

from community import indval
from ml import (SEED, SETTINGS, importance_null, main_model, make_models,
                oof_permutation_importance, partial_dependence)
from paths import save_figure
from report import p_txt, f2, f3, p_fmt, panel_label, pct, save_results, save_table, set_style
from wasps import AUTHORITY, COLOR, LABEL, LANDSCAPES, load_records, site_matrix, species_tribe

set_style()
rec = load_records()
X, meta = site_matrix(rec)
y = meta["landscape"].to_numpy()
Xv = X.to_numpy(float)
species = list(X.columns)
tribe = species_tribe(rec)

# ---------------------------------------------------------------- 1. permutation importance
t0 = time.time()
imp = oof_permutation_importance(main_model(), Xv, y)                # repetitions x species
print(f"importance: {imp.shape[0]} repetitions [{time.time() - t0:.0f}s]")
t0 = time.time()
null = importance_null(main_model(n_trees=SETTINGS["n_trees_perm"]), Xv, y)   # perms x species
print(f"importance null: {null.shape[0]} label permutations [{time.time() - t0:.0f}s]")
obs = imp.mean(0)
p_imp = (1 + (null >= obs[None, :] - 1e-12).sum(0)) / (1 + null.shape[0])

# ---------------------------------------------------------------- 2. partial dependence
full = clone(main_model(n_trees=1000)).fit(Xv, y)
pd_tab = partial_dependence(full, Xv, values=(0, 1, 2))
# the first draft used an unweighted forest; its partial dependence is kept for comparison, because
# with unequal class sizes the affinity of widespread species can follow the larger class
full_unw = clone(make_models(n_trees=1000)["rf_default"].estimator).fit(Xv, y)
pd_unw = partial_dependence(full_unw, Xv, values=(0, 1))
prev = (X > 0).groupby(y).mean().reindex(LANDSCAPES)                  # landscapes x species
n_pts = (X > 0).sum(axis=0)

# ---------------------------------------------------------------- 3. IndVal
iv = indval(X.to_numpy(), y, SETTINGS["n_perm_community"], SEED)

rows = []
for j, sp in enumerate(species):
    p0 = pd_tab[(pd_tab.j == j) & (pd_tab.value == 0)].iloc[0]
    p1 = pd_tab[(pd_tab.j == j) & (pd_tab.value == 1)].iloc[0]
    delta = {c: p1[c] - p0[c] for c in LANDSCAPES}
    affinity = max(delta, key=delta.get) if n_pts[sp] >= 2 else "rare"
    q0 = pd_unw[(pd_unw.j == j) & (pd_unw.value == 0)].iloc[0]
    q1 = pd_unw[(pd_unw.j == j) & (pd_unw.value == 1)].iloc[0]
    d_unw = {c: q1[c] - q0[c] for c in LANDSCAPES}
    affinity_unw = max(d_unw, key=d_unw.get) if n_pts[sp] >= 2 else "rare"
    lo, hi = np.quantile(imp[:, j], [0.025, 0.975])
    rows.append({
        "tribe": tribe.get(sp, ""), "species": sp, "n_points": int(n_pts[sp]),
        **{f"prev_{c}": prev.loc[c, sp] for c in LANDSCAPES},
        **{f"pd_{c}": p1[c] for c in LANDSCAPES},
        **{f"pd0_{c}": p0[c] for c in LANDSCAPES},
        **{f"pd_unw_{c}": q1[c] for c in LANDSCAPES},
        "affinity": affinity, "affinity_unweighted": affinity_unw, "importance": obs[j], "imp_lo": lo, "imp_hi": hi, "p_imp": p_imp[j],
        "indval_group": iv["group"][j], "indval": iv["stat"][j], "indval_p": iv["p"][j],
    })
tab = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
tab.to_csv("data/results/04_species.csv", index=False)
print(tab[["species", "n_points", "affinity", "importance", "p_imp", "indval_group", "indval",
           "indval_p"]].round(3).to_string())

# ---------------------------------------------------------------- Table S1
NPI, NPC = null.shape[0], SETTINGS["n_perm_community"]
SHORTLAB = {"floodplain": "Fpl", "pantanal": "Pan", "cerrado": "Cer"}
tex = ["\\begin{tabular}{llrrrrrrllrrlr}", "\\toprule",
       " & & \\multicolumn{3}{c}{Occurrence (\\% points)} & "
       "\\multicolumn{3}{c}{P(landscape $\\mid$ recorded)} & \\multicolumn{2}{c}{Affinity} & "
       "\\multicolumn{2}{c}{Permutation importance}"
       " & \\multicolumn{2}{c}{IndVal.g}\\\\",
       "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-10}\\cmidrule(lr){11-12}"
       "\\cmidrule(lr){13-14}",
       "Tribe & Species & Fpl & Pan & Cer & Fpl & Pan & Cer & BRF & RF$_{\\mathrm{unw}}$ & "
       "Mean [95\\% int.] & $p$ & "
       "Group & $p$\\\\", "\\midrule"]
for r in tab.itertuples():
    sig = (lambda p: "\\textbf{" if p < 0.05 else "{")
    tex.append(
        f"{r.tribe} & \\textit{{{r.species}}} {AUTHORITY.get(r.species, '')} & "
        + " & ".join(f"{100 * getattr(r, f'prev_{c}'):.0f}" for c in LANDSCAPES) + " & "
        + " & ".join(f"{getattr(r, f'pd_{c}'):.2f}" for c in LANDSCAPES) + " & "
        + f"{SHORTLAB.get(r.affinity, 'rare')} & {SHORTLAB.get(r.affinity_unweighted, 'rare')} & "
        + f"{r.importance:.3f} [{r.imp_lo:.3f}, {r.imp_hi:.3f}] & "
        + sig(r.p_imp) + p_fmt(r.p_imp, NPI) + "} & "
        + f"{SHORTLAB[r.indval_group]} ({r.indval:.2f}) & " + sig(r.indval_p) + p_fmt(r.indval_p, NPC)
        + "}\\\\")
tex += ["\\bottomrule", "\\end{tabular}"]
save_table("tab_species", "\n".join(tex))

# ---------------------------------------------------------------- figure 4
top = tab.copy()
fig, axs = plt.subplots(1, 2, figsize=(7.0, 4.0), gridspec_kw={"width_ratios": [1.5, 1]},
                        sharey=True)
ax = axs[0]
yy = np.arange(len(top))
for i, r in top.iterrows():
    col = COLOR.get(r.affinity, "#9AA0A6")
    ax.plot([r.imp_lo, r.imp_hi], [i, i], color=col, lw=1.2)
    ax.plot(r.importance, i, "o", color=col, ms=4.5, mfc=col if r.p_imp < 0.05 else "white")
ax.axvline(0, color="gray", lw=0.6)
ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
ax.set_yticks(yy, [f"$\\it{{{s.replace(' ', '~')}}}$" for s in top.species])
ax.invert_yaxis()
ax.set_xlabel("Drop in balanced accuracy when shuffled\n(held-out points; filled: $p$ < 0.05)")
for g in LANDSCAPES:
    ax.plot([], [], "o", color=COLOR[g], label=f"affinity: {LABEL[g]}")
ax.plot([], [], "o", color="#9AA0A6", label="rare (1 point)")
ax.legend(frameon=False, loc="lower right", fontsize=6.5)
panel_label(ax, "a")

ax = axs[1]
H = top[[f"prev_{c}" for c in LANDSCAPES]].to_numpy()
ax.imshow(H, cmap="Greens", vmin=0, vmax=1, aspect="auto")
for i in range(H.shape[0]):
    for j in range(3):
        if H[i, j] > 0:
            ax.text(j, i, f"{100 * H[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if H[i, j] > 0.6 else "black")
ax.set_xticks(range(3), [LABEL[c] for c in LANDSCAPES], rotation=0)
ax.set_title("Points with the species (%)", loc="left")
ax.tick_params(length=0)
ax.spines[:].set_visible(False)
panel_label(ax, "b")
save_figure(fig, "fig4_importance")

# ---------------------------------------------------------------- figure 5
key = [s for s in tab.species if tab.set_index("species").loc[s, "n_points"] >= 2][:4]
fig, axs = plt.subplots(1, len(key), figsize=(7.0, 2.2), sharey=True)
for ax, sp in zip(np.atleast_1d(axs), key):
    j = species.index(sp)
    d = pd_tab[pd_tab.j == j]
    for c in LANDSCAPES:
        ax.plot(d.value, d[c], "-o", color=COLOR[c], ms=3, lw=1.2, label=LABEL[c])
    obs_vals = X[sp].to_numpy()
    ax.plot(obs_vals + np.random.default_rng(0).uniform(-0.05, 0.05, len(obs_vals)),
            np.full(len(obs_vals), -0.03), "|", color="black", ms=4, alpha=0.4)
    ax.set_title(f"$\\it{{{sp.replace(' ', '~')}}}$", fontsize=7.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xlabel("Independent records")
    ax.set_ylim(-0.06, 1)
np.atleast_1d(axs)[0].set_ylabel("Partial dependence\n(predicted probability)")
np.atleast_1d(axs)[0].legend(frameon=False, fontsize=6.5, loc="upper left")
save_figure(fig, "fig5_partial_dependence")

# ---------------------------------------------------------------- SHAP (optional dependency)
shap_ok = False
try:
    import shap

    rf = clone(make_models(n_trees=1000)["rf_weighted"].estimator).fit(Xv, y)
    sv = shap.TreeExplainer(rf).shap_values(Xv)
    sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)   # n x p x K
    cls = list(rf.classes_)
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.6), sharey=True)
    order = np.argsort(-np.abs(sv).mean((0, 2)))
    for ax, c in zip(axs, ("pantanal", "cerrado")):
        S = sv[:, :, cls.index(c)]
        for i, j in enumerate(order):
            pres = Xv[:, j] > 0
            jit = np.random.default_rng(j).uniform(-0.25, 0.25, len(S))
            ax.scatter(S[~pres, j], i + jit[~pres], s=5, color="#BDC1C6", lw=0)
            ax.scatter(S[pres, j], i + jit[pres], s=8, color=COLOR[c], lw=0)
        ax.axvline(0, color="gray", lw=0.6)
        ax.set_title(f"SHAP values for P({LABEL[c]})", loc="left")
        ax.set_xlabel("Contribution to the predicted probability")
    axs[0].set_yticks(range(len(order)),
                      [f"$\\it{{{species[j].replace(' ', '~')}}}$" for j in order])
    axs[0].invert_yaxis()
    axs[0].scatter([], [], s=8, color="black", label="species recorded")
    axs[0].scatter([], [], s=8, color="#BDC1C6", label="species absent")
    axs[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    save_figure(fig, "figS1_shap")
    shap_ok = True
except ImportError:
    print("shap not installed: figS1_shap skipped (pip install shap)")

# ---------------------------------------------------------------- numbers for the text
def row(sp):
    return tab.set_index("species").loc[sp] if sp in set(tab.species) else None


vals = {"nrep-importance": SETTINGS["n_rep_importance"], "nperm-importance": NPI,
        "n-sig-importance": int((tab.p_imp < 0.05).sum()),
        "n-sig-indval": int((tab.indval_p < 0.05).sum()),
        "n-singletons": int((tab.n_points == 1).sum()),
        "top-species": f"\\textit{{{tab.species.iloc[0]}}}",
        "sig-species": ", ".join(f"\\textit{{{s}}}" for s in tab.species[tab.p_imp < 0.05])}
for sp, k in (("Agelaia bequaerti", "abeq"), ("Agelaia pallipes", "apal"),
              ("Polybia ruficeps", "pruf"), ("Polybia jurinei", "pjur"),
              ("Polybia ignobilis", "pign"), ("Polistes lanio", "plan")):
    r = row(sp)
    if r is None:
        continue
    vals.update({
        f"{k}-imp": f3(r.importance), f"{k}-imp-p": p_txt(r.p_imp, NPI),
        f"{k}-affinity": LABEL.get(r.affinity, "rare"),
        f"{k}-indval-group": LABEL[r.indval_group], f"{k}-indval": f2(r.indval),
        f"{k}-indval-p": p_txt(r.indval_p, SETTINGS["n_perm_community"]),
        **{f"{k}-prev-{c[:3]}": pct(r[f"prev_{c}"], 0) for c in LANDSCAPES},
        **{f"{k}-pd-{c[:3]}": pct(r[f"pd_{c}"], 0) for c in LANDSCAPES},
        **{f"{k}-pd0-{c[:3]}": pct(r[f"pd0_{c}"], 0) for c in LANDSCAPES},
        **{f"{k}-pdunw-{c[:3]}": pct(r[f"pd_unw_{c}"], 0) for c in LANDSCAPES},
        f"{k}-affinity-unw": LABEL.get(r.affinity_unweighted, "rare"),
    })
vals["shap"] = "yes" if shap_ok else "no"
save_results("04", vals, raw={"species": tab.to_dict("records")})
print("done: fig4_importance, fig5_partial_dependence, figS1_shap, tab_species, res_04")
