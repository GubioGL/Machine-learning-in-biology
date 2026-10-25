"""Step 1 - Describe the data and the assemblies with classical community ecology.

    python run/01_community_description.py

Outputs
  paper/figures/fig1_community.pdf     PCoA (Jaccard) + mean within/between-landscape dissimilarity
  paper/generated/tab_sampling.tex     sampling design per landscape (points, campaigns, records)
  paper/generated/tab_permanova.tex    global and pairwise PERMANOVA, PERMDISP
  paper/generated/res_01.tex           numbers quoted in the text;  data/results/01.json

Why this step: it shows the biologists' standard view of the data (does composition differ?) and
gives the reference against which the machine-learning results are read in the Discussion.
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
from matplotlib.patches import Ellipse

from community import jaccard, mean_dissimilarity, pairwise_permanova, pcoa, permanova, permdisp
from ml import SEED, SETTINGS
from paths import save_figure
from report import p_txt, f2, f3, p_fmt, panel_label, pct, save_results, save_table, set_style
from wasps import COLOR, LABEL, LANDSCAPES, PROTECTED_AREA, load_records, site_matrix, summary

set_style()
NP = SETTINGS["n_perm_community"]

rec = load_records()
X, meta = site_matrix(rec)
y = meta["landscape"].to_numpy()
info = summary(X, meta)
print("points / species / independent records:", info)
if (info["n_points"], info["n_species"], info["n_records"]) != (46, 17, 99):
    print("WARNING: counts differ from the first draft (46 points, 17 species, 99 records).")

# ---------------------------------------------------------------- sampling design table
rows = []
susp = rec[rec["Armadilha"] == "suspension"]
for g in LANDSCAPES:
    m = meta[meta.landscape == g]
    camp = susp[susp.Paisagem == g].drop_duplicates(["point", "campaign"])
    camp_txt = ", ".join(f"{c} ({n})" for c, n in camp.groupby("campaign").point.nunique().items())
    Xg = X.loc[m.index]
    rows.append({
        "landscape": g, "points": len(m), "campaigns": camp_txt,
        "records": int(Xg.values.sum()), "species": int((Xg.sum(axis=0) > 0).sum()),
        "richness": (Xg > 0).sum(axis=1).mean(),
    })
samp = pd.DataFrame(rows)
tex = ["\\begin{tabular}{llrlrrr}", "\\toprule",
       "Landscape & Protected area & Points & Campaigns (points) & Records & Species & "
       "Species/point\\\\", "\\midrule"]
for r in samp.itertuples():
    tex.append(f"{LABEL[r.landscape]} & {PROTECTED_AREA[r.landscape]} & {r.points} & "
               f"{r.campaigns} & {r.records} & {r.species} & {r.richness:.2f}\\\\")
tex += ["\\midrule", f"Total & & {info['n_points']} & & {info['n_records']} & "
        f"{info['n_species']} & {(X > 0).sum(axis=1).mean():.2f}\\\\", "\\bottomrule", "\\end{tabular}"]
save_table("tab_sampling", "\n".join(tex))

# ---------------------------------------------------------------- PERMANOVA / PERMDISP
D = jaccard(X)
glob = permanova(D, y, NP, SEED)
pairs = pairwise_permanova(D, y, NP, SEED)
disp = permdisp(D, y, NP, SEED)
print("PERMANOVA", glob)
for r in pairs:
    print(f"  {r['a']} x {r['b']}: F={r['F']:.2f} R2={r['R2']:.3f} p={r['p']:.4f} p_adj={r['p_adj']:.4f}")
print("PERMDISP", disp)

tex = ["\\begin{tabular}{lrrrr}", "\\toprule",
       "Test & Pseudo-$F$ & $R^2$ & $p$ & $p_{\\mathrm{Holm}}$\\\\", "\\midrule",
       f"PERMANOVA, all landscapes & {glob['F']:.2f} & {glob['R2']:.3f} & {p_fmt(glob['p'], NP)} & --\\\\"]
for r in pairs:
    tex.append(f"\\quad {LABEL[r['a']]} $\\times$ {LABEL[r['b']]} & {r['F']:.2f} & {r['R2']:.3f} & "
               f"{p_fmt(r['p'], NP)} & {p_fmt(r['p_adj'])}\\\\")
tex += [f"PERMDISP (dispersion) & {disp['F']:.2f} & -- & {p_fmt(disp['p'], NP)} & --\\\\",
        "\\bottomrule", "\\end{tabular}"]
save_table("tab_permanova", "\n".join(tex))

M = mean_dissimilarity(D, y, LANDSCAPES)
iP, iF, iC = (LANDSCAPES.index(k) for k in ("pantanal", "floodplain", "cerrado"))

# ---------------------------------------------------------------- species pools (all methods)
pools = {g: set(rec.loc[rec.Paisagem == g, "Especie"]) for g in LANDSCAPES}
pool_j = {}
for a, b in (("pantanal", "cerrado"), ("pantanal", "floodplain"), ("cerrado", "floodplain")):
    pool_j[f"{a}-{b}"] = len(pools[a] & pools[b]) / len(pools[a] | pools[b])

# ---------------------------------------------------------------- geographic distances
# The three areas are single sites; distance between them is a competing explanation for any
# similarity pattern (distance decay), so it is reported next to the dissimilarities.
def _haversine(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


centre = meta.groupby("landscape")[["lat", "lon"]].mean()
dist_km = {f"{a}-{b}": _haversine(centre.loc[a].to_numpy(), centre.loc[b].to_numpy())
           for a, b in (("pantanal", "cerrado"), ("pantanal", "floodplain"), ("cerrado", "floodplain"))}
print("distances between areas (km):", {k: round(v) for k, v in dist_km.items()})

# ---------------------------------------------------------------- figure 1
coords, eig, expl = pcoa(D)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.35, 1]})
ax = axs[0]
rng = np.random.default_rng(SEED)
jit = rng.normal(0, 0.006, coords[:, :2].shape)        # identical species lists overlap exactly
for g in LANDSCAPES:
    m = y == g
    pts = coords[m, :2] + jit[m]
    ax.scatter(pts[:, 0], pts[:, 1], s=22, color=COLOR[g], alpha=0.85, lw=0.4, ec="white",
               label=f"{LABEL[g]} (n = {m.sum()})")
    c = coords[m, :2].mean(0)
    cov = np.cov(coords[m, :2].T)
    w, v = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(v[1, -1], v[0, -1]))
    ax.add_patch(Ellipse(c, 2 * 1.96 * np.sqrt(w[-1]), 2 * 1.96 * np.sqrt(w[0]), angle=ang,
                         fc=COLOR[g], ec=COLOR[g], alpha=0.12, lw=0.8))
ax.set_xlabel(f"PCoA 1 ({100 * expl[0]:.1f}%)")
ax.set_ylabel(f"PCoA 2 ({100 * expl[1]:.1f}%)")
ax.legend(frameon=False, loc="best")
ax.set_title("Composition of the assemblies (Jaccard)", loc="left")
panel_label(ax, "a")

ax = axs[1]
im = ax.imshow(M, cmap="Greys", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                color="white" if M[i, j] > 0.6 else "black")
ax.set_xticks(range(3), [LABEL[g] for g in LANDSCAPES])
ax.set_yticks(range(3), [LABEL[g] for g in LANDSCAPES])
ax.set_title("Mean Jaccard dissimilarity", loc="left")
ax.spines[:].set_visible(False)
ax.tick_params(length=0)
panel_label(ax, "b")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
save_figure(fig, "fig1_community")

# ---------------------------------------------------------------- numbers for the text
pair = {f"{r['a']}-{r['b']}": r for r in pairs}
occ = (X > 0).groupby(y).any()                          # landscape x species: recorded at all?


def n_shared(a, b):
    """Species recorded (suspended traps) in both landscapes."""
    return int((occ.loc[a] & occ.loc[b]).sum())


def only_bequaerti(g, sp="Agelaia bequaerti"):
    """Points whose whole species list is the dominant shared species (text: 'x of n points')."""
    Xg = X[y == g] > 0
    if sp not in X.columns:
        return "--"
    return f"{int((Xg[sp] & (Xg.sum(axis=1) == 1)).sum())} of {len(Xg)}"


pc = pair.get("cerrado-pantanal")
pf = pair.get("floodplain-pantanal")
cf = pair.get("cerrado-floodplain")
vals = {
    "n-points": info["n_points"], "n-species": info["n_species"], "n-records": info["n_records"],
    "n-floodplain": info["n_floodplain"], "n-pantanal": info["n_pantanal"],
    "n-cerrado": info["n_cerrado"],
    "n-individuals-suspension": int((rec.Armadilha == "suspension").sum()),
    "n-individuals-all": len(rec), "n-species-all": rec.Especie.nunique(),
    "permanova-F": f2(glob["F"]), "permanova-R2": f3(glob["R2"]), "permanova-p": p_txt(glob["p"], NP),
    "permanova-pancer-F": f2(pc["F"]), "permanova-pancer-R2": f3(pc["R2"]),
    "permanova-pancer-padj": p_txt(pc["p_adj"]),
    "permanova-panfpl-F": f2(pf["F"]), "permanova-panfpl-R2": f3(pf["R2"]),
    "permanova-panfpl-padj": p_txt(pf["p_adj"]),
    "permanova-cerfpl-F": f2(cf["F"]), "permanova-cerfpl-R2": f3(cf["R2"]),
    "permanova-cerfpl-padj": p_txt(cf["p_adj"]),
    "permdisp-F": f2(disp["F"]), "permdisp-p": p_txt(disp["p"], NP),
    "nperm-community": NP,
    "pcoa-axis1": f"{100 * expl[0]:.1f}\\%", "pcoa-axis2": f"{100 * expl[1]:.1f}\\%",
    "diss-pan-cer": f2(M[iP, iC]), "diss-pan-fpl": f2(M[iP, iF]), "diss-cer-fpl": f2(M[iC, iF]),
    "diss-within-pan": f2(M[iP, iP]), "diss-within-fpl": f2(M[iF, iF]),
    "diss-within-cer": f2(M[iC, iC]),
    "pool-jac-pan-cer": f2(pool_j["pantanal-cerrado"]),
    "pool-jac-pan-fpl": f2(pool_j["pantanal-floodplain"]),
    "pool-jac-cer-fpl": f2(pool_j["cerrado-floodplain"]),
    "dist-pan-cer": f"{dist_km['pantanal-cerrado']:.0f}",
    "dist-pan-fpl": f"{dist_km['pantanal-floodplain']:.0f}",
    "dist-cer-fpl": f"{dist_km['cerrado-floodplain']:.0f}",
    "pool-n-pan": len(pools["pantanal"]), "pool-n-fpl": len(pools["floodplain"]),
    "pool-n-cer": len(pools["cerrado"]),
    "shared-pan-fpl-suspension": n_shared("pantanal", "floodplain"),
    "shared-pan-cer-suspension": n_shared("pantanal", "cerrado"),
    "shared-cer-fpl-suspension": n_shared("cerrado", "floodplain"),
    "only-bequaerti-pan": only_bequaerti("pantanal"),
    "only-bequaerti-cer": only_bequaerti("cerrado"),
    "richness-pan": f2(samp.set_index("landscape").loc["pantanal", "richness"]),
    "richness-fpl": f2(samp.set_index("landscape").loc["floodplain", "richness"]),
    "richness-cer": f2(samp.set_index("landscape").loc["cerrado", "richness"]),
}
save_results("01", vals, raw={"permanova": glob, "pairwise": pairs, "permdisp": disp,
                              "mean_dissimilarity": M, "species_pool_jaccard": pool_j,
                              "sampling": samp.to_dict("records")})
# the point-level matrix with readable IDs, handy for checks and for R users. It is field data,
# so it stays in data/raw/ (git-ignored) with the raw file until the data are published.
X.set_index(meta["point_id"]).assign(landscape=meta["landscape"].to_numpy(),
                                      campaign=meta["campaign"].to_numpy()) \
    .to_csv("data/raw/site_by_species.csv")
print("done: fig1_community, tab_sampling, tab_permanova, res_01")
