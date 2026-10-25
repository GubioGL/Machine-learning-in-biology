"""Classical community-ecology tools (pure numpy, no R needed).

Jaccard dissimilarity, principal coordinates analysis (PCoA), PERMANOVA (Anderson 2001), PERMDISP
(Anderson 2006) and the group-equalised indicator value IndVal.g (Dufrene & Legendre 1997;
De Caceres & Legendre 2009). They are the 'descriptive' layer of the paper and the bridge to the
companion manuscript, which used the same tests in R (vegan / indicspecies).
"""
from itertools import combinations

import numpy as np
from scipy.spatial.distance import pdist, squareform


# ---------------------------------------------------------------- distances and ordination
def jaccard(X):
    """Square matrix of binary Jaccard dissimilarities, d = 1 - |A n B| / |A u B|."""
    B = np.asarray(X) > 0
    return squareform(pdist(B, metric="jaccard"))


def pcoa(D):
    """Classical (Torgerson-Gower) PCoA.

    Returns coordinates on the axes with positive eigenvalues, all eigenvalues, and the share of
    the positive inertia explained by each retained axis.
    """
    D = np.asarray(D, float)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    pos = w > 1e-10
    coords = V[:, pos] * np.sqrt(w[pos])
    explained = w[pos] / w[pos].sum()
    return coords, w, explained


# ---------------------------------------------------------------- PERMANOVA / PERMDISP
def _pseudo_f(D2, labels):
    labels = np.asarray(labels)
    n = len(labels)
    groups = np.unique(labels)
    ss_t = D2[np.triu_indices(n, 1)].sum() / n
    ss_w = 0.0
    for g in groups:
        idx = np.where(labels == g)[0]
        sub = D2[np.ix_(idx, idx)]
        ss_w += sub[np.triu_indices(len(idx), 1)].sum() / len(idx)
    ss_a = ss_t - ss_w
    k = len(groups)
    F = (ss_a / (k - 1)) / (ss_w / (n - k))
    return F, ss_a / ss_t


def permanova(D, labels, n_perm=9999, seed=0):
    """One-way PERMANOVA. Returns dict with F, R2 and permutation p-value."""
    D2 = np.asarray(D, float) ** 2
    labels = np.asarray(labels)
    F, R2 = _pseudo_f(D2, labels)
    rng = np.random.default_rng(seed)
    exceed = sum(_pseudo_f(D2, rng.permutation(labels))[0] >= F for _ in range(n_perm))
    return {"F": F, "R2": R2, "p": (exceed + 1) / (n_perm + 1), "n_perm": n_perm,
            "df1": len(np.unique(labels)) - 1, "df2": len(labels) - len(np.unique(labels))}


def pairwise_permanova(D, labels, n_perm=9999, seed=0, adjust="holm"):
    """PERMANOVA for every pair of groups, with Holm (default) or Bonferroni adjusted p-values."""
    labels = np.asarray(labels)
    out = []
    for a, b in combinations(sorted(np.unique(labels)), 2):
        idx = np.where(np.isin(labels, [a, b]))[0]
        r = permanova(np.asarray(D)[np.ix_(idx, idx)], labels[idx], n_perm, seed)
        out.append({"a": a, "b": b, **r})
    p = np.array([r["p"] for r in out])
    m = len(p)
    if adjust == "bonferroni":
        padj = np.minimum(1, p * m)
    else:                                       # Holm step-down
        order = np.argsort(p)
        padj = np.empty(m)
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, (m - rank) * p[i])
            padj[i] = min(1.0, running)
    for r, pa in zip(out, padj):
        r["p_adj"] = pa
    return out


def _dist_to_centroids(D, labels):
    """Distance of each point to its group centroid in the full PCoA space (negative axes handled)."""
    D = np.asarray(D, float)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(G)
    keep = np.abs(w) > 1e-10
    w, V = w[keep], V[:, keep]
    Y = V * np.sqrt(np.abs(w))
    sign = np.sign(w)
    z = np.empty(n)
    for g in np.unique(labels):
        idx = np.where(labels == g)[0]
        c = Y[idx].mean(0)
        sq = ((Y[idx] - c) ** 2 * sign).sum(1)
        z[idx] = np.sqrt(np.maximum(sq, 0))
    return z


def _anova_f(z, labels):
    groups = np.unique(labels)
    n, k = len(z), len(groups)
    grand = z.mean()
    ss_a = sum(len(z[labels == g]) * (z[labels == g].mean() - grand) ** 2 for g in groups)
    ss_w = sum(((z[labels == g] - z[labels == g].mean()) ** 2).sum() for g in groups)
    return (ss_a / (k - 1)) / (ss_w / (n - k))


def permdisp(D, labels, n_perm=9999, seed=0):
    """Test of homogeneity of multivariate dispersions (betadisper + permutest in vegan)."""
    labels = np.asarray(labels)
    z = _dist_to_centroids(D, labels)
    F = _anova_f(z, labels)
    means = {g: z[labels == g].mean() for g in np.unique(labels)}
    fitted = np.array([means[g] for g in labels])
    resid = z - fitted                 # vegan::permutest.betadisper permutes the residuals
    rng = np.random.default_rng(seed)
    exceed = sum(_anova_f(rng.permutation(resid), labels) >= F for _ in range(n_perm))
    return {"F": F, "p": (exceed + 1) / (n_perm + 1),
            "mean_dist": means}


def mean_dissimilarity(D, labels, order):
    """Mean dissimilarity within (diagonal) and between (off-diagonal) groups."""
    D = np.asarray(D)
    labels = np.asarray(labels)
    M = np.zeros((len(order), len(order)))
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            ia, ib = np.where(labels == a)[0], np.where(labels == b)[0]
            sub = D[np.ix_(ia, ib)]
            M[i, j] = sub[np.triu_indices(len(ia), 1)].mean() if a == b else sub.mean()
    return M


# ---------------------------------------------------------------- indicator species
def _indval_g(X, labels, groups):
    X = np.asarray(X, float)
    means = np.vstack([X[labels == g].mean(0) for g in groups])            # groups x species
    tot = means.sum(0)
    A = np.divide(means, tot, out=np.zeros_like(means), where=tot > 0)
    B = np.vstack([(X[labels == g] > 0).mean(0) for g in groups])
    return np.sqrt(A * B)                                                   # groups x species


def indval(X, labels, n_perm=9999, seed=0):
    """Group-equalised IndVal for single groups (indicspecies::multipatt, duleg = TRUE).

    Returns per species: best group, IndVal statistic, A (specificity), B (fidelity) and the
    permutation p-value of the maximum statistic.
    """
    labels = np.asarray(labels)
    groups = sorted(np.unique(labels))
    S = _indval_g(X, labels, groups)
    stat = S.max(0)
    best = S.argmax(0)
    rng = np.random.default_rng(seed)
    exceed = np.zeros(S.shape[1])
    for _ in range(n_perm):
        exceed += _indval_g(X, rng.permutation(labels), groups).max(0) >= stat - 1e-12
    p = (exceed + 1) / (n_perm + 1)
    X = np.asarray(X, float)
    means = np.vstack([X[labels == g].mean(0) for g in groups])
    tot = means.sum(0)
    A = np.divide(means, tot, out=np.zeros_like(means), where=tot > 0)
    B = np.vstack([(X[labels == g] > 0).mean(0) for g in groups])
    cols = np.arange(S.shape[1])
    return {"group": [groups[b] for b in best], "stat": stat, "A": A[best, cols],
            "B": B[best, cols], "p": p}
