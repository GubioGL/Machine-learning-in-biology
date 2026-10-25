"""Machine-learning layer: models, repeated cross-validation, permutation tests, pairwise
separability (the homogenisation test), out-of-fold permutation importance and partial dependence.

Design choices (they are stated in the Methods, keep them in sync):
- Hyperparameters are fixed a priori to standard values (no tuning), so the cross-validated
  scores are not optimistic and no nested CV is needed. Sensitivity to them is run in run/05.
- Every model is evaluated on the SAME splits (seed + repeat index), so differences between models
  are paired comparisons.
- Metrics are computed on the pooled out-of-fold (OOF) predictions of each repetition (all 46 points
  predicted once per repetition), which is more stable than averaging tiny folds.
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

try:                                    # imbalanced-learn provides the Balanced Random Forest
    from imblearn.ensemble import BalancedRandomForestClassifier
except ImportError:                     # pragma: no cover
    BalancedRandomForestClassifier = None

SEED = 42

# Analysis sizes. WASPS_QUICK=1 shrinks everything for a smoke test (minutes instead of ~30 min).
SETTINGS = dict(
    n_trees=500,            # trees per forest
    n_splits=5,             # folds per repetition
    n_repeats=50,           # CV repetitions for the model benchmark
    n_perm_cheap=999,       # label permutations, fast models (logistic, k-NN)
    n_perm_forest=199,      # label permutations, forests (smallest attainable p = 0.005)
    n_trees_perm=300,       # trees per forest inside permutation tests (observed and null alike)
    n_rep_pairwise=20,      # CV repetitions for the pairwise AUC test
    n_boot=2000,            # bootstrap resamples of the pairwise AUCs
    n_rep_importance=10,    # CV repetitions for permutation importance
    n_feature_perm=10,      # shuffles of each species inside each test fold
    n_perm_importance=99,   # label permutations for importance p-values (Altmann et al. 2010)
    n_perm_community=9999,  # PERMANOVA / PERMDISP / IndVal permutations
)
if os.environ.get("WASPS_QUICK"):
    SETTINGS.update(n_trees=100, n_repeats=5, n_perm_cheap=49, n_perm_forest=19, n_trees_perm=50,
                    n_rep_pairwise=3, n_boot=200, n_rep_importance=2, n_feature_perm=3,
                    n_perm_importance=9, n_perm_community=199)

N_JOBS = int(os.environ.get("WASPS_JOBS", "-1"))


# ---------------------------------------------------------------- models
def presence(X):
    """Counts -> presence/absence (module-level so that it can be pickled for parallel runs)."""
    return np.asarray(X) > 0


@dataclass
class ModelSpec:
    key: str
    label: str
    estimator: object
    role: str           # "null", "baseline", "main", "robustness"


def make_models(n_trees=None, seed=SEED):
    """All classifiers of the benchmark, from the null reference to the most flexible model."""
    n_trees = n_trees or SETTINGS["n_trees"]
    specs = [
        ModelSpec("null_prior", "Null (majority class)",
                  DummyClassifier(strategy="prior"), "null"),
        ModelSpec("logistic", "Logistic regression (L2)",
                  make_pipeline(StandardScaler(),
                                LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)),
                  "baseline"),
        ModelSpec("knn", "k-NN (Jaccard, k = 5)",
                  make_pipeline(FunctionTransformer(presence),
                                KNeighborsClassifier(n_neighbors=5, metric="jaccard")),
                  "baseline"),
        ModelSpec("rf_default", "Random forest (unweighted)",
                  RandomForestClassifier(n_estimators=n_trees, max_features="sqrt",
                                         random_state=seed, n_jobs=1),
                  "baseline"),
        ModelSpec("rf_weighted", "Random forest (class-weighted)",
                  RandomForestClassifier(n_estimators=n_trees, max_features="sqrt",
                                         class_weight="balanced_subsample", random_state=seed,
                                         n_jobs=1),
                  "main"),
        ModelSpec("boosting", "Gradient boosting (depth 2)",
                  HistGradientBoostingClassifier(max_depth=2, learning_rate=0.05, max_iter=200,
                                                 min_samples_leaf=4, class_weight="balanced",
                                                 random_state=seed),
                  "robustness"),
    ]
    if BalancedRandomForestClassifier is not None:
        specs.insert(5, ModelSpec(
            "brf", "Balanced random forest",
            BalancedRandomForestClassifier(n_estimators=n_trees, max_features="sqrt",
                                           sampling_strategy="all", replacement=True,
                                           bootstrap=False, random_state=seed, n_jobs=1),
            "main"))
    return {s.key: s for s in specs}


def main_model(n_trees=None, seed=SEED):
    """The model used for the homogenisation test and the interpretation (Balanced RF if available)."""
    m = make_models(n_trees, seed)
    return m["brf"].estimator if "brf" in m else m["rf_weighted"].estimator


# ---------------------------------------------------------------- repeated out-of-fold predictions
def stratified_splits(y, n_splits, n_repeats, seed=SEED):
    """List (one item per repetition) of lists of (train, test) index pairs."""
    y = np.asarray(y)
    return [list(StratifiedKFold(n_splits, shuffle=True, random_state=seed + r).split(y, y))
            for r in range(n_repeats)]


def _align_proba(model, proba, classes):
    out = np.zeros((proba.shape[0], len(classes)))
    mc = list(model.classes_)
    for j, c in enumerate(classes):
        if c in mc:
            out[:, j] = proba[:, mc.index(c)]
    return out


def _oof_one(model, X, y, splits, classes):
    P = np.full((len(y), len(classes)), np.nan)
    yhat = np.empty(len(y), dtype=object)
    for tr, te in splits:
        m = clone(model).fit(X[tr], y[tr])
        P[te] = _align_proba(m, m.predict_proba(X[te]), classes)
        yhat[te] = m.predict(X[te])
    return P, yhat


def repeated_oof(model, X, y, n_repeats=None, n_splits=None, seed=SEED, classes=None,
                 splits=None, n_jobs=None):
    """Out-of-fold class probabilities and predictions for each CV repetition.

    Returns P (repeats x points x classes) and Yhat (repeats x points).
    `splits` (from stratified_splits or spatial_block_splits) overrides the random splits.
    """
    X, y = np.asarray(X, float), np.asarray(y)
    classes = classes or [c for c in _class_order(y)]
    if splits is None:
        splits = stratified_splits(y, n_splits or SETTINGS["n_splits"],
                                   n_repeats or SETTINGS["n_repeats"], seed)
    res = Parallel(n_jobs=N_JOBS if n_jobs is None else n_jobs)(
        delayed(_oof_one)(model, X, y, s, classes) for s in splits)
    P = np.stack([r[0] for r in res])
    Yhat = np.stack([r[1] for r in res])
    return P, Yhat, classes


def _class_order(y):
    from wasps import LANDSCAPES
    present = list(pd.unique(np.asarray(y)))
    return [c for c in LANDSCAPES if c in present] + sorted(c for c in present if c not in LANDSCAPES)


def spatial_block_splits(meta, n_blocks=3, n_repeats=10, seed=SEED):
    """Spatially blocked CV inside each landscape (Roberts et al. 2017).

    In every landscape the points are grouped into `n_blocks` spatial clusters (k-means on the
    coordinates). Fold b holds out cluster b of every landscape, so a test point never has its
    nearest neighbours in the training set while all landscapes stay represented. Repetitions
    change the k-means initialisation and the random pairing of clusters across landscapes.
    """
    from sklearn.cluster import KMeans
    lat, lon = meta["lat"].to_numpy(), meta["lon"].to_numpy()
    xy = np.column_stack([lat * 111.0, lon * 111.0 * np.cos(np.deg2rad(lat))])   # ~km
    land = meta["landscape"].to_numpy()
    out = []
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        fold = np.empty(len(meta), int)
        for g in np.unique(land):
            idx = np.where(land == g)[0]
            k = min(n_blocks, len(idx))
            lab = KMeans(k, n_init=10, random_state=seed + r).fit_predict(xy[idx])
            fold[idx] = rng.permutation(k)[lab]          # random pairing of clusters to folds
        out.append([(np.where(fold != b)[0], np.where(fold == b)[0]) for b in range(n_blocks)])
    return out


# ---------------------------------------------------------------- metrics
def oof_metrics(y, yhat, P, classes, eps=1e-6):
    """Metrics of one repetition (all points predicted once)."""
    y = np.asarray(y)
    yhat = np.asarray(yhat).astype(str)
    Pc = np.clip(P, eps, 1)
    Pc = Pc / Pc.sum(1, keepdims=True)
    idx = np.array([classes.index(c) for c in y])
    out = {
        "accuracy": accuracy_score(y, yhat),
        "balanced_accuracy": balanced_accuracy_score(y, yhat),
        "f1_macro": f1_score(y, yhat, average="macro", labels=classes, zero_division=0),
        "kappa": cohen_kappa_score(y, yhat, labels=classes),
        "log_loss": float(-np.log(Pc[np.arange(len(y)), idx]).mean()),
    }
    cm = confusion_matrix(y, yhat, labels=classes)
    for i, c in enumerate(classes):
        out[f"recall_{c}"] = cm[i, i] / cm[i].sum()
    if "pantanal" in classes:
        i = classes.index("pantanal")
        for c in classes:
            if c != "pantanal":
                out[f"pantanal_to_{c}"] = cm[i, classes.index(c)] / cm[i].sum()
    return out


def metrics_table(y, P, Yhat, classes):
    """One row of metrics per repetition."""
    return pd.DataFrame([oof_metrics(y, Yhat[r], P[r], classes) for r in range(len(P))])


def mean_confusion(y, Yhat, classes):
    """Confusion matrix (counts) averaged over repetitions; rows = observed, columns = predicted."""
    return np.mean([confusion_matrix(y, Yhat[r].astype(str), labels=classes)
                    for r in range(len(Yhat))], axis=0)


def interval(x, level=0.95):
    """Mean and central interval of a vector (CV repetitions or bootstrap resamples)."""
    x = np.asarray(x, float)
    a = (1 - level) / 2
    return float(np.mean(x)), float(np.quantile(x, a)), float(np.quantile(x, 1 - a))


# ---------------------------------------------------------------- permutation test
def _perm_stat(model, X, y, n_splits, seed, classes):
    splits = stratified_splits(y, n_splits, 1, seed)[0]
    _, yhat = _oof_one(model, X, y, splits, classes)
    return balanced_accuracy_score(y, yhat.astype(str))


def permutation_test(model, X, y, n_perm, n_splits=None, seed=SEED, n_jobs=None):
    """Label-permutation test of the cross-validated balanced accuracy (Ojala & Garriga 2010).

    The observed statistic and every null statistic use the same CV procedure (one 5-fold split).
    Returns observed score, null scores and p = (1 + #null >= observed) / (1 + n_perm).
    """
    X, y = np.asarray(X, float), np.asarray(y)
    n_splits = n_splits or SETTINGS["n_splits"]
    classes = _class_order(y)
    obs = _perm_stat(model, X, y, n_splits, seed, classes)
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(y) for _ in range(n_perm)]
    null = np.array(Parallel(n_jobs=N_JOBS if n_jobs is None else n_jobs)(
        delayed(_perm_stat)(model, X, yp, n_splits, seed + 1 + k, classes)
        for k, yp in enumerate(perms)))
    p = (1 + np.sum(null >= obs - 1e-12)) / (1 + n_perm)
    return {"observed": obs, "null": null, "p": p, "n_perm": n_perm}


# ---------------------------------------------------------------- pairwise separability (H2)
def auc_mw(pos, neg):
    """AUC as the Mann-Whitney probability P(score_pos > score_neg) (ties count 1/2)."""
    pos, neg = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((pos > neg) + 0.5 * (pos == neg)).mean())


def pairwise_scores(model, X, y, positive, negative, n_repeats=None, seed=SEED, split_fn=None):
    """Binary classifier positive-vs-negative using only those two landscapes.

    Returns the OOF probability of `positive` for each point of the pair (mean over repetitions),
    the per-repetition AUCs and the indices of the points in the full data.
    `split_fn(idx)` may return custom CV splits for the subset `idx` (e.g. spatial blocks).
    """
    X, y = np.asarray(X, float), np.asarray(y)
    idx = np.where(np.isin(y, [positive, negative]))[0]
    yp = y[idx]
    P, _, classes = repeated_oof(model, X[idx], yp, n_repeats or SETTINGS["n_rep_pairwise"],
                                 seed=seed, classes=[negative, positive],
                                 splits=split_fn(idx) if split_fn else None)
    score = P[:, :, 1]
    is_pos = yp == positive
    aucs = np.array([auc_mw(s[is_pos], s[~is_pos]) for s in score])
    return {"idx": idx, "is_pos": is_pos, "score": score.mean(0), "auc_reps": aucs,
            "auc": auc_mw(score.mean(0)[is_pos], score.mean(0)[~is_pos])}


def paired_auc_bootstrap(pair_a, pair_b, shared="pantanal", y=None, n_boot=None, seed=SEED):
    """Paired bootstrap of AUC_a - AUC_b when both pairs share the positive class (Pantanal).

    The Pantanal points are resampled once and used in both AUCs, so the difference accounts for
    the shared sample. Returns bootstrap AUCs for each pair and their difference.
    """
    n_boot = n_boot or SETTINGS["n_boot"]
    rng = np.random.default_rng(seed)
    pos_a = pair_a["idx"][pair_a["is_pos"]]
    pos_b = pair_b["idx"][pair_b["is_pos"]]
    assert np.array_equal(np.sort(pos_a), np.sort(pos_b)), "pairs must share the positive points"
    sa = dict(zip(pair_a["idx"], pair_a["score"]))
    sb = dict(zip(pair_b["idx"], pair_b["score"]))
    neg_a = pair_a["idx"][~pair_a["is_pos"]]
    neg_b = pair_b["idx"][~pair_b["is_pos"]]
    A, B = np.empty(n_boot), np.empty(n_boot)
    for k in range(n_boot):
        p = rng.choice(pos_a, len(pos_a))
        na = rng.choice(neg_a, len(neg_a))
        nb = rng.choice(neg_b, len(neg_b))
        A[k] = auc_mw([sa[i] for i in p], [sa[i] for i in na])
        B[k] = auc_mw([sb[i] for i in p], [sb[i] for i in nb])
    return {"auc_a": A, "auc_b": B, "diff": A - B}


def bootstrap_auc(pair, n_boot=None, seed=SEED):
    """Stratified bootstrap of a single pairwise AUC."""
    n_boot = n_boot or SETTINGS["n_boot"]
    rng = np.random.default_rng(seed)
    s, pos = pair["score"], pair["is_pos"]
    ip, ineg = np.where(pos)[0], np.where(~pos)[0]
    return np.array([auc_mw(s[rng.choice(ip, len(ip))], s[rng.choice(ineg, len(ineg))])
                     for _ in range(n_boot)])


# ---------------------------------------------------------------- interpretation
def _importance_one(model, X, y, splits, n_feature_perm, seed, classes):
    """Permutation importance from one CV repetition, computed on held-out points only."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    base = np.empty(n, dtype=object)
    perm = np.empty((p, n_feature_perm, n), dtype=object)
    for tr, te in splits:
        m = clone(model).fit(X[tr], y[tr])
        base[te] = m.predict(X[te])
        Xte = X[te]
        blocks = []
        for j in range(p):
            for k in range(n_feature_perm):
                Z = Xte.copy()
                Z[:, j] = rng.permutation(Z[:, j])
                blocks.append(Z)
        pred = m.predict(np.vstack(blocks)).reshape(p, n_feature_perm, len(te))
        perm[:, :, te] = pred
    s0 = balanced_accuracy_score(y, base.astype(str))
    imp = np.array([np.mean([s0 - balanced_accuracy_score(y, perm[j, k].astype(str))
                             for k in range(n_feature_perm)]) for j in range(p)])
    return imp


def oof_permutation_importance(model, X, y, n_repeats=None, n_feature_perm=None, seed=SEED):
    """Drop in OOF balanced accuracy when one species is shuffled among held-out points.

    Returns an array (repetitions x species).
    """
    X, y = np.asarray(X, float), np.asarray(y)
    n_repeats = n_repeats or SETTINGS["n_rep_importance"]
    n_feature_perm = n_feature_perm or SETTINGS["n_feature_perm"]
    classes = _class_order(y)
    splits = stratified_splits(y, SETTINGS["n_splits"], n_repeats, seed)
    return np.array(Parallel(n_jobs=N_JOBS)(
        delayed(_importance_one)(model, X, y, s, n_feature_perm, seed + r, classes)
        for r, s in enumerate(splits)))


def importance_null(model, X, y, n_perm=None, n_feature_perm=None, seed=SEED):
    """Altmann et al. (2010): importances recomputed after permuting the landscape labels.

    One CV repetition per permutation; returns an array (permutations x species).
    """
    X, y = np.asarray(X, float), np.asarray(y)
    n_perm = n_perm or SETTINGS["n_perm_importance"]
    n_feature_perm = max(3, (n_feature_perm or SETTINGS["n_feature_perm"]) // 2)
    rng = np.random.default_rng(seed + 999)
    classes = _class_order(y)
    jobs = []
    for k in range(n_perm):
        yp = rng.permutation(y)
        s = stratified_splits(yp, SETTINGS["n_splits"], 1, seed + 1000 + k)[0]
        jobs.append(delayed(_importance_one)(model, X, yp, s, n_feature_perm, seed + k, classes))
    return np.array(Parallel(n_jobs=N_JOBS)(jobs))


def partial_dependence(model, X, values=(0, 1, 2)):
    """Partial dependence of every class probability on each species (Friedman 2001).

    For species j and value v, set x_j = v in every point and average the predicted probabilities.
    Returns a DataFrame with columns species, value and one column per class.
    """
    X = np.asarray(X, float)
    rows = []
    for j in range(X.shape[1]):
        for v in values:
            Z = X.copy()
            Z[:, j] = v
            proba = model.predict_proba(Z).mean(0)
            rows.append({"j": j, "value": v, **dict(zip(model.classes_, proba))})
    return pd.DataFrame(rows)
