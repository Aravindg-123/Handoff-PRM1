"""Statistics helpers for workstream B (no I/O, no heavy models)."""
import numpy as np
from scipy.stats import binomtest, rankdata


def auc(y, s):
    """Fast AUROC (rank formula, ties averaged)."""
    y = np.asarray(y)
    n1 = int((y == 1).sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _groups(df, mask=None):
    d = df if mask is None else df[mask]
    return d, [np.asarray(ix) for ix in d.groupby("task_id").indices.values()]


def task_bootstrap(df, scores, metric, mask=None, n_boot=10_000, seed=42):
    """Task-resampled percentile CI. Returns (lo, hi, n_valid, point)."""
    d, idx = _groups(df.assign(_s=scores), mask)
    y, s = d["label"].values, d["_s"].values
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        p = np.concatenate([idx[i] for i in rng.choice(len(idx), len(idx))])
        if y[p].min() != y[p].max():
            vals.append(metric(y[p], s[p]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi, len(vals), metric(y, s)


def paired_bootstrap_diff(df, s_a, s_b, metric, mask=None, n_boot=10_000, seed=42):
    """CI for metric(s_a) - metric(s_b) on the SAME resampled tasks.
    Returns (lo, hi, n_valid, point_diff)."""
    d, idx = _groups(df.assign(_a=s_a, _b=s_b), mask)
    y, a, b = d["label"].values, d["_a"].values, d["_b"].values
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        p = np.concatenate([idx[i] for i in rng.choice(len(idx), len(idx))])
        if y[p].min() != y[p].max():
            vals.append(metric(y[p], a[p]) - metric(y[p], b[p]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi, len(vals), metric(y, a) - metric(y, b)


def ece_equal_mass(y, p, n_bins=10):
    o = np.argsort(p); y, p = np.asarray(y)[o], np.asarray(p)[o]
    bins = np.array_split(np.arange(len(p)), n_bins)
    return sum(len(b) / len(p) * abs(y[b].mean() - p[b].mean()) for b in bins)


def clopper_pearson(k, n, alpha=0.05):
    if n == 0:
        return (np.nan, np.nan)
    ci = binomtest(int(k), int(n)).proportion_ci(1 - alpha, method="exact")
    return ci.low, ci.high


def block_metrics(y, s, t):
    """Gate blocks when score < t.  label 1 = good handoff, 0 = bad.
    Returns TP (bad blocked), FN, FP (good blocked), TN, tpr, fbr, precision, f1."""
    y = np.asarray(y); blk = np.asarray(s) < t
    tp = int((blk & (y == 0)).sum()); fn = int((~blk & (y == 0)).sum())
    fp = int((blk & (y == 1)).sum()); tn = int((~blk & (y == 1)).sum())
    tpr = tp / max(tp + fn, 1); fbr = fp / max(fp + tn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * tpr / (prec + tpr) if prec + tpr else 0.0
    return dict(TP=tp, FN=fn, FP=fp, TN=tn, tpr=tpr, fbr=fbr, precision=prec, f1=f1)


def threshold_fbr_cap(y, s, cap=0.05):
    """Rule: largest t whose clean false-block rate <= cap (max detection s.t. cap).
    Block when score < t."""
    good = np.sort(np.asarray(s)[np.asarray(y) == 1])
    k = int(np.floor(cap * len(good)))          # at most k good rows may be blocked
    # blocking exactly the k lowest good scores => t just above good[k-1], at/below good[k]
    return float(good[k]) if k < len(good) else float(good[-1]) + 1e-9


def block_precision(pi, tpr, fbr):
    return pi * tpr / (pi * tpr + (1 - pi) * fbr)
