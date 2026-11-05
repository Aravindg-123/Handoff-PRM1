import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from common.cv_utils import task_folds, oof_scores, task_bootstrap_ci
from common.stats_utils import (auc, task_bootstrap, paired_bootstrap_diff, ece_equal_mass,
                                block_metrics, threshold_fbr_cap)
from sklearn.linear_model import LogisticRegression


def _df(n=60, seed=0):
    r = np.random.RandomState(seed)
    t = np.repeat([f"t{i}" for i in range(n // 2)], 2)
    return pd.DataFrame({"task_id": t, "label": [1, 0] * (n // 2), "x": r.randn(n)})


def test_folds_never_split_a_task():
    d = _df(); f = task_folds(d)
    assert all(d.task_id[f == k].isin(d.task_id[f != k]).sum() == 0 for k in range(5))


def test_oof_runs_and_leak_assert_holds():
    d = _df(); d["x"] += d.label * 2
    s = oof_scores(d, ["x"], LogisticRegression)
    assert roc_auc_score(d.label, s) > 0.8


def test_fast_auc_matches_sklearn():
    d = _df(200, 1); s = np.random.RandomState(2).rand(200)
    assert abs(auc(d.label, s) - roc_auc_score(d.label, s)) < 1e-12


def test_bootstrap_ci_brackets_point_and_agrees_between_impls():
    d = _df(120, 3); d["x"] += d.label * 1.0
    lo, hi, n, pt = task_bootstrap(d, d.x.values, roc_auc_score, n_boot=500)
    ci, n2 = task_bootstrap_ci(d, d.x.values, roc_auc_score, n_boot=500)
    assert lo <= pt <= hi and n == n2 and np.allclose([lo, hi], ci)


def test_paired_diff_of_identical_scores_is_zero():
    d = _df(120, 4); s = d.x.values
    lo, hi, _, pt = paired_bootstrap_diff(d, s, s, roc_auc_score, n_boot=200)
    assert lo == hi == pt == 0


def test_ece_perfect_and_threshold_rule():
    y = np.array([0, 1] * 50); assert ece_equal_mass(y, y.astype(float)) == 0
    s = np.r_[np.linspace(.5, 1, 100), np.linspace(0, .6, 100)]; yy = np.r_[np.ones(100), np.zeros(100)]
    t = threshold_fbr_cap(yy, s, 0.05)
    assert block_metrics(yy, s, t)["fbr"] <= 0.05
