"""Nested task-grouped calibration. For each outer fold the calibrator is fit on
INNER task-grouped OOF scores of the training tasks only (CalibratedClassifierCV's
default CV ignores groups, so this is done by hand)."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from common.cv_utils import oof_scores, task_folds

_logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


class Platt:
    def fit(self, s, y):
        self.m = LogisticRegression(C=1e6).fit(_logit(s)[:, None], y); return self
    def predict(self, s):
        return self.m.predict_proba(_logit(s)[:, None])[:, 1]


class Iso:
    def fit(self, s, y):
        self.m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s, y); return self
    def predict(self, s):
        return self.m.predict(s)


def nested_scores(df, feats, make_model, seed=0, n_splits=5):
    """Returns dict: raw, iso, platt (outer OOF arrays), fold, inner (list of
    (train_row_positions, inner_oof_raw_scores) per outer fold)."""
    fold = task_folds(df, n_splits, seed)
    raw, iso, pl = (np.full(len(df), np.nan) for _ in range(3)); inner = []
    for k in range(n_splits):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        dtr = df.iloc[tr].reset_index(drop=True)
        s_in = oof_scores(dtr, feats, make_model, seed=1000 + 10 * seed + k)
        inner.append((tr, s_in))
        m = make_model().fit(df.iloc[tr][feats], df.iloc[tr]["label"])
        s_te = m.predict_proba(df.iloc[te][feats])[:, list(m.classes_).index(1)]
        raw[te] = s_te
        iso[te] = Iso().fit(s_in, dtr.label.values).predict(s_te)
        pl[te] = Platt().fit(s_in, dtr.label.values).predict(s_te)
    assert not np.isnan(raw).any()
    return dict(raw=raw, iso=iso, platt=pl, fold=fold, inner=inner)


def prior_shift(p, pi_dev, pi_deploy):
    """Shift P(label=1) from dev prevalence of label 1 to deployment prevalence."""
    z = _logit(p) + np.log(pi_deploy / (1 - pi_deploy)) - np.log(pi_dev / (1 - pi_dev))
    return 1 / (1 + np.exp(-z))
