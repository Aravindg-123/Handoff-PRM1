"""
common/cv_utils.py
==================
Write once, import everywhere.

Shared cross-validation primitives for handoff-prmv2:

    task_folds          -- assign every TASK to one fold (all rows stay together)
    oof_scores          -- task-grouped out-of-fold predictions
    task_bootstrap_ci   -- percentile CI by resampling TASKS (not rows)

Design rules
------------
* No side effects (no I/O, no prints).
* All randomness goes through explicit seeds.
* The ``drop_from_train`` mechanism supports held-out-type experiments
  without changing the fold structure.
* CI is reported over TASKS so that the unit of independence matches the
  unit of data collection (one task can contribute many rows).

Usage
-----
>>> from common.cv_utils import task_folds, oof_scores, task_bootstrap_ci
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.metrics import roc_auc_score
>>>
>>> fold = task_folds(df)
>>> scores = oof_scores(df, FEAT_COLS, LogisticRegression)
>>> ci, n_valid = task_bootstrap_ci(df, scores, roc_auc_score)
>>> print(f"AUROC 95% CI: {ci[0]:.3f} – {ci[1]:.3f}  (n_boot={n_valid})")
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score   # noqa: F401 – re-exported for convenience


# ---------------------------------------------------------------------------
# 1.  Task fold assignment
# ---------------------------------------------------------------------------

def task_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Return a fold index array (0 … n_splits-1) aligned to ``df``.

    Every row that belongs to the same ``task_id`` receives the **same**
    fold index, so there is never a task split across train and validation.

    Parameters
    ----------
    df:
        DataFrame that must contain a ``task_id`` column.
    n_splits:
        Number of folds (default: 5).
    seed:
        Random seed that controls the task-to-fold shuffle.

    Returns
    -------
    np.ndarray of shape ``(len(df),)`` with dtype int.
    """
    tasks = np.unique(df["task_id"].values)
    rng = np.random.RandomState(seed)
    rng.shuffle(tasks)
    fold_of = {t: i % n_splits for i, t in enumerate(tasks)}
    return df["task_id"].map(fold_of).values


# ---------------------------------------------------------------------------
# 2.  Out-of-fold scorer
# ---------------------------------------------------------------------------

def oof_scores(
    df: pd.DataFrame,
    feats: list[str],
    make_model,
    *,
    n_splits: int = 5,
    seed: int = 0,
    drop_from_train: np.ndarray | None = None,
) -> np.ndarray:
    """Task-grouped out-of-fold predicted probabilities.

    Parameters
    ----------
    df:
        Feature DataFrame. Must contain columns ``task_id`` and ``label``.
    feats:
        List of column names used as model inputs.
    make_model:
        **Zero-argument callable** that returns a fresh, unfitted
        sklearn-compatible estimator (must expose ``fit`` and
        ``predict_proba``).  Example::

            from xgboost import XGBClassifier
            make_model = lambda: XGBClassifier(n_estimators=50, random_state=42)

    n_splits:
        Number of CV folds (default: 5).
    seed:
        Passed to :func:`task_folds` to control the task shuffle.
    drop_from_train:
        Optional boolean array of shape ``(len(df),)``.  Rows where this
        is ``True`` are excluded from **training only** – they still
        receive a prediction in the fold where they are in the test set.
        Useful for leave-corruption-out experiments.

    Returns
    -------
    np.ndarray of shape ``(len(df),)`` with float OOF scores.
    All positions are filled (``np.nan``-free).

    Raises
    ------
    AssertionError:
        If any task appears in both the train and test sets of a fold
        ("task leakage"), or if the returned array contains NaN values.
    """
    required = ["task_id", "label", *feats]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"oof_scores: missing required columns: {missing}")
    if len(df) == 0:
        raise ValueError("oof_scores: cannot score an empty DataFrame.")
    if not df["task_id"].notna().all():
        raise ValueError("oof_scores: task_id contains missing values.")
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("oof_scores: labels must be encoded as 0 and 1.")

    fold = task_folds(df, n_splits, seed)
    out = np.full(len(df), np.nan)
    g = df["task_id"].values
    if len(fold) != len(df) or len(g) != len(df):
        raise ValueError("oof_scores: fold assignments and task groups are misaligned.")

    for k in range(n_splits):
        te = fold == k
        tr = fold != k
        if drop_from_train is not None:
            if len(drop_from_train) != len(df):
                raise ValueError(
                    "oof_scores: drop_from_train must have one value per DataFrame row."
                )
            tr = tr & ~drop_from_train

        # Sanity check: zero task-id overlap between train and test
        overlap = set(g[tr]) & set(g[te])
        if overlap:
            raise RuntimeError(f"Task leakage detected in fold {k}: {overlap}")

        m = make_model()
        x_train = df.loc[tr, feats]
        y_train = df.loc[tr, "label"]
        x_test = df.loc[te, feats]
        if len(x_train) != int(tr.sum()) or len(x_test) != int(te.sum()):
            raise RuntimeError(f"oof_scores: positional row selection failed in fold {k}.")
        m.fit(x_train, y_train)
        classes = np.asarray(m.classes_)
        positive_class = np.flatnonzero(classes == 1)
        if len(positive_class) != 1:
            raise ValueError(
                f"oof_scores: fold {k} estimator classes are {classes.tolist()}, "
                "expected exactly one positive class 1."
            )
        probabilities = m.predict_proba(x_test)
        if probabilities.shape != (int(te.sum()), len(classes)):
            raise RuntimeError(
                f"oof_scores: fold {k} predict_proba shape {probabilities.shape} "
                f"does not match {int(te.sum())} test rows and {len(classes)} classes."
            )
        out[te] = probabilities[:, positive_class[0]]

    if np.isnan(out).any():
        raise RuntimeError(
            "oof_scores: NaN values remain after all folds. "
            "Check that every task appears in at least one test fold."
        )
    return out


# ---------------------------------------------------------------------------
# 3.  Task-level bootstrap confidence interval
# ---------------------------------------------------------------------------

def task_bootstrap_ci(
    df: pd.DataFrame,
    scores: np.ndarray,
    metric,
    *,
    mask: np.ndarray | None = None,
    n_boot: int = 10_000,
    seed: int = 42,
) -> tuple[np.ndarray, int]:
    """Percentile CI from resampling **tasks** (not rows) on fixed OOF scores.

    Parameters
    ----------
    df:
        Source DataFrame with a ``task_id`` and ``label`` column.
    scores:
        OOF predicted probabilities aligned to ``df`` (e.g., from
        :func:`oof_scores`).
    metric:
        Callable ``metric(y_true, y_score) -> float``.  Must accept 1-D
        numpy arrays.  E.g., ``sklearn.metrics.roc_auc_score``.
    mask:
        Optional boolean array of shape ``(len(df),)``.  When given, only
        the rows where ``mask`` is ``True`` are considered (applied before
        grouping by task).
    n_boot:
        Number of bootstrap replicates (default: 10 000).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    ci : np.ndarray of shape (2,)
        ``[2.5th-percentile, 97.5th-percentile]`` of the bootstrap
        distribution.
    n_valid : int
        Number of bootstrap replicates that produced a valid metric value
        (i.e., the bootstrap resample had both label classes present).
        Report this alongside the CI so readers can judge coverage.

    Notes
    -----
    Resampling is done at the **task** level (with replacement), which is
    the correct unit because rows within a task are not independent.
    """
    d = df.assign(_s=scores)
    if mask is not None:
        d = d[mask]

    groups = {t: g for t, g in d.groupby("task_id")}
    tasks = np.array(list(groups))
    rng = np.random.RandomState(seed)
    vals: list[float] = []

    for _ in range(n_boot):
        sampled = pd.concat([groups[t] for t in rng.choice(tasks, len(tasks))])
        if sampled["label"].nunique() == 2:
            vals.append(metric(sampled["label"].values, sampled["_s"].values))

    ci = np.percentile(vals, [2.5, 97.5])
    return ci, len(vals)
