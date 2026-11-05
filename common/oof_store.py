"""
common/oof_store.py
===================
Single-writer, multi-reader OOF prediction store.

All scripts that generate out-of-fold scores write one block of rows to
``oof_predictions_v2.csv`` via :func:`append_oof`.  Downstream consumers
(Harshil's live-pilot cascade, paper-table scripts) load the file once
via :func:`load_oof` and query by model/seed – no re-training required.

Schema of ``oof_predictions_v2.csv``
--------------------------------------
row_id    : int   – original DataFrame index (preserves linkage to features.csv)
task_id   : str   – e.g. "task_042"
model     : str   – canonical model name, e.g. "XGBoost", "LogisticRegression"
cv_seed   : int   – the ``seed`` passed to task_folds / oof_scores
score     : float – predicted probability of label=1 (OOF)

Usage – writing
---------------
>>> from common.oof_store import append_oof
>>> scores = oof_scores(df, FEAT_COLS, make_xgb)          # from cv_utils
>>> append_oof(df, scores, model="XGBoost", cv_seed=0)

Usage – reading
---------------
>>> from common.oof_store import load_oof
>>> oof = load_oof()
>>> xgb_oof = oof.query("model == 'XGBoost' and cv_seed == 0")
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Canonical location, relative to the project root.
# Scripts that import this module from a subdirectory will still resolve the
# path correctly because we anchor to this file's location.
_HERE = Path(__file__).resolve().parent.parent   # project root
OOF_CSV = _HERE / "oof_predictions_v2.csv"

_COLUMNS = ["row_id", "task_id", "model", "cv_seed", "score"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_oof(path: str | Path | None = None) -> pd.DataFrame:
    """Load the canonical OOF store.

    Parameters
    ----------
    path:
        Override the default ``oof_predictions_v2.csv`` location.
        Useful for testing or for reading a one-off export.

    Returns
    -------
    DataFrame with columns: row_id, task_id, model, cv_seed, score.
    Returns an **empty** DataFrame (with the correct columns) if the file
    does not exist yet.
    """
    p = Path(path) if path is not None else OOF_CSV
    if not p.exists():
        return pd.DataFrame(columns=_COLUMNS)
    df = pd.read_csv(p, dtype={"row_id": int, "task_id": str,
                                "model": str, "cv_seed": int, "score": float})
    # Ensure column order is consistent regardless of how the file was created
    return df[_COLUMNS]


def append_oof(
    df: pd.DataFrame,
    scores: np.ndarray,
    *,
    model: str,
    cv_seed: int,
    path: str | Path | None = None,
    overwrite_existing: bool = False,
) -> pd.DataFrame:
    """Append a block of OOF predictions to the shared store.

    Parameters
    ----------
    df:
        The DataFrame that was passed to :func:`~common.cv_utils.oof_scores`.
        Must have a ``task_id`` column.  The DataFrame index is used as
        ``row_id``.
    scores:
        OOF predicted probabilities aligned to ``df``
        (output of :func:`~common.cv_utils.oof_scores`).
    model:
        Canonical model name string (e.g. ``"XGBoost"``).  Use the same
        string consistently across scripts so queries work.
    cv_seed:
        The ``seed`` value that was passed to
        :func:`~common.cv_utils.oof_scores`.
    path:
        Override the default file location.
    overwrite_existing:
        If ``True``, drop any existing rows with the same ``(model, cv_seed)``
        key before appending.  If ``False`` (default), duplicate keys are
        **not** written (idempotent re-runs are safe).

    Returns
    -------
    The updated full DataFrame (all models, all seeds).

    Raises
    ------
    ValueError:
        If ``scores`` contains NaN values.
    """
    if np.isnan(scores).any():
        raise ValueError("append_oof: ``scores`` must not contain NaN values.")

    p = Path(path) if path is not None else OOF_CSV
    existing = load_oof(p)

    # Build new block
    new_rows = pd.DataFrame({
        "row_id":  df.index.to_numpy(),
        "task_id": df["task_id"].values,
        "model":   model,
        "cv_seed": int(cv_seed),
        "score":   scores.astype(float),
    })

    if overwrite_existing and len(existing) > 0:
        mask = ~((existing["model"] == model) & (existing["cv_seed"] == cv_seed))
        existing = existing[mask]

    # Idempotency: skip if the (model, cv_seed) block already present
    if not overwrite_existing and len(existing) > 0:
        already = ((existing["model"] == model) & (existing["cv_seed"] == cv_seed)).any()
        if already:
            print(
                f"[oof_store] ({model}, seed={cv_seed}) already in {p.name} – skipping."
            )
            return existing

    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined[_COLUMNS].to_csv(p, index=False)
    print(f"[oof_store] Wrote {len(new_rows)} rows -> {p}  "
          f"(total {len(combined)} rows, {combined['model'].nunique()} models)")
    return combined[_COLUMNS]


def query_oof(
    model: str,
    cv_seed: int = 0,
    path: str | Path | None = None,
) -> pd.Series:
    """Convenience helper: return the ``score`` Series for one (model, seed).

    Parameters
    ----------
    model:
        Model name exactly as stored (e.g. ``"XGBoost"``).
    cv_seed:
        CV seed (default: 0).
    path:
        Override file location.

    Returns
    -------
    pd.Series of float scores, indexed by ``row_id``.

    Raises
    ------
    KeyError:
        If the requested ``(model, cv_seed)`` combination is not found.
    """
    df = load_oof(path)
    sub = df[(df["model"] == model) & (df["cv_seed"] == cv_seed)]
    if sub.empty:
        available = df[["model", "cv_seed"]].drop_duplicates().to_string(index=False)
        raise KeyError(
            f"No OOF scores found for model={model!r}, cv_seed={cv_seed}.\n"
            f"Available combinations:\n{available}"
        )
    return sub.set_index("row_id")["score"]
