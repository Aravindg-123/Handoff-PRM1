"""Shared dataset, feature lists and model factories for workstream B."""
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

FEAT_3 = ["cosine_similarity", "entity_overlap", "length_ratio"]
FEAT_11 = FEAT_3 + [
    "sentence_count_ratio", "section_coverage", "verbatim_copy_rate",
    "trailing_specificity", "constraint_count", "role_pronoun_rate",
    "novel_api_rate", "function_name_preserved",
]
FEAT_12 = FEAT_11 + ["nli_entailment_mean"]
RAW_CSV = ROOT / "final_dataset_full_features.csv"
V2_CSV = ROOT / "features_v2.csv"
V2_JSON = ROOT / "features_v2.json"


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_CSV)


def load_v2():
    """Frozen dev dataset + feature list (written by b08_nli_decision.py)."""
    return pd.read_csv(V2_CSV), json.loads(V2_JSON.read_text())["features"]


def make_rf():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("clf", RandomForestClassifier(
                         n_estimators=300, max_depth=6, class_weight="balanced",
                         random_state=42, n_jobs=-1))])


def make_lr():
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                random_state=42))])


def oof_multi(df, feats, make_model, seeds=range(5)):
    """Mean task-grouped OOF score over several fold seeds (less fold noise)."""
    from common.cv_utils import oof_scores
    import numpy as np
    return np.mean([oof_scores(df, feats, make_model, seed=s) for s in seeds], axis=0)
