"""Optional 10-second check: leave-one-corruption-type-out with TASK-GROUPED splits
(the held-out type's tasks, including their clean siblings, never appear in training).
Results are printed rounded to 2 d.p. to match paper table values.
Usage: python grouped_loto_check.py path/to/final_dataset_full_features.csv
"""
import sys, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

f = pd.read_csv(sys.argv[1], encoding_errors="replace")

feats = [
    "cosine_similarity", "entity_overlap", "length_ratio",
    "sentence_count_ratio", "section_coverage", "verbatim_copy_rate",
    "trailing_specificity", "constraint_count", "role_pronoun_rate",
    "novel_api_rate", "function_name_preserved",
]

mk = {
    "LR": lambda: Pipeline([("s", StandardScaler()),
                            ("m", LogisticRegression(max_iter=1000))]),
    "RF": lambda: RandomForestClassifier(n_estimators=300,
                                         min_samples_leaf=2,
                                         random_state=42),
}

for c in sorted(set(f.corruption_type) - {"none"}):
    T  = set(f[f.corruption_type == c].task_id)
    te = f[f.task_id.isin(T) & f.corruption_type.isin([c, "none"])]
    tr = f[~f.task_id.isin(T) & (f.corruption_type != c)]
    out = {
        k: round(roc_auc_score(
            te.label,
            m().fit(tr[feats], tr.label).predict_proba(te[feats])[:, 1],
        ), 2)
        for k, m in mk.items()
    }
    print(f"{c:26s} n_neg={int((te.label == 0).sum()):3d}  {out}")
