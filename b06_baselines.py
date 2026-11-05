"""B6 (1-3): length-threshold, exact-signature, embedding-LR baselines on the same task-grouped folds/metrics.
Operating point for continuous scorers: B3 rule (largest t with clean false-block rate <= 5%) on the
method's own pooled OOF scores (optimistic by construction, the same caveat as B3's headline)."""
import re
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from common.cv_utils import oof_scores
from common.data import load_v2, load_raw, RES
from common.stats_utils import auc, task_bootstrap, block_metrics, threshold_fbr_cap

d, feats = load_v2(); y = d.label.values
raw = load_raw()
ch = pd.read_csv("corrupted_handoffs.csv"); ep = ch.drop_duplicates("task_id").set_index("task_id").entry_point
out = {}

# 1. length-only: depth-2 tree on length_ratio (learns the dev thresholds; handles both too-short and too-long)
out["length threshold (tree on length_ratio)"] = np.mean(
    [oof_scores(d, ["length_ratio"], lambda: DecisionTreeClassifier(max_depth=2, min_samples_leaf=10, random_state=0), seed=s)
     for s in range(5)], axis=0)

# 2. exact signature match: is `def <entry_point>(` verbatim in the handoff?  (binary)
out["exact signature match (binary)"] = np.array(
    [float(bool(re.search(rf"def\s+{re.escape(ep[t])}\s*\(", h))) for t, h in zip(raw.task_id, raw.handoff)])

# 3. embedding-only logistic regression on [e(h), e(t), |e(h)-e(t)|, e(h)*e(t)]
from sentence_transformers import SentenceTransformer
st = SentenceTransformer("all-MiniLM-L6-v2")
eh = st.encode(raw.handoff.tolist(), normalize_embeddings=True, batch_size=64)
et = st.encode(raw.problem.tolist(), normalize_embeddings=True, batch_size=64)
E = np.hstack([eh, et, np.abs(eh - et), eh * et])
cols = [f"e{i}" for i in range(E.shape[1])]
de = pd.concat([d[["task_id", "label"]].reset_index(drop=True), pd.DataFrame(E, columns=cols)], axis=1)
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced"))
out["MiniLM embeddings + LR"] = np.mean([oof_scores(de, cols, mk, seed=s) for s in range(5)], axis=0)
np.save(RES / "b06_embedding_lr_oof.npy", out["MiniLM embeddings + LR"])
pd.DataFrame(out).to_csv(RES / "b06_baseline_oof_scores.csv", index=False)

rows = []
for k, s in out.items():
    lo, hi, nv, pt = task_bootstrap(d, s, auc, n_boot=5000)
    t = 0.5 if s.max() == 1 and set(np.unique(s)) <= {0.0, 1.0} else threshold_fbr_cap(y, s, 0.05)
    m = block_metrics(y, s, t)
    rows.append(dict(method=k, auroc=pt, ci=f"[{lo:.3f}, {hi:.3f}]", detection=m["tpr"], false_block=m["fbr"], op_threshold=round(t, 4)))
res = pd.DataFrame(rows).round(4); res.to_csv(RES / "b06_baselines_1to3.csv", index=False); print(res.to_string())
