"""B6 (6): end-to-end latency of the deployed path at batch size 1, warm models, CPU only:
spaCy NER/noun-chunks + MiniLM embedding + 11 feature functions + RF predict_proba.
Run with the GPU hidden so the number is a CPU number."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import time
import numpy as np, pandas as pd
import feature_extraction as fx
from common.data import load_v2, load_raw, make_rf, RES

d, feats = load_v2(); raw = load_raw()
ch = pd.read_csv("corrupted_handoffs.csv"); ep = ch.drop_duplicates("task_id").set_index("task_id").entry_point
rf = make_rf().fit(d[feats], d.label)


def one(h, p, e):
    f = {"cosine_similarity": fx.compute_cosine_similarity(h, p),
         "entity_overlap": fx.compute_entity_overlap(h, p),
         "length_ratio": fx.compute_length_ratio(h, p),
         "sentence_count_ratio": fx.compute_sentence_count_ratio(h, p),
         "section_coverage": fx.compute_section_coverage(h, p, e),
         "verbatim_copy_rate": fx.compute_verbatim_copy_rate(h, p),
         "trailing_specificity": fx.compute_trailing_specificity(h, p),
         "constraint_count": fx.compute_constraint_count(h, p),
         "role_pronoun_rate": fx.compute_role_pronoun_rate(h, p),
         "novel_api_rate": fx.compute_novel_api_rate(h, p),
         "function_name_preserved": fx.compute_function_name_preserved(h, p, e)}
    return rf.predict_proba(pd.DataFrame([f])[feats])[0, 1]


rf.set_params(clf__n_jobs=1)                      # single-example scoring: no thread fan-out
for i in range(10): one(raw.handoff[i], raw.problem[i], ep[raw.task_id[i]])      # warm-up
T = []
for h, p, t in zip(raw.handoff, raw.problem, raw.task_id):
    a = time.perf_counter(); one(h, p, ep[t]); T.append((time.perf_counter() - a) * 1e3)
T = np.array(T)
# classifier alone, for the old "predict time only" ratio
X1 = d[feats].iloc[[0]]; C = []
for _ in range(500):
    a = time.perf_counter(); rf.predict_proba(X1); C.append((time.perf_counter() - a) * 1e3)
res = dict(n=len(T), e2e_ms_p50=np.percentile(T, 50), e2e_ms_p95=np.percentile(T, 95), e2e_ms_mean=T.mean(),
           classifier_only_ms_p50=np.percentile(C, 50),
           ratio_vs_3933ms_judge_e2e=3933 / np.percentile(T, 50), ratio_vs_3933ms_judge_clf_only=3933 / np.percentile(C, 50))
pd.Series(res).round(3).to_csv(RES / "b06_latency.csv", header=["value"]); print(pd.Series(res).round(3))
