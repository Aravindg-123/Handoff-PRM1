# Workstream B: Implementation and Results (paper-ready)

Everything below was produced by code in the `Handoff-PRM1` clone (commit `443cd45`, which added the 483-row dataset), on the rebuilt 483-row dataset (248 tasks, 248 clean / 235 corrupted). All numbers are regenerable with the commands in §12. Interval = 95% task-level bootstrap unless stated. Dev pool only; the live/holdout set was never touched.

**Status at a glance**

| Item | Status |
|---|---|
| Shared utilities + tests (`common/`) | Done, 8 new tests + 24 existing tests pass |
| B1 leave-one-type-out | Done, 20 seeds |
| B2 `function_name_preserved` | Done, cause identified |
| B8 NLI decision, `features_v2` frozen | Done: NLI dropped, 11 features |
| B4 bootstrap CIs | Done |
| B5 calibration | Done (Platt chosen) |
| B3 threshold protocol | Done, threshold frozen |
| B9 prevalence sweep | Done |
| B7 length-ratio confound | Done (matched pair + within-type) |
| B6 baselines | 1–4 and latency done. **LLM judge (item 5): script written, not run (no `GEMINI_API_KEY` in this environment).** $ per 1k handoffs therefore not measured |
| B10 organic evaluation | **Not done: blocked on Avaneesh's organic dataset (A1).** Procedure unchanged from the plan |

---

## 0. Things in the current paper that these results contradict (fix before submission)

1. **Tables XXVIII vs XXXII "unseen beats seen" was an artefact.** Cause (found in `held_out_corruption_eval.py`, the XXXII script): it trains on every row whose type ≠ X, which includes **all clean rows**, then tests on `all clean rows + type X`. So clean rows are scored **in-sample**, with no task-grouped folds. It also uses a different model (StandardScaler + LogisticRegression on every non-label column, including NLI) than XXVIII (RF, 12 features). With one code path (B1, §2) unseen is never reliably higher than seen: U − S ≤ +0.022 on every type, inside the seed spread and CIs.
2. **`function_name_preserved` numbers in the paper are stale.** Paper: label 0 mean 0.434, label 1 mean 0.996, `signature_rename` AUROC 0.45. Repo features (verified recomputed = stored, 100%): label 0 = 0.209, label 1 = 0.649, and all 14 `signature_rename` rows score 0 (see §3). The 0.45 was the substring bug (`name` ⊂ `name_impl`), already fixed in `feature_extraction.py` and regression-tested.
3. **The 0.968 RF AUROC (Table XXX) is not reproduced** by the shipped data and features: 0.918 [0.883, 0.948] (§6). The paper's value was computed with the older feature version (label-1 `function_name_preserved` ≈ 0.996). I did not verify that this is the whole cause, so report the new number.
4. **"393,000×" latency claim.** The honest end-to-end CPU latency is p50 252 ms (§9.6). The old figure compared classifier `predict` time with the judge's end-to-end time.
5. **Table XI is optimistic.** On the 483-row dev set at a 5% false-block rate the gate detects 78% of bad handoffs, not 87% (§7). Precision of blocks at 5% prevalence is 0.47, not 0.62 (§10).
6. **F1 0.9464 vs 0.9455.** From Table XI counts (TP 59, FN 9, FP 3, TN 106), F1 for the **good** class is 212/224 = 0.9464 and F1 for the **bad** class is 0.908. 0.9455 cannot be reproduced from those counts, so use 0.9464 and say which class it is for. The new §7 table reports both.

---

## 1. Implementation map

| File | Purpose |
|---|---|
| `common/cv_utils.py` | `task_folds`, `oof_scores` (leak assertion), `task_bootstrap_ci`. Bootstrap loop rewritten with index arrays (identical draws, ~50× faster) |
| `common/stats_utils.py` | fast AUROC, `task_bootstrap`, `paired_bootstrap_diff`, `ece_equal_mass`, Clopper–Pearson, `block_metrics`, `threshold_fbr_cap`, `block_precision` |
| `common/calib.py` | nested task-grouped Platt/isotonic calibration, prior shift |
| `common/data.py` | feature lists, RF/LR factories, `features_v2` loader |
| `common/oof_store.py` | existing; `oof_predictions_v2.csv` regenerated with `generate_oof_v2.py` |
| `b01_loto.py` … `b09_prevalence.py` | one script per item, each writes `results/bXX_*.csv` and a figure in `results/figures/` |
| `b06_*.py` | baselines (`baselines`, `finetune_encoder`, `latency`, `llm_judge`) |
| `features_v2.csv`, `features_v2.json` | frozen dev dataset and feature list |
| `results/frozen_threshold.json` | the frozen operating threshold (used by Harshil's C2) |
| `tests/test_stats_utils.py` | folds never split a task, leak assertion, fast AUROC = sklearn, bootstrap agrees with the old implementation, paired diff of identical scores = 0, ECE of perfect scores = 0, threshold rule respects the cap |

Model used throughout: Random Forest (300 trees, depth 6, balanced, median imputer), 11 `features_v2` features, 5 task-grouped folds. Seeds: fold seed 0 for headline OOF; 20 seeds (0–19) for B1; bootstrap seed 42, 10,000 resamples (5,000 where marked).

---

## 2. B1: Leave-one-corruption-type-out, redone

Protocol S trains on all types; Protocol U drops type X from training only. Both use `oof_scores`, the same task folds, the same model and features, clean rows scored out-of-fold, 20 fold seeds. AUROC is on `{clean, X}`. CI from task bootstrap of seed-averaged scores. Paired win rate = within task, clean row scores higher than the type-X row (protocol U), with Clopper–Pearson CI on wins/(wins+losses). **This table replaces Tables XXVIII and XXXII.**

| Type | n | AUROC-S [CI] | AUROC-U [CI] | U − S | Paired wins/ties/losses | Win rate [CI] |
|---|---|---|---|---|---|---|
| over_summarization | 63 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 | 63/0/0 | 1.00 [0.94, 1.00] |
| tool_result_drop | 65 | 0.9996 [0.999, 1.000] | 0.999 [0.999, 1.000] | −0.0002 | 65/0/0 | 1.00 [0.94, 1.00] |
| truncation | 33 | 0.994 [0.983, 1.000] | 0.977 [0.954, 0.997] | −0.017 | 33/0/0 | 1.00 [0.89, 1.00] |
| fake_completion | 11 | 0.917 [0.879, 0.969] | 0.804 [0.754, 0.882] | −0.113 | 8/0/3 | 0.73 [0.39, 0.94] |
| entity_omission | 16 | 0.885 [0.738, 0.983] | 0.875 [0.731, 0.982] | −0.009 | 14/0/2 | 0.88 [0.62, 0.98] |
| signature_rename | 14 | 0.833 [0.696, 0.948] | 0.761 [0.613, 0.889] | −0.072 | 14/0/0 | 1.00 [0.77, 1.00] |
| inject_false_constraint | 5 | 0.686 [0.530, 0.865] | 0.687 [0.518, 0.880] | +0.0003 | 3/0/2 | 0.60 [0.15, 0.95] |
| negate_edge_case | 6 | 0.552 [0.196, 0.847] | 0.574 [0.235, 0.852] | +0.022 | 2/0/4 | 0.33 [0.04, 0.78] |
| invert_objective | 14 | 0.534 [0.361, 0.703] | 0.536 [0.373, 0.699] | +0.0015 | 9/2/3 | 0.75 [0.43, 0.95] |
| wrong_algorithm_name | 8 | 0.520 [0.295, 0.749] | 0.508 [0.287, 0.755] | −0.012 | 1/0/7 | 0.13 [0.003, 0.53] |

Done-when checks: S and U share one code path; the leak assertion passes; U ≤ S within CI for every type (largest positive gap +0.022, with a U seed SD of 0.027 for that type).

**Paper text.** *Under a single protocol, held-out corruption types are never detected better than seen ones (U − S ≤ +0.02, within seed and bootstrap spread). The earlier Table XXXII scored clean rows in-sample, which produced the apparent reversal. Volumetric types (over-summarization, tool-result drop, truncation) are detected almost perfectly and transfer to unseen training. Length-preserving semantic types (invert objective, negate edge case, wrong algorithm name) are statistically indistinguishable from chance; their intervals are wide (n = 6–14) and we make no claim beyond "no detectable signal".*

Figure: `results/figures/b01_loto.png`.

---

## 3. B2: `function_name_preserved` diagnosis

**Cause: there is no extractor bug in the current data; the paper's 0.45 came from the old substring bug, and the feature's real weakness is a task-level naming effect.** Evidence:

* All 14 `signature_rename` rows: the original handoff contained the canonical name (13 of 14; `mbpp_002` already used a different name), the corrupted handoff contains `name_impl` and **zero** whole-word mentions, feature = 0 for all 14. The rename was applied to every mention. The stored feature equals a recomputation on the stored text for 100% of rows, so it is not computed on pre-corruption text. (Full 14-row table: `results/b02_signature_rename_rows.csv`.)
* The feature is a **task property**, not a corruption detector: for length-preserving types it equals its value on the clean sibling. **35% of clean rows (87/248) score 0**, because Agent A used a different function name than the canonical entry point (86 of 87 have a `def` line with another name). By source: MBPP 81%, hand_easy 33%, HumanEval 2.5%, hand_hard 0%.
* The overall separation comes from volumetric types dropping the signature (`over_summarization` 0.05, `tool_result_drop` 0.03 mean), exactly the "another volumetric proxy" hypothesis.

| Type | n | fnp mean | fnp on the pre-corruption handoff |
|---|---|---|---|
| none (clean) | 248 | 0.649 | 0.649 |
| signature_rename | 14 | 0.000 | 0.929 |
| over_summarization | 63 | 0.048 | 0.762 |
| tool_result_drop | 65 | 0.031 | 0.831 |
| truncation | 33 | 0.788 | 0.818 |
| entity_omission | 16 | 0.250 | 0.688 |
| invert_objective | 14 | 0.500 | 0.500 |
| negate_edge_case | 6 | 0.333 | 0.333 |
| wrong_algorithm_name | 8 | 0.625 | 0.625 |
| fake_completion / inject_false_constraint | 11 / 5 | 0.000 | 0.000 |

Single-feature AUROC (task bootstrap): clean vs signature_rename **0.825 [0.794, 0.853]** (capped, because 35% of clean rows also score 0); clean vs truncation 0.431 [0.365, 0.501]; clean vs invert_objective 0.575 [0.439, 0.707]; all corrupted vs clean 0.720 [0.686, 0.754].

Unit tests (`tests/test_function_name_preserved.py`, 24 pass): clean handoff = 1; full rename = 0; substring trap (`name` vs `name_impl`) = 0; missing/NaN entry point = 0.5; includes the integration check over every `signature_rename` row. Parameter-only rename is not covered by this feature (it checks the name only; the parameter-diff feature was dropped at 65% missing).

**Paper implication.** The feature is also the most important one in the model (dropping it costs 0.044 AUROC, §6.3), so a large part of the 11-feature gain is the MBPP naming pattern. Report the per-source AUROC (§6.4) and say so.

---

## 4. B8: NLI feature decision (and freezing `features_v2`)

**Decision rule, written before looking:** keep `nli_entailment_mean` only if (a) the within-task paired direction is correct (clean entailment > corrupted, sign test p < 0.05) and (b) the paired task-bootstrap CI of ΔAUROC (12-feature minus 11-feature RF) excludes 0 on the positive side. Otherwise drop it.

| | Result |
|---|---|
| Class means (label 0 / label 1) | 0.126 / 0.105 (wrong direction) |
| Paired sign test, all 235 pairs | clean higher in 150, lower in 85 (63.8%, p < 0.001), but **driven only by over_summarization (46/63) and tool_result_drop (48/65)**; on six other types the direction is reversed or not significant; mean paired difference −0.003 |
| ΔAUROC, 12 vs 11 features (5 fold-seeds averaged, paired CI) | **+0.0003 [−0.0049, +0.0061]** (AUROC 0.9145 vs 0.9142) |
| Correlation with volume | Spearman(NLI, length_ratio) = −0.012, with sentence_count_ratio = 0.081 (so the ≤16-pair mean is not simply a length proxy) |

**Decision: DROP** (the CI includes 0). Frozen set `features_v2` = the 11 gate features (`features_v2.json`, `features_v2.csv`). This removes the DeBERTa NLI model from the latency path and makes Tables XXV/XXVI/XXVIII (12 features) and the gate (11) consistent: use 11 everywhere. Per-type means are in `results/b08_nli_by_type.csv`.

Not done: the "20 hand-checked premise/hypothesis examples" test requires manual labelling. By code inspection the pairs are `(problem_sentence, handoff_sentence)` = (premise, hypothesis), and the label order `[contradiction, entailment, neutral]` is the one documented for the cross-encoder; the decision above does not depend on this.

Figure: `results/figures/b08_nli_by_type.png`.

---

## 5. B5: Calibration

Nested: for each outer fold, the calibrator is fitted on inner task-grouped OOF scores of the training tasks only (done by hand, since `CalibratedClassifierCV`'s default CV ignores groups).

| Scores | ECE (10 equal-mass bins) | Brier | AUROC |
|---|---|---|---|
| Raw RF | 0.0403 | 0.1017 | 0.9178 |
| Isotonic | 0.0481 | 0.0992 | 0.9036 |
| **Platt** | **0.0382** | 0.1003 | 0.9177 |

The RF is already close to calibrated on the dev set; isotonic is worse (ties collapse ranking, AUROC −0.014). Platt scores are used for B3 and B9 (lowest ECE, no ties). Calibration is not a meaningful improvement (ECE 0.040 → 0.038); report it as "well calibrated at 51/49 prevalence".

**Prior shift.** Dev is 51.4% good. Without logit adjustment, a dev-calibrated probability threshold is wrong at deployment prevalence; with `logit shift = log(π_deploy/(1−π_deploy)) − log(π_dev/(1−π_dev))` the mean P(good) moves 0.506 → 0.618 / 0.691 / 0.744 at 20% / 5% / 1% bad rate (`results/b05_prior_shift.csv`). The frozen threshold below is a **dev-prevalence** threshold; for a deployed system either re-apply the shift or choose the threshold on the false-block rate (as done here, which is prevalence-free).

Figure (reliability and per-type score violins): `results/figures/b05_calibration.png`. The violins show `over_summarization`/`tool_result_drop` at the bottom and tier-3 types overlapping clean rows.

---

## 6. B4: Task-level bootstrap CIs

Intervals reflect **evaluation-sample uncertainty** (tasks resampled on fixed OOF scores), not training variance. 10,000 resamples, seed 42; every replicate was valid except where noted.

### 6.1 Model comparison (replaces Table VIII), seed-0 OOF

| Model | 3 features | 11 features |
|---|---|---|
| Logistic regression | 0.836 [0.788, 0.879] | 0.897 [0.861, 0.930] |
| Random forest | 0.851 [0.811, 0.889] | **0.918 [0.883, 0.948]** |
| XGBoost | 0.839 [0.793, 0.881] | 0.903 [0.865, 0.937] |

### 6.2 Paired differences (same resampled tasks)

| Comparison | ΔAUROC [CI] |
|---|---|
| RF − XGBoost (11 feat) | +0.015 [+0.003, +0.028] |
| RF − LR (11 feat) | +0.020 [−0.004, +0.047] (**not distinguishable**) |
| XGBoost − LR (11 feat) | +0.005 [−0.020, +0.032] |
| RF 11 − RF 3 features | **+0.067 [+0.038, +0.098]** |
| XGBoost 11 − XGBoost 3 | +0.063 [+0.030, +0.100] |

So "11 features beat 3" is supported; "RF beats LR" is not.

### 6.3 Ablation of the RF (replaces Table XXXI)

| Feature | AUROC alone [CI] | ΔAUROC when dropped [paired CI] |
|---|---|---|
| function_name_preserved | 0.698 [0.657, 0.737] | **−0.044 [−0.067, −0.023]** |
| section_coverage | 0.760 [0.721, 0.798] | **−0.012 [−0.021, −0.003]** |
| novel_api_rate | 0.714 [0.677, 0.750] | −0.004 [−0.010, +0.001] |
| **length_ratio** | **0.816 [0.769, 0.860]** | **−0.004 [−0.012, +0.003]** |
| sentence_count_ratio | 0.787 [0.740, 0.831] | −0.002 [−0.007, +0.002] |
| entity_overlap | 0.584 [0.537, 0.631] | −0.002 [−0.007, +0.003] |
| cosine_similarity | 0.656 [0.609, 0.704] | −0.002 [−0.008, +0.004] |
| role_pronoun_rate | 0.533 [0.482, 0.581] | −0.001 [−0.005, +0.002] |
| constraint_count | 0.748 [0.700, 0.794] | −0.001 [−0.006, +0.003] |
| trailing_specificity | 0.727 [0.682, 0.770] | +0.001 [−0.005, +0.006] |
| verbatim_copy_rate | 0.675 [0.633, 0.718] | +0.005 [+0.0002, +0.010] (dropping it helps slightly) |

The claim "removing length ratio costs only 0.003" is supported: the paired CI is [−0.012, +0.003], tight around 0. The reason is redundancy (sentence_count_ratio and others carry the same volume information), not irrelevance: length ratio alone reaches 0.816.

### 6.4 Per source (replaces Table XXXIII), RF seed-0 OOF

| Source | n | negatives | AUROC [CI] |
|---|---|---|---|
| hand_hard | 127 | 83 | 0.989 [0.970, 1.000] |
| hand_easy | 64 | 31 | 0.907 [0.792, 0.976] |
| humaneval | 141 | 61 | 0.894 [0.805, 0.966] |
| mbpp | 151 | 60 | 0.825 [0.746, 0.902] |

Per-type intervals are in §2. Types with n = 5–14 have wide CIs; where the interval contains 0.5 write "CI is wide; no detectable signal", not "systematically fooled".

---

## 7. B3: Threshold protocol

**Rule (fixed first):** block when score < t; choose the largest t such that the **false-block rate on clean handoffs ≤ 5%**, which gives the maximum detection subject to that cap. Headline t chosen on the pooled Platt-calibrated dev OOF scores, then frozen. Nested estimate: per outer fold, t chosen on inner-fold OOF scores of the training tasks only, applied to the outer test fold, pooled.

| | Frozen headline (Platt scale) | Same rule on raw RF scores | **Nested estimate** |
|---|---|---|---|
| Threshold | **0.475** | 0.550 | mean 0.556 (SD 0.006) across folds |
| TP / FN / FP / TN | 183 / 52 / 12 / 236 | 184 / 51 / 12 / 236 | 186 / 49 / 13 / 235 |
| Detection rate on bad | 0.779 [0.710, 0.843] | 0.783 [0.717, 0.846] | 0.792 [0.727, 0.853] |
| **False-block rate on good** | 0.048 [0.024, 0.077] | 0.048 [0.024, 0.077] | 0.052 [0.028, 0.081] |
| Precision of blocks | 0.939 [0.903, 0.969] | 0.939 [0.904, 0.970] | 0.935 [0.899, 0.966] |
| F1 (bad class) | 0.851 [0.805, 0.891] | 0.854 | 0.857 [0.813, 0.895] |
| F1 (good class) | 0.881 | 0.882 | 0.884 |

The nested estimate agrees with the headline (detection +0.013, false-block +0.004), so selecting the threshold on the same OOF scores is not materially optimistic here. **Frozen for the live run:** `results/frozen_threshold.json` (`threshold_platt` 0.475 on Platt-calibrated scores; `threshold_raw` 0.550 on raw RF scores).

For context, the five thresholds quoted in the paper, on raw OOF scores (`results/b03_paper_thresholds_on_raw_oof.csv`):

| t | detection | false-block |
|---|---|---|
| 0.330 | 0.685 | 0.004 |
| 0.450 | 0.736 | 0.020 |
| 0.500 | 0.766 | 0.020 |
| 0.695 | 0.864 | 0.190 |
| 0.7972 | 0.919 | 0.427 |

Figure (ROC and PR with the chosen point): `results/figures/b03_roc_pr.png`.

---

## 8. B7: Length-ratio confound

Under the discard policy type determines the label, so `label ~ features + C(type)` is perfectly separable; not fitted. Used instead:

**Matched-pair model.** For each of the 235 negatives, the within-task difference (clean − corrupted) of the standardized features, random sign flip, no-intercept logistic regression predicting the sign, task-grouped CV.

| Features | Pair AUROC [CI] |
|---|---|
| all 11 | 0.9905 [0.980, 0.997] |
| all minus length_ratio | 0.9892 [0.978, 0.997] |
| minus length_ratio and sentence_count_ratio | 0.9849 [0.970, 0.995] |
| length_ratio only | 0.9643 [0.940, 0.982] |

Adding length ratio to the other ten features changes pair AUROC by **+0.0013 [−0.0003, +0.0041]**.

**Within-type, length-matched controls** (each type-X row matched to the nearest unused clean row by length ratio; "model" = the OOF RF score):

| Type | n | LR AUROC unmatched → matched | Model AUROC on matched set [CI] |
|---|---|---|---|
| truncation | 33 | 0.966 → 0.824 | 0.976 [0.941, 0.997] |
| entity_omission | 16 | 0.769 → 0.520 | 0.816 [0.669, 0.955] |
| signature_rename | 14 | 0.563 → 0.500 | 0.867 [0.762, 0.979] |
| fake_completion | 11 | 0.208 → 0.500 | 0.860 [0.769, 0.967] |
| invert_objective | 14 | 0.462 → 0.500 | 0.513 [0.474, 0.574] |
| wrong_algorithm_name | 8 | 0.570 → 0.516 | 0.445 [0.135, 0.736] |
| negate_edge_case | 6 | 0.365 → 0.514 | 0.333 [0.037, 0.714] |
| inject_false_constraint | 5 | 0.230 → 0.500 | 0.440 [0.200, 0.680] |

**Conclusion (paragraph for the paper).** *Length ratio does not carry signal beyond what the other features and the corruption type already supply: in a matched-pair model that controls for task, adding it to the other ten features changes AUROC by +0.001 (CI includes 0). Alone it is a strong volumetric proxy (pair AUROC 0.96), but it is redundant with sentence-count and section features. After matching on length, the model still separates entity omission, signature rename and fake completion from clean handoffs (AUROC 0.82–0.87), through the signature, coverage and trailing-specificity features, and it does not separate invert-objective, wrong-algorithm, negate-edge-case or false-constraint corruptions from clean ones (CIs include 0.5). Detection is therefore not purely volumetric, but the semantic tier is not detected.* Caveats: for length-preserving types the sibling length ratio is identical by construction, so the matched-length AUROC of exactly 0.500 there is expected, not evidence. The after-A3 likelihood-ratio test (robust-positive dataset, both labels per type) is pending that dataset.

---

## 9. B6: Baselines (same task-grouped folds, same metrics)

Operating point for continuous scorers = the B3 rule on that method's own pooled OOF scores (optimistic in the same way as the headline). Fine-tuned encoder: MiniLM-L6, 3 lower layers frozen, 4 epochs, lr 5e-5, 3 seeds × 5 folds, GPU. Embedding LR: C = 0.01, not tuned.

| Method | AUROC [CI] | Detection / false-block at operating point |
|---|---|---|
| **Handoff PRM (RF, 11 features)** | **0.918 [0.883, 0.948]** | 0.783 / 0.048 |
| Fine-tuned MiniLM-L6 (3 seeds avg; seeds 0.852, 0.875, 0.851) | 0.862 [0.818, 0.901] | 0.706 / 0.048 |
| MiniLM embeddings + LR | 0.883 [0.851, 0.912] | 0.689 / 0.048 |
| Length threshold (depth-2 tree on length_ratio) | 0.781 [0.727, 0.831] | 0.660 / 0.048 |
| Exact signature match (binary; `def <entry_point>(` in handoff) | 0.720 [0.687, 0.752] | 0.847 / 0.407 (single operating point) |
| LLM judge (3 prompts, temp 0, 0–100 score) | **not run** (script `b06_llm_judge.py`, needs `GEMINI_API_KEY`) | n/a |

Hand-built features beat a plain learned probe (paired ΔAUROC vs embedding LR +0.035 [+0.004, +0.066]) and a fine-tuned encoder (+0.056 [+0.021, +0.092]). The exact signature baseline catches most bad handoffs only by blocking 41% of good ones.

### 9.6 End-to-end latency (CPU only, GPU hidden, batch size 1, warm models, 483 handoffs)

spaCy + MiniLM + 11 feature functions + RF `predict_proba` (single thread):

| p50 | p95 | mean | classifier-only p50 |
|---|---|---|---|
| **252 ms** | 358 ms | 264 ms | 121 ms |

Against the previously reported judge latency of 3,933 ms (177 rows, unstated prompt, not re-measured), the end-to-end ratio is about **16×**, not 393,000×. The RF with 300 trees dominates classifier time; a smaller forest or XGBoost would be faster and was not tuned. Judge API latency and dollar cost per 1,000 handoffs are **not measured**. Report "latency", not "cost".

---

## 10. B9: Prevalence sweep (frozen threshold)

TPR 0.779 [0.710, 0.843], FBR 0.048 [0.024, 0.077] (§7). Live quantities from the 94-task pilot (`live_pilot_results.csv`): among blocked handoffs that would have failed ungated, the fallback succeeded in 13/23 (r = 0.565); among blocked handoffs that would have passed, the fallback failed in 2/24 (h = 0.083). `ΔPass@1 ≈ π·TPR·r − (1−π)·FBR·h`. Intervals come from joint task-bootstrap of TPR/FBR and row-bootstrap of r and h.

| Bad-handoff prevalence | Precision of blocks | NPV of passes | False blocks per true catch | ΔPass@1 (pp) |
|---|---|---|---|---|
| 1% | 0.150 [0.093, 0.249] | 0.998 [0.997, 0.998] | 5.7 | +0.04 [−0.69, +0.51] |
| 5% | **0.470 [0.348, 0.633]** | 0.988 [0.984, 0.991] | 1.1 | **+1.8 [+0.8, +2.8]** |
| 20% | 0.804 [0.717, 0.891] | 0.945 [0.929, 0.960] | 0.24 | +8.5 [+5.3, +11.7] |
| 50% | 0.942 [0.910, 0.970] | 0.812 [0.767, 0.858] | 0.06 | +21.8 [+14.0, +29.9] |

**Break-even prevalence** (ΔPass@1 = 0): **0.9% [0%, 2.7%]**. Below about 1% bad handoffs the gate nets negative or zero. The decision curve (net benefit vs cost ratio of a false block to a miss) is in `results/figures/b09_prevalence.png`.

Caveats to state: (1) r and h rest on 23 and 24 handoffs; their uncertainty is included in the CI but the estimates are fragile. (2) The live pilot used the two-stage cascade with its own stage-1 threshold, not the frozen B3 threshold, so combining them is an approximation. (3) The corruption mix here is 235 negatives of which 54% (128/235) are the two volumetric types, so organic prevalence of *detectable* failures may be lower than the nominal π. Precision of blocks at 5% prevalence is **0.47**, not the 0.62 obtained from Table XI.

---

## 11. B10: Organic evaluation

Not run: needs the organic dataset (A1). Procedure to run when it lands: score with the frozen classifier (`handoff_prm.joblib` retrained on `features_v2` only, threshold from `results/frozen_threshold.json`), task-bootstrap AUROC overall / per Agent-A variant / per source, block/false-block/recall/precision at the observed failure rate, comparison against every B6 baseline on the same rows, per-feature synthetic-vs-organic distribution plots (especially length ratio), hand-label ~50 organic negatives into the corruption categories or "other", and check A2 label agreement. Gate 1: organic AUROC ≥ ~0.75 keeps the detection framing; ~0.5–0.65 switches to the measurement/negative-result framing. **Given §2 (semantic tier at chance) and §3 (a large part of the signal is the MBPP naming pattern), the organic result is the key test.**

---

## 12. Reproduction

```powershell
cd Handoff-PRM1
python -m venv --system-site-packages .venv
.venv\Scripts\python -m pip install sentence-transformers spacy statsmodels shap tqdm sumy
.venv\Scripts\python -m spacy download en_core_web_sm

.venv\Scripts\python b02_function_name_diagnosis.py      # B2
.venv\Scripts\python b08_nli_decision.py                 # B8, writes features_v2.csv/json
.venv\Scripts\python generate_oof_v2.py                  # shared oof_predictions_v2.csv
.venv\Scripts\python b01_loto.py                         # B1 (20 seeds, ~45 min on CPU)
.venv\Scripts\python b05_calibration.py                  # B5, writes results/b05_nested.pkl
.venv\Scripts\python b03_threshold.py                    # B3, writes frozen_threshold.json
.venv\Scripts\python b04_bootstrap_cis.py                # B4
.venv\Scripts\python b07_confound.py                     # B7
.venv\Scripts\python b09_prevalence.py                   # B9
.venv\Scripts\python b06_baselines.py                    # B6 1-3
.venv\Scripts\python b06_finetune_encoder.py             # B6 4 (GPU)
.venv\Scripts\python b06_latency.py                      # B6 6
# B6 5, needs GEMINI_API_KEY: python b06_llm_judge.py --prompt generic (then targeted, with_problem, then --summarize)
.venv\Scripts\python -m pytest tests -q
```

Seeds: fold seed 0 (headline), 0–19 (B1), 0–4 averaged (B8, length-only and embedding baselines), fine-tune seeds 0/1/2, bootstrap seed 42, B7 sign-flip seed 0. Models: RF(300, depth 6, balanced), `all-MiniLM-L6-v2` (features, embedding LR, fine-tune), `en_core_web_sm`. Python 3.11, scikit-learn 1.8, torch 2.5.1+cu121.

## 13. Deliverables checklist

- [x] `common/` utilities with unit tests (task overlap assertion, bootstrap)
- [x] `features_v2` frozen (B2, B8): 11 features
- [x] One script per item (`b01`…`b09`; B10 pending A1), each writing `results/bXX_*.csv` and a figure
- [x] `oof_predictions_v2.csv` regenerated and shared
- [x] This consolidated results file
- [x] Reproduction commands, seeds, models
- [ ] B6 item 5 (LLM judge) and $ per 1k: script ready, needs an API key
- [ ] B10 organic evaluation: needs A1
- [ ] Hand-checked NLI examples (B8 step 3): manual labelling required
- [ ] B7 likelihood-ratio tests: need the A3 robust-positive dataset
