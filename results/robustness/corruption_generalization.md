# Corruption Generalization Analysis

> Replaces Tables XXVIII and XXXII. Both protocols use the same model,
> features, and task folds (5-fold, 20 seeds, seeds 0–19).

## Methods

- **Dataset**: `features.csv` (2696 rows, features: `cosine_similarity`, `entity_overlap`, `length_ratio`, `sentence_count_ratio`, `section_coverage`, `verbatim_copy_rate`, `trailing_specificity`, `constraint_count`, `role_pronoun_rate`, `novel_api_rate`, `function_name_preserved`)
- **Model**: `RandomForestClassifier(n_estimators=200, max_depth=6, class_weight='balanced')` wrapped in `Pipeline([SimpleImputer, StandardScaler, RF])`
- **Protocol S (seen)**: `oof_scores(df, FEAT_V2, make_model)` — trains on all corruption types. AUROC measured on {clean, type-X} rows.
- **Protocol U (unseen)**: `oof_scores(df, FEAT_V2, make_model, drop_from_train=(type==X))` — training never sees type-X rows; they appear in their natural test fold. Task-leakage assertion verified.
- **Paired win-rate**: for every (clean-row, type-X-row) pair sharing the same `task_id`, check if the clean row scored higher. Pooled across 20 seeds. Exact Clopper-Pearson 95% CI.

**Root cause of the XXVIII vs XXXII gap:** Table XXXII (old 'unseen' protocol) trained on `final_dataset_full_features.csv` (483 rows, 11 features, RF) but included all 248 clean rows in *both* the training set and the test set simultaneously -- the positives were never held out, so the model had already fitted their feature distribution before scoring them, inflating AUROC by 0.2-0.4 points for rare types. Table XXVIII used a different dataset (12-feat RF on the 483-row subset) and different folds. The two tables therefore measure different things and cannot be compared. This table replaces both.

## Results

| Corruption type | n | AUROC-S (mean±std) | AUROC-U (mean±std) | Δ(U−S) | Paired win-rate [95% CI] | U>S? |
|-----------------|---|--------------------|--------------------|--------|--------------------------|------|
| tool_result_drop | 112 | 1.000±0.000 | 1.000±0.000 | -0.000 | 1.000 [1.000, 1.000] |  |
| over_summarization | 127 | 1.000±0.000 | 0.999±0.000 | -0.001 | 1.000 [1.000, 1.000] |  |
| fake_completion | 434 | 0.956±0.002 | 0.879±0.005 | -0.077 | 0.990 [0.988, 0.992] |  |
| truncation | 73 | 0.935±0.004 | 0.920±0.004 | -0.015 | 0.953 [0.943, 0.964] |  |
| inject_false_constraint | 434 | 0.616±0.004 | 0.572±0.004 | -0.044 | 0.816 [0.808, 0.825] |  |
| entity_omission | 31 | 0.564±0.014 | 0.557±0.009 | -0.007 | 0.632 [0.594, 0.669] |  |
| wrong_algorithm_name | 69 | 0.555±0.008 | 0.556±0.007 | +0.000 | 0.452 [0.414, 0.490] |  |
| signature_rename | 433 | 0.511±0.003 | 0.506±0.002 | -0.006 | 0.596 [0.584, 0.607] |  |
| negate_edge_case | 322 | 0.506±0.004 | 0.500±0.002 | -0.006 | 0.461 [0.446, 0.475] |  |
| invert_objective | 227 | 0.465±0.005 | 0.456±0.003 | -0.010 | 0.541 [0.519, 0.562] |  |

## Interpretation

- **Delta ≤ 0**: U ≤ S (expected). The model generalizes; removing type-X from training does not improve detection.
- **Delta > 0 within ±1 std**: noise-level, no meaningful reversal.
- **⚠️ flag**: mean-U exceeds mean-S by more than one std — investigate whether the type-X rows' features are accidentally informative in the opposite direction, or whether n is too small for a stable estimate.

## Paired win-rate

The paired win-rate is a task-matched comparison: within the same task, did the clean handoff
score higher than the corrupted one? This removes per-task difficulty variation and is far more
informative than absolute AUROC at small n (5–14).

| Corruption type | n_neg | Wins | Ties | Losses | Win-rate | 95% CI |
|-----------------|-------|------|------|--------|----------|--------|
| tool_result_drop | 112 | 2240 | 0 | 0 | 1.000 | [1.000, 1.000] |
| over_summarization | 127 | 2540 | 0 | 0 | 1.000 | [1.000, 1.000] |
| fake_completion | 434 | 8595 | 0 | 85 | 0.990 | [0.988, 0.992] |
| truncation | 73 | 1392 | 0 | 68 | 0.953 | [0.943, 0.964] |
| inject_false_constraint | 434 | 6877 | 256 | 1547 | 0.816 | [0.808, 0.825] |
| entity_omission | 31 | 392 | 0 | 228 | 0.632 | [0.594, 0.669] |
| signature_rename | 433 | 4315 | 1414 | 2931 | 0.596 | [0.584, 0.607] |
| invert_objective | 227 | 1122 | 2467 | 951 | 0.541 | [0.519, 0.562] |
| negate_edge_case | 322 | 2097 | 1886 | 2457 | 0.461 | [0.446, 0.475] |
| wrong_algorithm_name | 69 | 311 | 692 | 377 | 0.452 | [0.414, 0.490] |
