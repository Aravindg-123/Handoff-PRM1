# G vs Badamalva: The Workstream B Chronicles

*A README-style change report. Aravind G walks in with a 483-row dataset and a 10-item to-do list; the paper's old numbers walk out with fewer excuses.*

Full paper-ready tables, CIs and prose live in [`workstream_B_results.md`](workstream_B_results.md). This file is the "what did you actually change" ledger.

## Cast

| Role | Who |
|---|---|
| **G** | the new offline-validity code and stats (this branch) |
| **Badamalva** | the old numbers: Table XXXII's in-sample clean rows, the 0.45 `signature_rename` AUROC, the 393,000× latency claim |

## Round results

| Round | Item | Verdict |
|---|---|---|
| 1 | B1 leave-one-type-out | Unseen never beats seen once clean rows are scored out-of-fold (U − S ≤ +0.02). Badamalva's "reversal" was in-sample scoring. |
| 2 | B2 `function_name_preserved` | Extractor is fine now. The old 0.45 was the `name`/`name_impl` substring bug. Real weakness: 35% of clean rows score 0 (MBPP naming). |
| 3 | B8 NLI | Dropped (ΔAUROC +0.0003, CI [−0.005, +0.006]). `features_v2` = 11 features. |
| 4 | B4 bootstrap CIs | Every AUROC now has an interval. RF vs LR is not distinguishable; 11 vs 3 features is. |
| 5 | B5 calibration | RF already near-calibrated (ECE 0.040); Platt marginally best, isotonic worse. |
| 6 | B3 threshold | Frozen at 0.475 (Platt scale) / 0.550 (raw): detection 0.78, false-block 4.8%. Nested estimate agrees. |
| 7 | B9 prevalence | Precision of blocks at 5% prevalence is 0.47, not 0.62. Break-even ≈ 0.9% bad handoffs. |
| 8 | B7 length confound | Length ratio adds +0.001 AUROC beyond the other features (CI includes 0). |
| 9 | B6 baselines | PRM 0.918 beats embeddings+LR 0.883, fine-tuned MiniLM 0.862, length 0.781, exact signature 0.720. End-to-end CPU latency p50 252 ms. |
| 10 | B10 organic | Not run, needs the organic dataset (A1). |

## Files changed

**Modified**
- `common/cv_utils.py`: `task_bootstrap_ci` now resamples with index arrays (identical draws, about 50× faster).
- `oof_predictions_v2.csv`: regenerated with `generate_oof_v2.py` on the 483-row dataset.

**New: shared code**
- `common/stats_utils.py`: fast AUROC, task and paired bootstrap, ECE, Clopper–Pearson, block metrics, threshold rule, block precision.
- `common/calib.py`: nested task-grouped Platt/isotonic calibration, prior shift.
- `common/data.py`: feature lists, RF/LR factories, `features_v2` loader.
- `tests/test_stats_utils.py`: 6 new tests (fold integrity, leak assertion, AUROC = sklearn, bootstrap parity, paired diff, ECE and threshold rule).

**New: one script per item**
`b01_loto.py`, `b02_function_name_diagnosis.py`, `b03_threshold.py`, `b04_bootstrap_cis.py`, `b05_calibration.py`, `b06_baselines.py`, `b06_finetune_encoder.py`, `b06_latency.py`, `b06_llm_judge.py` (written, not run: no API key), `b07_confound.py`, `b08_nli_decision.py`, `b09_prevalence.py`.

**New: frozen artefacts and outputs**
- `features_v2.csv`, `features_v2.json`: the frozen 11-feature dev set.
- `results/frozen_threshold.json`: the threshold for the live run.
- `results/b*.csv`, `results/figures/b*.png`: every table and figure.
- `results/workstream_B_results.md`: the paper-ready report.

## Things the paper must correct

1. Tables XXVIII/XXXII: replace both with the B1 table (single protocol).
2. `function_name_preserved` class means and the 0.45 AUROC are stale.
3. RF AUROC 0.968 is not reproduced; the shipped data gives 0.918 [0.883, 0.948].
4. Drop the "393,000×" claim; end-to-end latency is p50 252 ms.
5. Table XI is optimistic (78% detection at 5% false-block, 0.47 precision at 5% prevalence).
6. F1 0.9464 is the good-class F1 from Table XI's counts; 0.9455 is not reproducible.

## Not done

- LLM-judge baseline and its dollar cost (script ready, needs `GEMINI_API_KEY`).
- B10 organic evaluation (needs A1).
- Hand-checked NLI examples (manual labelling).
- B7 likelihood-ratio tests (need the A3 robust-positive dataset).

## Run it yourself

```powershell
python b02_function_name_diagnosis.py; python b08_nli_decision.py; python generate_oof_v2.py
python b01_loto.py; python b05_calibration.py; python b03_threshold.py
python b04_bootstrap_cis.py; python b07_confound.py; python b09_prevalence.py
python b06_baselines.py; python b06_finetune_encoder.py; python b06_latency.py
python -m pytest tests -q
```

Seeds, model IDs and package versions are in §12 of `workstream_B_results.md`.
