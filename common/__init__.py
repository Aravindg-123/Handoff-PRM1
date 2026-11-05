"""
common/ -- shared utilities for handoff-prmv2.

Import from anywhere in the project:

    from common.cv_utils import task_folds, oof_scores, task_bootstrap_ci
    from common.oof_store import load_oof, append_oof

Nothing here makes API calls or loads heavy models -- keep it that way.
"""
