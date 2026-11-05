"""
Patch existing scripts to add `from common.cv_utils import ...` and
`from common.oof_store import ...` lines after their existing import blocks.
Run once: python common/_patch_imports.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

COMMON_IMPORTS = (
    "\n# Shared CV utilities -- write once, use everywhere\n"
    "from common.cv_utils import task_folds, oof_scores, task_bootstrap_ci  # noqa: F401\n"
    "from common.oof_store import append_oof, load_oof                       # noqa: F401\n"
)

# Each entry: (file, unique anchor pattern to insert AFTER)
PATCHES = [
    (
        "model_comparison.py",
        r"import time\r?\n",
    ),
    (
        "feature_ablation.py",
        r"from sklearn\.linear_model import LogisticRegression\r?\n",
    ),
    (
        "held_out_corruption_eval.py",
        r"from scipy import stats\r?\n",
    ),
    (
        "model_sweep_12features_final.py",
        r"from sklearn\.metrics import \(",     # start of the metrics import block
    ),
    (
        "model_comparison_sweep.py",
        r"from sklearn\.metrics import \(",
    ),
]


def patch(filename: str, anchor_pattern: str) -> bool:
    path = ROOT / filename
    if not path.exists():
        print(f"  SKIP (not found): {filename}")
        return False

    text = path.read_text(encoding="utf-8", errors="replace")

    # Already patched?
    if "from common.cv_utils import" in text:
        print(f"  SKIP (already patched): {filename}")
        return False

    m = re.search(anchor_pattern, text)
    if not m:
        print(f"  SKIP (anchor not found): {filename}  pattern={anchor_pattern!r}")
        return False

    # Find the end of the full line containing the anchor
    line_end = text.find("\n", m.end())
    if line_end == -1:
        line_end = len(text)
    else:
        line_end += 1  # include the newline

    new_text = text[:line_end] + COMMON_IMPORTS + text[line_end:]
    path.write_text(new_text, encoding="utf-8")
    print(f"  PATCHED: {filename}")
    return True


if __name__ == "__main__":
    print("Patching scripts with common/ imports...\n")
    for fname, pattern in PATCHES:
        patch(fname, pattern)
    print("\nDone.")
