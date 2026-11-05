"""
tests/test_function_name_preserved.py
======================================
Unit tests for compute_function_name_preserved + the signature_rename corruption.
Run with:  python -m pytest tests/test_function_name_preserved.py -v

Covers the four scenarios from the B2 diagnostic note plus regression tests
for the substring-trap and entry_point=None bugs.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from feature_extraction import compute_function_name_preserved as fnp
from corruption import signature_rename


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _fnp(handoff, entry_point):
    return fnp(handoff, reference_text="", entry_point=entry_point)


# ---------------------------------------------------------------------------
# T1: Clean handoff — entry_point name appears as a whole word → 1.0
# ---------------------------------------------------------------------------
class TestCleanHandoff:
    def test_name_in_def_line(self):
        h = "def is_prime(n: int) -> bool:\n    pass"
        assert _fnp(h, "is_prime") == 1.0

    def test_name_in_prose(self):
        h = "Implement is_prime. It should return True for prime inputs."
        assert _fnp(h, "is_prime") == 1.0

    def test_name_in_backtick(self):
        h = "Call `is_prime(n)` to check primality."
        assert _fnp(h, "is_prime") == 1.0


# ---------------------------------------------------------------------------
# T2: Full rename via signature_rename() — original must NOT match (whole-word)
# This was the main BUG: `is_prime in 'is_prime_impl'` returned True
# ---------------------------------------------------------------------------
class TestSignatureRename:
    CLEAN = (
        "Write a Python function `is_prime(n)` that returns True if n is prime.\n"
        "def is_prime(n: int) -> bool:\n"
        "    # your code here\n"
        "Handle edge cases: n <= 1 returns False."
    )

    def test_rename_produces_wrong_name_in_def(self):
        renamed = signature_rename(self.CLEAN, entry_point="is_prime")
        assert "is_prime_impl" in renamed, "corruption did not introduce wrong name"

    def test_renamed_handoff_returns_zero(self):
        """REGRESSION: old `in` check returned 1.0 because 'is_prime' ⊂ 'is_prime_impl'."""
        renamed = signature_rename(self.CLEAN, entry_point="is_prime")
        # Confirm original name is gone as whole word
        assert re.search(r"\bis_prime\b", renamed) is None, \
            "original name still present as whole word after rename"
        assert _fnp(renamed, "is_prime") == 0.0

    def test_all_247_rename_rows_return_zero(self):
        """Integration: every signature_rename row in corrupted_handoffs.csv scores 0."""
        import pandas as pd
        ch_path = ROOT / "corrupted_handoffs.csv"
        if not ch_path.exists():
            pytest.skip("corrupted_handoffs.csv not found")
        ch = pd.read_csv(ch_path)
        sig = ch[ch["corruption_type"] == "signature_rename"]
        for _, row in sig.iterrows():
            ep = str(row.get("entry_point", ""))
            h  = str(row.get("handoff", ""))
            val = _fnp(h, ep)
            assert val == 0.0, (
                f"task={row['task_id']} ep={ep!r}: fnp={val} "
                f"(expected 0.0 — original name must not appear after rename)"
            )


# ---------------------------------------------------------------------------
# T3: Substring trap — 'sort' must not match inside 'sort_impl'
# ---------------------------------------------------------------------------
class TestSubstringTrap:
    def test_sort_not_in_sort_impl(self):
        h = "def sort_impl(arr): pass\nsort_impl an array."
        assert _fnp(h, "sort") == 0.0, \
            "'sort' should not match as whole word when only 'sort_impl' is present"

    def test_find_not_in_find_all(self):
        h = "Use find_all(text) to locate items. def find_all(t): pass"
        assert _fnp(h, "find_all") == 1.0
        # Whole-word matching is intentional: "find" is not a standalone
        # identifier when it only appears as the prefix of a longer function name.
        assert _fnp(h, "find") == 0.0
        h2 = "def find_all(t): pass"
        assert _fnp(h2, "find") == 0.0, \
            "'find' must not match inside 'find_all'"

    def test_longer_name_match(self):
        h = "Implement second_largest(arr). def second_largest(a): pass"
        assert _fnp(h, "second_largest") == 1.0
        # A shorter token is not the canonical entry point when it is only a
        # prefix of a longer function name.
        assert _fnp(h, "second") == 0.0
        h_renamed = "def second_largest_impl(a): pass. Use second_largest_impl."
        assert _fnp(h_renamed, "second_largest") == 0.0


# ---------------------------------------------------------------------------
# T4: Parameter-only rename — feature must still return 1.0 (name unchanged)
# ---------------------------------------------------------------------------
class TestParamOnlyRename:
    def test_param_renamed_name_unchanged(self):
        h = "def is_prime(x: int) -> bool:\n    # x renamed from n\n    pass"
        assert _fnp(h, "is_prime") == 1.0

    def test_feature_ignores_param_names(self):
        """The feature only checks the function NAME, not parameter names."""
        h_a = "def compute(a, b, c): pass"
        h_b = "def compute(x, y, z): pass"
        assert _fnp(h_a, "compute") == _fnp(h_b, "compute") == 1.0


# ---------------------------------------------------------------------------
# T5: Unknown entry_point → neutral 0.5
# ---------------------------------------------------------------------------
class TestUnknownEntryPoint:
    def test_none_returns_half(self):
        assert _fnp("some handoff text", None) == 0.5

    def test_empty_string_returns_half(self):
        assert _fnp("some handoff text", "") == 0.5

    def test_zero_string_returns_half(self):
        # Failed joins and sentinel strings should be treated as unknown entry
        # points, not as valid names.
        assert _fnp("some handoff nan text", "nan") == 0.5

    def test_nan_float_returns_half(self):
        assert _fnp("some handoff text", float("nan")) == 0.5


# ---------------------------------------------------------------------------
# T6: Multiple mentions — name must appear as whole word at least once
# ---------------------------------------------------------------------------
class TestMultipleMentions:
    def test_name_in_prose_and_def(self):
        h = "The function is_prime checks primality.\ndef is_prime(n): pass"
        assert _fnp(h, "is_prime") == 1.0

    def test_one_mention_renamed_rest_prose_still_present(self):
        """If corruption renamed only the def line but left prose intact,
        the fixed extractor correctly returns 1.0 (name still present)."""
        h = "is_prime should return True for primes.\ndef is_prime_impl(n): pass"
        assert _fnp(h, "is_prime") == 1.0  # 'is_prime' appears in prose



# ---------------------------------------------------------------------------
# T7: Partial rename — Agent A wrote a synonym def but mentioned the real name
#     in prose. This models the actual clean-zero pattern we discovered:
#     Agent A used an alias (e.g., merge_sorted_lists) in the def line
#     but never wrote the canonical entry_point (merge_sorted) anywhere.
#     The feature should return 0.0 — the name IS missing.
#     This is NOT a bug; the handoff is imprecise but not corrupted.
#     Documented here so we know what boundary behaviour to expect.
# ---------------------------------------------------------------------------
class TestPartialRename:
    def test_alias_def_no_canonical_name_returns_zero(self):
        """Agent used merge_sorted_lists as alias; entry_point=merge_sorted is absent."""
        h = "def merge_sorted_lists(list1, list2): pass\nMerge two sorted lists."
        assert _fnp(h, "merge_sorted") == 0.0, \
            "canonical name absent → 0.0 (alias is a clean-zero, not a corruption)"

    def test_camelcase_alias_returns_zero(self):
        """Agent wrote longestCommonPrefix; entry_point=longest_common_prefix absent."""
        h = "def longestCommonPrefix(strs): pass"
        assert _fnp(h, "longest_common_prefix") == 0.0

    def test_partial_rename_one_mention_survives(self):
        """Def line uses alias but prose still mentions canonical name → 1.0."""
        h = "Implement merge_sorted to combine two sorted lists.\ndef merge_sorted_lists(a, b): pass"
        assert _fnp(h, "merge_sorted") == 1.0

    def test_rename_only_adds_suffix_no_prose_copy(self):
        """
        signature_rename removes ALL occurrences (whole-word regex in the corruptor).
        Scenario: corruptor ran re.sub(r'\\bmerge_sorted\\b', 'merge_sorted_impl', h)
        Result: 'merge_sorted' never appears as whole word → 0.0.
        """
        h = "Implement merge_sorted_impl to combine two sorted lists.\ndef merge_sorted_impl(a, b): pass"
        assert _fnp(h, "merge_sorted") == 0.0


# ---------------------------------------------------------------------------
# Regression: features.csv after recompute_fnp.py
# ---------------------------------------------------------------------------
class TestFeaturesCSV:
    def test_no_neutral_values(self):
        """After recompute_fnp.py, no row should have the old 0.5 neutral value."""
        import pandas as pd
        fc_path = ROOT / "features.csv"
        if not fc_path.exists():
            pytest.skip("features.csv not found")
        fc = pd.read_csv(fc_path)
        neutrals = (fc["function_name_preserved"] == 0.5).sum()
        assert neutrals == 0, \
            f"{neutrals} rows still have function_name_preserved=0.5 (entry_point not resolved)"

    def test_signature_rename_all_zero(self):
        """Every signature_rename row must have function_name_preserved=0.0."""
        import pandas as pd
        fc_path = ROOT / "features.csv"
        if not fc_path.exists():
            pytest.skip("features.csv not found")
        fc = pd.read_csv(fc_path)
        sig = fc[fc["corruption_type"] == "signature_rename"]
        non_zero = (sig["function_name_preserved"] != 0.0).sum()
        assert non_zero == 0, \
            f"{non_zero} signature_rename rows have function_name_preserved != 0"

    def test_clean_rows_majority_one(self):
        """Clean rows should mostly have function_name_preserved=1.0."""
        import pandas as pd
        fc_path = ROOT / "features.csv"
        if not fc_path.exists():
            pytest.skip("features.csv not found")
        fc = pd.read_csv(fc_path)
        clean = fc[fc["corruption_type"] == "none"]
        mean_val = clean["function_name_preserved"].mean()
        assert mean_val > 0.4, \
            f"Clean rows mean={mean_val:.3f} unexpectedly low after fix"
