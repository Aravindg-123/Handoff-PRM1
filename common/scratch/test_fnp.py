"""
Unit tests for compute_function_name_preserved + signature_rename corruption.
Covers the four specific scenarios from B2.
"""
import sys, re
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from feature_extraction import compute_function_name_preserved
from corruption import signature_rename

PASS = "PASS"
FAIL = "FAIL"

results = []

def check(name, got, expected, note=""):
    ok = got == expected
    results.append((PASS if ok else FAIL, name, got, expected, note))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fnp(handoff, entry_point):
    return compute_function_name_preserved(handoff, reference_text="", entry_point=entry_point)

# ---------------------------------------------------------------------------
# T1: Clean handoff — entry_point in handoff → should return 1.0
# ---------------------------------------------------------------------------
clean_h = (
    "Write a Python function `is_prime(n)` that returns True if n is prime.\n"
    "def is_prime(n: int) -> bool:\n"
    "    # your code here\n"
    "Handle edge cases: n <= 1 returns False."
)
check("T1_clean_explicit_def",   fnp(clean_h, "is_prime"), 1.0,
      "entry_point clearly in handoff")

# ---------------------------------------------------------------------------
# T2: Name fully renamed in def line — but original name still in prose
# This is the ACTUAL signature_rename output: regex replaces \b word boundaries
# ---------------------------------------------------------------------------
renamed_h = signature_rename(clean_h, entry_point="is_prime")
print(f"\nT2 renamed handoff:\n{renamed_h}\n")
orig_in_prose = "is_prime" in renamed_h and "is_prime_impl" in renamed_h
print(f"Original name still appears in prose: {orig_in_prose}")
print(f"Wrong name (is_prime_impl) appears: {'is_prime_impl' in renamed_h}")

# Current extractor: checks entry_point IN handoff (substring)
got_T2 = fnp(renamed_h, "is_prime")
print(f"fnp() returns: {got_T2}  (expected 0.0)")
check("T2_full_rename_current_extractor",
      got_T2, 0.0,  # expected 0.0 — but will it actually be 0.0 or 1.0?
      "original name must NOT appear if extractor is correct")

# ---------------------------------------------------------------------------
# T3: Entry_point appears only in prose, not in any def line
# (Partial rename — rename applied only once; original still in prose)
# ---------------------------------------------------------------------------
partial_h = (
    "Write is_prime function. The function name in the signature is different.\n"
    "def is_prime_impl(n: int) -> bool:\n"
    "    pass"
)
got_T3 = fnp(partial_h, "is_prime")
print(f"\nT3 partial rename: fnp()={got_T3}  (original still in prose)")
check("T3_partial_rename_in_prose", got_T3, 1.0,
      "current extractor returns 1.0 because 'is_prime' is a substring of 'is_prime_impl' — FALSE POSITIVE")

# ---------------------------------------------------------------------------
# T4: Parameter-only rename (function name same, only param name changed)
# Does the feature even look at params?
# ---------------------------------------------------------------------------
param_renamed = (
    "def is_prime(x: int) -> bool:\n"   # param renamed n -> x
    "    # checks if x is prime\n"
    "Handle edge cases carefully."
)
got_T4 = fnp(param_renamed, "is_prime")
check("T4_param_only_rename", got_T4, 1.0,
      "feature ignores params (correct: feature only checks name)")

# ---------------------------------------------------------------------------
# T5: entry_point=None → neutral 0.5
# ---------------------------------------------------------------------------
check("T5_no_entry_point", fnp(clean_h, None), 0.5,
      "unknown entry_point returns neutral 0.5")

check("T5b_empty_entry_point", fnp(clean_h, ""), 0.5,
      "empty string entry_point also returns 0.5")

# ---------------------------------------------------------------------------
# T6: Substring trap — 'sort' in 'sorted', 'sort_impl' in 'sorted_impl'
# The \b word boundary in corruption but 'in' substring check in extractor
# ---------------------------------------------------------------------------
h_sort_clean = "Implement `sort(arr)` to sort an array.\ndef sort(arr): pass"
h_sort_renamed = signature_rename(h_sort_clean, entry_point="sort")
print(f"\nT6 sort rename:\nOriginal: {h_sort_clean}")
print(f"Renamed:  {h_sort_renamed}")
in_check = "sort" in h_sort_renamed   # 'sort' IS in 'sort_impl' and in 'sorted'
print(f"'sort' in renamed handoff (substring): {in_check}")
got_T6 = fnp(h_sort_renamed, "sort")
check("T6_substring_trap", got_T6, 0.0,
      "'sort' substring appears in 'sort_impl', so extractor returns 1.0 — BUG if sort replaced everywhere")

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
print("\n" + "="*72)
print("UNIT TEST RESULTS")
print("="*72)
passed = failed = 0
for status, name, got, expected, note in results:
    sym = "OK" if status == PASS else "BUG"
    print(f"  [{sym}] {name}")
    if status == FAIL:
        print(f"       got={got!r}  expected={expected!r}")
    if note:
        print(f"       note: {note}")
    if status == PASS:
        passed += 1
    else:
        failed += 1
print(f"\n{passed} passed, {failed} failed")
print()

# ---------------------------------------------------------------------------
# Diagnostic: how many signature_rename rows in corrupted_handoffs.csv
# would the FIXED extractor correctly classify?
# Fixed: use word-boundary check instead of substring 'in'
# ---------------------------------------------------------------------------
print("="*72)
print("DIAGNOSTIC: corrupted_handoffs.csv signature_rename rows")
print("="*72)
ch = pd.read_csv('corrupted_handoffs.csv')
sig = ch[ch['corruption_type'] == 'signature_rename'].copy()
print(f"n signature_rename rows: {len(sig)}")

def fnp_fixed(handoff, entry_point):
    """Correct extractor: use word-boundary to avoid substring false positives."""
    if not entry_point:
        return 0.5
    # Must find the name as a whole word (not as part of entry_point_impl)
    pattern = r'\b' + re.escape(entry_point) + r'\b'
    return 1.0 if re.search(pattern, handoff) else 0.0

sig['ep'] = sig['entry_point'].astype(str)
sig['fnp_buggy']   = sig.apply(lambda r: fnp(r['handoff'], r['ep']), axis=1)
sig['fnp_fixed']   = sig.apply(lambda r: fnp_fixed(r['handoff'], r['ep']), axis=1)
sig['wrong_name']  = sig['ep'].apply(lambda e: e + '_impl')
sig['wrong_in_h']  = sig.apply(lambda r: r['wrong_name'] in r['handoff'], axis=1)

print("\nBuggy extractor (substring 'in'):")
print("  value_counts:", sig['fnp_buggy'].value_counts().to_dict())
print(f"  mean: {sig['fnp_buggy'].mean():.4f}")
print()
print("Fixed extractor (word-boundary):")
print("  value_counts:", sig['fnp_fixed'].value_counts().to_dict())
print(f"  mean: {sig['fnp_fixed'].mean():.4f}")
print()

# Show rows where buggy=1.0 but fixed=0.0 (the false positives we fixed)
fp = sig[(sig['fnp_buggy'] == 1.0) & (sig['fnp_fixed'] == 0.0)]
print(f"Rows where buggy=1 but fixed=0 (TRUE false positives from substring trap): {len(fp)}")
fn = sig[(sig['fnp_buggy'] == 0.0) & (sig['fnp_fixed'] == 1.0)]
print(f"Rows where buggy=0 but fixed=1 (genuine preserve despite rename): {len(fn)}")

# Show cases where orig name is still in prose even after rename
still_orig = sig[sig.apply(lambda r: re.search(r'\b' + re.escape(r['ep']) + r'\b', r['handoff']) is not None, axis=1)]
print(f"\nRows where original name still appears (word-boundary) after rename: {len(still_orig)}")
print("Examples:")
for _, row in still_orig.head(3).iterrows():
    snippet = row['handoff'][:400]
    print(f"  task={row['task_id']}  ep={row['ep']!r}")
    # Find where in the handoff the original name appears
    for m in re.finditer(r'\b' + re.escape(row['ep']) + r'\b', row['handoff']):
        ctx = row['handoff'][max(0, m.start()-40):m.end()+40]
        print(f"    context: ...{ctx!r}...")
    print()
