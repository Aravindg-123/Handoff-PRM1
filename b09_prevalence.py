"""B9: prevalence sweep at the frozen B3 threshold; decision curve; downstream dPass@1 and break-even."""
import json, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common.data import load_v2, RES
from common.stats_utils import block_metrics, block_precision

d, _ = load_v2(); y = d.label.values
ns = pickle.load(open(RES / "b05_nested.pkl", "rb")); s = ns["platt"]
t = json.load(open(RES / "frozen_threshold.json"))["threshold_platt"]
m = block_metrics(y, s, t); tpr, fbr = m["tpr"], m["fbr"]

# task-bootstrap draws of (tpr, fbr)
idx = [np.asarray(v) for v in d.groupby("task_id").indices.values()]; rng = np.random.RandomState(42)
B = []
for _ in range(5000):
    p = np.concatenate([idx[i] for i in rng.choice(len(idx), len(idx))]); mm = block_metrics(y[p], s[p], t)
    B.append((mm["tpr"], mm["fbr"]))
B = np.array(B)

# live quantities (94-task held-out pilot): r = P(fallback ok | blocked & would fail), h = P(fallback fails | blocked & would pass)
l = pd.read_csv("live_pilot_results.csv"); b = l[l.gate_decision == "BLOCK"]
f, p_ = b[~b.ungated_passed], b[b.ungated_passed]
r_hat, h_hat = f.gated_passed.mean(), (~p_.gated_passed).mean()
live = dict(n_blocked_would_fail=len(f), recoveries=int(f.gated_passed.sum()), r=r_hat,
            n_blocked_would_pass=len(p_), harms=int((~p_.gated_passed).sum()), h=h_hat)
print(live)
rb = rng.choice(f.gated_passed.values, (5000, len(f))).mean(1)
hb = (~rng.choice(p_.gated_passed.values, (5000, len(p_)))).mean(1)

rows = []
q = lambda a: f"{np.mean(a):.3f} [{np.percentile(a, 2.5):.3f}, {np.percentile(a, 97.5):.3f}]"
for pi in [0.01, 0.05, 0.20, 0.50]:
    prec = block_precision(pi, B[:, 0], B[:, 1])
    npv = (1 - pi) * (1 - B[:, 1]) / ((1 - pi) * (1 - B[:, 1]) + pi * (1 - B[:, 0]))
    dpass = pi * B[:, 0] * rb - (1 - pi) * B[:, 1] * hb
    rows.append(dict(prevalence_bad=pi, block_precision=q(prec), npv_of_passes=q(npv),
                     false_blocks_per_true_catch=round((1 - prec.mean()) / prec.mean(), 2), dPass1_pp=q(100 * dpass)))
sw = pd.DataFrame(rows); sw.to_csv(RES / "b09_prevalence_sweep.csv", index=False); print(sw.to_string())
be = lambda tp, fb, r, h: fb * h / (tp * r + fb * h)
be_pt = be(tpr, fbr, r_hat, h_hat); be_b = be(B[:, 0], B[:, 1], np.maximum(rb, 1e-9), hb)
res = dict(threshold=t, tpr=tpr, fbr=fbr, tpr_ci=np.percentile(B[:, 0], [2.5, 97.5]).tolist(),
           fbr_ci=np.percentile(B[:, 1], [2.5, 97.5]).tolist(), **live,
           breakeven_prevalence=be_pt, breakeven_ci=np.percentile(be_b, [2.5, 97.5]).tolist(),
           precision_at_5pct=block_precision(.05, tpr, fbr), precision_at_20pct=block_precision(.2, tpr, fbr))
json.dump(res, open(RES / "b09_summary.json", "w"), indent=1); print(json.dumps(res, indent=1))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2)); g = np.linspace(.001, .6, 200)
ax[0].plot(g, block_precision(g, tpr, fbr), label="precision of blocks")
ax[0].plot(g, (1 - g) * (1 - fbr) / ((1 - g) * (1 - fbr) + g * (1 - tpr)), label="NPV of passes")
ax[0].set_xlabel("bad-handoff prevalence"); ax[0].legend(); ax[0].set_title("B9 prevalence sweep")
ax[1].plot(g, 100 * (g * tpr * r_hat - (1 - g) * fbr * h_hat)); ax[1].axhline(0, c="k", ls=":")
ax[1].axvline(be_pt, c="r", ls="--", label=f"break-even {be_pt:.3f}"); ax[1].legend()
ax[1].set_xlabel("prevalence"); ax[1].set_ylabel("dPass@1 (pp)"); ax[1].set_title("Downstream impact")
w = np.logspace(-2, 1, 200)
for pi in [.01, .05, .2, .5]:
    ax[2].plot(w, pi * tpr - (1 - pi) * fbr * w, label=f"gate, pi={pi}")
ax[2].plot(w, .05 - .95 * w, "k--", lw=.8, label="block all (pi=.05)"); ax[2].axhline(0, c="gray")
ax[2].set_xscale("log"); ax[2].set_ylim(-.15, .25); ax[2].set_xlabel("cost(false block)/cost(miss)"); ax[2].set_ylabel("net benefit")
ax[2].legend(fontsize=7); ax[2].set_title("Decision curve")
plt.tight_layout(); plt.savefig(RES / "figures" / "b09_prevalence.png", dpi=150)
