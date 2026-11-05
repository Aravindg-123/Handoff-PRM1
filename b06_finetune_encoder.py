"""B6 (4): fine-tuned small encoder (MiniLM-L6) on (problem, handoff) pairs. Task-grouped 5-fold CV, 3 seeds.
Lower 3 of 6 layers + embeddings frozen, 4 epochs, lr 5e-5, max_len 384."""
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from common.cv_utils import task_folds
from common.data import load_v2, load_raw, RES
from common.stats_utils import auc, task_bootstrap, block_metrics, threshold_fbr_cap

dev = "cuda" if torch.cuda.is_available() else "cpu"
NAME = "sentence-transformers/all-MiniLM-L6-v2"
d, _ = load_v2(); raw = load_raw(); y = d.label.values
tok = AutoTokenizer.from_pretrained(NAME)
enc = tok(raw.problem.tolist(), raw.handoff.tolist(), truncation="only_second", max_length=384,
          padding="max_length", return_tensors="pt")
Y = torch.tensor(y)


def run(seed, k, fold):
    torch.manual_seed(seed)
    m = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=2).to(dev)
    for p in m.bert.embeddings.parameters(): p.requires_grad = False
    for l in m.bert.encoder.layer[:3]:
        for p in l.parameters(): p.requires_grad = False
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=5e-5, weight_decay=0.01)
    tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
    g = np.random.RandomState(seed)
    m.train()
    for ep in range(4):
        g.shuffle(tr)
        for i in range(0, len(tr), 16):
            b = tr[i:i + 16]
            out = m(input_ids=enc["input_ids"][b].to(dev), attention_mask=enc["attention_mask"][b].to(dev),
                    token_type_ids=enc["token_type_ids"][b].to(dev), labels=Y[b].to(dev))
            out.loss.backward(); opt.step(); opt.zero_grad()
    m.eval(); ps = []
    with torch.no_grad():
        for i in range(0, len(te), 32):
            b = te[i:i + 32]
            lg = m(input_ids=enc["input_ids"][b].to(dev), attention_mask=enc["attention_mask"][b].to(dev),
                   token_type_ids=enc["token_type_ids"][b].to(dev)).logits
            ps.append(torch.softmax(lg, -1)[:, 1].cpu().numpy())
    return te, np.concatenate(ps)


S = np.zeros((3, len(d)))
for si, seed in enumerate([0, 1, 2]):
    fold = task_folds(d, 5, seed)
    for k in range(5):
        te, p = run(seed, k, fold); S[si, te] = p
    print("seed", seed, "AUROC", round(auc(y, S[si]), 4), flush=True)
s = S.mean(0); np.save(RES / "b06_finetune_oof.npy", S)
lo, hi, nv, pt = task_bootstrap(d, s, auc, n_boot=5000); t = threshold_fbr_cap(y, s, .05); m = block_metrics(y, s, t)
res = pd.DataFrame([dict(method="fine-tuned MiniLM-L6 (3 seeds avg)", auroc=pt, ci=f"[{lo:.3f}, {hi:.3f}]",
                         per_seed=str([round(auc(y, S[i]), 4) for i in range(3)]),
                         detection=m["tpr"], false_block=m["fbr"], op_threshold=round(t, 4))]).round(4)
res.to_csv(RES / "b06_baseline_finetune.csv", index=False); print(res.T)
