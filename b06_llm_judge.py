"""B6 (5): LLM-judge baseline, done properly. NOT RUN in this session (no GEMINI_API_KEY available).

Design: temperature 0, fixed model id, three prompts (generic / targeted at the 4 known failure modes /
generic-with-raw-problem), 0-100 sufficiency score -> AUROC. Resumable (appends to results/b06_llm_judge_raw.csv).
Run:  python b06_llm_judge.py --model gemini-2.5-flash --prompt generic|targeted|with_problem
Then: python b06_llm_judge.py --summarize
"""
import argparse, os, re, time
import pandas as pd
from common.data import load_v2, load_raw, RES

PROMPTS = {
    "generic": "You are reviewing a handoff message written by a planning agent for a coding agent.\n"
               "Rate from 0 to 100 how sufficient the HANDOFF is for the coding agent to implement a correct solution "
               "(100 = fully sufficient). Reply with ONLY the integer.\n\nHANDOFF:\n{handoff}",
    "targeted": "You review handoffs for a coding agent. Check for: (1) dropped constraints or edge cases, (2) inverted or "
                "changed objective, (3) wrong algorithm named, (4) wrong function name/signature. Rate 0-100 how sufficient "
                "and correct the HANDOFF is (100 = no defect). Reply with ONLY the integer.\n\nHANDOFF:\n{handoff}",
    "with_problem": "Original problem:\n{problem}\n\nHandoff written from it:\n{handoff}\n\nRate 0-100 how faithfully and "
                    "sufficiently the handoff conveys the problem for a coding agent (100 = fully). Reply with ONLY the integer.",
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--prompt", choices=PROMPTS); ap.add_argument("--summarize", action="store_true"); a = ap.parse_args()
    out = RES / "b06_llm_judge_raw.csv"
    if a.summarize:
        from common.stats_utils import auc, task_bootstrap
        d, _ = load_v2(); r = pd.read_csv(out); rows = []
        for (m, p), g in r.groupby(["model", "prompt"]):
            g = g.set_index("row_id").reindex(d.index)
            lo, hi, _, pt = task_bootstrap(d, g.score.values / 100, auc, n_boot=5000)
            rows.append(dict(model=m, prompt=p, auroc=pt, lo=lo, hi=hi, mean_latency_ms=g.latency_ms.mean()))
        print(pd.DataFrame(rows).round(4)); return
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    raw = load_raw(); done = set(pd.read_csv(out).query("model==@a.model and prompt==@a.prompt").row_id) if out.exists() else set()
    for i, r in raw.iterrows():
        if i in done: continue
        t0 = time.perf_counter()
        resp = client.models.generate_content(model=a.model, contents=PROMPTS[a.prompt].format(handoff=r.handoff, problem=r.problem),
                                              config=types.GenerateContentConfig(temperature=0))
        ms = (time.perf_counter() - t0) * 1e3
        m = re.search(r"\d+", resp.text or ""); s = min(100, int(m.group())) if m else None
        pd.DataFrame([dict(row_id=i, model=a.model, prompt=a.prompt, score=s, latency_ms=ms)]).to_csv(
            out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    main()
