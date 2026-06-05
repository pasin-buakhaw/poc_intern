#!/usr/bin/env python3
"""Print the full benchmark table (CLI) — non-FC approaches computed live, the 3
Extract-Law variants loaded from data/extract_law_bench.json — each evaluated
against its OWN relevance basis. Mirrors pages/7_Metrics_Summary.py.

    python show_benchmark.py [k]      # default k=3
"""

import json
import math
import os
import sys

import search_core as sc

DEPTH = 150
_HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(_HERE, "data", "extract_law_bench.json")


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def _approach_queries(a, row):
    if a["key"] == "subfacts":
        return sc.subfact_list(row)
    if a.get("match") == "set":
        return [sc.query_items(a["key"], row)]
    return [sc.query_text(a["key"], row)]


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    b = sc.build_indexes()
    qdf = b["query_df"]
    bench = json.load(open(BENCH, encoding="utf-8")) if os.path.exists(BENCH) else {"rankings": {}}

    rankings = {}
    included = []
    for a in sc.APPROACHES:
        if a.get("fourcorners"):
            pre = bench.get("rankings", {}).get(a["key"])
            if pre is None:
                continue
            rankings[a["key"]] = pre
        else:
            idx = b["indexes"][a["key"]]
            rankings[a["key"]] = [
                [sc.ranked_uids(idx.search(q, k=DEPTH)) for q in _approach_queries(a, row)]
                for _, row in qdf.iterrows()
            ]
        included.append(a)

    sc_col, thr = "relevance_score", 1  # common basis for every approach
    print(f"\nBenchmark @k={k}  (all approaches vs {sc_col}>={thr})\n")
    hdr = f"{'approach':28} {'nDCG':>6} {'Hit':>6} {'Recall':>7} {'Prec':>6} {'MRR':>6}"
    print(hdr); print("-" * len(hdr))
    table = []
    for a in included:
        H, R, P, M, N = [], [], [], [], []
        for i, (_, row) in enumerate(qdf.iterrows()):
            rl = rankings[a["key"]][i]
            rel = sc.relevant_uids(row, score_col=sc_col, thr=thr)
            gr = sc.graded_rel(row, score_col=sc_col)
            H.append(_mean([sc.hit_at_k(r, rel, k) for r in rl]))
            R.append(_mean([sc.recall_at_k(r, rel, k) for r in rl]))
            P.append(_mean([sc.precision_at_k(r, rel, k) for r in rl]))
            M.append(_mean([sc.mrr_at_k(r, rel, k) for r in rl]))
            N.append(_mean([sc.ndcg_at_k(r, gr, k) for r in rl]))
        table.append((a["label"], _mean(N), _mean(H), _mean(R), _mean(P), _mean(M)))
    for lab, n, h, r, p, m in sorted(table, key=lambda t: -t[2]):
        print(f"{lab:28} {n:6.3f} {h:6.3f} {r:7.3f} {p:6.3f} {m:6.3f}")
    print()


if __name__ == "__main__":
    main()
