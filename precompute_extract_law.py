#!/usr/bin/env python3
"""Precompute the 3 Extract-Law variants' rankings over all queries and persist
them to data/extract_law_bench.json, so the Metrics page can show the benchmark
without a live FourCorners call (or token).

Each variant: source text -> FourCorners semantic search (k_results) -> มาตรา ->
Laws set-overlap index -> ranked uids. The subfact variant runs one search per
subfact string (several rankings per query, matching the metrics averaging).

Run with the SSH tunnel up:
    FOURCORNERS_BASE_URL=http://localhost:6767 \
    FOURCORNERS_TOKEN=<token-from-run/.env> \
    python precompute_extract_law.py
or rely on the env fallback in fourcorners.get_config() / run_local.sh.

The output JSON shape matches the metrics `rankings[key]`:
    {key: [ per_query ]}  where per_query[i] = [ranked_uids_subsearch1, ...]
"""

import datetime
import json
import os
import sys

import fourcorners as fc
import search_core as sc

DEPTH = 150
K_RESULTS = 3
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "data", "extract_law_bench.json")


def _texts(approach, row):
    if approach.get("granularity") == "subfact":
        return sc.subfact_list(row)
    return [sc.source_text(approach.get("source_field", "legal_fact_result"), row)]


def main():
    token, base_url = fc.get_config()
    if not token:
        sys.exit("ไม่มี FourCorners token (ตั้ง FOURCORNERS_TOKEN หรือ run/.env TOOLKIT_API_TOKENS)")
    ok, msg = fc.health(base_url)
    print(f"health {base_url}: {msg}")
    if not ok:
        sys.exit("เชื่อม API ไม่ได้ — เปิด SSH tunnel ก่อน")

    bundle = sc.build_indexes()
    qdf = bundle["query_df"]
    variants = [a for a in sc.APPROACHES if a.get("fourcorners")]

    cache = {}  # text -> ranked uids (dedup identical texts across queries/variants)
    calls = 0
    rankings = {}
    for a in variants:
        idx = bundle["indexes"][a["reuses_index"]]
        per_query = []
        for qi, (_, row) in enumerate(qdf.iterrows()):
            rankings_for_q = []
            for text in _texts(a, row) or [""]:
                if text not in cache:
                    calls += 1
                    try:
                        laws, _, _ = fc.extract_laws_from_text(
                            text, token, base_url=base_url, k_results=K_RESULTS)
                        cache[text] = sc.ranked_uids(idx.search(laws, k=DEPTH)) if laws else []
                    except Exception as e:  # noqa: BLE001
                        print(f"  ! {a['key']} q{qi}: {e}")
                        cache[text] = []
                rankings_for_q.append(cache[text])
            per_query.append(rankings_for_q or [[]])
        rankings[a["key"]] = per_query
        print(f"[{a['key']}] done · {len(per_query)} queries · {calls} API calls so far")

    out = {
        "k_results": K_RESULTS,
        "depth": DEPTH,
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "n_queries": len(qdf),
        "api_calls": calls,
        "rankings": rankings,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"wrote {OUT} · {calls} total API calls")


if __name__ == "__main__":
    main()
