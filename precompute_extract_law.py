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
import time

import fourcorners as fc
import search_core as sc

DEPTH = 150
API_K = 20   # pull as many sections as the API allows (k off)
TOP_K = 3    # the "top-3" variants slice to this
RETRIES = 4  # transient 502/connection-reset retries with backoff

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "data", "extract_law_bench.json")

# (source_field, granularity, top-k key, all key) — one API call per text serves both
SOURCES = [
    ("long_text", "case", "extract_law_long", "extract_law_long_all"),
    ("legal_fact_result", "case", "extract_law_legal", "extract_law_legal_all"),
    ("subfacts", "subfact", "extract_law_subfact", "extract_law_subfact_all"),
]


def _texts(source_field, granularity, row):
    if granularity == "subfact":
        return sc.subfact_list(row)
    return [sc.source_text(source_field, row)]


def _full_laws_with_retry(text, token, base_url):
    """Get ALL extracted laws (k=API_K, no cap) with retries on transient errors."""
    last = None
    for attempt in range(RETRIES):
        try:
            laws, _, _ = fc.extract_laws_from_text(
                text, token, base_url=base_url, k_results=API_K, cap=False)
            return laws
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise last


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
    idx = bundle["indexes"]["laws"]

    laws_cache = {}  # text -> full laws list (one API call per unique text)
    calls = 0
    rankings = {}
    for field, gran, top_key, all_key in SOURCES:
        per_query_top, per_query_all = [], []
        for qi, (_, row) in enumerate(qdf.iterrows()):
            top_rk, all_rk = [], []
            for text in _texts(field, gran, row) or [""]:
                if text not in laws_cache:
                    calls += 1
                    try:
                        laws_cache[text] = _full_laws_with_retry(text, token, base_url)
                    except Exception as e:  # noqa: BLE001 — give up after retries
                        print(f"  ! {top_key} q{qi} (after {RETRIES} tries): {e}")
                        laws_cache[text] = []
                laws = laws_cache[text]
                top_rk.append(sc.ranked_uids(idx.search(laws[:TOP_K], k=DEPTH)) if laws else [])
                all_rk.append(sc.ranked_uids(idx.search(laws, k=DEPTH)) if laws else [])
            per_query_top.append(top_rk or [[]])
            per_query_all.append(all_rk or [[]])
        rankings[top_key] = per_query_top
        rankings[all_key] = per_query_all
        print(f"[{field}] done · top-3 + all · {calls} API calls so far")

    out = {
        "top_k": TOP_K,
        "api_k": API_K,
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
