"""Metrics Summary — compare every retrieval approach over the full query set.

Each approach is evaluated the SAME way it is searched on its page:
  - Long text / Legal fact : one query = that whole field
  - Crimes / Laws          : one query = all the query's keywords (= tag-all)
  - Subfacts               : one search PER single subfact, averaged per query
                             (matches the page where you search one subfact)
"""

import json
import math
import os

import pandas as pd
import streamlit as st

import fourcorners as fc
import search_core as sc

_BENCH_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "extract_law_bench.json")

st.set_page_config(page_title="Metrics Summary · Retrieval PoC", layout="wide")

bundle = sc.build_indexes()
qdf = bundle["query_df"]

# approaches that retrieve from a BM25 index directly (no network)
STD_APPROACHES = [a for a in sc.APPROACHES if not a.get("fourcorners")]
FC_APPROACHES = [a for a in sc.APPROACHES if a.get("fourcorners")]

st.title("Metrics Summary")
st.caption(
    f"รันทุก approach กับ {len(qdf)} queries (`{bundle['query_src']}`) ค้นจาก corpus เต็ม "
    f"{len(bundle['cand_df'])} คดี — วัดแบบเดียวกับที่หน้า approach ค้นจริง "
    "(Subfacts = ค้นทีละ subfact แล้วเฉลี่ยต่อ query), ผล dedup เป็นอันดับ 'คดี' (uid) ก่อนคิดคะแนน"
)

RETRIEVAL_DEPTH = 150


def _approach_queries(approach, row):
    """ข้อความค้นของ approach นี้ต่อ 1 query — เหมือนที่หน้าเว็บทำ.

    Subfacts -> หลาย query (1 ต่อ subfact); ที่เหลือ -> 1 query (field รวม).
    """
    if approach["key"] == "subfacts":
        out = []
        for e in sc.parse_cell(row.get("subfacts", "")) or []:
            if not isinstance(e, dict):
                continue
            subs = e.get("subfacts") or []
            if isinstance(subs, str):
                subs = [subs]
            out.extend(s.strip() for s in (str(x) for x in subs) if s.strip())
        return out
    if approach.get("match") == "set":
        return [sc.query_items(approach["key"], row)]  # one query = the item list
    return [sc.query_text(approach["key"], row)]


@st.cache_data(show_spinner="Running retrieval for all approaches ...")
def all_rankings(depth=RETRIEVAL_DEPTH):
    b = sc.build_indexes()
    out = {}
    for a in STD_APPROACHES:
        idx = b["indexes"][a["key"]]
        per_query = []
        for _, row in b["query_df"].iterrows():
            qs = _approach_queries(a, row)
            per_query.append([sc.ranked_uids(idx.search(q, k=depth)) for q in qs])
        out[a["key"]] = per_query
    return out


def _fc_texts(approach, row):
    """Source text(s) for an Extract-Law variant: per-subfact list or single field."""
    if approach.get("granularity") == "subfact":
        return sc.subfact_list(row)
    return [sc.source_text(approach.get("source_field", "legal_fact_result"), row)]


@st.cache_data(show_spinner="Calling FourCorners semantic search per query ...")
def fourcorners_rankings(approach_key, token, base_url, k_results,
                         depth=RETRIEVAL_DEPTH):
    """Live Extract-Law rankings: per query, text(s) -> API -> laws -> set-overlap.

    Subfact variant runs one search per subfact (-> several rankings/query). Cached
    on (token, base, k). A failed/empty search contributes an empty ranking (miss).
    """
    b = sc.build_indexes()
    a = sc.APPROACH_BY_KEY[approach_key]
    idx = b["indexes"][a["reuses_index"]]
    cache = {}
    per_query = []
    for _, row in b["query_df"].iterrows():
        rankings_for_q = []
        for text in _fc_texts(a, row) or [""]:
            if text not in cache:
                try:
                    laws, _, _ = fc.extract_laws_from_text(
                        text, token, base_url=base_url, k_results=k_results)
                    cache[text] = sc.ranked_uids(idx.search(laws, k=depth)) if laws else []
                except Exception:  # noqa: BLE001 — failed/empty search = miss
                    cache[text] = []
            rankings_for_q.append(cache[text])
        per_query.append(rankings_for_q or [[]])
    return per_query


@st.cache_data(show_spinner=False)
def load_precomputed():
    """Load data/extract_law_bench.json (precomputed Extract-Law rankings) if present."""
    if not os.path.exists(_BENCH_FILE):
        return None
    with open(_BENCH_FILE, encoding="utf-8") as f:
        return json.load(f)


rankings = all_rankings()
included = list(STD_APPROACHES)

# ---- Extract Law variants: precomputed by default, optional live recompute ----
if FC_APPROACHES:
    st.divider()
    st.markdown("#### Extract Law from text (FourCorners semantic search)")
    bench = load_precomputed()
    token, base_url = fc.render_token_input(st, key_prefix="metrics_fc")
    run_fc = st.checkbox(
        f"🔁 recompute live — เรียก API ใหม่ทุก query (ช้า, ต้องมี token)",
        value=False, disabled=not token, key="run_fc_metrics")
    fc_k = st.slider("k (จำนวนมาตราที่ดึงจาก search)", 3, 20, 3, key="fc_k_metrics")

    if run_fc and token:
        for a in FC_APPROACHES:
            rankings[a["key"]] = fourcorners_rankings(a["key"], token, base_url, fc_k)
            included.append(a)
        st.caption(f"คิดสดจาก API (k_results={fc_k})")
    elif bench and bench.get("rankings"):
        for a in FC_APPROACHES:
            pre = bench["rankings"].get(a["key"])
            if pre is not None:
                rankings[a["key"]] = pre
                included.append(a)
        st.caption(f"ใช้ผล precomputed (k_results={bench.get('k_results')}, "
                   f"สร้างเมื่อ {bench.get('generated', '-')}) · ติ๊กด้านบนเพื่อคิดสดด้วย token")
    else:
        st.caption("ยังไม่มีผล precomputed — รัน `precompute_extract_law.py` หรือติ๊ก recompute live")
    st.divider()

# --- controls: only k (every approach is scored on relevance_score >= 1) -------
RELEVANCE = ("relevance_score", 1)
k = st.selectbox("k (top-k)", [1, 3, 4, 5, 10], index=2)
st.caption(
    "ℹ️ ทุก approach วัด relevant ด้วยเกณฑ์เดียวกัน: **`relevance_score ≥ 1`** "
    "(= candidate ที่ถูก label ว่าเกี่ยวข้องอย่างน้อย 1 มิติ) — เทียบ apples-to-apples · "
    "nDCG ใช้ graded `relevance_score` (1/2/3) เป็นน้ำหนัก"
)


def _mean(xs):
    xs = [x for x in xs
          if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


# --- compute per-approach averages (all vs relevance_score >= 1) ---------------
score_col, thr = RELEVANCE
rows = []
for a in included:
    hit, rec, prec, mrr, ndcg = [], [], [], [], []
    for i, (_, row) in enumerate(qdf.iterrows()):
        ranking_list = rankings[a["key"]][i]   # list of ranked-uid lists (1+ per query)
        rel = sc.relevant_uids(row, score_col=score_col, thr=thr)
        graded = sc.graded_rel(row, score_col=score_col)
        # metric per sub-search, then average within this query
        hit.append(_mean([sc.hit_at_k(r, rel, k) for r in ranking_list]))
        rec.append(_mean([sc.recall_at_k(r, rel, k) for r in ranking_list]))
        prec.append(_mean([sc.precision_at_k(r, rel, k) for r in ranking_list]))
        mrr.append(_mean([sc.mrr_at_k(r, rel, k) for r in ranking_list]))
        ndcg.append(_mean([sc.ndcg_at_k(r, graded, k) for r in ranking_list]))
    rows.append({
        "approach": a["label"],
        "granularity": a["granularity"],
        f"nDCG@{k}": round(_mean(ndcg), 3),
        f"Hit@{k}": round(_mean(hit), 3),
        f"Recall@{k}": round(_mean(rec), 3),
        f"Precision@{k}": round(_mean(prec), 3),
        f"MRR@{k}": round(_mean(mrr), 3),
    })

df = pd.DataFrame(rows).sort_values(f"nDCG@{k}", ascending=False).reset_index(drop=True)

# --- headline: best approach by the common comparator (nDCG@k) ----------------
best = df.iloc[0]
st.success(f"approach ที่ดีสุด (nDCG@{k}) = **{best['approach']}**  ·  nDCG@{k} = {best[f'nDCG@{k}']}")

num_cols = [c for c in df.columns if c not in ("approach", "granularity", "relevant basis")]
st.dataframe(
    df.style.highlight_max(subset=num_cols, color="#91e1a0", axis=0).format(precision=3),
    hide_index=True, use_container_width=True,
)

with st.expander("นิยาม metric"):
    st.markdown(
        f"""
- **nDCG@{k}** — ตัวกลางเปรียบเทียบ (ใช้ graded `relevance_score` 1/2/3 เป็นน้ำหนัก) → บอกว่าวิธีไหนจัดอันดับดีสุด
- **Hit@{k}** — top-{k} มี candidate ที่ relevant อย่างน้อย 1 (0/1) เฉลี่ยทุก query
- **Recall@{k}** — สัดส่วน relevant ที่เจอใน top-{k}
- **Precision@{k}** — สัดส่วน top-{k} ที่ relevant
- **MRR@{k}** — 1/อันดับของ relevant ตัวแรก
- **Subfacts / Extract Law (subfact)** วัดแบบค้นทีละ subfact แล้วเฉลี่ยต่อ query (ตรงกับหน้า approach)
- ทุก approach วัด relevant ด้วย **`relevance_score ≥ 1`** เหมือนกัน · nDCG ใช้ graded `relevance_score` (1/2/3)
"""
    )
