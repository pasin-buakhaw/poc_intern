"""Metrics Summary — compare every retrieval approach over the full query set.

Each approach is evaluated the SAME way it is searched on its page:
  - Long text / Legal fact : one query = that whole field
  - Crimes / Laws          : one query = all the query's keywords (= tag-all)
  - Subfacts               : one search PER single subfact, averaged per query
                             (matches the page where you search one subfact)
"""

import math

import pandas as pd
import streamlit as st

import fourcorners as fc
import search_core as sc

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


@st.cache_data(show_spinner="Calling FourCorners semantic search per query ...")
def fourcorners_rankings(approach_key, token, base_url, k_results,
                         depth=RETRIEVAL_DEPTH):
    """extract_law rankings: per query, text -> API -> laws -> Laws index ranking.

    Cached on (token, base, params) so toggling controls doesn't re-hit the API.
    A query whose API call fails contributes an empty ranking (counts as a miss).
    """
    b = sc.build_indexes()
    a = sc.APPROACH_BY_KEY[approach_key]
    idx = b["indexes"][a["reuses_index"]]
    field = a.get("source_field", "legal_fact_result")
    per_query = []
    for _, row in b["query_df"].iterrows():
        text = sc.source_text(field, row)
        try:
            laws, _, _ = fc.extract_laws_from_text(
                text, token, base_url=base_url, k_results=k_results)
            ranked = sc.ranked_uids(idx.search(fc.laws_to_query(laws), k=depth)) if laws else []
        except Exception:  # noqa: BLE001 — a failed/empty query just scores as a miss
            ranked = []
        per_query.append([ranked])
    return per_query


rankings = all_rankings()

# ---- opt-in: include the FourCorners "Extract Law" approach (needs token) -----
included = list(STD_APPROACHES)
if FC_APPROACHES:
    st.divider()
    st.markdown("#### Extract Law from text (FourCorners semantic search)")
    token, base_url = fc.render_token_input(st, key_prefix="metrics_fc")
    run_fc = st.checkbox(
        f"รวมในตาราง — เรียก `search_legal_corpus` {len(qdf)} ครั้ง (ช้า, ต้องมี token)",
        value=False, disabled=not token, key="run_fc_metrics")
    fc_k = st.slider("k (จำนวนมาตราที่ดึงจาก search)", 3, 20, 10, key="fc_k_metrics")
    st.caption("ข้อความทั้งก้อนถูกส่งเป็น topic เดียวเข้า semantic search")
    if run_fc and token:
        for a in FC_APPROACHES:
            rankings[a["key"]] = fourcorners_rankings(
                a["key"], token, base_url, fc_k)
            included.append(a)
    else:
        st.caption("ยังไม่รวม Extract Law ในตาราง (ติ๊กช่องด้านบนหลังใส่ token เพื่อคิดคะแนน)")
    st.divider()

# --- controls -----------------------------------------------------------------
# relevance_score มาจาก 2 มิติ: MF = subfact ตรง, LF = legal fact ตรง
#   0 = ไม่เกี่ยว · 1 = MF อย่างเดียว · 2 = LF อย่างเดียว · 3 = ตรงทั้ง MF+LF
_BASIS = {
    "ทั้ง subfact + legal fact เกี่ยวข้อง": ("relevance_score", 3),
    "อย่างน้อย legal fact ต้องเกี่ยวข้อง": ("relevance_score", 2),
    "เกี่ยวข้องกับ subfact หรือ legal factก็ได้": ("relevance_score", 1),
    "แค่ subfact เกี่ยวข้องเท่านั้น": ("subfacts_score", 1),
    "แค่ legal fact เกี่ยวข้องเท่านั้น": ("legal_fact_result_score", 1),
}
c1, c2 = st.columns(2)
k = c1.selectbox("k (top-k)", [1, 3, 4, 5, 10], index=2)
basis = c2.selectbox("วิธีการนับ candidate ว่าเกี่ยวข้อง(relevant)",
                     list(_BASIS.keys()), index=1)
score_col, thr = _BASIS[basis]

st.caption(
    "ℹ️ **relevance score** มาจาก 2 มิติ — **MF** = ข้อเท็จจริงย่อย (subfact) ตรง, "
    "**LF** = ผลข้อเท็จจริงทางกฎหมาย (legal fact) ตรง · "
    "**0** ไม่เกี่ยว · **1** MF อย่างเดียว · **2** LF อย่างเดียว · **3** ตรงทั้งคู่ "
    "(ตัวเลือกด้านบนใช้กับ Hit/Recall/Precision/MRR; nDCG ใช้ score 0–3 เป็นน้ำหนักเสมอ)"
)


def _mean(xs):
    xs = [x for x in xs
          if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


# --- compute per-approach averages --------------------------------------------
rows = []
for a in included:
    hit, rec, prec, mrr, ndcg = [], [], [], [], []
    for i, (_, row) in enumerate(qdf.iterrows()):
        ranking_list = rankings[a["key"]][i]   # list of ranked-uid lists (1+ per query)
        rel = sc.relevant_uids(row, score_col=score_col, thr=thr)
        graded = sc.graded_rel(row)            # nDCG always uses graded relevance_score
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

num_cols = [c for c in df.columns if c not in ("approach", "granularity")]
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
- **Subfacts** วัดแบบค้นทีละ subfact แล้วเฉลี่ยต่อ query (ตรงกับหน้า approach) — ค่าจึงต่ำกว่าเดิมที่รวมทุก subfact
- relevant (binary) = `{basis}` · nDCG ใช้ graded `relevance_score` เสมอ
"""
    )
