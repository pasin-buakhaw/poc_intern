"""Metrics Summary — compare every retrieval approach over the full query set."""

import pandas as pd
import streamlit as st

import search_core as sc

st.set_page_config(page_title="Metrics Summary · Retrieval PoC", layout="wide")

bundle = sc.build_indexes()
qdf = bundle["query_df"]

st.title("Metrics Summary")
st.caption(
    f"รันทุก approach กับ {len(qdf)} queries (`{bundle['query_src']}`) ค้นจาก corpus เต็ม "
    f"{len(bundle['cand_df'])} คดี แล้วเทียบ metric — ผลถูก dedup เป็นอันดับ 'คดี' (uid) ก่อนคิดคะแนน"
)

# --- retrieval is fixed; cache the ranked uid lists per approach/query ---------
RETRIEVAL_DEPTH = 150


@st.cache_data(show_spinner="Running retrieval for all approaches ...")
def all_rankings(depth=RETRIEVAL_DEPTH):
    b = sc.build_indexes()
    out = {}
    for a in sc.APPROACHES:
        idx = b["indexes"][a["key"]]
        lst = []
        for _, row in b["query_df"].iterrows():
            res = idx.search(sc.query_text(a["key"], row), k=depth)
            lst.append(sc.ranked_uids(res))
        out[a["key"]] = lst
    return out


rankings = all_rankings()

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
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


# --- compute per-approach averages --------------------------------------------
rows = []
for a in sc.APPROACHES:
    hit, rec, prec, mrr, ndcg = [], [], [], [], []
    for i, (_, row) in enumerate(qdf.iterrows()):
        ranked = rankings[a["key"]][i]
        rel = sc.relevant_uids(row, score_col=score_col, thr=thr)
        graded = sc.graded_rel(row)  # nDCG always uses graded relevance_score
        hit.append(sc.hit_at_k(ranked, rel, k))
        rec.append(sc.recall_at_k(ranked, rel, k))
        prec.append(sc.precision_at_k(ranked, rel, k))
        mrr.append(sc.mrr_at_k(ranked, rel, k))
        ndcg.append(sc.ndcg_at_k(ranked, graded, k))
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
- relevant (binary) = `{basis}` · nDCG ใช้ graded `relevance_score` เสมอ
"""
    )
