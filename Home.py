"""Legal Case Retrieval PoC (BM25) — overview / landing page."""

import streamlit as st

import search_core as sc

st.set_page_config(page_title="Legal Retrieval PoC", layout="wide")

bundle = sc.build_indexes()
n_cases = len(bundle["cand_df"])
n_queries = len(bundle["query_df"])

st.title("Legal Case Retrieval — PoC (BM25)")
st.caption(
    f"ฐานข้อมูล {n_cases} คดี (`candidate.csv`) · ชุดทดสอบ {n_queries} queries "
    f"(`{bundle['query_src']}`) · เฟสนี้ใช้ BM25 + Thai tokenizer (pythainlp newmm)"
)

st.markdown(
    """
ทดลอง **หลาย retrieval approach** โดยแต่ละ approach index คนละ "มุม" ของคดี แล้วเทียบกันว่า
มุมไหนค้นได้ตรงที่สุด — ใช้ sidebar ซ้ายเปิดแต่ละหน้า

**แต่ละหน้า approach** แบ่งเป็น 2 ฝั่ง: ซ้าย = คำอธิบาย + ตัวอย่าง query พร้อมเฉลย (label),
ขวา = ช่องค้นหาอิสระ + ผลลัพธ์ top-4 (คลิกดู case info จริงได้)
"""
)

st.divider()
st.subheader("Retrieval approaches")
rows = []
for a in sc.APPROACHES:
    rows.append({
        "approach": a["label"],
        "depth level": a["granularity"],
        "คำอธิบาย": a["desc"].replace("`", ""),
    })
st.dataframe(rows, hide_index=True, use_container_width=True)

st.divider()
st.subheader("Metrics")
st.markdown(
    """
หน้า **Metrics Summary** รันทุก approach กับชุด query ทั้งหมด เทียบ **Hit / Recall / Precision /
MRR / nDCG @k** — ใช้ **nDCG@k** เป็นตัวกลางบอกว่า approach ไหนดีสุด (relevant ดูจาก
`relevance_score`, ปรับ threshold/k ได้ในหน้านั้น)
"""
)
