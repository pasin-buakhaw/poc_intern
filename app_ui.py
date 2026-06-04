"""2-panel approach page: left = explanation + demo query, right = search + correctness."""

import streamlit as st

import search_core as sc
from ui_common import render_result

K = 4  # top-k shown


def _query_option_label(qdf, i):
    row = qdf.iloc[i]
    crimes = sc.parse_cell(row.get("crimes", ""))
    crime_str = ", ".join(str(c) for c in crimes[:2]) if crimes else ""
    return f"uid {int(row['uid'])} · ฎีกา {row.get('deka_no', '-')} · {crime_str}"


def _render_left(approach, bundle):
    """Explanation + which field is searched + pick a demo query (with label)."""
    key = approach["key"]
    qdf, cases = bundle["query_df"], bundle["cases"]

    st.subheader(approach["label"])
    st.caption(approach["desc"])
    # ★ ระบุชัดเจนว่า approach นี้เอา "คอลัมน์ไหนของ query" ไปค้น
    st.info(f"🔎 ค้นด้วยข้อมูล: **{approach['field']}**\n\n"
            f"(index ฝั่งคดี = คอลัมน์เดียวกันของ candidate · granularity: {approach['granularity']})")

    st.divider()
    st.markdown("**Demo query** — เลือกตัวอย่าง แล้วกดส่งข้อความด้านล่างไปค้นทางขวา")
    qi = st.selectbox("query ตัวอย่าง", options=list(range(len(qdf))),
                      format_func=lambda i: _query_option_label(qdf, i),
                      key=f"demo_sel_{key}")
    row = qdf.iloc[qi]
    qtext = sc.query_text(key, row)

    # ★ ปุ่มส่งไปค้น อยู่ใต้ dropdown ทันที
    if st.button("ใช้ query นี้ค้นหา →", key=f"use_{key}", use_container_width=True):
        st.session_state[f"q_{key}"] = qtext
        st.session_state[f"demo_qtext_{key}"] = qtext
        st.session_state[f"demo_qi_{key}"] = qi
        st.rerun()

    st.caption(f"ข้อความที่จะใช้ค้น (= {approach['field'].split(' (')[0]} ของ query นี้)")
    st.text_area("query text", value=qtext, height=120, disabled=True,
                 key=f"demo_txt_{key}", label_visibility="collapsed")

    # label answer key — which candidates are relevant (no metric numbers)
    rel = sc.relevant_uids(row, thr=2)
    graded = sc.graded_rel(row)
    st.caption("เฉลย (label): candidate ของ query นี้ — ✓ = relevant (ควรค้นเจอ)")
    rows = []
    for uid in sc.query_candidates(row):
        c = cases.get(uid, {})
        crimes = c.get("crimes") or []
        rows.append({
            "uid": uid,
            "ฎีกา": c.get("deka_no", "-"),
            "ฐานความผิด": ", ".join(str(x) for x in crimes[:2]),
            "relevance": int(graded.get(uid, 0)),
            "relevant?": "✓" if uid in rel else "·",
        })
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_right(approach, bundle):
    """Free-form search (+ keyword facets for Crimes/Laws) + correctness marks."""
    key = approach["key"]
    index, cases, cand_df = bundle["indexes"][key], bundle["cases"], bundle["cand_df"]
    qdf = bundle["query_df"]

    st.subheader("ค้นหา")
    extra = ""
    facet = approach.get("keyword_facet")
    if facet:
        crimes, laws = sc.collect_keywords(cand_df)
        opts = crimes if facet == "crimes" else laws
        label = "ฐานความผิด" if facet == "crimes" else "มาตรากฎหมาย"
        picks = st.multiselect(f"เลือก {label} — เลือกหลายอันได้", options=opts, key=f"kw_{key}")
        extra = " ".join(picks)

    q = st.text_input("พิมพ์ค้นหา (ภาษาไทย)", key=f"q_{key}",
                      placeholder="เช่น ปลอมเอกสาร, เมทแอมเฟตามีน, ฉ้อโกง")
    query = f"{q} {extra}".strip()

    if not query:
        st.info("พิมพ์คำค้น หรือเลือก demo query ทางซ้ายแล้วกด 'ใช้ query นี้ค้นหา'")
        return

    # demo mode: search text มาจาก demo query -> รู้เฉลย จึงเช็คได้ว่า "ค้นถูกไหม"
    demo_qtext = st.session_state.get(f"demo_qtext_{key}", "")
    demo_qi = st.session_state.get(f"demo_qi_{key}")
    is_demo = bool(demo_qtext) and not extra and q.strip() == demo_qtext.strip()
    relevant = sc.relevant_uids(qdf.iloc[demo_qi], thr=2) if is_demo else set()

    results = index.search(query, k=K)
    if not results:
        st.warning("ไม่พบผลลัพธ์")
        return

    if is_demo:
        ranked = sc.ranked_uids(results)
        found = sorted(relevant & set(ranked[:K]))
        miss = sorted(relevant - set(ranked[:K]))
        msg = f"ค้นถูกไหม: พบคดีที่ relevant **{len(found)}/{len(relevant)}** ใน top-{K}"
        (st.success if found else st.warning)(msg)
        exp = []
        for uid in sorted(relevant):
            deka = cases.get(uid, {}).get("deka_no", "-")
            exp.append(f"{'✓' if uid in found else '✗'} ฎีกา {deka} (uid {uid})")
        st.caption("เฉลยที่ควรเจอ: " + " · ".join(exp))

    st.caption(f"ผลลัพธ์ top {K}" + (" — ✓ = คดีที่ relevant ตามเฉลย" if is_demo else ""))
    for rank, (unit, score) in enumerate(results):
        st.divider()
        mark = None
        if is_demo:
            mark = "✓" if int(unit["uid"]) in relevant else "·"
        render_result(unit, score, cases, key=f"{key}_{rank}", mark=mark)


def render_approach_page(key):
    approach = sc.APPROACH_BY_KEY[key]
    st.set_page_config(page_title=f"{approach['label']} · Retrieval PoC", layout="wide")
    bundle = sc.build_indexes()

    st.title(f"Approach: {approach['label']}")
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        _render_left(approach, bundle)
    with right:
        _render_right(approach, bundle)
