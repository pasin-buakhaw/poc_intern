"""2-panel approach page: left = explanation + demo query, right = search + correctness.

Input style differs by approach:
  - Subfacts        : pick / type ONE subfact
  - Crimes / Laws   : tag keywords (multiselect) — no free text
  - Long text / Legal fact : free text
"""

import streamlit as st

import fourcorners as fc
import search_core as sc
from ui_common import render_result

K = 4  # top-k shown


def _query_option_label(qdf, i):
    row = qdf.iloc[i]
    crimes = sc.parse_cell(row.get("crimes", ""))
    crime_str = ", ".join(str(c) for c in crimes[:2]) if crimes else ""
    return f"uid {int(row['uid'])} · ฎีกา {row.get('deka_no', '-')} · {crime_str}"


def _facet_keywords(facet, row):
    """The query's own crime/law keywords (list)."""
    col = "crimes" if facet == "crimes" else "laws_list_matra"
    return [str(x).strip() for x in (sc.parse_cell(row.get(col, "")) or []) if str(x).strip()]


def _subfact_options(row):
    """[(label, value)] one entry per individual subfact string of the query."""
    opts = []
    for e in sc.parse_cell(row.get("subfacts", "")) or []:
        if not isinstance(e, dict):
            continue
        crime = e.get("crime", "")
        subs = e.get("subfacts") or []
        if isinstance(subs, str):
            subs = [subs]
        for s in subs:
            s = str(s).strip()
            if s:
                short = s[:55] + ("…" if len(s) > 55 else "")
                opts.append((f"[{crime}] {short}", s))
    return opts


# --------------------------------------------------------------------------- #
# left panel
# --------------------------------------------------------------------------- #
def _render_left(approach, bundle):
    key = approach["key"]
    qdf, cases = bundle["query_df"], bundle["cases"]

    st.subheader(approach["label"])
    st.caption(approach["desc"])
    st.info(f"🔎 ค้นด้วยข้อมูล: **{approach['field']}**\n\n"
            f"(index ฝั่งคดี = คอลัมน์เดียวกันของ candidate · granularity: {approach['granularity']})")

    st.divider()
    st.markdown("**Demo query** — เลือกตัวอย่าง แล้วส่งไปค้นทางขวา")
    qi = st.selectbox("query ตัวอย่าง", options=list(range(len(qdf))),
                      format_func=lambda i: _query_option_label(qdf, i),
                      key=f"demo_sel_{key}")
    row = qdf.iloc[qi]
    facet = approach.get("keyword_facet")

    # ---- input depends on approach ----
    if facet:  # Crimes / Laws — tag keywords
        tags = _facet_keywords(facet, row)
        if st.button("ใช้ keyword เหล่านี้ค้นหา →", key=f"use_{key}",
                     use_container_width=True, disabled=not tags):
            st.session_state[f"kw_{key}"] = tags
            st.session_state[f"demo_qi_{key}"] = qi
            st.session_state[f"demo_sig_{key}"] = tuple(sorted(tags))
            st.rerun()
        st.caption("keyword ของ query นี้ (กดปุ่มเพื่อ tag ไปค้นทางขวา)")
        st.markdown(" ".join(f"`{t}`" for t in tags) if tags else "—")

    elif key == "subfacts":  # one subfact only
        opts = _subfact_options(row)
        st.caption("เลือก **1 subfact** เพื่อใช้ค้น")
        chosen = ""
        if opts:
            si = st.selectbox("subfact", options=list(range(len(opts))),
                              format_func=lambda i: opts[i][0], key=f"sfsel_{key}")
            chosen = opts[si][1]
            with st.container(height=110, border=True):
                st.write(chosen)
        else:
            st.write("—")
        if st.button("ใช้ subfact นี้ค้นหา →", key=f"use_{key}",
                     use_container_width=True, disabled=not opts):
            st.session_state[f"q_{key}"] = chosen
            st.session_state[f"demo_qi_{key}"] = qi
            st.session_state[f"demo_sig_{key}"] = chosen.strip()
            st.rerun()

    else:  # Long text / Legal fact — free text
        qtext = sc.query_text(key, row)
        if st.button("ใช้ query นี้ค้นหา →", key=f"use_{key}", use_container_width=True):
            st.session_state[f"q_{key}"] = qtext
            st.session_state[f"demo_qi_{key}"] = qi
            st.session_state[f"demo_sig_{key}"] = qtext.strip()
            st.rerun()
        st.caption(f"ข้อความที่จะใช้ค้น (= {approach['field'].split(' (')[0]} ของ query นี้)")
        with st.container(height=220, border=True):
            st.write(qtext or "—")

    # ---- label answer key: candidate + relevance_score (relevant = score >= 1) ----
    graded = sc.graded_rel(row)  # relevance_score 1/2/3
    st.caption("เฉลย (label): candidate ของ query นี้ + คะแนน `relevance_score` "
               "(relevant = score ≥ 1)")
    rows = []
    for uid in sc.query_candidates(row):
        c = cases.get(uid, {})
        crimes = c.get("crimes") or []
        rows.append({
            "uid": uid,
            "ฎีกา": c.get("deka_no", "-"),
            "ฐานความผิด": ", ".join(str(x) for x in crimes[:2]),
            "relevance_score": int(graded.get(uid, 0)),
        })
    st.dataframe(rows, hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------- #
# right panel
# --------------------------------------------------------------------------- #
def _render_right(approach, bundle):
    key = approach["key"]
    index, cases, cand_df = bundle["indexes"][key], bundle["cases"], bundle["cand_df"]
    qdf = bundle["query_df"]
    facet = approach.get("keyword_facet")

    st.subheader("ค้นหา")
    if facet:  # Crimes / Laws — tag picker only
        crimes, laws = sc.collect_keywords(cand_df)
        base = crimes if facet == "crimes" else laws
        sel = st.session_state.get(f"kw_{key}", [])
        options = sorted(set(base) | set(sel))  # keep demo-set tags valid even if not in corpus
        label = "ฐานความผิด" if facet == "crimes" else "มาตรากฎหมาย"
        picks = st.multiselect(f"แท็ก {label} (เลือกได้หลายอัน)", options=options, key=f"kw_{key}")
        query = " ".join(picks)
        search_query = picks  # set-overlap matches on the item list, not the joined string
        sig = tuple(sorted(picks))
    else:  # Subfacts / Long text / Legal fact — free text
        ph = "พิมพ์ subfact" if key == "subfacts" else "พิมพ์ค้นหา (ภาษาไทย)"
        q = st.text_input(ph, key=f"q_{key}",
                          placeholder="เช่น ปลอมเอกสาร, เมทแอมเฟตามีน, ฉ้อโกง")
        query = q.strip()
        search_query = query
        sig = query

    if not query:
        msg = ("เลือกแท็ก keyword หรือกด 'ใช้ keyword เหล่านี้ค้นหา' ทางซ้าย"
               if facet else "พิมพ์คำค้น หรือเลือก demo query ทางซ้ายแล้วกดปุ่มส่ง")
        st.info(msg)
        return

    # demo mode: ค้นมาจาก demo query (signature ตรงกัน) -> รู้เฉลย เช็คได้ว่าค้นถูกไหม
    demo_qi = st.session_state.get(f"demo_qi_{key}")
    demo_sig = st.session_state.get(f"demo_sig_{key}")
    is_demo = demo_qi is not None and demo_sig == sig
    relevant = sc.relevant_uids(qdf.iloc[demo_qi], thr=1) if is_demo else set()

    results = index.search(search_query, k=K)
    if not results:
        st.warning("ไม่พบผลลัพธ์")
        return

    if is_demo:
        ranked = sc.ranked_uids(results)
        found = sorted(relevant & set(ranked[:K]))
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
        mark = ("✓" if int(unit["uid"]) in relevant else "·") if is_demo else None
        render_result(unit, score, cases, key=f"{key}_{rank}", mark=mark, approach=key)


def render_approach_page(key):
    approach = sc.APPROACH_BY_KEY[key]
    if approach.get("fourcorners"):
        return render_extract_law_page(preselect=key)
    st.set_page_config(page_title=f"{approach['label']} · Retrieval PoC", layout="wide")
    bundle = sc.build_indexes()

    st.title(f"Approach: {approach['label']}")
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        _render_left(approach, bundle)
    with right:
        _render_right(approach, bundle)


# --------------------------------------------------------------------------- #
# Extract Law from text (FourCorners semantic search -> Laws index)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _cached_extract(text, token, base_url, k_results):
    """Cache one FourCorners call per (text, token, base, k) so reruns are free."""
    return fc.extract_laws_from_text(text, token, base_url=base_url, k_results=k_results)


def render_extract_law_page(preselect=None):
    """Single page for all 3 Extract-Law variants (source = long_text / legal_fact /
    per-subfact). Pick a variant -> extract มาตรา via FourCorners -> set-overlap search."""
    st.set_page_config(page_title="Extract Law · Retrieval PoC", layout="wide")
    bundle = sc.build_indexes()
    qdf, cases = bundle["query_df"], bundle["cases"]

    variants = [a for a in sc.APPROACHES if a.get("fourcorners")]
    vkeys = [a["key"] for a in variants]
    default_idx = vkeys.index(preselect) if preselect in vkeys else 0

    st.title("Approach: Extract Law from text")
    vi = st.selectbox("variant (ใช้ข้อความจากแหล่งไหนไปดึงมาตรา)",
                      options=list(range(len(variants))),
                      format_func=lambda i: variants[i]["label"], index=default_idx,
                      key="extract_variant")
    approach = variants[vi]
    key = approach["key"]
    source_field = approach["source_field"]
    score_col, thr = "relevance_score", 1  # measure relevant the same everywhere
    laws_index = bundle["indexes"][approach["reuses_index"]]

    st.caption(approach["desc"] + "  ·  relevant = `relevance_score ≥ 1`")
    token, base_url = fc.render_token_input(st)

    left, right = st.columns([1, 1.3], gap="large")

    # -------- left: pick a demo query + the text to extract laws from -------- #
    with left:
        st.subheader("ข้อความตั้งต้น")
        st.info("🔎 ขั้นตอน: **ข้อความ → semantic search (FourCorners) → มาตรา → "
                "ค้นคดีที่อ้างมาตราเดียวกัน (co-cite, set-overlap)**")
        qi = st.selectbox("query ตัวอย่าง", options=list(range(len(qdf))),
                          format_func=lambda i: _query_option_label(qdf, i),
                          key=f"demo_sel_{key}")
        row = qdf.iloc[qi]

        if source_field == "subfacts":  # per-subfact: pick ONE subfact string
            subs = sc.subfact_list(row)
            st.caption("เลือก **1 subfact** เพื่อดึงมาตรา (benchmark รันทุก subfact แล้วเฉลี่ย)")
            if subs:
                si = st.selectbox("subfact", options=list(range(len(subs))),
                                  format_func=lambda i: subs[i][:60], key=f"sfsel_{key}")
                text = st.text_area("ข้อความ subfact (แก้ได้)", value=subs[si],
                                    height=140, key=f"text_{key}")
            else:
                text = st.text_area("ข้อความ subfact", value="", height=140, key=f"text_{key}")
        else:
            text = st.text_area("ข้อความที่จะส่งเข้า semantic search (แก้ได้)",
                                value=sc.source_text(source_field, row), height=180,
                                key=f"text_{key}")

        k_results = st.slider("k (จำนวนมาตราที่ดึงจาก search)", 3, 20, 3, key=f"kres_{key}")
        st.caption("ข้อความถูกส่งเป็น topic เดียวเข้า semantic search")
        go = st.button("ดึงมาตรา แล้วค้นคดี →", use_container_width=True,
                       disabled=not (token and text.strip()), key=f"go_{key}")

        # label answer key: candidate + relevance_score (relevant = score >= 1)
        graded = sc.graded_rel(row, score_col=score_col)
        st.caption("เฉลย (label): candidate ของ query นี้ + คะแนน `relevance_score` "
                   "(relevant = score ≥ 1)")
        rows = []
        for uid in sc.query_candidates(row):
            c = cases.get(uid, {})
            crimes = c.get("crimes") or []
            rows.append({
                "uid": uid, "ฎีกา": c.get("deka_no", "-"),
                "ฐานความผิด": ", ".join(str(x) for x in crimes[:2]),
                "relevance_score": int(graded.get(uid, 0)),
            })
        st.dataframe(rows, hide_index=True, use_container_width=True)

    # -------- right: run the pipeline, show extracted laws + results -------- #
    with right:
        st.subheader("ผลลัพธ์")
        if go:
            st.session_state[f"run_{key}"] = {
                "text": text, "qi": qi, "k_results": k_results}

        run = st.session_state.get(f"run_{key}")
        if not run:
            st.info("เลือก query และกด 'ดึงมาตรา แล้วค้นคดี →' (ต้องใส่ token ก่อน)")
            return

        try:
            laws, topics, raw_md = _cached_extract(
                run["text"], token, base_url, run["k_results"])
        except Exception as e:  # noqa: BLE001 — show API errors to the user
            st.error(f"เรียก FourCorners ไม่สำเร็จ: {e}")
            return

        with st.expander("ข้อความที่ส่งเข้า semantic search (topic เดียว)", expanded=False):
            st.write(topics[0] if topics else "—")
        if not laws:
            st.warning("semantic search ไม่คืนมาตราที่ parse ได้ — ลองปรับข้อความ/topic")
            with st.expander("ดู raw markdown จาก API"):
                st.code(raw_md or "(ว่าง)")
            return
        laws = laws[:run["k_results"]]  # use only top-k extracted มาตรา for the search
        st.success(f"ใช้ **top-{run['k_results']}** ({len(laws)} มาตรา) → ค้นคดีที่อ้างมาตราเดียวกัน")
        st.markdown(" ".join(f"`{l}`" for l in laws))

        rrow = qdf.iloc[run["qi"]]
        relevant = sc.relevant_uids(rrow, score_col=score_col, thr=thr)
        results = laws_index.search(laws, k=K)  # exact set-overlap on extracted มาตรา
        if not results:
            st.warning("ไม่พบคดีที่อ้างมาตราเหล่านี้")
            return

        ranked = sc.ranked_uids(results)
        found = sorted(relevant & set(ranked[:K]))
        msg = f"ค้นถูกไหม: พบคดีที่ relevant **{len(found)}/{len(relevant)}** ใน top-{K}"
        (st.success if found else st.warning)(msg)
        st.caption(f"ผลลัพธ์ top {K} — ✓ = คดีที่ relevant ตามเฉลย")
        for rank, (unit, score) in enumerate(results):
            st.divider()
            mark = "✓" if int(unit["uid"]) in relevant else "·"
            render_result(unit, score, cases, key=f"{key}_{rank}", mark=mark, approach=key)
