"""Shared Streamlit render helpers (clean & minimal) for both pages."""

import streamlit as st


def _as_list(x):
    return x if isinstance(x, list) else ([] if x in (None, "") else [x])


def render_case_info(case):
    """Render the real case info inside an already-open container/expander."""
    if not case:
        st.info("ไม่พบข้อมูลคดี (case info) สำหรับ uid นี้")
        return

    cols = st.columns(3)
    cols[0].caption("ฎีกาที่")
    cols[0].write(case.get("deka_no", "-"))
    cols[1].caption("ปี")
    cols[1].write(case.get("year", "-"))
    cols[2].caption("uid")
    cols[2].write(case.get("uid", "-"))

    # คอลัมน์ link ในข้อมูลถูกเปลี่ยนเป็น Google search ที่ key ด้วยเลขฎีกาแล้ว
    # (ลิงก์ detail/{uid} เดิมเปิดไม่ได้ -> 404 และเว็บ deka ทางการไม่มี permalink ต่อคดี)
    deka_no = str(case.get("deka_no", "")).strip()
    link = str(case.get("link", "")).strip()
    if deka_no:
        st.caption("เลขฎีกา (คัดลอกไปค้นที่เว็บ deka ทางการได้)")
        st.code(deka_no, language=None)  # มีปุ่ม copy ในตัว
    if link:
        st.markdown(f"[ค้นหาคำพิพากษาใน Google ↗]({link})")

    crimes = _as_list(case.get("crimes"))
    if crimes:
        st.caption("ฐานความผิด")
        st.write(" · ".join(str(c) for c in crimes))

    lfr = case.get("legal_fact_result")
    if lfr:
        st.caption("ผลข้อเท็จจริงทางกฎหมาย")
        st.write(str(lfr))

    subfacts = _as_list(case.get("subfacts"))
    if subfacts:
        st.caption("subfacts ทั้งหมดของคดี")
        for entry in subfacts:
            if not isinstance(entry, dict):
                continue
            st.markdown(f"**{entry.get('crime', '')}**")
            for s in _as_list(entry.get("subfacts")):
                st.write(f"- {s}")

    long_text = case.get("long_text")
    if long_text:
        st.caption("เนื้อหาคดีฉบับเต็ม")
        st.write(str(long_text))


def _chips(items):
    """Render a list as inline code 'chips' (`tag` `tag`), or — when empty."""
    items = [str(x).strip() for x in _as_list(items) if str(x).strip()]
    return " ".join(f"`{x}`" for x in items) if items else "—"


def _render_facet(approach, unit, case):
    """Show the facet of the matched case that THIS approach searched on.

    Makes it obvious what each approach matched: crimes/laws as tags, subfacts
    as ข้อหา + ข้อเท็จจริงย่อย, long_text/legal_fact as text.
    """
    if approach == "subfacts":
        crime = unit.get("crime") or ""
        if crime:
            st.markdown(f"**ข้อหา:** {crime}")
        st.caption("ข้อเท็จจริงย่อย (subfact) ที่ตรง")
        st.write(unit.get("snippet") or "—")
    elif approach == "crimes":
        st.caption("ฐานความผิดของคดีนี้ (tags)")
        st.markdown(_chips(case.get("crimes")))
    elif approach in ("laws", "extract_law"):
        st.caption("มาตรากฎหมายที่คดีนี้อ้าง (tags)")
        st.markdown(_chips(case.get("laws")))
    elif approach == "legal_fact":
        st.caption("ผลข้อเท็จจริงทางกฎหมาย")
        st.write(case.get("legal_fact_result") or unit.get("snippet") or "—")
    elif approach == "long_text":
        st.caption("คำพิพากษา (ตัดตอน)")
        txt = str(case.get("long_text") or unit.get("snippet") or "")
        st.write((txt[:400] + "…") if len(txt) > 400 else (txt or "—"))
    else:  # unknown approach -> generic snippet
        snippet = unit.get("snippet") or unit.get("subfact") or ""
        st.write(snippet or "—")


def render_result(unit, score, cases, key, mark=None, approach=None):
    """Render one result block + expander with real case info.

    Title is always the ฎีกา number; the body below it changes with `approach`
    so it's clear what data was matched. `mark` is an optional status string
    shown next to the header (e.g. "✓" / "·").
    """
    case = cases.get(int(unit["uid"])) or {}
    head = f"**ฎีกา {unit.get('deka_no', '-')}**"
    if mark:
        head = f"{mark}  {head}"
    st.markdown(head)
    st.caption(f"ปี {unit.get('year', '-')} · uid {unit['uid']} · BM25 {score:.3f}")
    _render_facet(approach, unit, case)
    with st.expander("ดู case info จริง"):
        render_case_info(case)
