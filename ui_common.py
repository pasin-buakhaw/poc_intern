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


def render_result(unit, score, cases, key, mark=None):
    """Render one result block (generic unit) + expander with real case info.

    Works for both case-level units (title="ฎีกา ...", snippet=field text) and
    subfact units (title=crime, snippet=subfact). `mark` is an optional status
    string shown next to the header (e.g. "✓" / "·").
    """
    title = unit.get("title") or unit.get("crime") or f"ฎีกา {unit.get('deka_no', '-')}"
    snippet = unit.get("snippet") or unit.get("subfact") or ""
    head = f"**{title}**"
    if mark:
        head = f"{mark}  {head}"
    st.markdown(head)
    st.caption(f"ฎีกา {unit.get('deka_no', '-')} · ปี {unit.get('year', '-')} · "
               f"uid {unit['uid']} · BM25 {score:.3f}")
    if snippet:
        st.write(snippet)
    with st.expander("ดู case info จริง"):
        render_case_info(cases.get(int(unit["uid"])))
