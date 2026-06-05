"""FourCorners Toolkit API client for the "Extract Law from text" approach.

Pipeline this module powers:
    free text  --search_legal_corpus (semantic / hybrid)-->  law sections (มาตรา)
    law sections  -->  query string fed to the Laws BM25 index  -->  co-citing cases

Only the pieces the PoC needs are implemented:
  - get_config / render_token_input : token + base URL (UI field, env fallback)
  - search_legal_corpus             : POST /tools/search_legal_corpus
  - parse_law_sections              : markdown result -> ["<law> มาตรา <n>", ...]
  - text_to_topics                  : long Thai text -> a few short topic phrases
  - extract_laws_from_text          : the two steps above, glued together

All parsing helpers are pure (no streamlit / no network) so they can be unit
tested without reaching the private API.
"""

import os
import re

try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - surfaced in the UI instead
    requests = None

# Default from USAGE 2.md / manifest.json (private network — needs a tunnel).
DEFAULT_BASE_URL = "http://10.204.100.77:6767"


# --------------------------------------------------------------------------- #
# config (token + base url): UI field first, env var fallback
# --------------------------------------------------------------------------- #
def get_config():
    """Return (token, base_url): streamlit session_state -> env vars -> default.

    `base` starts empty so an env override actually wins over DEFAULT_BASE_URL
    (the hardcoded default is only the last resort).
    """
    token, base = "", ""
    try:
        import streamlit as st

        token = (st.session_state.get("fc_token") or "").strip()
        base = (st.session_state.get("fc_base_url") or "").strip()
    except Exception:  # pragma: no cover - no streamlit runtime (CLI/tests)
        pass
    token = token or os.environ.get("FOURCORNERS_TOKEN", "").strip()
    base = base or os.environ.get("FOURCORNERS_BASE_URL", "").strip() or DEFAULT_BASE_URL
    return token, base.rstrip("/")


def render_token_input(st, *, key_prefix="fc"):
    """Render the FourCorners token + base-url fields. Returns (token, base_url).

    The user pastes their bearer token here (USAGE 2.md: 'your lead will hand
    you the token'). Stored only in session_state, never written to disk.
    """
    with st.expander("🔑 FourCorners API — ตั้งค่า token", expanded=not get_config()[0]):
        st.caption(
            "ใส่ **Bearer token** ที่ได้รับจาก lead เพื่อเรียก semantic search "
            "(`search_legal_corpus`) ของ FourCorners · เก็บไว้ใน session เท่านั้น ไม่บันทึกลงดิสก์"
        )
        st.text_input("Bearer token", type="password", key="fc_token",
                      placeholder="วาง token ที่นี่")
        st.text_input("Base URL", key="fc_base_url",
                      placeholder=DEFAULT_BASE_URL)
        token, base = get_config()
        c1, c2 = st.columns([1, 3])
        if c1.button("ทดสอบการเชื่อมต่อ", key=f"{key_prefix}_health"):
            ok, msg = health(base)
            (c2.success if ok else c2.error)(msg)
        if not token:
            st.warning("ยังไม่มี token — approach นี้จะเรียก API ไม่ได้จนกว่าจะใส่ token")
    return get_config()


# --------------------------------------------------------------------------- #
# network calls
# --------------------------------------------------------------------------- #
def _require_requests():
    if requests is None:
        raise RuntimeError("ต้องติดตั้ง `requests` ก่อน (pip install requests)")


def health(base_url):
    """GET /health (no auth). Returns (ok: bool, message: str)."""
    if requests is None:
        return False, "ยังไม่ได้ติดตั้ง requests"
    try:
        r = requests.get(f"{base_url.rstrip('/')}/health", timeout=10)
        r.raise_for_status()
        j = r.json()
        emb = j.get("embedding", {}).get("ok")
        neo = j.get("neo4j", {}).get("ok")
        return True, f"ok · neo4j={neo} · embedding={emb}"
    except Exception as e:  # noqa: BLE001 - report whatever went wrong to the UI
        return False, f"เชื่อมต่อไม่ได้: {e}"


def search_legal_corpus(topics, token, base_url=DEFAULT_BASE_URL, k=10, *,
                        as_of_date=None, include_inactive=False, timeout=60):
    """POST /tools/search_legal_corpus -> markdown `result` string.

    `topics` = 1-5 short Thai keyword phrases (manifest constraint). Raises on
    HTTP / network error so the caller can show it.
    """
    _require_requests()
    if not token:
        raise RuntimeError("ไม่มี FourCorners token")
    topics = [t for t in (topics or []) if str(t).strip()][:5]
    if not topics:
        raise ValueError("ต้องมีอย่างน้อย 1 topic")
    body = {"topics": topics, "k": int(k)}
    if as_of_date:
        body["as_of_date"] = as_of_date
    if include_inactive:
        body["include_inactive"] = True
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    r = requests.post(f"{base_url.rstrip('/')}/tools/search_legal_corpus",
                      headers=headers, json=body, timeout=timeout)
    if r.status_code == 401:
        raise RuntimeError("401 unauthorized — token ผิดหรือหมดอายุ")
    r.raise_for_status()
    return r.json().get("result", "")


# --------------------------------------------------------------------------- #
# pure parsing helpers
# --------------------------------------------------------------------------- #
# law-name prefixes seen in the corpus / Thai statutes
_LN = r"[ก-๙A-Za-z0-9 .]"  # Thai/Latin letters, digits (พ.ศ. years), dots, spaces
_LAW_NAME_RE = re.compile(
    r"(ประมวลกฎหมาย" + _LN + r"+?|"
    r"พระราชบัญญัติ" + _LN + r"+?|พ\.ร\.บ\.?" + _LN + r"+?|"
    r"พระราชกำหนด" + _LN + r"+?|พ\.ร\.ก\.?" + _LN + r"+?|"
    r"ประกาศ" + _LN + r"+?|รัฐธรรมนูญ" + _LN + r"*?)"
    r"(?=\s*(?:—|–|-|\||:|มาตรา|$|\n))"
)
# section labels incl. family suffixes: 'มาตรา 288', 'มาตรา 81/3', 'มาตรา 7 ทวิ'
_MATRA_RE = re.compile(
    r"มาตรา\s*[0-9]+(?:/[0-9]+)?"
    r"(?:\s*(?:ทวิ|ตรี|จัตวา|เบญจ|ฉ|สัตต|อัฏฐ|นว|ทศ))?"
)
# section URI inside a result bullet: `th/law/<code>/section-<label>`
_URI_SECTION_RE = re.compile(r"th/law/[^\s`)/]+/section-([^\s`)>]+)")
_HEADING_RE = re.compile(r"^#{2,6}\s*(.+)$")


def _clean_law_name(name):
    name = re.sub(r"\(\(.*?\)\)", "", str(name))            # ((unknown law)) annotation
    name = re.sub(r"\s*[—–-]\s*\d+\s*match\w*.*$", "", name)  # "— 3 matches" suffix
    name = name.replace("**", "")
    return " ".join(name.split()).strip(" :*—–-|")


def _label_to_matra(label):
    """URI section label -> 'มาตรา <label>' (e.g. '25' -> 'มาตรา 25')."""
    return "มาตรา " + str(label).strip().strip("-")


def parse_law_sections(markdown):
    """Parse `search_legal_corpus` markdown -> ordered unique ["<law> มาตรา <n>"].

    The real output groups results by law under `## <law name> — N matches`
    headings, with each match carrying a `th/law/<code>/section-<label>` URI.
    The section number is read from the URI (authoritative) and paired with the
    current heading's law name — NOT from the `> ...` statute-text previews,
    which mention unrelated มาตรา and would add noise.

    Falls back to a prose `มาตรา N` scan if the markdown has no section URIs.
    """
    if not markdown:
        return []
    out, seen = [], set()

    def add(law, matra):
        item = " ".join(f"{law} {matra}".split()) if law else matra.strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)

    current_law = ""
    for raw in str(markdown).splitlines():
        line = raw.strip()
        if not line:
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            name = _clean_law_name(heading.group(1))
            if name and _LAW_NAME_RE.search(name):
                current_law = name
            continue
        for label in _URI_SECTION_RE.findall(line):
            add(current_law, _label_to_matra(label))

    if out:
        return out
    return _parse_law_sections_prose(markdown)  # markdown without URIs (fallback)


def _parse_law_sections_prose(markdown):
    """Fallback: scan headings for law names + 'มาตรา N' tokens line by line."""
    out, seen, current_law = [], set(), ""
    for raw in str(markdown).splitlines():
        line = raw.strip()
        if not line:
            continue
        heading = re.match(r"^#{1,6}\s*(.+)$", line)
        candidate = heading.group(1) if heading else line
        m_name = _LAW_NAME_RE.search(candidate)
        matras = _MATRA_RE.findall(line)
        if m_name and (heading or not matras):
            current_law = _clean_law_name(m_name.group(1))
        line_law = _clean_law_name(m_name.group(1)) if m_name else current_law
        for matra in matras:
            item = " ".join(f"{line_law} {matra}".split()) if line_law else matra
            if item and item not in seen:
                seen.add(item)
                out.append(item)
    return out


def extract_laws_from_text(text, token, base_url=DEFAULT_BASE_URL, *,
                           k_results=3, max_chars=2000):
    """text -> (law_sections, topics_sent, raw_markdown).

    The whole text is sent as a SINGLE topic to `search_legal_corpus` (no phrase
    splitting), then the returned markdown is parsed into law-section strings
    ready for the Laws index. `k_results` caps how many sections the API returns;
    `max_chars` truncates very long inputs (e.g. a full long_text judgment).
    """
    text = " ".join(str(text or "").split())[:max_chars]
    if not text:
        return [], [], ""
    topics = [text]
    md = search_legal_corpus(topics, token, base_url=base_url, k=k_results)
    # the API may return sections across several law groups (more than k for long
    # inputs); cap to the top k_results so the extracted count is predictable.
    return parse_law_sections(md)[:k_results], topics, md
