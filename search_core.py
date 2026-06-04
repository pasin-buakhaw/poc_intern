"""
Core retrieval logic for the multi-approach legal-case search PoC (BM25 phase).

Each "approach" indexes a different representation of a case (long_text,
crimes+laws, legal_fact_result, subfacts, all-fields). Candidate and query share
the same columns, so an approach is symmetric: the same field function builds the
candidate document text AND the query text.

All functions here are pure (no streamlit) except the cached `build_indexes()`
wrapper at the bottom, so they can be unit-tested and reused when embedding /
hybrid retrievers are added later.

Generic index unit:
    {uid, deka_no, year, link, title, snippet, text}
    (subfact units additionally keep `crime` / `crime_norm`)
BM25 is built over the tokenized `unit["text"]`.
"""

import math
import os

import numpy as np
import pandas as pd
from pythainlp.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from label_helpers import parse_cell, norm_crime

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
_DATA = os.path.join(_HERE, "data")


def _first_existing(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


# bundled data/ (self-contained deploy) -> fall back to the parent repo (local dev)
CANDIDATE_CSV = _first_existing(
    os.path.join(_DATA, "candidate.csv"),
    os.path.join(_PARENT, "candidate.csv"),
)
QUERY_CSV_CANDIDATES = [
    os.path.join(_DATA, "query_clean.csv"),
    os.path.join(_PARENT, "query_clean.csv"),
    os.path.join(_PARENT, "query_labeled.csv"),
    os.path.join(_PARENT, "query.csv"),
]


# --------------------------------------------------------------------------- #
# tokenization
# --------------------------------------------------------------------------- #
def tokenize(text):
    """Thai word tokenization (newmm) -> lowercased, whitespace-free tokens."""
    if not text:
        return []
    toks = word_tokenize(str(text), engine="newmm")
    return [t.lower() for t in toks if t and not t.isspace()]


# --------------------------------------------------------------------------- #
# field representations (work on a dict-like row: pandas Series or dict)
# --------------------------------------------------------------------------- #
def _get(row, col, default=""):
    val = row.get(col, default) if hasattr(row, "get") else getattr(row, col, default)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    return val


def _join_list(cell):
    return " ".join(str(x) for x in (parse_cell(cell) or []) if str(x).strip())


def text_long(row):
    return str(_get(row, "long_text")).strip()


def text_crimes(row):
    return _join_list(_get(row, "crimes"))


def text_laws(row):
    return _join_list(_get(row, "laws_list_matra"))


def text_legalfact(row):
    return str(_get(row, "legal_fact_result")).strip()


def text_subfacts_concat(row):
    """Flatten subfacts -> "<crime> <subfact> <subfact> ...". for the whole case."""
    parts = []
    for entry in parse_cell(_get(row, "subfacts")) or []:
        if not isinstance(entry, dict):
            continue
        parts.append(str(entry.get("crime", "")))
        subs = entry.get("subfacts", []) or []
        if isinstance(subs, str):
            subs = [subs]
        parts.extend(str(s) for s in subs)
    return " ".join(p for p in parts if p and p.strip()).strip()


# --------------------------------------------------------------------------- #
# approach registry
# --------------------------------------------------------------------------- #
APPROACHES = [
    {
        "key": "long_text", "label": "Long text", "granularity": "case",
        "repr": text_long, "field": "long_text (คำพิพากษาฉบับเต็มของ query)",
        "desc": "คำพิพากษาฉบับเต็ม (`long_text`) — 1 document ต่อ 1 คดี "
            
    },
    {
        "key": "subfacts", "label": "Subfacts", "granularity": "subfact",
        "repr": text_subfacts_concat, "field": "subfacts (ข้อเท็จจริงย่อยทุกอันของ query รวมกัน)",
        "desc": "ข้อเท็จจริงย่อย (subfact) ที่เกิดขึ้นในคดี แยกเป็นตามข้อหา"
    },
    {
        "key": "crimes", "label": "Crimes", "granularity": "case",
        "repr": text_crimes, "field": "crimes (ฐานความผิดของ query)", "keyword_facet": "crimes",
        "desc": "ข้อหา — keyword-based เลือกฐานความผิดหลายอัน "
                "เพื่อหาคดีที่เกี่ยวข้องได้",
    },
    {
        "key": "laws", "label": "Laws", "granularity": "case",
        "repr": text_laws, "field": "laws_list_matra (มาตรากฎหมายของ query)", "keyword_facet": "laws",
        "desc": "มาตรากฎหมาย (`laws_list_matra`) — keyword-based เลือกมาตราหลายอัน "
                "เพื่อหาคดีที่เกี่ยวข้องได้",
    },
    {
        "key": "legal_fact", "label": "Legal fact result", "granularity": "case",
        "repr": text_legalfact, "field": "legal_fact_result (ผลข้อเท็จจริงทางกฎหมายของ query)",
        "desc": "การตีความข้อเท็จจริงตามมาตรา(วินิจฉัย)",
    },
]
APPROACH_BY_KEY = {a["key"]: a for a in APPROACHES}


def query_text(approach_key, row):
    """Build the query text for an approach from a query row (symmetric repr)."""
    return APPROACH_BY_KEY[approach_key]["repr"](row)


# --------------------------------------------------------------------------- #
# index construction
# --------------------------------------------------------------------------- #
def build_subfact_units(cand_df):
    """One generic unit per individual subfact string (subfact granularity)."""
    units = []
    for row in cand_df.itertuples():
        uid = int(row.uid)
        for entry in parse_cell(getattr(row, "subfacts", "")) or []:
            if not isinstance(entry, dict):
                continue
            crime = entry.get("crime", "")
            subs = entry.get("subfacts", []) or []
            if isinstance(subs, str):
                subs = [subs]
            for s in subs:
                s = str(s).strip()
                if not s:
                    continue
                units.append({
                    "uid": uid,
                    "deka_no": getattr(row, "deka_no", ""),
                    "year": getattr(row, "year", ""),
                    "link": getattr(row, "link", ""),
                    "crime": crime,
                    "crime_norm": norm_crime(crime),
                    "title": crime,
                    "snippet": s,
                    "text": f"{crime} {s}",
                })
    return units


def build_case_units(cand_records, approach):
    """One unit per case, text = approach field representation (case granularity)."""
    units = []
    for r in cand_records:
        txt = approach["repr"](r)
        deka = _get(r, "deka_no")
        units.append({
            "uid": int(r["uid"]),
            "deka_no": deka,
            "year": _get(r, "year"),
            "link": _get(r, "link"),
            "title": f"ฎีกา {deka}",
            "snippet": txt[:300],
            "text": txt,
        })
    return units


def load_cases(cand_df):
    """uid -> full case row (dict) for showing real case info on click."""
    cases = {}
    for row in cand_df.itertuples():
        cases[int(row.uid)] = {
            "uid": int(row.uid),
            "deka_no": getattr(row, "deka_no", ""),
            "year": getattr(row, "year", ""),
            "link": getattr(row, "link", ""),
            "crimes": parse_cell(getattr(row, "crimes", "")),
            "laws": parse_cell(getattr(row, "laws_list_matra", "")),
            "legal_fact_result": getattr(row, "legal_fact_result", ""),
            "subfacts": parse_cell(getattr(row, "subfacts", "")),
            "long_text": getattr(row, "long_text", ""),
        }
    return cases


class BM25Index:
    """Thin wrapper: holds units + BM25 model and answers top-k searches."""

    def __init__(self, units):
        self.units = units
        corpus = [tokenize(u["text"]) for u in units]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query, k=4):
        """Return list of (unit, score) for the top-k units, score desc."""
        toks = tokenize(query)
        if not toks or self.bm25 is None:
            return []
        scores = self.bm25.get_scores(toks)
        k = min(k, len(self.units))
        top = np.argsort(scores)[::-1][:k]
        return [(self.units[i], float(scores[i])) for i in top]


# --------------------------------------------------------------------------- #
# keyword facets (for the Crimes / Laws pages)
# --------------------------------------------------------------------------- #
def collect_keywords(cand_df):
    """Distinct crime labels and law-matra strings across the corpus, sorted."""
    crimes, laws = set(), set()
    for row in cand_df.itertuples():
        for c in parse_cell(getattr(row, "crimes", "")) or []:
            if str(c).strip():
                crimes.add(str(c).strip())
        for m in parse_cell(getattr(row, "laws_list_matra", "")) or []:
            if str(m).strip():
                laws.add(str(m).strip())
    return sorted(crimes), sorted(laws)


# --------------------------------------------------------------------------- #
# retrieval metrics (pure; operate on hashable keys — uids for case-level eval)
# --------------------------------------------------------------------------- #
def ranked_uids(results):
    """[(unit, score), ...] -> list of unique uids, first-occurrence order."""
    out, seen = [], set()
    for u, _ in results:
        uid = int(u["uid"])
        if uid not in seen:
            seen.add(uid)
            out.append(uid)
    return out


def hit_at_k(ranked_keys, relevant_keys, k=4):
    """1.0 if any of the top-k keys is relevant, else 0.0. None if no relevant."""
    rel = set(relevant_keys)
    if not rel:
        return None
    return 1.0 if any(key in rel for key in ranked_keys[:k]) else 0.0


def recall_at_k(ranked_keys, relevant_keys, k=4):
    """|relevant ∩ top-k| / |relevant|. None if there are no relevant keys."""
    rel = set(relevant_keys)
    if not rel:
        return None
    return len(rel.intersection(ranked_keys[:k])) / len(rel)


def precision_at_k(ranked_keys, relevant_keys, k=4):
    """|relevant ∩ top-k| / k."""
    rel = set(relevant_keys)
    hits = sum(1 for key in ranked_keys[:k] if key in rel)
    return hits / k


def mrr_at_k(ranked_keys, relevant_keys, k=4):
    """Reciprocal rank of the first relevant key within top-k (0 if none)."""
    rel = set(relevant_keys)
    if not rel:
        return None
    for i, key in enumerate(ranked_keys[:k], start=1):
        if key in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_keys, graded, k=4):
    """nDCG@k using graded relevance dict {uid: gain}. None if no graded items."""
    if not graded:
        return None

    def dcg(gains):
        return sum(g / math.log2(i + 2) for i, g in enumerate(gains))

    gains = [float(graded.get(key, 0.0)) for key in ranked_keys[:k]]
    ideal = sorted((float(v) for v in graded.values()), reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg == 0:
        return None
    return dcg(gains) / idcg


# --------------------------------------------------------------------------- #
# query-side label helpers
# --------------------------------------------------------------------------- #
def query_candidates(query_row):
    """List of candidate uids presented for this query."""
    return [int(u) for u in parse_cell(_get(query_row, "query_candidate"))]


def _scores(query_row, col):
    return parse_cell(_get(query_row, col))


def relevant_uids(query_row, score_col="relevance_score", thr=2):
    """Set of candidate uids whose `score_col` value >= thr."""
    rel = set()
    for uid, s in zip(query_candidates(query_row), _scores(query_row, score_col)):
        try:
            if float(s) >= thr:
                rel.add(int(uid))
        except (TypeError, ValueError):
            continue
    return rel


def graded_rel(query_row, score_col="relevance_score"):
    """{uid: graded relevance} for nDCG (uses relevance_score by default)."""
    g = {}
    for uid, s in zip(query_candidates(query_row), _scores(query_row, score_col)):
        try:
            g[int(uid)] = float(s)
        except (TypeError, ValueError):
            g[int(uid)] = 0.0
    return g


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def load_candidate_df():
    return pd.read_csv(CANDIDATE_CSV)


def load_query_df():
    """Load the first available query CSV (prefers query_clean.csv)."""
    for path in QUERY_CSV_CANDIDATES:
        if os.path.exists(path):
            return pd.read_csv(path), os.path.basename(path)
    raise FileNotFoundError("No query CSV found: " + ", ".join(QUERY_CSV_CANDIDATES))


def _build_indexes_impl():
    cand_df = load_candidate_df()
    cand_records = cand_df.to_dict("records")
    cases = load_cases(cand_df)
    query_df, query_src = load_query_df()

    indexes = {}
    for a in APPROACHES:
        if a["granularity"] == "subfact":
            units = build_subfact_units(cand_df)
        else:
            units = build_case_units(cand_records, a)
        indexes[a["key"]] = BM25Index(units)

    return {
        "indexes": indexes,
        "cases": cases,
        "cand_df": cand_df,
        "query_df": query_df,
        "query_src": query_src,
    }


# Streamlit cache if available; otherwise a plain (uncached) call for CLI/tests.
try:
    import streamlit as st

    @st.cache_resource(show_spinner="Building BM25 indexes ...")
    def build_indexes():
        return _build_indexes_impl()

except ModuleNotFoundError:  # pragma: no cover - non-streamlit usage
    def build_indexes():
        return _build_indexes_impl()
