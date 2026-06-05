# Legal Case Retrieval PoC (BM25) — multi-approach

ทดลอง **หลาย retrieval approach** บนคดีศาลฎีกา (ภาษาไทย) โดยใช้ `../candidate.csv` (148 คดี)
เป็นฐานข้อมูล และ `../query_clean.csv` (77 queries) เป็นชุดทดสอบ เฟสนี้ใช้ **BM25** +
Thai tokenizer (pythainlp `newmm`). แต่ละ approach index คนละ "มุม" ของคดี แล้วเทียบกันว่ามุมไหนดีสุด

## Approaches (แต่ละอัน = 1 หน้า)
| approach | คอลัมน์ที่ index | granularity |
|---|---|---|
| Long text | `long_text` (คำพิพากษาเต็ม) | case — 1 doc/คดี |
| Subfacts | subfact ราย string | subfact — หลาย doc/คดี |
| Crimes | `crimes` (keyword) | case |
| Laws | `laws_list_matra` (keyword) | case |
| Legal fact result | `legal_fact_result` | case |
| Extract Law from text | ดึงมาตราจากข้อความด้วย **FourCorners semantic search** → ค้นด้วย Laws index | case |

## โครงสร้าง
```
poc_search/
  search_core.py          # APPROACHES registry + build BM25 ทุก approach + metric fns (pure, testable)
  app_ui.py               # render_approach_page(key): 2-panel (อธิบาย+demo+label / search+เช็คถูก)
  ui_common.py            # render ผลลัพธ์ (generic unit) + case info
  Home.py                 # overview/landing
  pages/
    1_Long_text.py  2_Subfacts.py  3_Crimes.py  4_Laws.py  5_Legal_fact.py
    6_Metrics_Summary.py  # เทียบทุก approach
```

## วิธีรัน (local)
```bash
pip install -r requirements.txt
streamlit run Home.py     # เปิด http://localhost:8501
```
sidebar มี: Home, 5 หน้า approach, และ Metrics Summary

## Deploy (Streamlit Community Cloud)
repo นี้ self-contained แล้ว (ข้อมูลอยู่ใน `data/`, ไม่พึ่งโค้ดนอก repo)
1. ไปที่ https://share.streamlit.io → **New app** → เลือก repo `pasin-buakhaw/poc_intern`
   (repo เป็น private — กด authorize ให้ Streamlit เข้าถึง GitHub)
2. ตั้ง **Main file path** = `Home.py` · branch = `main` · Python 3.11/3.12
3. Deploy — `requirements.txt` จะถูกติดตั้งให้อัตโนมัติ

โครงสร้างที่จำเป็นต่อ deploy: `Home.py`, `pages/`, `search_core.py`, `app_ui.py`,
`ui_common.py`, `label_helpers.py`, `requirements.txt`, `data/{candidate,query_clean}.csv`

## หน้า approach (2 panel)
- **ซ้าย:** ระบุชัดว่า approach นี้ **เอาคอลัมน์ไหนของ query ไปค้น** (เช่น Long text ใช้ `long_text`
  ของ query) + เลือก **demo query** ตัวอย่าง → เห็นข้อความที่จะใช้ค้น และ **เฉลย (label)** ว่า
  candidate ตัวไหน relevant (`relevance_score`) — กดปุ่ม **"ใช้ query นี้ค้นหา →"** ส่งไปช่องขวา
- **ขวา:** ช่อง **free-form search** → top-4 ผลลัพธ์ คลิก **"ดู case info จริง"** ได้ ·
  หน้า **Crimes / Laws** มี `multiselect` keyword เลือกหลายอันได้ ·
  เมื่อค้นจาก demo query จะ **บอกว่าค้นถูกไหม**: banner "พบคดี relevant N/ทั้งหมด ใน top-k" +
  `✓` หน้าผลลัพธ์ที่เป็นคดี relevant ตามเฉลย
- ไม่มีตัวเลข metric เต็ม ๆ บนหน้านี้ (อยู่ที่หน้า Metrics Summary)

## หน้า Metrics Summary
รันทุก approach × 77 queries ค้นจาก corpus เต็ม แล้ว dedup เป็นอันดับ **คดี (uid)** ก่อนคิดคะแนน
เทียบ **nDCG / Hit / Recall / Precision / MRR @k** (ปรับ `k` และนิยาม relevant ได้) —
ใช้ **nDCG@k** (graded จาก `relevance_score` 1/2/3) เป็นตัวกลางบอกว่า approach ไหนดีสุด
ไฮไลต์ค่าดีสุดต่อคอลัมน์ + สรุป approach ที่ดีสุด

baseline (k=4, relevant = `relevance_score>=2`): nDCG@4 — Subfacts 0.83 > Crimes 0.55 >
Long text 0.48 > Legal fact 0.36 > Laws 0.12

### นิยาม metric (case-level, uid-based)
- **nDCG@k** — graded relevance (`relevance_score`) เป็นน้ำหนัก, ตัวเปรียบเทียบหลัก
- **Hit@k** = top-k มี relevant ≥1 (0/1) · **Recall@k** = |rel ∩ top-k|/|rel| · **Precision@k** = |rel ∩ top-k|/k
- **MRR@k** = 1/อันดับ relevant ตัวแรก · relevant (binary) ปรับ threshold ได้ในหน้า

## หมายเหตุเรื่องลิงก์คำพิพากษา
ลิงก์ `detail/{uid}` เดิม **เปิดไม่ได้ (404)** — `uid` เป็น id ภายในของ scraper และเว็บ deka
ทางการไม่มี permalink ต่อคดี คอลัมน์ `link` ใน `candidate.csv`/`query_clean.csv` จึงเปลี่ยนเป็น
**Google search** key ด้วยเลขฎีกา · case info แสดงเลขฎีกาให้คัดลอก + ลิงก์ Google ·
ต้นฉบับสำรองที่ `*.csv.bak`

## Extract Law from text (FourCorners semantic search)
approach นี้จำลอง pipeline จริง: **ข้อความ → semantic search → มาตรา → คดีที่ co-cite**
1. ส่งข้อความ (default `legal_fact_result`) เข้า `search_legal_corpus` ของ FourCorners
   (hybrid vector + fulltext) — แตกเป็น topic phrase สั้น ๆ ก่อนส่ง
2. parse markdown ที่ได้กลับมาเป็นรายการมาตรา (`<law> มาตรา <n>`)
3. ใช้รายการมาตรานั้นเป็น query ของ **Laws BM25 index** → ได้คดีที่อ้างมาตราเดียวกัน
   (เหมือน approach Laws แต่มาตรามาจาก search ไม่ใช่ gold ของ query)

**Token:** หน้านี้ (และหน้า Metrics) มีช่อง 🔑 ให้วาง Bearer token ของ FourCorners เอง —
เก็บใน session เท่านั้น ไม่บันทึกลงดิสก์ · ตั้ง env `FOURCORNERS_TOKEN` / `FOURCORNERS_BASE_URL`
แทนได้ · base URL ดีฟอลต์ `http://10.204.100.77:6767` (เป็น private network ต้องมี tunnel)

**คะแนนใน Metrics Summary:** ติ๊กช่อง "รวม Extract Law" หลังใส่ token เพื่อให้เรียก API
ทีละ query (cache ต่อ token+พารามิเตอร์) แล้วคิด nDCG/Hit/Recall/Precision/MRR เทียบ approach อื่น —
query ที่ API ล้มเหลว/ว่างนับเป็น miss · client อยู่ใน `fourcorners.py` (parser เป็น pure ฟังก์ชัน)

## เฟสถัดไป (out of scope)
- retriever แบบ embedding / hybrid / re-rank — เพิ่มเข้า `APPROACHES` ใน `search_core.py` ได้เลย
- persist index ลงดิสก์ (ตอนนี้ build in-memory ตอน start)

## Dependencies (ติดตั้งครบแล้ว)
`streamlit`, `rank-bm25`, `pythainlp`, `pandas`, `numpy`
