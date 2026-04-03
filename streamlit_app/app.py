import os
import io
import json
import uuid
import hashlib
import traceback
import pandas as pd
from typing import Dict, Any, List, Tuple

# Import modul buatan Anda (Pastikan file ini ada di folder yang sama)
from preprocessing import (
    Preprocessor,
    build_stopwords,
    load_slang_map_from_json,
    load_slang_map_from_excel,
)
from similarity_engine import SimilarityEngine
from sentiment_engine import SentimentEngine

# =========================
# PATH & CONFIGURATION
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Sesuaikan folder model (karena file-file ini biasanya ada di folder 'model')
MODEL_DIR = os.path.join(BASE_DIR, "model")

SENTIMENT_MAP = {"-1": "Negatif", "0": "Netral", "1": "Positif"}
DEFAULT_DIM_THR = 0.0
DEFAULT_SENT_THR = 0.3

# =========================
# HELPER FUNCTIONS
# =========================
def _safe_dim_label(s: str) -> str:
    return str(s).strip().upper().replace(" ", "_")

def _read_csv_bytes(csv_bytes: bytes) -> pd.DataFrame:
    bio = io.BytesIO(csv_bytes)
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            bio.seek(0)
            return pd.read_csv(bio, encoding=enc)
        except:
            continue
    return pd.read_csv(bio)

def _pick_text_column(df: pd.DataFrame) -> str:
    candidates = ["text_final", "text", "teks", "review", "komentar", "content"]
    for c in candidates:
        if c in df.columns: return c
    str_cols = [c for c in df.columns if df[c].dtype == "object"]
    return max(str_cols, key=lambda c: df[c].astype(str).str.len().mean()) if str_cols else df.columns[0]

def _dim_keywords_str(dim_label: str, dim_keywords_dict: dict, limit: int = 15) -> str:
    kws = dim_keywords_dict.get(_safe_dim_label(dim_label), [])
    return ", ".join(kws[:limit]) if kws else ""

# =========================
# LOAD ASSETS (Singleton Pattern)
# =========================
# Fungsi ini akan dipanggil oleh UI agar model di-load hanya sekali
def init_all_engines():
    sim = SimilarityEngine(
        model_dir=MODEL_DIR,
        lexicon_filename="keyword_dimension.json",
        vectorizer_filename="tfidf_vectorizer_dimensi.joblib",
    )
    sent = SentimentEngine(
        model_dir=MODEL_DIR,
        vectorizer_filename="tfidf_vectorizer_sentiment.joblib",
        model_filename="logreg_sentiment.joblib",
    )
    
    # Load Keywords
    kw_path = os.path.join(MODEL_DIR, "keyword_dimension.json")
    with open(kw_path, "r", encoding="utf-8") as f:
        kw_raw = json.load(f)
    dim_kws = {_safe_dim_label(k): [str(x).strip() for x in (v or [])] for k, v in kw_raw.items()}

    # Load Preprocessor
    kamus_xlsx = os.path.join(MODEL_DIR, "kamuskatabaku (2).xlsx")
    slang_map = load_slang_map_from_excel(kamus_xlsx) if os.path.exists(kamus_xlsx) else {}
    pre = Preprocessor(slang_map=slang_map, stopwords=build_stopwords())

    return sim, sent, pre, dim_kws

# Variabel Global untuk menyimpan engine (akan diisi oleh ui_app.py)
sim_engine, sent_engine, pre, DIM_KEYWORDS = None, None, None, None

def set_engines(s, se, p, dk):
    global sim_engine, sent_engine, pre, DIM_KEYWORDS
    sim_engine, sent_engine, pre, DIM_KEYWORDS = s, se, p, dk

# =========================
# CORE LOGIC FUNCTIONS (Pengganti API Endpoints)
# =========================

def predict_text_logic(text: str) -> Dict[str, Any]:
    """Pengganti endpoint /predict-text"""
    text_final = pre.preprocess_text(text)
    if not text_final.strip():
        return {"ok": False, "message": "Teks kosong setelah preprocessing"}

    dres = sim_engine.predict_one(text_final, unknown_threshold=DEFAULT_DIM_THR)
    sres = sent_engine.predict_one(text_final, unknown_threshold=DEFAULT_SENT_THR)

    sent_raw = str(sres.label)
    return {
        "ok": True,
        "input_text": text,
        "text_final": text_final,
        "dimensi_prediksi": str(dres.label),
        "dimensi_score": float(dres.score),
        "dimensi_scores": dict(dres.scores),
        "dimensi_keywords": DIM_KEYWORDS.get(_safe_dim_label(str(dres.label)), [])[:25],
        "sentimen_raw": sent_raw,
        "sentimen_prediksi": SENTIMENT_MAP.get(sent_raw, sent_raw),
        "sentimen_score": float(sres.score),
        "sentimen_proba": dict(sres.scores),
    }

def analyze_csv_logic(csv_bytes: bytes, mode: str = "gabungan") -> Dict[str, Any]:
    """Pengganti endpoint /analyze-dimensi, /analyze-sentimen, & /analyze-gabungan"""
    df = _read_csv_bytes(csv_bytes)
    if df.empty: return {"ok": False, "message": "CSV Kosong"}

    text_col = _pick_text_column(df)
    raw_texts = df[text_col].fillna("").astype(str).tolist()
    
    # Preprocessing & Prediction
    results = []
    for t in raw_texts:
        clean_t = pre.preprocess_text(t) if t.strip() else ""
        if not clean_t:
            results.append({"dim": "UNKNOWN", "sent": "Netral", "ds": 0, "ss": 0})
            continue
        
        d = sim_engine.predict_one(clean_t, unknown_threshold=DEFAULT_DIM_THR)
        s = sent_engine.predict_one(clean_t, unknown_threshold=DEFAULT_SENT_THR)
        
        results.append({
            "text_final": clean_t,
            "dimensi_prediksi": d.label,
            "dimensi_score": d.score,
            "sentimen_prediksi": SENTIMENT_MAP.get(str(s.label), "Netral"),
            "sentimen_score": s.score
        })

    res_df = pd.DataFrame(results)
    final_df = pd.concat([df.reset_index(drop=True), res_df], axis=1)

    # Menghasilkan output yang mirip dengan JSON API Anda agar ui_app tidak perlu banyak berubah
    return {
        "ok": True,
        "rows_total": len(df),
        "rows_used": len(df),
        "text_col": text_col,
        "dimensi_distribution": final_df["dimensi_prediksi"].value_counts().to_dict(),
        "sentimen_distribution": final_df["sentimen_prediksi"].value_counts().to_dict(),
        "preview": final_df.head(20).to_dict(orient="records"),
        "full_df": final_df # Kita kirim DF aslinya agar Streamlit bisa download
    }