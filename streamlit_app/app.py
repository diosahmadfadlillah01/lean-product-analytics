import os  # untuk membaca path/folder file
import io  # untuk mengelola file/data bytes di memori
import json  # untuk membaca file JSON
import time  # untuk timestamp cache
import uuid  # untuk membuat token download unik
import hashlib  # untuk membuat fingerprint/md5 file upload
import traceback  # untuk menangkap detail error
from typing import Dict, Any, List, Tuple  # type hint agar kode lebih rapi

import pandas as pd  # untuk membaca dan mengolah CSV/DataFrame
from flask import Flask, request, jsonify, send_file  # komponen utama Flask API

from preprocessing import (
    Preprocessor,  # class preprocessing teks
    build_stopwords,  # fungsi membangun stopwords
    load_slang_map_from_json,  # load kamus slang dari JSON
    load_slang_map_from_excel,  # load kamus slang dari Excel
)
from similarity_engine import SimilarityEngine  # engine prediksi dimensi
from sentiment_engine import SentimentEngine  # engine prediksi sentimen


# =========================
# PATH
# =========================

# folder tempat app.py berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# folder model di dalam project
MODEL_DIR = os.path.join(BASE_DIR, "model")

# daftar file model yang wajib ada
# UBAH DI SINI kalau suatu saat nama file model berubah
REQUIRED_FILES = [
    os.path.join(MODEL_DIR, "keyword_dimension.json"),
    os.path.join(MODEL_DIR, "tfidf_vectorizer_dimensi.joblib"),
    os.path.join(MODEL_DIR, "tfidf_vectorizer_sentiment.joblib"),
    os.path.join(MODEL_DIR, "logreg_sentiment.joblib"),
]

# durasi cache dalam detik
# UBAH DI SINI kalau ingin cache lebih lama / lebih singkat
CACHE_TTL = int(os.environ.get("CACHE_TTL", "1800"))

# threshold internal
# UI tidak perlu mengirim threshold, semua diatur backend
# UBAH DI SINI kalau dosen minta threshold default diubah
DEFAULT_DIM_THR = float(os.environ.get("DEFAULT_DIM_THR", "0.0"))
DEFAULT_SENT_THR = float(os.environ.get("DEFAULT_SENT_THR", "0.3"))

# mapping hasil sentimen angka -> label teks
# UBAH DI SINI kalau label sentimen ingin diganti
SENTIMENT_MAP = {"-1": "Negatif", "0": "Netral", "1": "Positif"}


def _safe_dim_label(s: str) -> str:
    # fungsi ini merapikan label dimensi:
    # - strip spasi
    # - uppercase
    # - spasi jadi underscore
    return str(s).strip().upper().replace(" ", "_")


def _assert_files() -> None:
    # fungsi ini memastikan semua file model wajib tersedia
    missing = [p for p in REQUIRED_FILES if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("File model belum lengkap:\n- " + "\n- ".join(missing))


# jalankan pengecekan file model saat startup
_assert_files()


# =========================
# LOAD ASSETS (sekali)
# =========================

# load engine similarity sekali saat server menyala
sim_engine = SimilarityEngine(
    model_dir=MODEL_DIR,
    lexicon_filename="keyword_dimension.json",
    vectorizer_filename="tfidf_vectorizer_dimensi.joblib",
)

# load engine sentimen sekali saat server menyala
sent_engine = SentimentEngine(
    model_dir=MODEL_DIR,
    vectorizer_filename="tfidf_vectorizer_sentiment.joblib",
    model_filename="logreg_sentiment.joblib",
)

# baca file keyword dimensi dari JSON
with open(os.path.join(MODEL_DIR, "keyword_dimension.json"), "r", encoding="utf-8") as f:
    _kw_raw = json.load(f)

# bentuk kamus dimensi -> daftar keyword
DIM_KEYWORDS: Dict[str, List[str]] = {
    _safe_dim_label(k): [str(x).strip() for x in (v or []) if str(x).strip()]
    for k, v in _kw_raw.items()
}


def _dim_keywords_str(dim_label: str, limit: int = 15) -> str:
    # fungsi ini mengambil daftar keyword sebuah dimensi lalu dijadikan string
    kws = DIM_KEYWORDS.get(_safe_dim_label(dim_label), [])
    if not kws:
        return ""
    return ", ".join(kws[:limit])


# path kamus slang/baku
# NOTE:
# jika file kamus Anda sebenarnya berada di model/ dan bernama lain,
# bagian ini yang perlu disesuaikan
KAMUS_JSON = os.path.join(MODEL_DIR, "kamuskatabaku.json")
KAMUS_XLSX = os.path.join(BASE_DIR, "kamuskatabaku.xlsx")

# load kamus slang dari JSON jika ada
# kalau tidak ada, coba dari Excel
# kalau dua-duanya tidak ada, slang_map = {}
if os.path.exists(KAMUS_JSON):
    slang_map = load_slang_map_from_json(KAMUS_JSON)
elif os.path.exists(KAMUS_XLSX):
    slang_map = load_slang_map_from_excel(KAMUS_XLSX)
else:
    slang_map = {}

# stopwords tambahan khusus project ini
extra_stopwords = {
    "ya", "nih", "deh", "dong", "loh", "lah", "kok", "sih",
    "aja", "saja", "juga", "lagi", "banget",
    "gue", "gua", "aku", "kamu", "dia", "kita", "kami",
    "yang", "dan", "di", "ke", "dari", "untuk", "pada", "dengan",
    "itu", "ini", "ada", "akan", "tapi", "atau", "karena"
}

# kata yang ingin dipertahankan walaupun biasanya bisa ikut dibuang
keep_words = {"tidak", "kopi", "arabika", "robusta"}

# bangun stopwords final
stopwords = build_stopwords(extra=extra_stopwords, keep=keep_words)

# inisialisasi preprocessor utama
pre = Preprocessor(slang_map=slang_map, stopwords=stopwords)


# =========================
# CACHES
# =========================

# cache untuk hasil analisis CSV
ANALYZE_CACHE: Dict[str, Dict[str, Any]] = {}

# cache untuk hasil file download
DOWNLOAD_CACHE: Dict[str, Dict[str, Any]] = {}


def _now() -> float:
    # fungsi helper untuk ambil waktu sekarang
    return time.time()


def _cleanup_cache() -> None:
    # fungsi ini membersihkan cache yang sudah melewati TTL
    t = _now()

    # hapus cache analisis yang expired
    for k in list(ANALYZE_CACHE.keys()):
        if t - float(ANALYZE_CACHE[k]["ts"]) > CACHE_TTL:
            del ANALYZE_CACHE[k]

    # hapus cache download yang expired
    for k in list(DOWNLOAD_CACHE.keys()):
        if t - float(DOWNLOAD_CACHE[k]["ts"]) > CACHE_TTL:
            del DOWNLOAD_CACHE[k]


def _md5_bytes(b: bytes) -> str:
    # fungsi ini membuat fingerprint file upload berdasarkan bytes
    return hashlib.md5(b).hexdigest()


def _read_csv_bytes(csv_bytes: bytes) -> pd.DataFrame:
    # fungsi ini membaca CSV dari bytes upload
    # mencoba beberapa encoding agar lebih aman
    bio = io.BytesIO(csv_bytes)
    try:
        bio.seek(0)
        return pd.read_csv(bio, encoding="utf-8")
    except Exception:
        pass
    try:
        bio.seek(0)
        return pd.read_csv(bio, encoding="utf-8-sig")
    except Exception:
        pass
    bio.seek(0)
    return pd.read_csv(bio, encoding="latin-1")


def _pick_text_column(df: pd.DataFrame) -> str:
    # fungsi ini menentukan kolom teks terbaik dari CSV
    # dipakai saat user upload CSV mentah

    candidates = [
        "text_final", "text", "teks", "full_text", "tweet", "tweet_text",
        "content", "caption", "body", "komentar", "review",
        "filtered_text", "text_clean_2", "text_clean", "tokens_final",
        "Text", "FullText"
    ]

    # prioritas pertama: nama kolom yang umum
    for c in candidates:
        if c in df.columns:
            return c

    # jika tidak ada, cari semua kolom string
    str_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not str_cols:
        raise ValueError("Tidak ada kolom teks yang bisa dipakai (tipe string).")

    # pilih kolom string dengan rata-rata panjang teks terbesar
    means = {c: df[c].astype(str).str.len().mean() for c in str_cols}
    return max(means, key=means.get)


def _get_thresholds_internal() -> Tuple[float, float]:
    # fungsi ini mengembalikan threshold internal backend
    return float(DEFAULT_DIM_THR), float(DEFAULT_SENT_THR)


def _predict_batch_fast(cleaned_texts: List[str], dim_thr: float, sent_thr: float) -> Dict[str, Any]:
    # fungsi inti untuk prediksi batch:
    # - dimensi
    # - sentimen
    # dari list teks yang sudah dipreprocess

    n = len(cleaned_texts)

    # hanya teks non-kosong yang diproses model
    valid_idx = [i for i, t in enumerate(cleaned_texts) if str(t).strip()]

    # default output dimensi
    dim_pred = ["UNKNOWN"] * n
    dim_score = [0.0] * n
    dim_margin = [0.0] * n
    dim_scores_list: List[Dict[str, float]] = [{} for _ in range(n)]

    # default output sentimen
    sent_raw = ["UNKNOWN"] * n
    sent_pred = ["UNKNOWN"] * n
    sent_score = [0.0] * n
    sent_proba_list: List[Dict[str, float]] = [{} for _ in range(n)]

    # kalau tidak ada teks valid, kembalikan output default
    if not valid_idx:
        return {
            "dim_pred": dim_pred,
            "dim_score": dim_score,
            "dim_margin": dim_margin,
            "dim_scores_list": dim_scores_list,
            "sent_raw": sent_raw,
            "sent_pred": sent_pred,
            "sent_score": sent_score,
            "sent_proba_list": sent_proba_list,
        }

    valid_texts = [cleaned_texts[i] for i in valid_idx]

    # ===== DIMENSI =====
    # vectorize teks
    X_text = sim_engine.vectorizer.transform(valid_texts)  # type: ignore

    # hitung similarity ke semua label dimensi
    sim = (X_text @ sim_engine.X_dim.T).toarray()  # type: ignore

    # daftar label dimensi
    labels = list(sim_engine.labels)  # type: ignore

    for j, row in enumerate(sim):
        i = valid_idx[j]

        # simpan seluruh skor dimensi per baris
        score_dict = {_safe_dim_label(labels[k]): float(row[k]) for k in range(len(labels))}

        # urutkan skor dari terbesar
        order = sorted(range(len(labels)), key=lambda k: row[k], reverse=True)
        top1 = order[0]
        top2 = order[1] if len(order) > 1 else order[0]

        best_label = _safe_dim_label(labels[top1])
        best_score = float(row[top1])

        # margin = selisih skor pertama dan kedua
        margin = float(row[top1] - row[top2]) if len(order) > 1 else float(row[top1])

        # jika skor di bawah threshold -> UNKNOWN
        dim_pred[i] = "UNKNOWN" if best_score <= float(dim_thr) else best_label
        dim_score[i] = best_score
        dim_margin[i] = margin
        dim_scores_list[i] = score_dict

    # ===== SENTIMEN =====
    # vectorize teks untuk model sentimen
    Xs = sent_engine.vectorizer.transform(valid_texts)  # type: ignore

    # prediksi probabilitas sentimen
    proba = sent_engine.model.predict_proba(Xs)  # type: ignore

    # daftar class model sentimen
    classes = [str(c) for c in list(sent_engine.model.classes_)]  # type: ignore

    for j in range(len(valid_texts)):
        i = valid_idx[j]
        row = proba[j]

        # class dengan probabilitas tertinggi
        idx = int(row.argmax())
        raw = classes[idx]
        score = float(row[idx])

        # simpan seluruh probabilitas sentimen
        proba_dict = {classes[k]: float(row[k]) for k in range(len(classes))}

        # jika skor di bawah threshold -> UNKNOWN
        if score < float(sent_thr):
            sent_raw[i] = "UNKNOWN"
            sent_pred[i] = "UNKNOWN"
            sent_score[i] = score
        else:
            sent_raw[i] = raw
            sent_pred[i] = SENTIMENT_MAP.get(raw, raw)
            sent_score[i] = score

        sent_proba_list[i] = proba_dict

    return {
        "dim_pred": dim_pred,
        "dim_score": dim_score,
        "dim_margin": dim_margin,
        "dim_scores_list": dim_scores_list,
        "sent_raw": sent_raw,
        "sent_pred": sent_pred,
        "sent_score": sent_score,
        "sent_proba_list": sent_proba_list,
    }


def _build_pred_df_from_csv_bytes(csv_bytes: bytes, dim_thr: float, sent_thr: float) -> Tuple[pd.DataFrame, str]:
    # fungsi ini:
    # 1) baca CSV upload
    # 2) cari kolom teks
    # 3) preprocessing
    # 4) prediksi dimensi + sentimen
    # 5) membangun DataFrame hasil lengkap

    df = _read_csv_bytes(csv_bytes)
    if df.empty:
        raise ValueError("CSV kosong.")

    # hapus kolom Unnamed
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]

    # tentukan kolom teks
    text_col = _pick_text_column(df)

    # ambil teks mentah
    raw_texts = df[text_col].fillna("").astype(str).tolist()

    # preprocessing hemat: hanya per teks unik
    unique_texts = pd.Series(raw_texts).astype(str).unique().tolist()
    cleaned_map: Dict[str, str] = {}
    for t in unique_texts:
        cleaned_map[t] = pre.preprocess_text(t) if str(t).strip() else ""

    # map hasil preprocessing kembali ke urutan awal
    cleaned = [cleaned_map.get(t, "") for t in raw_texts]

    # prediksi batch
    pred = _predict_batch_fast(cleaned, dim_thr=dim_thr, sent_thr=sent_thr)

    # kumpulkan semua label dimensi yang muncul dalam skor
    all_dim_labels = sorted({k for d in pred["dim_scores_list"] for k in d.keys()})

    # siapkan kolom similarity per label dimensi
    sim_cols = {f"similarity_{lab}": [] for lab in all_dim_labels}
    for d in pred["dim_scores_list"]:
        for lab in all_dim_labels:
            sim_cols[f"similarity_{lab}"].append(float(d.get(lab, 0.0)))

    # siapkan kolom keyword dimensi
    dim_kw_col = []
    for lab in pred["dim_pred"]:
        dim_kw_col.append(_dim_keywords_str(lab, limit=15))

    # menandai baris valid
    ok_row = [1 if str(x).strip() else 0 for x in cleaned]

    # buat DataFrame hasil
    df_out = df.copy()
    df_out["ok_row"] = ok_row
    df_out["text_final"] = cleaned

    # kolom hasil dimensi
    df_out["dimensi_prediksi"] = pred["dim_pred"]
    df_out["dimensi_score"] = pred["dim_score"]
    df_out["dimensi_margin"] = pred["dim_margin"]
    df_out["dimensi_keywords"] = dim_kw_col

    # kolom hasil sentimen
    df_out["sentimen_raw"] = pred["sent_raw"]
    df_out["sentimen_prediksi"] = pred["sent_pred"]
    df_out["sentimen_score"] = pred["sent_score"]

    # kolom probabilitas sentimen
    proba_neg, proba_net, proba_pos = [], [], []
    for p in pred["sent_proba_list"]:
        proba_neg.append(float(p.get("-1", 0.0)))
        proba_net.append(float(p.get("0", 0.0)))
        proba_pos.append(float(p.get("1", 0.0)))

    df_out["proba_negatif"] = proba_neg
    df_out["proba_netral"] = proba_net
    df_out["proba_positif"] = proba_pos

    # tambahkan seluruh kolom similarity_*
    for c, values in sim_cols.items():
        df_out[c] = values

    return df_out, text_col


def _get_df_cached(csv_bytes: bytes, dim_thr: float, sent_thr: float) -> Tuple[pd.DataFrame, str]:
    # fungsi ini mengambil hasil analisis dari cache jika ada
    # jika belum ada, bangun hasil baru lalu simpan ke cache

    _cleanup_cache()

    # fingerprint file
    fp = _md5_bytes(csv_bytes)

    # cache key mempertimbangkan file + threshold
    cache_key = f"{fp}|dim{dim_thr:.3f}|sent{sent_thr:.3f}"

    # kalau ada di cache, ambil langsung
    if cache_key in ANALYZE_CACHE:
        return ANALYZE_CACHE[cache_key]["df"], ANALYZE_CACHE[cache_key]["text_col"]

    # kalau belum ada, proses ulang
    df_pred, text_col = _build_pred_df_from_csv_bytes(csv_bytes, dim_thr=dim_thr, sent_thr=sent_thr)
    ANALYZE_CACHE[cache_key] = {"ts": _now(), "df": df_pred, "text_col": text_col}
    return df_pred, text_col


def _top_counts(series: pd.Series, top_n: int, label_field: str) -> List[Dict[str, Any]]:
    # fungsi helper untuk membuat top counts ke format list of dict
    vc = series.value_counts(dropna=False).head(top_n)
    top = vc.reset_index(name="jumlah")
    top = top.rename(columns={top.columns[0]: label_field})
    return [{label_field: str(r[label_field]), "jumlah": int(r["jumlah"])} for _, r in top.iterrows()]


def _top_dimensions_with_keywords(dfv: pd.DataFrame, top_n: int = 6) -> List[Dict[str, Any]]:
    # fungsi helper untuk membuat top dimensi lengkap dengan keyword
    base = _top_counts(dfv["dimensi_prediksi"], top_n=top_n, label_field="dimensi")
    out = []
    for r in base:
        dim = str(r["dimensi"])
        out.append({
            "dimensi": dim,
            "jumlah": int(r["jumlah"]),
            "kata_kunci": _dim_keywords_str(dim, limit=15)
        })
    return out


def _shorten_text(s: str, limit: int = 220) -> str:
    # fungsi ini memendekkan teks agar preview contoh teks tidak terlalu panjang
    s = "" if s is None else str(s)
    s = s.replace("\n", " ").replace("\r", " ").strip()
    if len(s) <= limit:
        return s
    return s[:limit].rstrip() + "…"


# =========================
# FLASK
# =========================

# inisialisasi aplikasi Flask
app = Flask(__name__)


@app.get("/")
def health():
    # endpoint healthcheck
    # dipakai UI untuk mengecek backend aktif atau tidak
    return jsonify({
        "ok": True,
        "service": "LEAN API",
        "version": "4.3",
        "endpoints": {
            "POST /predict-text": "1 teks",
            "POST /analyze-dimensi": "CSV → similarity dimensi",
            "POST /analyze-sentimen": "CSV → sentiment",
            "POST /analyze-gabungan": "CSV → insight + bukti similarity + download",
            "GET /download/<token>": "download CSV hasil gabungan"
        },
        "thresholds": {"dimensi": DEFAULT_DIM_THR, "sentimen": DEFAULT_SENT_THR}
    })


@app.post("/predict-text")
def predict_text():
    # endpoint untuk memprediksi 1 teks langsung
    # dipakai oleh mode "Prediksi Text" di Streamlit
    try:
        payload = request.get_json(silent=True) or {}

        raw_text = payload.get("text", "")
        raw_text = "" if raw_text is None else str(raw_text)

        # preprocessing teks
        text_final = pre.preprocess_text(raw_text)

        # jika kosong setelah preprocessing -> error 400
        if not text_final.strip():
            return jsonify({
                "ok": False,
                "message": "Teks kosong setelah preprocessing",
                "input_text": raw_text,
                "text_final": text_final
            }), 400

        # ambil threshold internal
        dim_thr, sent_thr = _get_thresholds_internal()

        # prediksi dimensi dan sentimen untuk 1 teks
        dres = sim_engine.predict_one(text_final, unknown_threshold=float(dim_thr))
        sres = sent_engine.predict_one(text_final, unknown_threshold=float(sent_thr))

        sent_raw = str(sres.label)
        sent_pred = SENTIMENT_MAP.get(sent_raw, sent_raw)

        # kirim hasil ke frontend
        return jsonify({
            "ok": True,
            "input_text": raw_text,
            "text_final": text_final,

            "dimensi_prediksi": str(dres.label),
            "dimensi_score": float(dres.score),
            "dimensi_margin": float(getattr(dres, "margin", 0.0)),
            "dimensi_scores": dict(dres.scores),
            "dimensi_keywords": DIM_KEYWORDS.get(_safe_dim_label(str(dres.label)), [])[:25],

            "sentimen_raw": sent_raw,
            "sentimen_prediksi": sent_pred,
            "sentimen_score": float(sres.score),
            "sentimen_proba": dict(sres.scores),
        })
    except (ValueError, KeyError) as e:
        # error input user -> 400
        return jsonify({"ok": False, "message": str(e)}), 400
    except Exception as e:
        # error internal -> 500
        tb = traceback.format_exc()
        return jsonify({"ok": False, "message": "internal error", "error": str(e), "traceback": tb}), 500


@app.post("/analyze-dimensi")
def analyze_dimensi():
    # endpoint untuk analisis CSV fokus pada dimensi
    # dipakai mode "Similarity" di Streamlit
    try:
        # file wajib ada
        if "file" not in request.files:
            return jsonify({"ok": False, "message": "Form-data harus ada field file (CSV)."}), 400

        dim_thr, sent_thr = _get_thresholds_internal()

        f = request.files["file"]
        csv_bytes = f.read()
        if not csv_bytes:
            return jsonify({"ok": False, "message": "File kosong."}), 400

        # ambil hasil prediksi dari cache / bangun baru
        df_pred, text_col = _get_df_cached(csv_bytes, dim_thr=dim_thr, sent_thr=sent_thr)

        rows_total = int(len(df_pred))

        # pakai hanya baris valid
        dfv = df_pred[df_pred["ok_row"] == 1].copy()
        rows_used = int(len(dfv))

        # distribusi dimensi
        dist = dfv["dimensi_prediksi"].value_counts(dropna=False).to_dict()
        dist = {str(k): int(v) for k, v in dist.items()}

        # top dimensi + keyword
        top_list = _top_dimensions_with_keywords(dfv, top_n=6)

        # preview data
        prev_cols = [text_col, "text_final", "dimensi_prediksi", "dimensi_score", "dimensi_margin"]
        prev_cols = [c for c in prev_cols if c in df_pred.columns]
        preview = df_pred[prev_cols].head(20).to_dict(orient="records")

        return jsonify({
            "ok": True,
            "rows_total": rows_total,
            "rows_used": rows_used,
            "text_col": text_col,
            "dimensi_distribution": dist,
            "top_dimensions": top_list,
            "preview": preview
        })
    except (ValueError, KeyError) as e:
        return jsonify({"ok": False, "message": str(e)}), 400
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"ok": False, "message": "internal error", "error": str(e), "traceback": tb}), 500


@app.post("/analyze-sentimen")
def analyze_sentimen():
    # endpoint untuk analisis CSV fokus pada sentimen
    # dipakai mode "Sentiment" di Streamlit
    try:
        # file wajib ada
        if "file" not in request.files:
            return jsonify({"ok": False, "message": "Form-data harus ada field file (CSV)."}), 400

        dim_thr, sent_thr = _get_thresholds_internal()

        f = request.files["file"]
        csv_bytes = f.read()
        if not csv_bytes:
            return jsonify({"ok": False, "message": "File kosong."}), 400

        # ambil hasil prediksi dari cache / bangun baru
        df_pred, text_col = _get_df_cached(csv_bytes, dim_thr=dim_thr, sent_thr=sent_thr)

        rows_total = int(len(df_pred))

        # pakai hanya baris valid
        dfv = df_pred[df_pred["ok_row"] == 1].copy()
        rows_used = int(len(dfv))

        # distribusi sentimen
        dist = dfv["sentimen_prediksi"].value_counts(dropna=False).to_dict()
        dist = {str(k): int(v) for k, v in dist.items()}

        # top sentimen
        top_list = _top_counts(dfv["sentimen_prediksi"], top_n=10, label_field="sentimen")

        # preview data
        prev_cols = [text_col, "text_final", "sentimen_prediksi", "sentimen_score",
                     "proba_negatif", "proba_netral", "proba_positif"]
        prev_cols = [c for c in prev_cols if c in df_pred.columns]
        preview = df_pred[prev_cols].head(20).to_dict(orient="records")

        return jsonify({
            "ok": True,
            "rows_total": rows_total,
            "rows_used": rows_used,
            "text_col": text_col,
            "sentimen_distribution": dist,
            "top_sentimen": top_list,
            "preview": preview
        })
    except (ValueError, KeyError) as e:
        return jsonify({"ok": False, "message": str(e)}), 400
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"ok": False, "message": "internal error", "error": str(e), "traceback": tb}), 500


@app.post("/analyze-gabungan")
def analyze_gabungan():
    # endpoint untuk analisis gabungan:
    # - dimensi
    # - sentimen
    # - insight utama
    # - contoh teks
    # - bukti similarity
    # - token download CSV final
    # dipakai mode "Insight" di Streamlit
    try:
        # file wajib ada
        if "file" not in request.files:
            return jsonify({"ok": False, "message": "Form-data harus ada field file (CSV)."}), 400

        dim_thr, sent_thr = _get_thresholds_internal()

        f = request.files["file"]
        csv_bytes = f.read()
        if not csv_bytes:
            return jsonify({"ok": False, "message": "File kosong."}), 400

        # ambil hasil prediksi dari cache / bangun baru
        df_pred, text_col = _get_df_cached(csv_bytes, dim_thr=dim_thr, sent_thr=sent_thr)

        rows_total = int(len(df_pred))

        # pakai hanya baris valid
        dfv = df_pred[df_pred["ok_row"] == 1].copy()
        rows_used = int(len(dfv))

        # distribusi dimensi
        dist_dim = dfv["dimensi_prediksi"].value_counts(dropna=False).to_dict()
        dist_dim = {str(k): int(v) for k, v in dist_dim.items()}

        # distribusi sentimen
        dist_sent = dfv["sentimen_prediksi"].value_counts(dropna=False).to_dict()
        dist_sent = {str(k): int(v) for k, v in dist_sent.items()}

        # top dimensi dan top sentimen
        top_dimensions = _top_dimensions_with_keywords(dfv, top_n=6)
        top_sentimen = _top_counts(dfv["sentimen_prediksi"], top_n=10, label_field="sentimen")

        # kombinasi dimensi x sentimen
        gb = (
            dfv.groupby(["dimensi_prediksi", "sentimen_prediksi"])
            .size()
            .reset_index(name="jumlah")
            .sort_values("jumlah", ascending=False)
        )

        gb_records = []
        for _, r in gb.iterrows():
            dim = str(r["dimensi_prediksi"])
            sent = str(r["sentimen_prediksi"])
            gb_records.append({
                "dimensi": dim,
                "sentimen": sent,
                "jumlah": int(r["jumlah"]),
                "kata_kunci": _dim_keywords_str(dim, limit=15)
            })

        # insight utama
        dim_top = top_dimensions[0] if top_dimensions else {"dimensi": "-", "jumlah": 0, "kata_kunci": ""}
        sent_top = top_sentimen[0] if top_sentimen else {"sentimen": "-", "jumlah": 0}
        combo_top = gb_records[0] if gb_records else {"dimensi": "-", "sentimen": "-", "jumlah": 0, "kata_kunci": ""}

        insight_highlights = {
            "dimensi_terbanyak": dim_top,
            "sentimen_dominan": sent_top,
            "kombinasi_teratas": combo_top
        }

        # contoh teks untuk kombinasi teratas
        top_combo_examples = []
        for item in gb_records[:6]:
            dim = item["dimensi"]
            sent = item["sentimen"]
            subset = dfv[(dfv["dimensi_prediksi"] == dim) & (dfv["sentimen_prediksi"] == sent)]
            samples = subset[text_col].head(3).astype(str).tolist()
            samples = [_shorten_text(x, limit=220) for x in samples]

            top_combo_examples.append({
                "dimensi": dim,
                "sentimen": sent,
                "jumlah": int(item["jumlah"]),
                "kata_kunci": item["kata_kunci"],
                "contoh_teks": samples
            })

        # preview data
        prev_cols = [text_col, "text_final", "dimensi_prediksi", "dimensi_score",
                     "sentimen_prediksi", "sentimen_score", "dimensi_keywords"]
        prev_cols = [c for c in prev_cols if c in df_pred.columns]
        preview = df_pred[prev_cols].head(20).to_dict(orient="records")

        # ===== BUKTI TRANSPARANSI =====
        # tampilkan kolom similarity_* + dimensi_prediksi
        sim_cols = sorted([c for c in df_pred.columns if str(c).startswith("similarity_")])
        proof_cols = sim_cols + (["dimensi_prediksi"] if "dimensi_prediksi" in df_pred.columns else [])
        similarity_proof = df_pred[proof_cols].head(20).to_dict(orient="records")

        # siapkan file CSV final untuk didownload
        csv_out = df_pred.to_csv(index=False).encode("utf-8")
        token = uuid.uuid4().hex
        DOWNLOAD_CACHE[token] = {"ts": _now(), "bytes": csv_out}

        return jsonify({
            "ok": True,
            "rows_total": rows_total,
            "rows_used": rows_used,
            "text_col": text_col,

            "dimensi_distribution": dist_dim,
            "sentimen_distribution": dist_sent,
            "top_dimensions": top_dimensions,
            "top_sentimen": top_sentimen,

            "insight_highlights": insight_highlights,
            "groupby_dimensi_sentimen": gb_records,
            "top_combo_examples": top_combo_examples,

            "preview": preview,
            "similarity_proof": similarity_proof,
            "download_token": token
        })
    except (ValueError, KeyError) as e:
        return jsonify({"ok": False, "message": str(e)}), 400
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"ok": False, "message": "internal error", "error": str(e), "traceback": tb}), 500


@app.get("/download/<token>")
def download(token: str):
    # endpoint untuk download CSV hasil gabungan berdasarkan token
    try:
        _cleanup_cache()

        # cek token valid atau tidak
        if token not in DOWNLOAD_CACHE:
            return jsonify({"ok": False, "message": "Token download tidak valid atau sudah expired."}), 404

        # ambil bytes file dari cache
        data = DOWNLOAD_CACHE[token]["bytes"]

        # kirim sebagai file download
        mem = io.BytesIO(data)
        mem.seek(0)
        return send_file(
            mem,
            mimetype="text/csv",
            as_attachment=True,
            download_name="hasil_gabungan_dimensi_sentimen.csv"
        )
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"ok": False, "message": "internal error", "error": str(e), "traceback": tb}), 500


# =========================
# MENJALANKAN SERVER FLASK
# =========================
if __name__ == "__main__":
    # UBAH DI SINI kalau ingin ganti host/port backend
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)