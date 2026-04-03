import os  # untuk membaca path/folder file
import io  # untuk operasi data berbasis bytes/file di memori
import json  # untuk kirim/terima data JSON ke API
import ast  # untuk mengubah string dict menjadi dict Python secara aman

import pandas as pd  # untuk membaca dan mengolah CSV/DataFrame
import requests  # untuk komunikasi HTTP ke backend Flask
import streamlit as st  # framework UI dashboard
import matplotlib.pyplot as plt  # untuk membuat grafik pie chart
from PIL import Image  # untuk membuka logo gambar


# =========================
# KONFIGURASI UTAMA
# =========================

API_BASE = "http://127.0.0.1:5000"  # alamat backend Flask
CONNECT_TIMEOUT = 20  # batas waktu koneksi ke backend
READ_TIMEOUT = 5000  # batas waktu menunggu respons backend

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder tempat ui_app.py berada

# path file CSV hasil analisis tersimpan
# UBAH DI SINI kalau ingin ganti file hasil data latih
TRAIN_RESULT_PATH = os.path.join(BASE_DIR, "model", "Hasil_Final_full.csv")

# UBAH DI SINI kalau ingin ganti file hasil data uji
TEST_RESULT_PATH = os.path.join(BASE_DIR, "model", "hasil_2025_final.csv")

# mapping label sentimen angka -> teks
# UBAH DI SINI kalau ingin ganti label sentimen
SENTIMENT_LABEL_MAP = {
    "-1": "Negatif",
    "0": "Netral",
    "1": "Positif",
    -1: "Negatif",
    0: "Netral",
    1: "Positif",
}


# =========================
# FUNGSI BANTU UNTUK TIMEOUT REQUEST
# =========================
def req_timeout():
    # fungsi ini mengembalikan tuple timeout untuk request API
    return (CONNECT_TIMEOUT, READ_TIMEOUT)


# =========================
# CEK APAKAH BACKEND FLASK HIDUP
# =========================
def api_healthcheck():
    # fungsi ini memanggil endpoint root "/" untuk cek backend aktif atau tidak
    try:
        r = requests.get(f"{API_BASE}/", timeout=(10, 10))
        return r.ok, r.text
    except Exception as e:
        return False, str(e)


# =========================
# MENAMPILKAN ERROR HTTP DARI BACKEND DENGAN LEBIH JELAS
# =========================
def friendly_http_error(e: requests.HTTPError) -> str:
    # fungsi ini mengambil isi body error dari backend agar lebih mudah dibaca
    base = str(e)
    try:
        body = e.response.text
        return f"{base}\n\nServer response:\n{body}"
    except Exception:
        return base


# =========================
# KIRIM 1 TEKS KE BACKEND UNTUK DIPREDIKSI
# =========================
def post_predict_text(text: str):
    # fungsi ini dipakai oleh mode "Prediksi Text"
    payload = {"text": text}
    r = requests.post(
        f"{API_BASE}/predict-text",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=(CONNECT_TIMEOUT, 120),
    )
    r.raise_for_status()
    return r.json()


# =========================
# KIRIM FILE CSV KE BACKEND UNTUK DIANALISIS
# =========================
def post_analyze(endpoint: str, file_bytes: bytes, filename: str):
    # fungsi ini dipakai oleh mode Similarity, Sentiment, dan Insight
    files = {"file": (filename, file_bytes, "text/csv")}
    r = requests.post(
        f"{API_BASE}/{endpoint}",
        files=files,
        timeout=req_timeout(),
    )
    r.raise_for_status()
    return r.json()


# =========================
# AMBIL FILE CSV HASIL DARI BACKEND BERDASARKAN TOKEN
# =========================
def get_download(token: str) -> bytes:
    # fungsi ini dipakai untuk download CSV final dari mode Insight
    r = requests.get(f"{API_BASE}/download/{token}", timeout=req_timeout())
    r.raise_for_status()
    return r.content


# =========================
# MEMBUAT PIE CHART DARI DICTIONARY
# =========================
def pie_from_dict(title: str, d: dict, legend_title: str = "Label"):
    # fungsi ini membuat pie chart untuk distribusi data, misalnya distribusi dimensi
    if not d:
        st.info("Data kosong.")
        return

    labels = list(d.keys())
    sizes = list(d.values())

    fig, ax = plt.subplots(figsize=(5.2, 3.8))

    # menampilkan persen hanya jika cukup besar
    def _autopct(pct: float):
        return f"{pct:.1f}%" if pct >= 3 else ""

    wedges, texts, autotexts = ax.pie(
        sizes,
        autopct=_autopct,
        startangle=90,
        pctdistance=0.75
    )
    ax.set_title(title, fontsize=13)
    ax.axis("equal")  # memastikan pie chart bulat proporsional

    ax.legend(
        wedges,
        labels,
        title=legend_title,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        title_fontsize=10
    )

    st.pyplot(fig, clear_figure=True)


# =========================
# MEMBUAT BAR CHART SEDERHANA DARI DICTIONARY
# =========================
def bar_from_dict(title: str, d: dict):
    # fungsi ini dipakai untuk distribusi sentimen dan data kategori lain
    if not d:
        st.info("Data kosong.")
        return
    s = pd.Series(d).sort_values(ascending=False)
    st.markdown(f"**{title}**")
    st.bar_chart(s)


# =========================
# MEMBACA CSV LOKAL (HASIL LATIH / HASIL UJI)
# =========================
def read_local_csv(path: str) -> pd.DataFrame:
    # fungsi ini membaca CSV lokal tanpa melalui backend
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        pass
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        pass
    return pd.read_csv(path, encoding="latin-1")


# =========================
# MENENTUKAN KOLOM TEKS DARI CSV LOKAL
# =========================
def pick_text_col_local(df: pd.DataFrame) -> str:
    # fungsi ini mencari kolom teks yang paling cocok untuk contoh teks/preview
    candidates = [
        "full_text", "text", "text_final", "teks", "tweet", "tweet_text",
        "content", "caption", "body", "komentar", "review"
    ]
    for c in candidates:
        if c in df.columns:
            return c

    # kalau tidak ada nama kolom yang cocok, ambil kolom string terpanjang rata-rata
    str_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not str_cols:
        return df.columns[0]
    means = {c: df[c].astype(str).str.len().mean() for c in str_cols}
    return max(means, key=means.get)


# =========================
# MENORMALKAN LABEL SENTIMEN
# =========================
def normalize_sentiment_label(x):
    # fungsi ini mengubah label sentimen menjadi format yang rapi
    if pd.isna(x):
        return "UNKNOWN"
    s = str(x).strip()
    return SENTIMENT_LABEL_MAP.get(s, s)


# =========================
# MENGUBAH STRING DICT MENJADI DICT PYTHON
# =========================
def safe_parse_dict(v):
    # fungsi ini dipakai saat kolom scores_dimensi disimpan sebagai string dict
    if isinstance(v, dict):
        return v
    if pd.isna(v):
        return {}
    s = str(v).strip()
    if not s:
        return {}
    try:
        out = ast.literal_eval(s)
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


# =========================
# MENGAMBIL TOP COUNT DARI SERIES
# =========================
def top_counts_local(series: pd.Series, top_n: int, label_field: str):
    # fungsi ini dipakai untuk membuat top dimensi / top sentimen
    vc = series.fillna("UNKNOWN").astype(str).value_counts(dropna=False).head(top_n)
    top = vc.reset_index(name="jumlah")
    top = top.rename(columns={top.columns[0]: label_field})
    return [{label_field: str(r[label_field]), "jumlah": int(r["jumlah"])} for _, r in top.iterrows()]


# =========================
# MEMBANGUN RINGKASAN ANALISIS DARI CSV FINAL
# =========================
def build_saved_analysis_from_df(df: pd.DataFrame) -> dict:
    # fungsi ini adalah inti visualisasi untuk:
    # 1) Hasil Data Latih
    # 2) Hasil Data Uji

    df = df.copy()

    # menghapus kolom Unnamed yang biasanya muncul dari CSV export Excel
    df = df.loc[:, ~df.columns.astype(str).str.contains(r"^Unnamed")]

    # normalisasi nama kolom jika file lama memakai nama berbeda
    rename_map = {}
    if "score_dimensi" in df.columns and "dimensi_score" not in df.columns:
        rename_map["score_dimensi"] = "dimensi_score"
    if "score_sentimen" in df.columns and "sentimen_score" not in df.columns:
        rename_map["score_sentimen"] = "sentimen_score"
    df = df.rename(columns=rename_map)

    # menentukan kolom teks utama
    text_col = pick_text_col_local(df)

    # memastikan kolom hasil prediksi memang ada
    required_cols = ["dimensi_prediksi", "sentimen_prediksi"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "CSV hasil analisis belum siap divisualisasikan. "
            f"Kolom yang belum ada: {', '.join(missing)}"
        )

    # membersihkan isi kolom prediksi
    df["dimensi_prediksi"] = df["dimensi_prediksi"].fillna("UNKNOWN").astype(str)
    df["sentimen_prediksi"] = df["sentimen_prediksi"].apply(normalize_sentiment_label)

    # total baris data
    rows_total = int(len(df))
    rows_used = rows_total

    # distribusi dimensi
    dist_dim = df["dimensi_prediksi"].value_counts(dropna=False).to_dict()
    dist_dim = {str(k): int(v) for k, v in dist_dim.items()}

    # distribusi sentimen
    dist_sent = df["sentimen_prediksi"].value_counts(dropna=False).to_dict()
    dist_sent = {str(k): int(v) for k, v in dist_sent.items()}

    # top dimensi dan top sentimen
    top_dimensions = top_counts_local(df["dimensi_prediksi"], top_n=6, label_field="dimensi")
    top_sentimen = top_counts_local(df["sentimen_prediksi"], top_n=10, label_field="sentimen")

    # tabel kombinasi dimensi x sentimen
    gb = (
        df.groupby(["dimensi_prediksi", "sentimen_prediksi"])
        .size()
        .reset_index(name="jumlah")
        .sort_values("jumlah", ascending=False)
    )

    gb_records = []
    for _, r in gb.iterrows():
        gb_records.append({
            "dimensi": str(r["dimensi_prediksi"]),
            "sentimen": str(r["sentimen_prediksi"]),
            "jumlah": int(r["jumlah"]),
        })

    # insight utama
    dim_top = top_dimensions[0] if top_dimensions else {"dimensi": "-", "jumlah": 0}
    sent_top = top_sentimen[0] if top_sentimen else {"sentimen": "-", "jumlah": 0}
    combo_top = gb_records[0] if gb_records else {"dimensi": "-", "sentimen": "-", "jumlah": 0}

    insight_highlights = {
        "dimensi_terbanyak": dim_top,
        "sentimen_dominan": sent_top,
        "kombinasi_teratas": combo_top,
    }

    # contoh teks dari kombinasi teratas
    top_combo_examples = []
    for item in gb_records[:6]:
        dim = item["dimensi"]
        sent = item["sentimen"]
        subset = df[(df["dimensi_prediksi"] == dim) & (df["sentimen_prediksi"] == sent)]
        sample_texts = subset[text_col].fillna("").astype(str).head(3).tolist()

        top_combo_examples.append({
            "dimensi": dim,
            "sentimen": sent,
            "jumlah": int(item["jumlah"]),
            "contoh_teks": sample_texts,
        })

    # kolom preview yang ingin ditampilkan
    preview_cols = [
        text_col,
        "text_final",
        "dimensi_prediksi",
        "dimensi_score",
        "sentimen_prediksi",
        "sentimen_score",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]
    preview = df[preview_cols].head(20).to_dict(orient="records")

    # bukti skor dimensi
    score_proof = []
    if "scores_dimensi" in df.columns:
        # jika ada kolom scores_dimensi berbentuk dict/string dict
        proof_rows = []
        for _, row in df.head(20).iterrows():
            score_dict = safe_parse_dict(row.get("scores_dimensi"))
            one = {f"score_{k}": float(v) for k, v in score_dict.items()}
            one["dimensi_prediksi"] = row.get("dimensi_prediksi", "")
            proof_rows.append(one)
        score_proof = proof_rows
    else:
        # jika skor dimensi tersimpan dalam banyak kolom similarity_*
        score_cols = [c for c in df.columns if str(c).startswith("similarity_")]
        if score_cols:
            proof_cols = score_cols + ["dimensi_prediksi"]
            proof_cols = [c for c in proof_cols if c in df.columns]
            score_proof = df[proof_cols].head(20).to_dict(orient="records")

    # hasil akhir yang akan dipakai render UI
    return {
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
        "score_proof": score_proof,
        "all_columns": df.columns.tolist(),
    }


# =========================
# MEMASTIKAN MODE TERTENTU HANYA BISA JALAN JIKA API AKTIF
# =========================
def api_required_or_stop(api_ok: bool, api_msg: str):
    # fungsi ini dipakai oleh mode yang membutuhkan backend Flask
    if api_ok:
        return
    st.error("Mode ini membutuhkan backend Flask aktif.")
    st.code(api_msg)
    st.stop()


# =========================
# MENAMPILKAN HALAMAN HASIL DATA LATIH / HASIL DATA UJI
# =========================
def render_saved_analysis(title: str, csv_path: str, cache_key: str, show_raw: bool):
    # fungsi ini digunakan untuk menampilkan visualisasi cepat dari CSV final

    st.subheader(title)  # judul halaman
    st.write("Baca CSV hasil akhir → tampilkan visualisasi pembuktian analisis secara cepat.")
    st.caption(f"Sumber file: {csv_path}")  # menampilkan sumber file

    # tombol untuk membaca file dan menampilkan hasil
    run_btn = st.button(f"Tampilkan {title}", type="primary", key=f"btn_{cache_key}")

    if run_btn:
        with st.spinner("Membaca file hasil analisis..."):
            try:
                df_saved = read_local_csv(csv_path)
                out = build_saved_analysis_from_df(df_saved)
            except Exception as e:
                st.error("Gagal membaca hasil analisis.")
                st.code(str(e))
                st.session_state[cache_key] = None
                return
        st.session_state[cache_key] = out

    out = st.session_state.get(cache_key)
    if out:
        # ringkasan metrik
        st.markdown("### Ringkasan")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows total", out.get("rows_total", 0))
        c2.metric("Rows used", out.get("rows_used", 0))
        c3.metric("Kolom teks", out.get("text_col", "-"))

        # insight utama
        st.markdown("### Insight Utama")
        hi = out.get("insight_highlights", {}) or {}
        dim_top = hi.get("dimensi_terbanyak", {}) or {}
        sent_top = hi.get("sentimen_dominan", {}) or {}
        combo_top = hi.get("kombinasi_teratas", {}) or {}

        a, b, c = st.columns(3, gap="large")
        with a:
            st.metric("Dimensi terbanyak", dim_top.get("dimensi", "-"), f"{dim_top.get('jumlah', 0)}")
        with b:
            st.metric("Sentimen dominan", sent_top.get("sentimen", "-"), f"{sent_top.get('jumlah', 0)}")
        with c:
            st.metric(
                "Kombinasi teratas",
                f"{combo_top.get('dimensi', '-')} × {combo_top.get('sentimen', '-')}",
                f"{combo_top.get('jumlah', 0)}"
            )

        # grafik distribusi dimensi
        st.markdown("### Distribusi Dimensi")
        pie_from_dict("Distribusi Dimensi", out.get("dimensi_distribution", {}), legend_title="Dimensi")

        # tabel top dimensi
        st.markdown("### Top Dimensi")
        top_dim = out.get("top_dimensions", [])
        if top_dim:
            st.dataframe(pd.DataFrame(top_dim), use_container_width=True)
        else:
            st.info("Top dimensi kosong.")

        # grafik distribusi sentimen
        st.markdown("### Distribusi Sentimen")
        bar_from_dict("Distribusi Sentimen", out.get("sentimen_distribution", {}))

        # tabel top sentimen
        st.markdown("### Top Sentimen")
        top_sent = out.get("top_sentimen", [])
        if top_sent:
            st.dataframe(pd.DataFrame(top_sent), use_container_width=True)
        else:
            st.info("Top sentimen kosong.")

        # tabel kombinasi dimensi x sentimen
        st.markdown("### Kombinasi Dimensi × Sentimen")
        gb = out.get("groupby_dimensi_sentimen", [])
        if gb:
            st.dataframe(pd.DataFrame(gb).head(30), use_container_width=True)
        else:
            st.info("Data kombinasi kosong.")

        # contoh teks
        st.markdown("### Contoh Teks")
        examples = out.get("top_combo_examples", [])
        if examples:
            for item in examples:
                title_exp = f"{item.get('dimensi', '-')} × {item.get('sentimen', '-')} — {item.get('jumlah', 0)}"
                with st.expander(title_exp, expanded=False):
                    texts = item.get("contoh_teks", []) or []
                    if texts:
                        for t in texts:
                            st.write(f"- {t}")
                    else:
                        st.info("Contoh teks kosong.")
        else:
            st.info("Belum ada contoh teks.")

        # bukti skor dimensi
        st.markdown("### Bukti Skor Dimensi")
        proof = out.get("score_proof", [])
        if proof:
            st.dataframe(pd.DataFrame(proof).head(20), use_container_width=True)
        else:
            st.info("Kolom bukti skor belum tersedia pada CSV ini.")

        # preview 20 baris
        st.markdown("### Preview (20 baris)")
        prev = out.get("preview", [])
        if prev:
            st.dataframe(pd.DataFrame(prev), use_container_width=True)
        else:
            st.info("Preview kosong.")

        # jika checkbox detail dicentang, tampilkan data mentah
        if show_raw:
            st.json(out)
    else:
        st.info(f"Belum ada hasil. Klik tombol Tampilkan {title}.")


# =========================
# UI UTAMA
# =========================

# UBAH DI SINI kalau ingin ganti judul tab browser
st.set_page_config(page_title="Coffee Insight Dashboard", layout="wide")

# UBAH DI SINI kalau ingin ganti judul dashboard
st.title("Dashboard Intelligence Ide Konten Pemasaran Berbasis Analisis Sentiment dan Similarity")

# UBAH DI SINI kalau ingin ganti caption dashboard
st.caption("Analisis Instant Untuk Rekomendasi Konten Copywriter")

# cek apakah backend aktif
api_ok, api_msg = api_healthcheck()
if api_ok:
    st.success("API: Connected")
else:
    # walaupun backend mati, mode Hasil Analisis tetap bisa dipakai
    st.warning("API: Not Connected. Mode Hasil Analisis tetap bisa dipakai karena membaca CSV lokal.")
    with st.expander("Lihat detail koneksi API"):
        st.code(api_msg)

st.divider()

# session_state untuk menyimpan hasil sementara agar tidak hilang saat rerun
for k in ["m1_last", "m2_last", "m3_last", "m4_last", "latih_last", "uji_last"]:
    if k not in st.session_state:
        st.session_state[k] = None


# =========================
# MENAMPILKAN LOGO DI SIDEBAR
# =========================
def sidebar_logo():
    # fungsi ini mencari dan menampilkan logo di folder assets
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(base_dir, "assets", "Logo.png")

    if os.path.exists(logo_path):
        try:
            img = Image.open(logo_path)
            st.image(img, use_container_width=True)
        except Exception:
            st.warning("Logo ada, tapi gagal dibaca. Cek file Logo.png.")
    else:
        st.warning("Logo tidak ditemukan: assets/Logo.png")


# =========================
# SIDEBAR NAVIGASI
# =========================
with st.sidebar:
    sidebar_logo()  # menampilkan logo

    st.markdown("### Navigasi")

    # navigasi utama: memilih apakah user ingin masuk ke Mode atau Hasil Analisis
    nav_group = st.radio(
        "Pilih kategori",
        [
            "Mode",
            "Hasil Analisis",
        ],
        index=0,
        key="nav_group"
    )

    mode = None
    hasil_mode = None

    # jika user memilih kategori Mode
    if nav_group == "Mode":
        st.markdown("### Mode")
        mode = st.radio(
            "Pilih fitur",
            [
                "1) Similarity",
                "2) Sentiment",
                "3) Insight",
                "4) Prediksi Text",
            ],
            index=0,
            key="mode_radio"
        )

    # jika user memilih kategori Hasil Analisis
    elif nav_group == "Hasil Analisis":
        st.markdown("### Hasil Analisis")
        hasil_mode = st.radio(
            "Pilih hasil analisis",
            [
                "1) Hasil Data Latih",
                "2) Hasil Data Uji",
            ],
            index=0,
            key="hasil_radio"
        )

    st.markdown("---")

    # checkbox untuk menampilkan data mentah/detail
    show_raw = st.checkbox("Lihat Detail Data", value=False)

    # tombol untuk menghapus cache tampilan
    if st.button("Hapus Cache Data"):
        st.session_state["m1_last"] = None
        st.session_state["m2_last"] = None
        st.session_state["m3_last"] = None
        st.session_state["m4_last"] = None
        st.session_state["latih_last"] = None
        st.session_state["uji_last"] = None
        st.success("Cache dibersihkan.")


# =========================
# RENDER HALAMAN BERDASARKAN NAVIGASI
# =========================

# ===== jika user memilih kategori Hasil Analisis =====
if nav_group == "Hasil Analisis":
    # halaman Hasil Data Latih
    if hasil_mode == "1) Hasil Data Latih":
        render_saved_analysis(
            title="Hasil Data Latih",
            csv_path=TRAIN_RESULT_PATH,
            cache_key="latih_last",
            show_raw=show_raw,
        )

    # halaman Hasil Data Uji
    elif hasil_mode == "2) Hasil Data Uji":
        render_saved_analysis(
            title="Hasil Data Uji",
            csv_path=TEST_RESULT_PATH,
            cache_key="uji_last",
            show_raw=show_raw,
        )


# ===== jika user memilih kategori Mode =====
elif nav_group == "Mode":

    # =========================
    # MODE 1: SIMILARITY
    # =========================
    if mode == "1) Similarity":
        api_required_or_stop(api_ok, api_msg)  # mode ini butuh backend aktif

        st.subheader("Similarity (CSV)")  # judul halaman similarity
        st.write("Upload CSV → distribusi dimensi + top dimensi (dengan kata kunci) + preview data.")

        # upload file CSV
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="m1_file")

        # tombol proses similarity
        run_btn = st.button(
            "Proses Similarity",
            type="primary",
            disabled=(uploaded is None and st.session_state["m1_last"] is None)
        )

        # kirim file ke backend jika tombol ditekan
        if run_btn and uploaded is not None:
            with st.spinner("Memproses..."):
                try:
                    out = post_analyze("analyze-dimensi", uploaded.getvalue(), uploaded.name)
                except requests.HTTPError as e:
                    st.error("Gagal hit /analyze-dimensi")
                    st.code(friendly_http_error(e))
                    st.stop()
                except Exception as e:
                    st.error("Error")
                    st.code(str(e))
                    st.stop()
            st.session_state["m1_last"] = out

        # tampilkan hasil terakhir
        out = st.session_state["m1_last"]
        if out:
            st.markdown("### Ringkasan")
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows total", out.get("rows_total", 0))
            c2.metric("Rows used", out.get("rows_used", 0))
            c3.metric("Kolom teks", out.get("text_col", "-"))

            dist = out.get("dimensi_distribution", {})
            top = out.get("top_dimensions", [])

            st.markdown("### Hasil Similarity")
            left, right = st.columns([1.2, 1.0], gap="large")
            with left:
                pie_from_dict("Distribusi Dimensi", dist, legend_title="Dimensi")
            with right:
                st.markdown("**Top Dimensi + Kata Kunci**")
                if top:
                    df_top = pd.DataFrame(top)
                    cols = [c for c in ["dimensi", "jumlah", "kata_kunci"] if c in df_top.columns]
                    st.dataframe(df_top[cols], use_container_width=True)
                else:
                    st.info("Top dimensi kosong.")

            st.markdown("### Preview (20 baris)")
            prev = out.get("preview", [])
            if prev:
                st.dataframe(pd.DataFrame(prev), use_container_width=True)
            else:
                st.info("Preview kosong.")

            if show_raw:
                st.json(out)
        else:
            st.info("Belum ada hasil. Upload CSV lalu klik Proses Similarity.")


    # =========================
    # MODE 2: SENTIMENT
    # =========================
    elif mode == "2) Sentiment":
        api_required_or_stop(api_ok, api_msg)  # mode ini butuh backend aktif

        st.subheader("Sentiment (CSV)")  # judul halaman sentiment
        st.write("Upload CSV → distribusi sentimen + preview data.")

        # upload file CSV
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="m2_file")

        # tombol proses sentiment
        run_btn = st.button(
            "Proses Sentiment",
            type="primary",
            disabled=(uploaded is None and st.session_state["m2_last"] is None)
        )

        # kirim file ke backend jika tombol ditekan
        if run_btn and uploaded is not None:
            with st.spinner("Memproses..."):
                try:
                    out = post_analyze("analyze-sentimen", uploaded.getvalue(), uploaded.name)
                except requests.HTTPError as e:
                    st.error("Gagal hit /analyze-sentimen")
                    st.code(friendly_http_error(e))
                    st.stop()
                except Exception as e:
                    st.error("Error")
                    st.code(str(e))
                    st.stop()
            st.session_state["m2_last"] = out

        # tampilkan hasil terakhir
        out = st.session_state["m2_last"]
        if out:
            st.markdown("### Ringkasan")
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows total", out.get("rows_total", 0))
            c2.metric("Rows used", out.get("rows_used", 0))
            c3.metric("Kolom teks", out.get("text_col", "-"))

            dist = out.get("sentimen_distribution", {})
            st.markdown("### Distribusi Sentimen")
            bar_from_dict("Distribusi Sentimen", dist)

            top = out.get("top_sentimen", [])
            if top:
                st.markdown("### Top Sentimen")
                st.dataframe(pd.DataFrame(top), use_container_width=True)
            else:
                st.info("Top sentimen kosong.")

            st.markdown("### Preview (20 baris)")
            prev = out.get("preview", [])
            if prev:
                st.dataframe(pd.DataFrame(prev), use_container_width=True)
            else:
                st.info("Preview kosong.")

            if show_raw:
                st.json(out)
        else:
            st.info("Belum ada hasil. Upload CSV lalu klik Proses Sentiment.")


    # =========================
    # MODE 3: INSIGHT
    # =========================
    elif mode == "3) Insight":
        api_required_or_stop(api_ok, api_msg)  # mode ini butuh backend aktif

        st.subheader("Insight (CSV)")  # judul halaman insight
        st.write("Upload CSV → ringkasan insight + kombinasi dimensi×sentimen + contoh teks + bukti similarity + download CSV final.")

        # upload file CSV
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="m3_file")

        # tombol proses insight
        run_btn = st.button(
            "Proses Insight",
            type="primary",
            disabled=(uploaded is None and st.session_state["m3_last"] is None)
        )

        # kirim file ke backend jika tombol ditekan
        if run_btn and uploaded is not None:
            with st.spinner("Memproses... (data besar bisa lama)"):
                try:
                    out = post_analyze("analyze-gabungan", uploaded.getvalue(), uploaded.name)
                except requests.HTTPError as e:
                    st.error("Gagal hit /analyze-gabungan")
                    st.code(friendly_http_error(e))
                    st.stop()
                except Exception as e:
                    st.error("Error")
                    st.code(str(e))
                    st.stop()
            st.session_state["m3_last"] = out

        # tampilkan hasil terakhir
        out = st.session_state["m3_last"]
        if out:
            st.markdown("### Ringkasan")
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows total", out.get("rows_total", 0))
            c2.metric("Rows used", out.get("rows_used", 0))
            c3.metric("Kolom teks", out.get("text_col", "-"))

            st.markdown("### Insight Utama")
            hi = out.get("insight_highlights", {}) or {}
            dim_top = hi.get("dimensi_terbanyak", {}) or {}
            sent_top = hi.get("sentimen_dominan", {}) or {}
            combo_top = hi.get("kombinasi_teratas", {}) or {}

            a, b, c = st.columns(3, gap="large")
            with a:
                st.metric("Dimensi terbanyak", dim_top.get("dimensi", "-"), f"{dim_top.get('jumlah', 0)}")
                if dim_top.get("kata_kunci"):
                    st.caption(f"Kata kunci: {dim_top.get('kata_kunci')}")
            with b:
                st.metric("Sentimen dominan", sent_top.get("sentimen", "-"), f"{sent_top.get('jumlah', 0)}")
            with c:
                st.metric(
                    "Kombinasi teratas",
                    f"{combo_top.get('dimensi', '-')} × {combo_top.get('sentimen', '-')}",
                    f"{combo_top.get('jumlah', 0)}"
                )
                if combo_top.get("kata_kunci"):
                    st.caption(f"Kata kunci: {combo_top.get('kata_kunci')}")

            st.markdown("### Kombinasi Dimensi × Sentimen (dengan kata kunci)")
            gb = out.get("groupby_dimensi_sentimen", [])
            if gb:
                df_gb = pd.DataFrame(gb)
                cols = [c for c in ["dimensi", "sentimen", "jumlah", "kata_kunci"] if c in df_gb.columns]
                st.dataframe(df_gb[cols].head(30), use_container_width=True)
            else:
                st.info("Tabel kombinasi kosong.")

            st.markdown("### Contoh Teks untuk Copywriter (Top Kombinasi)")
            examples = out.get("top_combo_examples", [])
            if examples:
                for item in examples:
                    title = f"{item.get('dimensi', '-')} × {item.get('sentimen', '-')} — {item.get('jumlah', 0)}"
                    with st.expander(title, expanded=False):
                        kk = item.get("kata_kunci", "")
                        if kk:
                            st.write(f"**Kata kunci:** {kk}")
                        texts = item.get("contoh_teks", []) or []
                        if texts:
                            for t in texts:
                                st.write(f"- {t}")
                        else:
                            st.info("Contoh teks kosong.")
            else:
                st.info("Belum ada contoh teks.")

            st.markdown("### Bukti Transparansi Similarity (Dimensi dipilih dari skor terbesar)")
            proof = out.get("similarity_proof", [])
            if proof:
                df_proof = pd.DataFrame(proof)
                sim_cols = sorted([c for c in df_proof.columns if str(c).startswith("similarity_")])
                cols = sim_cols + (["dimensi_prediksi"] if "dimensi_prediksi" in df_proof.columns else [])
                st.dataframe(df_proof[cols].head(20), use_container_width=True)
            else:
                st.info("Bukti similarity belum tersedia.")

            st.markdown("### Preview (20 baris)")
            prev = out.get("preview", [])
            if prev:
                st.dataframe(pd.DataFrame(prev), use_container_width=True)
            else:
                st.info("Preview kosong.")

            # tombol ambil file final dari backend
            token = out.get("download_token")
            if token:
                if st.button("Ambil CSV Final"):
                    with st.spinner("Mengambil file..."):
                        try:
                            csv_bytes = get_download(token)
                        except requests.HTTPError as e:
                            st.error("Gagal download")
                            st.code(friendly_http_error(e))
                            st.stop()
                        except Exception as e:
                            st.error("Error")
                            st.code(str(e))
                            st.stop()

                    # tombol download file final
                    st.download_button(
                        label="Download hasil_gabungan_dimensi_sentimen.csv",
                        data=csv_bytes,
                        file_name="hasil_gabungan_dimensi_sentimen.csv",
                        mime="text/csv",
                    )
            else:
                st.warning("Token download tidak tersedia.")

            if show_raw:
                st.json(out)
        else:
            st.info("Belum ada hasil. Upload CSV lalu klik Proses Insight.")


    # =========================
    # MODE 4: PREDIKSI TEXT
    # =========================
    elif mode == "4) Prediksi Text":
        api_required_or_stop(api_ok, api_msg)  # mode ini butuh backend aktif

        st.subheader("Prediksi Text")  # judul halaman prediksi text
        st.write("Masukkan 1 teks → hasil dimensi (similarity) + sentimen.")

        # input teks manual
        text = st.text_area("Input text", height=140, placeholder="Contoh: kopi pahit banget")

        # tombol predict
        btn = st.button("Predict", type="primary", disabled=(not text.strip()))

        # kirim teks ke backend jika tombol ditekan
        if btn:
            with st.spinner("Memproses prediksi..."):
                try:
                    out = post_predict_text(text.strip())
                except requests.HTTPError as e:
                    st.error("Gagal hit /predict-text")
                    st.code(friendly_http_error(e))
                    st.stop()
                except Exception as e:
                    st.error("Error")
                    st.code(str(e))
                    st.stop()
            st.session_state["m4_last"] = out

        # tampilkan hasil terakhir
        out = st.session_state["m4_last"]
        if out:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Dimensi", out.get("dimensi_prediksi", "-"))
            c2.metric("Score Dimensi", f"{float(out.get('dimensi_score', 0.0)):.3f}")
            c3.metric("Sentimen", out.get("sentimen_prediksi", "-"))
            c4.metric("Score Sentimen", f"{float(out.get('sentimen_score', 0.0)):.3f}")

            st.write("**Text final:**")
            st.code(out.get("text_final", ""), language="text")

            left, right = st.columns([2, 1], gap="large")
            with left:
                st.subheader("Skor Dimensi (detail)")
                st.json(out.get("dimensi_scores", {}))
                st.subheader("Keyword Dimensi (top)")
                st.write(", ".join(out.get("dimensi_keywords", [])[:25]))

            with right:
                st.subheader("Proba Sentimen")
                st.json(out.get("sentimen_proba", {}))

            if show_raw:
                st.json(out)
        else:
            st.info("Belum ada hasil. Tulis teks lalu klik Predict.")