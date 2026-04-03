import os
import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Import file logika (app.py) yang sudah kita bersihkan dari Flask tadi
import app as logic_engine 

# =========================
# KONFIGURASI UTAMA
# =========================
st.set_page_config(page_title="Coffee Insight Dashboard", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Folder model diasumsikan berada di dalam folder yang sama dengan ui_app.py
MODEL_DIR = os.path.join(BASE_DIR, "model")

TRAIN_RESULT_PATH = os.path.join(MODEL_DIR, "Hasil_Final_full.csv")
TEST_RESULT_PATH = os.path.join(MODEL_DIR, "hasil_2025_final.csv")

# =========================
# LOAD MODELS & ENGINES (Sekali Saja)
# =========================
@st.cache_resource
def load_all_assets():
    # Memanggil fungsi inisialisasi dari app.py
    sim, sent, pre, d_kws = logic_engine.init_all_engines()
    # Mengeset engine ke variabel global di app.py
    logic_engine.set_engines(sim, sent, pre, d_kws)
    return True

# Jalankan loading asset saat aplikasi dibuka
with st.sidebar:
    with st.spinner("Memuat Model ML..."):
        assets_loaded = load_all_assets()
    if assets_loaded:
        st.success("🤖 Model ML Siap")

# =========================
# FUNGSI UI (PIE CHART, BAR, DLL)
# =========================
def pie_from_dict(title: str, d: dict):
    if not d:
        st.info("Data kosong.")
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(d.values(), labels=d.keys(), autopct='%1.1f%%', startangle=90)
    ax.set_title(title)
    st.pyplot(fig)

def sidebar_logo():
    logo_path = os.path.join(BASE_DIR, "assets", "Logo.png")
    if os.path.exists(logo_path):
        st.image(Image.open(logo_path), use_container_width=True)

# =========================
# UI SIDEBAR & NAVIGASI
# =========================
with st.sidebar:
    sidebar_logo()
    st.markdown("### Navigasi")
    nav_group = st.radio("Pilih kategori", ["Mode", "Hasil Analisis"], index=0)

    if nav_group == "Mode":
        mode = st.radio("Pilih fitur", ["1) Similarity", "2) Sentiment", "3) Insight", "4) Prediksi Text"])
    else:
        hasil_mode = st.radio("Pilih hasil analisis", ["1) Hasil Data Latih", "2) Hasil Data Uji"])

st.title("Dashboard Intelligence Ide Konten Pemasaran")
st.caption("Berbasis Analisis Sentiment dan Similarity — Tanpa Flask API")
st.divider()

# =========================
# LOGIKA RENDER HALAMAN
# =========================

if nav_group == "Hasil Analisis":
    # Menampilkan file CSV lokal (Latih/Uji)
    target_path = TRAIN_RESULT_PATH if "Latih" in hasil_mode else TEST_RESULT_PATH
    st.subheader(f"Review {hasil_mode}")
    
    if os.path.exists(target_path):
        df_saved = pd.read_csv(target_path)
        st.write(f"Menampilkan {len(df_saved)} baris data.")
        st.dataframe(df_saved.head(100))
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Top Dimensi**")
            st.bar_chart(df_saved['dimensi_prediksi'].value_counts())
        with c2:
            st.markdown("**Top Sentimen**")
            st.bar_chart(df_saved['sentimen_prediksi'].value_counts())
    else:
        st.error(f"File tidak ditemukan di: {target_path}")

elif nav_group == "Mode":

    # --- MODE 4: PREDIKSI TEXT ---
    if mode == "4) Prediksi Text":
        st.subheader("Cek Analisis Teks Satuan")
        text_input = st.text_area("Masukkan ulasan kopi:", placeholder="Contoh: Rasanya enak tapi mahal")
        
        if st.button("Predict", type="primary"):
            if text_input:
                # Memanggil langsung fungsi dari app.py
                res = logic_engine.predict_text_logic(text_input)
                if res["ok"]:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Dimensi", res["dimensi_prediksi"])
                    col2.metric("Sentimen", res["sentimen_prediksi"])
                    col3.metric("Score", f"{res['dimensi_score']:.2f}")
                    st.info(f"Teks Bersih: {res['text_final']}")
                    with st.expander("Detail Proba"):
                        st.json(res)
                else:
                    st.error(res["message"])

    # --- MODE 1, 2, 3: CSV BATCH ---
    else:
        st.subheader(f"Mode {mode}")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        
        if uploaded:
            if st.button(f"Proses {mode}", type="primary"):
                with st.spinner("Sedang Menganalisis..."):
                    # Memanggil langsung fungsi CSV dari app.py
                    res = logic_engine.analyze_csv_logic(uploaded.getvalue())
                    
                    if res["ok"]:
                        st.success("Analisis Berhasil!")
                        c1, c2 = st.columns(2)
                        with c1: pie_from_dict("Distribusi Dimensi", res["dimensi_distribution"])
                        with c2: st.bar_chart(res["sentimen_distribution"])
                        
                        st.write("Preview Data:")
                        st.dataframe(pd.DataFrame(res["preview"]))
                        
                        # Fitur Download (Langsung dari dataframe hasil)
                        csv_data = res["full_df"].to_csv(index=False).encode('utf-8')
                        st.download_button("Download Hasil Lengkap (.csv)", csv_data, "hasil_analisis.csv", "text/csv")
                    else:
                        st.error(res["message"])