import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import io

# ==============================================================================
# KONFIGURASI HALAMAN & TEMA UI
# ==============================================================================
st.set_page_config(
    page_title="Electricity Usage Classifier",
    page_icon="⚡",
    layout="centered"
)

# Custom CSS untuk mempercantik UI
st.markdown("""
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 16px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-box {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚡ Electricity Usage Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Aplikasi cerdas berbasis Machine Learning untuk memprediksi dan mengklasifikasikan efisiensi konsumsi listrik rumah tangga.</div>', unsafe_allow_html=True)

# ==============================================================================
# RETRIEVE & TRAIN MODEL SECARA OTOMATIS (BACKGROUND PROCESS)
# ==============================================================================
@st.cache_resource
def inisialisasi_dan_latih_model():
    try:
        # Membaca dataset utama dari repositori
        df_base = pd.read_excel('household_electricity_usage.xlsx')
        
        # Kalkulasi nilai dasar
        df_base['usage_score'] = df_base['power_watts'] * df_base['duration_minutes'] * df_base['occupancy_count']
        
        # Batas Kuantil (Sama seperti analisis sebelumnya)
        q1 = df_base['usage_score'].quantile(0.33)
        q2 = df_base['usage_score'].quantile(0.66)
        
        def hitung_kategori(score):
            if score <= q1: return "Rendah"
            elif score <= q2: return "Normal"
            else: return "Boros"
            
        df_base['kategori'] = df_base['usage_score'].apply(hitung_kategori)
        
        # Variabel Independen & Dependen
        X = df_base[['household_size', 'occupancy_count', 'power_watts', 'duration_minutes']]
        y = df_base['kategori']
        
        # Latih model
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.30, random_state=42)
        model_lr = LogisticRegression(max_iter=1000)
        model_lr.fit(X_train, y_train)
        
        return model_lr, q1, q2
    except Exception as e:
        st.error(f"Gagal melatih model di latar belakang. Pastikan file 'household_electricity_usage.xlsx' ada di GitHub. Error: {e}")
        return None, None, None

# Panggil fungsi training model
model, q1, q2 = inisialisasi_dan_latih_model()

# Definisikan fungsi klasifikasi manual berdasarkan nilai kuantil agar akurat
def klasifikasi_skor(score):
    if score <= q1: return "Rendah"
    elif score <= q2: return "Normal"
    else: return "Boros"

# ==============================================================================
# UI DUA PILIHAN INPUT (MENGGUNAKAN TABS)
# ==============================================================================
tab1, tab2 = st.tabs(["📝 Input Satu Data (Single)", "📂 Unggah File Excel (Bulk)"])

# ------------------------------------------------------------------------------
# PILIHAN 1: INPUT TUNGGAL (TAB 1)
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Formulir Input Data Baru")
    st.write("Silakan masukkan parameter penggunaan listrik untuk mendapatkan hasil prediksi instan.")
    
    # Form layout menggunakan kolom agar rapi
    col1, col2 = st.columns(2)
    with col1:
        power = st.number_input("Daya Alat Listrik (Power Watts)", min_value=0.0, step=10.0, value=450.0)
        duration = st.number_input("Durasi Penggunaan (Menit)", min_value=0, step=5, value=60)
    with col2:
        occupancy = st.number_input("Jumlah Penghuni Aktif (Occupancy Count)", min_value=0, step=1, value=2)
        h_size = st.number_input("Total Anggota Keluarga (Household Size)", min_value=1, step=1, value=4)
        
    if st.button("🚀 Hitung & Prediksi", type="primary"):
        if model is not None:
            # Hitung Usage Score secara eksak
            score_tunggal = power * duration * occupancy
            
            # Prediksi kategori menggunakan aturan kuantil data latih
            kategori_tunggal = klasifikasi_skor(score_tunggal)
            
            # Tampilkan Hasil dengan UI Box Indah
            st.markdown("---")
            st.write("### 📊 Hasil Analisis Data:")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="metric-box">
                    <span style='color: #6B7280; font-size: 14px;'>Usage Score</span><br>
                    <span style='font-size: 24px; font-weight: bold; color: #1E3A8A;'>{score_tunggal:,.2f}</span>
                </div>
                """, unsafe_allow_html=True)
                
            with c2:
                # Ganti warna box berdasarkan kategori hasil klasifikasi
                color_map = {"Rendah": "#10B981", "Normal": "#3B82F6", "Boros": "#EF4444"}
                warna = color_map.get(kategori_tunggal, "#1E3A8A")
                
                st.markdown(f"""
                <div class="metric-box" style="border-left: 5px solid {warna};">
                    <span style='color: #6B7280; font-size: 14px;'>Klasifikasi Kategori</span><br>
                    <span style='font-size: 24px; font-weight: bold; color: {warna};'>{kategori_tunggal}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Model belum siap, silakan periksa konfigurasi data Anda.")

# ------------------------------------------------------------------------------
# PILIHAN 2: UPLOAD FILE EXCEL (TAB 2)
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Pemrosesan Banyak Data Berbasis File")
    st.write("Unggah file Excel Anda yang berisi kolom: `household_size`, `occupancy_count`, `power_watts`, dan `duration_minutes`.")
    
    uploaded_file = st.file_uploader("Pilih file Excel (.xlsx)", type=["xlsx"])
    
    if uploaded_file is not None:
        try:
            # Membaca file excel yang diupload user
            df_user = pd.read_excel(uploaded_file)
            
            # Validasi kolom yang dibutuhkan
            required_cols = ['household_size', 'occupancy_count', 'power_watts', 'duration_minutes']
            if all(col in df_user.columns for col in required_cols):
                
                st.success("File berhasil diunggah dan diverifikasi! Memulai proses klasifikasi...")
                
                # 1. Hitung kolom usage_score baru
                df_user['usage_score'] = df_user['power_watts'] * df_user['duration_minutes'] * df_user['occupancy_count']
                
                # 2. Hitung kategori menggunakan standarisasi kuantil model
                df_user['kategori'] = df_user['usage_score'].apply(klasifikasi_skor)
                
                # Tampilkan cuplikan data yang sudah diproses di layar web
                st.write("### 👁️ Pratinjau Data Hasil Proses (5 Baris Pertama):")
                st.dataframe(df_user.head())
                
                # Konversi DataFrame hasil proses kembali ke bentuk Excel di dalam memori
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df_user.to_excel(writer, index=False, sheet_name='Hasil Klasifikasi')
                buffer.seek(0)
                
                # Tombol Download Excel Baru yang Estetik
                st.markdown("---")
                st.download_button(
                    label="📥 Unduh File Excel Hasil Klasifikasi",
                    data=buffer,
                    file_name="hasil_klasifikasi_listrik_baru.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )
            else:
                st.error(f"Format kolom file Anda salah. File harus memiliki kolom: {required_cols}")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file Anda: {e}")