import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 1. SETINGAN HALAMAN UTAMA (WIDE & DARK PRESET)
st.set_page_config(
    page_title="SPK Pemeliharaan Kendaraan Cerdas",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. INJEKSI CUSTOM CSS HIGH-FIDELITY & ANIMASI COLUMN
# ==========================================
st.markdown("""
    <style>
    /* Mengimpor Font Modern dari Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* RESET FONT GLOBAL & BACKGROUND */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        background: radial-gradient(circle at 90% 10%, #1e1e38 0%, #090910 100%) !important;
        color: #f1f5f9 !important;
    }
    
    /* STYLING SIDEBAR PREMIUM */
    [data-testid="stSidebar"] {
        background-color: rgba(13, 13, 25, 0.8) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
    }
    
    /* HEADER HERO BANNER */
    .hero-container {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
        background: linear-gradient(180deg, rgba(56, 189, 248, 0.05) 0%, rgba(0,0,0,0) 100%);
        border-radius: 24px;
        margin-bottom: 2rem;
        border: 1px solid rgba(56, 189, 248, 0.05);
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.05em;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        font-weight: 400;
        max-width: 700px;
        margin: 0 auto;
    }
    
    /* CARD GLASSMORPHISM UTAMA */
    .dashboard-card {
        background: rgba(22, 22, 43, 0.4);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.4);
        margin-bottom: 1.5rem;
    }
    .card-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding-bottom: 0.75rem;
    }

    /* =========================================================
       PENYESUAIAN WARNA & ANIMASI TIAP KOLOM INPUT (KONTROL BARU)
       ========================================================= */
    
    /* 1. Memperjelas Teks Label Kolom (Agar Terbaca Jelas) */
    div[data-testid="stWidgetLabel"] p {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.02em;
    }

    /* 2. Mengubah Kotak Putih Input Menjadi Gelap & Transparan (Glass Look) */
    div[data-baseweb="input"], div[data-baseweb="select"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    /* 3. Menyesuaikan Warna Teks di Dalam Kolom */
    div[data-baseweb="input"] input, div[data-baseweb="select"] div {
        color: #ffffff !important;
        font-weight: 500 !important;
    }

    /* 4. Efek Animasi & Glow Saat Kolom Diklik/Difokuskan */
    div[data-baseweb="input"]:focus-within, div[data-baseweb="select"]:focus-within {
        border-color: #38bdf8 !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.35) !important;
    }

    /* 5. Efek Animasi Hover (Mengangkat Kolom Secara Lembut Saat Kursor Lewat) */
    div[data-testid="stNumberInput"], div[data-testid="stSelectbox"] {
        transition: transform 0.3s ease !important;
    }
    div[data-testid="stNumberInput"]:hover, div[data-testid="stSelectbox"]:hover {
        transform: translateY(-3px);
    }

    /* 6. Animasi Fade-In Lembut Saat Form Pertama Kali Dimuat */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    form {
        animation: fadeInUp 0.6s ease-out forwards;
    }
    
    /* CUSTOM KARTU METRIK DI SIDEBAR */
    .metric-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 14px;
        padding: 1rem;
        margin-top: 1rem;
        text-align: center;
    }
    .metric-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; tracking-wider; font-weight: 600; }
    .metric-value { font-size: 1.8rem; font-weight: 800; color: #38bdf8; margin-top: 0.25rem; }
    
    /* KUSTOMISASI TOMBOL UTAMA (PERBAIKAN WARNA TEKS AGAR KONTRAS) */
    .stButton>button {
        background: linear-gradient(90deg, #0284c7 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.8rem 2rem !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        width: 100% !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 20px rgba(2, 132, 199, 0.25) !important;
        margin-top: 1.5rem;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.45) !important;
        color: #ffffff !important;
    }
    
    /* KARTU OUTPUT HASIL DIAGNOSIS ELEGAN */
    .result-card {
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 1rem;
        border-left: 5px solid;
    }
    .result-ringan { background: rgba(16, 185, 129, 0.08); border-color: #10b981; border-top: 1px solid rgba(16, 185, 129, 0.1); border-right: 1px solid rgba(16, 185, 129, 0.1); border-bottom: 1px solid rgba(16, 185, 129, 0.1); }
    .result-sedang { background: rgba(245, 158, 11, 0.08); border-color: #f59e0b; border-top: 1px solid rgba(245, 158, 11, 0.1); border-right: 1px solid rgba(245, 158, 11, 0.1); border-bottom: 1px solid rgba(245, 158, 11, 0.1); }
    .result-berat { background: rgba(239, 68, 68, 0.08); border-color: #ef4444; border-top: 1px solid rgba(239, 68, 68, 0.1); border-right: 1px solid rgba(239, 68, 68, 0.1); border-bottom: 1px solid rgba(239, 68, 68, 0.1); }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. CORE MACHINE LEARNING ENGINE (PSO-KNN)
# ==========================================
@st.cache_resource
def load_and_optimize_ai_model():
    file_name = "Tugas2 melatih Kecerdasan Buatan TI24E Dhimas.xlsx"
    if not os.path.exists(file_name):
        return None, None, None, None, None, 0, 0
        
    df = pd.read_excel(file_name)
    df_clean = df.drop(columns=['id', 'estimasi biaya'])
    
    le_kendaraan = LabelEncoder()
    df_clean['jenis kendaraan'] = le_kendaraan.fit_transform(df_clean['jenis kendaraan'])
    
    le_service = LabelEncoder()
    df_clean['jenis service'] = le_service.fit_transform(df_clean['jenis service'])
    
    le_target = LabelEncoder()
    df_clean['tingkat kerusakan'] = le_target.fit_transform(df_clean['tingkat kerusakan'])
    
    X = df_clean.drop(columns=['tingkat kerusakan']).values
    y = df_clean['tingkat kerusakan'].values
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    def fungsi_fitness(k_val):
        k_int = int(np.round(k_val))
        if k_int < 1: k_int = 1
        knn = KNeighborsClassifier(n_neighbors=k_int)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        return cross_val_score(knn, X_scaled, y, cv=kf, scoring='accuracy').mean()

    # Algoritma PSO Native
    np.random.seed(42)
    num_particles, max_iter = 10, 15
    w, c1, c2 = 0.5, 1.5, 1.5
    min_k, max_k = 1, 15

    particles_X = np.random.uniform(min_k, max_k, num_particles)
    particles_V = np.random.uniform(-1, 1, num_particles)
    pbest_X = np.copy(particles_X)
    pbest_fitness = np.array([fungsi_fitness(x) for x in pbest_X])

    gbest_X = pbest_X[np.argmax(pbest_fitness)]
    gbest_fitness = pbest_fitness[np.argmax(pbest_fitness)]

    for _ in range(max_iter):
        for p in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            particles_V[p] = w * particles_V[p] + c1 * r1 * (pbest_X[p] - particles_X[p]) + c2 * r2 * (gbest_X - particles_X[p])
            particles_X[p] = np.clip(particles_X[p] + particles_V[p], min_k, max_k)
            fit = fungsi_fitness(particles_X[p])
            if fit > pbest_fitness[p]:
                pbest_fitness[p] = fit
                pbest_X[p] = particles_X[p]
        if pbest_fitness[np.argmax(pbest_fitness)] > gbest_fitness:
            gbest_fitness = pbest_fitness[np.argmax(pbest_fitness)]
            gbest_X = pbest_X[np.argmax(pbest_fitness)]

    best_k = int(np.round(gbest_X))
    model_final = KNeighborsClassifier(n_neighbors=best_k)
    model_final.fit(X_scaled, y)
    
    return model_final, scaler, le_kendaraan, le_service, le_target, best_k, gbest_fitness * 100

# Eksekusi sistem cerdas
model, scaler, le_kendaraan, le_service, le_target, k_optimal, akurasi_final = load_and_optimize_ai_model()

# ==========================================
# 4. TAMPILAN ANTARMUKA (USER INTERFACE)
# ==========================================

# Hero Header Banner
st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🚗 SYSTEM ENGINE SPK KENDARAAN</div>
        <div class="hero-subtitle">Diagnosis Tingkat Kerusakan Suku Cadang Menggunakan Parameter K-NN Berbasis Optimasi Metaheuristik Particle Swarm Optimization</div>
    </div>
""", unsafe_allow_html=True)

if model is None:
    st.markdown("<div class='dashboard-card' style='border-color:#ef4444; text-align:center;'>❌ <b>Gagal Memuat Sistem:</b> File Excel Dataset tidak ditemukan di dalam direktori utama!</div>", unsafe_allow_html=True)
else:
    # Seting Informasi Panel Kontrol di Sidebar Kiri
    with st.sidebar:
        st.markdown("<div style='padding: 10px 0; font-size:1.1rem; font-weight:700; color:#38bdf8;'>🔮 PANEL KONTROL AI</div>", unsafe_allow_html=True)
        st.write("Status jaringan cerdas saat ini terkonfigurasi aktif menggunakan pustaka data terisolasi.")
        
        # Komponen Metrik Custom
        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">K-Optimal (Hasil PSO)</div>
                <div class="metric-value">K = {k_optimal}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Akurasi Validasi</div>
                <div class="metric-value" style="color:#818cf8;">{akurasi_final:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin-top: 3rem; font-size: 0.75rem; color:#64748b; text-align:center;'>Aplikasi Tugas Akhir Informatika © 2026</div>", unsafe_allow_html=True)

    # PEMBAGIAN KOLOM UTAMA DASHBOARD
    col_form, col_output = st.columns([11, 9], gap="large")

    with col_form:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header"><span>📥</span> Formulir Karakteristik Data Kendaraan</div>', unsafe_allow_html=True)
        
        opsi_kendaraan = ['Motor', 'Mobil']
        opsi_service = ['Service Rutin', 'Ganti Oli', 'Tune Up', 'Ganti Kampas Rem', 'Ganti Aki', 'Ganti Ban', 
                        'Ganti Timing Belt', 'Ganti Rantai', 'Ganti Suspensi', 'Ganti Radiator', 'Overhaul']
        
        with st.form("spk_premium_form", clear_on_submit=False):
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                input_km = st.number_input("Total Jarak Tempuh Berjalan (KM)", min_value=0, value=45000, step=500)
                input_kendaraan = st.selectbox("Klasifikasi Jenis Kendaraan", opsi_kendaraan)
            with f_col2:
                input_umur = st.number_input("Masa Pakai / Usia Kronologis (Tahun)", min_value=0, value=3, step=1)
                input_service = st.selectbox("Rencana Tindakan Operasional", opsi_service)
            
            st.markdown('</div>', unsafe_allow_html=True) # Penutup div dashboard-card
            submit_btn = st.form_submit_button("⚡ Jalankan Komputasi Prediksi")

    with col_output:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header"><span>📊</span> Panel Hasil Keputusan Akhir</div>', unsafe_allow_html=True)
        
        if submit_btn:
            encoded_kendaraan = le_kendaraan.transform([input_kendaraan])[0]
            encoded_service = le_service.transform([input_service])[0]
            
            raw_features = np.array([[input_km, input_umur, encoded_kendaraan, encoded_service]])
            scaled_features = scaler.transform(raw_features)
            
            pred_idx = model.predict(scaled_features)[0]
            label_kerusakan = le_target.classes_[pred_idx]
            
            if label_kerusakan == 'Berat':
                st.markdown(f"""
                    <div class="result-card result-berat">
                        <div style="font-size: 0.8rem; font-weight: 700; color: #ef4444; uppercase; tracking-wider;">Hasil Klasifikasi AI:</div>
                        <div style="font-size: 2.2rem; font-weight: 900; color: #ef4444; margin-bottom: 0.5rem;">{label_kerusakan.upper()}</div>
                        <div style="font-size: 0.95rem; font-weight: 700; margin-bottom: 0.5rem; color:#f8fafc;">STATUS: WAJIB DIGANTI SEGERA!</div>
                        <p style="font-size: 0.85rem; color: #cbd5e1; margin: 0; line-height: 1.6;">
                            Komponen suku cadang terdeteksi telah melewati batas pemakaian optimal dan mengalami tingkat keausan fisik yang parah. Segera lakukan penggantian sparepart baru demi menjaga keselamatan mutlak berkendara di jalan raya.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            elif label_kerusakan == 'Sedang':
                st.markdown(f"""
                    <div class="result-card result-sedang">
                        <div style="font-size: 0.8rem; font-weight: 700; color: #f59e0b; uppercase; tracking-wider;">Hasil Klasifikasi AI:</div>
                        <div style="font-size: 2.2rem; font-weight: 900; color: #f59e0b; margin-bottom: 0.5rem;">{label_kerusakan.upper()}</div>
                        <div style="font-size: 0.95rem; font-weight: 700; margin-bottom: 0.5rem; color:#f8fafc;">STATUS: JADWALKAN MAINTENANCE BERKALA.</div>
                        <p style="font-size: 0.85rem; color: #cbd5e1; margin: 0; line-height: 1.6;">
                            Komponen memasuki fase kritis dan mendekati batas ketahanan maksimal material. Disarankan untuk menjadwalkan proses penggantian suku cadang dalam waktu dekat sebelum memicu kerusakan fatal berkelanjutan.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-card result-ringan">
                        <div style="font-size: 0.8rem; font-weight: 700; color: #10b981; uppercase; tracking-wider;">Hasil Klasifikasi AI:</div>
                        <div style="font-size: 2.2rem; font-weight: 900; color: #10b981; margin-bottom: 0.5rem;">{label_kerusakan.upper()}</div>
                        <div style="font-size: 0.95rem; font-weight: 700; margin-bottom: 0.5rem; color:#f8fafc;">STATUS: KONDISI AMAN & LAYAK PAKAI.</div>
                        <p style="font-size: 0.85rem; color: #cbd5e1; margin: 0; line-height: 1.6;">
                            Kondisi struktural suku cadang dinilai masih sangat optimal dan beroperasi normal. Tidak diperlukan penggantian komponen baru, cukup pertahankan pola perawatan rutin secara berkala (Preventive Maintenance).
                        </p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="text-align: center; padding: 3.5rem 1rem; color: #64748b;">
                    <div style="font-size: 3.5rem; margin-bottom: 1rem; animation: pulse 2s infinite;">🔮</div>
                    <p style="font-size: 0.9rem; font-weight: 500; max-width: 280px; margin: 0 auto; line-height: 1.5;">
                        Sistem siap menerima parameter. Klik tombol di sebelah kiri untuk mengalkulasi keputusan SPK.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True) # Penutup div dashboard-card output
