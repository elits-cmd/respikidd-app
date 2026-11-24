import streamlit as st
import librosa
import librosa.display
import noisereduce as nr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import os

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="RespiKidd Mobile App",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Custom CSS: Mengubah Tampilan Menjadi Layar Mobile App Pastel
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@600;700;800&display=swap');

    /* Sembunyikan elemen bawaan Streamlit agar mirip aplikasi HP murni */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}

    /* Latar luar (Desktop) bernuansa pastel lembut */
    .stApp {
        background-color: #F3F6FA;
        font-family: 'Nunito', sans-serif;
    }

    /* Bingkai Layar Mobile App di Tengah */
    .block-container {
        max-width: 430px !important;
        background-color: #FFFDF9 !important;
        min-height: 92vh;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        padding: 1.8rem 1.4rem !important;
        border-radius: 36px;
        border: 8px solid #E2E8F0;
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.08);
    }

    h1, h2, h3, h4, p, div, span, label {
        font-family: 'Nunito', sans-serif !important;
        color: #334155;
    }

    /* Indikator Halaman (Step Bar) */
    .step-indicator {
        text-align: center;
        font-size: 0.8rem;
        font-weight: 700;
        color: #0284C7;
        background-color: #E0F2FE;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        display: inline-block;
        margin-bottom: 1rem;
    }

    /* Kartu Pastel Mobile */
    .mobile-card {
        background-color: #FFFFFF;
        border-radius: 22px;
        padding: 1.2rem;
        margin-bottom: 0.9rem;
        border: 2px solid #F1F5F9;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }

    .card-blue { background-color: #F0F9FF; border-color: #BAE6FD; }
    .card-peach { background-color: #FFF7ED; border-color: #FED7AA; }
    .card-mint { background-color: #ECFDF5; border-color: #A7F3D0; }
    .card-pink { background-color: #FFF1F2; border-color: #FECDD3; }

    /* Tombol Utama Mobile (Besar & Membulat) */
    .stButton>button {
        width: 100%;
        border-radius: 18px !important;
        padding: 0.75rem 1rem !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        border: none !important;
        background: linear-gradient(135deg, #38BDF8 0%, #60A5FA 100%) !important;
        color: white !important;
        box-shadow: 0 6px 15px rgba(56, 189, 248, 0.3);
        transition: 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# Ilustrasi Vektor SVG Beruang Pastel Lucu (Bukan Emoji)
BEAR_GUIDE_SVG = """
<div style="text-align:center; margin: 0.5rem 0;">
<svg width="110" height="110" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <!-- Telinga -->
  <circle cx="52" cy="55" r="24" fill="#D4A373"/>
  <circle cx="52" cy="55" r="13" fill="#FAEDCD"/>
  <circle cx="148" cy="55" r="24" fill="#D4A373"/>
  <circle cx="148" cy="55" r="13" fill="#FAEDCD"/>
  <!-- Kepala -->
  <circle cx="100" cy="105" r="62" fill="#E6B88A"/>
  <!-- Moncong -->
  <ellipse cx="100" cy="122" rx="26" ry="20" fill="#FEFAE0"/>
  <!-- Mata & Pipi -->
  <circle cx="76" cy="96" r="6" fill="#3D2C2E"/>
  <circle cx="78" cy="94" r="2" fill="#FFFFFF"/>
  <circle cx="124" cy="96" r="6" fill="#3D2C2E"/>
  <circle cx="126" cy="94" r="2" fill="#FFFFFF"/>
  <ellipse cx="63" cy="112" rx="9" ry="5" fill="#FFB5A7" opacity="0.7"/>
  <ellipse cx="137" cy="112" rx="9" ry="5" fill="#FFB5A7" opacity="0.7"/>
  <!-- Hidung & Senyum -->
  <ellipse cx="100" cy="114" rx="8" ry="5" fill="#5C4033"/>
  <path d="M 90 126 Q 100 135 110 126" stroke="#5C4033" stroke-width="3.5" fill="none" stroke-linecap="round"/>
  <!-- Syal Biru Pastel -->
  <path d="M 55 158 Q 100 175 145 158 L 138 180 Q 100 192 62 180 Z" fill="#90E0EF"/>
</svg>
</div>
"""

BEAR_EDU_SVG = """
<div style="text-align:center; margin: 0.5rem 0;">
<svg width="110" height="110" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <!-- Telinga Beruang Putih Krem -->
  <circle cx="52" cy="58" r="23" fill="#EDE0D4"/>
  <circle cx="52" cy="58" r="12" fill="#FFB5A7"/>
  <circle cx="148" cy="58" r="23" fill="#EDE0D4"/>
  <circle cx="148" cy="58" r="12" fill="#FFB5A7"/>
  <!-- Kepala -->
  <circle cx="100" cy="108" r="60" fill="#F5EBE0"/>
  <!-- Moncong -->
  <ellipse cx="100" cy="124" rx="24" ry="18" fill="#FFFFFF"/>
  <!-- Mata & Kacamata Bulat Lucu -->
  <circle cx="76" cy="98" r="5.5" fill="#334155"/>
  <circle cx="124" cy="98" r="5.5" fill="#334155"/>
  <circle cx="76" cy="98" r="16" stroke="#7BDFF2" stroke-width="3.5" fill="none"/>
  <circle cx="124" cy="98" r="16" stroke="#7BDFF2" stroke-width="3.5" fill="none"/>
  <line x1="92" y1="98" x2="108" y2="98" stroke="#7BDFF2" stroke-width="3.5"/>
  <!-- Pipi & Hidung -->
  <ellipse cx="62" cy="115" rx="8" ry="5" fill="#FFCAD4"/>
  <ellipse cx="138" cy="115" rx="8" ry="5" fill="#FFCAD4"/>
  <ellipse cx="100" cy="116" rx="7" ry="4.5" fill="#7F5539"/>
  <path d="M 92 127 Q 100 134 108 127" stroke="#7F5539" stroke-width="3" fill="none" stroke-linecap="round"/>
</svg>
</div>
"""

# Inisialisasi State Halaman (1 sampai 5)
if "page" not in st.session_state:
    st.session_state.page = 1
if "result_data" not in st.session_state:
    st.session_state.result_data = None

def go_to(page_num):
    st.session_state.page = page_num

@st.cache_resource
def load_respikidd_model():
    return tf.keras.models.load_model('respikidd_model.keras')

# ==============================================================================
# LAYAR 1: SPLASH & SAMBUTAN MASKOT UTAMA
# ==============================================================================
if st.session_state.page == 1:
    st.markdown("<div style='text-align:center;'><span class='step-indicator'>Selamat Datang • 1/5</span></div>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        if os.path.exists("mascot.png"):
            st.image("mascot.png", use_container_width=True)
        else:
            st.markdown("<h1 style='text-align:center;'>RespiKidd</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0 1.5rem 0;">
        <h2 style="color:#0369A1; margin-bottom:0.4rem;">Halo Ayah & Bunda!</h2>
        <p style="font-size:0.98rem; color:#475569; line-height:1.5;">
            <b>RespiKidd</b> membantu skrining dini gejala bronkitis pada batita lewat suara batuk dengan cara yang lembut, cepat, dan <b>tidak membuat anak takut</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.button("Mulai Sekarang ➔", on_click=go_to, args=(2,))

# ==============================================================================
# LAYAR 2: PANDUAN CARA PENGGUNAAN (DENGAN BERUANG LUCU)
# ==============================================================================
elif st.session_state.page == 2:
    st.markdown("<div style='text-align:center;'><span class='step-indicator'>Cara Pakai • 2/5</span></div>", unsafe_allow_html=True)
    st.markdown(BEAR_GUIDE_SVG, unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color:#0369A1; margin-top:0;'>Cara Menggunakan Aplikasi</h3>", unsafe_allow_html=True)

    st.markdown("""
    <div class="mobile-card card-blue">
        <b>1. Tenangkan Si Kecil</b><br>
        <span style="font-size:0.9rem;">Pastikan suasana sekitar cukup tenang agar suara batuk terekam jelas.</span>
    </div>
    <div class="mobile-card card-peach">
        <b>2. Rekam / Unggah Suara</b><br>
        <span style="font-size:0.9rem;">Dekatkan HP sekitar 15–20 cm saat anak batuk, atau pilih file rekaman audio.</span>
    </div>
    <div class="mobile-card card-mint">
        <b>3. Lihat Hasil & Edukasi</b><br>
        <span style="font-size:0.9rem;">AI akan menganalisis pola suara dalam hitungan detik beserta panduan langkah berikutnya.</span>
    </div>
    """, unsafe_allow_html=True)

    col_b1, col_b2 = st.columns([1, 2])
    with col_b1:
        st.button("⬅ Kembali", on_click=go_to, args=(1,))
    with col_b2:
        st.button("Lanjut Cek Batuk ➔", on_click=go_to, args=(3,))

# ==============================================================================
# LAYAR 3: REKAM / UNGGAH SUARA BATUK
# ==============================================================================
elif st.session_state.page == 3:
    st.markdown("<div style='text-align:center;'><span class='step-indicator'>Cek Suara • 3/5</span></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color:#0369A1;'>Masukkan Suara Batuk</h3>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:0.9rem;'>Pilih rekam langsung lewat mikrofon HP atau unggah rekaman (.wav / .mp3).</p>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Rekam Suara", "Unggah File"])
    audio_data = None

    with tab1:
        rec = st.audio_input("Tekan ikon mic untuk merekam:")
        if rec:
            audio_data = rec

    with tab2:
        upl = st.file_uploader("Pilih file audio batuk:", type=["wav", "mp3", "ogg"])
        if upl:
            audio_data = upl

    if audio_data is not None:
        st.audio(audio_data)
        if st.button("Analisis Suara Sekarang ✨"):
            with st.spinner("Sedang menganalisis suara batuk..."):
                try:
                    model = load_respikidd_model()
                    start_time = time.time()

                    y, sr = librosa.load(audio_data, sr=22050, duration=3.0)
                    y_clean = nr.reduce_noise(y=y, sr=sr)
                    y_norm = librosa.util.normalize(y_clean)

                    mel_spec = librosa.feature
