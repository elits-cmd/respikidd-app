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

# 2. Custom CSS: Background Cerah + Pola Doodle Gemoy + Layar Mobile Pastel
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600;700&family=Nunito:wght@600;700;800&display=swap');

    /* Sembunyikan elemen bawaan Streamlit */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}

    /* Background Luar Cerah dengan Pola Doodle SVG (Awan, Bintang, Hati Pastel) */
    .stApp {
        background-color: #E0F7FA;
        background-image: 
            url("data:image/svg+xml,%3Csvg width='180' height='180' viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3C!-- Awan Imut --%3E%3Cpath d='M35 55c-6.6 0-12-5.4-12-12 0-5.8 4.1-10.6 9.6-11.8C34.5 24.8 40.4 20 47.5 20c8.6 0 15.5 6.9 15.5 15.5 0 .6 0 1.2-.1 1.8 4.6 1.5 8.1 5.8 8.1 11 0 6.5-5.2 11.7-11.7 11.7H35z' fill='%23FFFFFF' fill-opacity='0.85'/%3E%3C!-- Bintang Kuning Pastel --%3E%3Cpath d='M135 32l3.2 6.5 7.2 1-5.2 5.1 1.2 7.2-6.4-3.4-6.4 3.4 1.2-7.2-5.2-5.1 7.2-1z' fill='%23FDE047' fill-opacity='0.9'/%3E%3C!-- Hati Pink Pastel --%3E%3Cpath d='M45 130c-3.5-4.5-10.5-3.5-12.5 1.5-2-5-9-6-12.5-1.5-3.5 4.5-1 11 12.5 18 13.5-7 16-13.5 12.5-18z' fill='%23F9A8D4' fill-opacity='0.75'/%3E%3C!-- Sparkle Mint --%3E%3Cpath d='M130 125q5 0 5-5 0 5 5 5-5 0-5 5 0-5-5-5z' stroke='%235EEAD4' stroke-width='3' stroke-linecap='round'/%3E%3C!-- Polka Dots --%3E%3Ccircle cx='90' cy='85' r='4' fill='%23FDBA74' fill-opacity='0.7'/%3E%3Ccircle cx='160' cy='95' r='3.5' fill='%2393C5FD' fill-opacity='0.8'/%3E%3Ccircle cx='18' cy='90' r='3.5' fill='%23C4B5FD' fill-opacity='0.7'/%3E%3C/g%3E%3C/svg%3E"),
            linear-gradient(135deg, #E0F2FE 0%, #FEF9C3 50%, #FCE7F3 100%);
        font-family: 'Nunito', sans-serif;
    }

    /* Bingkai Layar Mobile App Cerah dengan Doodle Halus di Dalam */
    /* Agar bingkai HP selalu pas di tengah layar */
    .main {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
    }

    /* Bingkai Layar Mobile App: Tinggi Otomatis Menyesuaikan Isi Teks */
/* Agar bingkai HP selalu pas di tengah layar */
    .main {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
    }

    /* Bingkai Layar Mobile App: Tinggi Otomatis Menyesuaikan Isi Teks */
    .block-container {
        max-width: 420px !important;
        width: 100% !important;
        height: auto !important;
        min-height: fit-content !important;
        background-color: #FFFDF7 !important;
        background-image: radial-gradient(#FDE68A 1.2px, transparent 1.2px), radial-gradient(#BAE6FD 1.2px, #FFFDF7 1.2px);
        background-size: 36px 36px;
        background-position: 0 0, 18px 18px;
        margin: 1.5rem auto !important;
        padding: 1.6rem 1.35rem 1.8rem 1.35rem !important;
        border-radius: 36px;
        border: 7px solid #FFFFFF;
        box-shadow: 0 18px 40px rgba(56, 189, 248, 0.22), 0 0 0 3px #BAE6FD;
    }

    h1, h2, h3 {
        font-family: 'Fredoka', sans-serif !important;
        color: #0369A1;
        letter-spacing: 0.3px;
    }

    p, div, span, label {
        font-family: 'Nunito', sans-serif !important;
        color: #334155;
    }

    /* Indikator Halaman Pill Ceria */
    .step-indicator {
        text-align: center;
        font-size: 0.82rem;
        font-weight: 800;
        color: #9A3412;
        background: linear-gradient(90deg, #FEF08A 0%, #FED7AA 100%);
        padding: 0.35rem 1rem;
        border-radius: 999px;
        border: 2px solid #FDE047;
        display: inline-block;
        margin-bottom: 0.6rem;
        box-shadow: 0 3px 8px rgba(250, 204, 21, 0.25);
    }

    /* Kartu Pastel Mobile Gemoy */
    .mobile-card {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 24px;
        padding: 1.15rem;
        margin-bottom: 0.85rem;
        border: 2.5px solid #F1F5F9;
        box-shadow: 0 6px 14px rgba(148, 163, 184, 0.1);
    }

    .card-blue { background-color: #F0F9FF; border-color: #7DD3FC; }
    .card-peach { background-color: #FFF7ED; border-color: #FDBA74; }
    .card-mint { background-color: #ECFDF5; border-color: #6EE7B7; }
    .card-pink { background-color: #FFF1F2; border-color: #FDA4AF; }
    .card-yellow { background-color: #FEFCE8; border-color: #FDE047; }

    /* Tombol Utama Mobile Ceria & Bouncy */
    .stButton>button {
        width: 100%;
        border-radius: 20px !important;
        padding: 0.75rem 1rem !important;
        font-family: 'Fredoka', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        border: 2px solid #FFFFFF !important;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%) !important;
        color: white !important;
        box-shadow: 0 6px 16px rgba(56, 189, 248, 0.35);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px) scale(1.02);
        background: linear-gradient(135deg, #0EA5E9 0%, #6366F1 100%) !important;
    }
</style>
""", unsafe_allow_html=True)

# Hiasan Doodle Awan & Bintang Gemoy untuk Bagian Atas Layar
TOP_DOODLE_SVG = """
<div style="text-align:center; margin-bottom:-0.3rem;">
<svg width="240" height="42" viewBox="0 0 240 42" xmlns="http://www.w3.org/2000/svg">
  <!-- Bintang Kiri -->
  <path d="M28 10l2.5 5 5.5.8-4 3.9.9 5.5-4.9-2.6-4.9 2.6.9-5.5-4-3.9 5.5-.8z" fill="#FACC15"/>
  <!-- Awan Senyum Kiri -->
  <path d="M58 32c-4.5 0-8-3.5-8-8 0-3.8 2.7-7 6.3-7.8 1.2-4.2 5.1-7.2 9.7-7.2 5.6 0 10.2 4.5 10.2 10.1 0 .4 0 .8-.1 1.2 3 1 5.3 3.8 5.3 7.2 0 4.2-3.4 7.5-7.6 7.5H58z" fill="#BAE6FD"/>
  <circle cx="63" cy="24" r="1.4" fill="#0369A1"/>
  <circle cx="71" cy="24" r="1.4" fill="#0369A1"/>
  <path d="M65.5 26.5q1.5 2 3 0" stroke="#0369A1" stroke-width="1.4" fill="none" stroke-linecap="round"/>
  <!-- Hati Tengah -->
  <path d="M124 14c-2.5-3.2-7.5-2.5-9 1-1.5-3.5-6.5-4.2-9-1-2.5 3.2-.8 7.8 9 12.8 9.8-5 11.5-9.6 9-12.8z" fill="#F472B6"/>
  <!-- Awan Senyum Kanan -->
  <path d="M162 32c-4.5 0-8-3.5-8-8 0-3.8 2.7-7 6.3-7.8 1.2-4.2 5.1-7.2 9.7-7.2 5.6 0 10.2 4.5 10.2 10.1 0 .4 0 .8-.1 1.2 3 1 5.3 3.8 5.3 7.2 0 4.2-3.4 7.5-7.6 7.5h-15.8z" fill="#FBCFE8"/>
  <circle cx="167" cy="24" r="1.4" fill="#9D174D"/>
  <circle cx="175" cy="24" r="1.4" fill="#9D174D"/>
  <path d="M169.5 26.5q1.5 2 3 0" stroke="#9D174D" stroke-width="1.4" fill="none" stroke-linecap="round"/>
  <!-- Bintang Kanan -->
  <path d="M212 10l2.5 5 5.5.8-4 3.9.9 5.5-4.9-2.6-4.9 2.6.9-5.5-4-3.9 5.5-.8z" fill="#4ADE80"/>
</svg>
</div>
"""

# Ilustrasi Beruang Pastel 1 (Panduan dengan Pita & Bintang)
BEAR_GUIDE_SVG = """
<div style="text-align:center; margin: 0.2rem 0;">
<svg width="115" height="115" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <!-- Doodle Bintang Kecil Sekitar Beruang -->
  <path d="M28 45l3 6 6.5 1-4.7 4.5 1.1 6.5-5.9-3.1-5.9 3.1 1.1-6.5-4.7-4.5 6.5-1z" fill="#FDE047"/>
  <circle cx="175" cy="50" r="6" fill="#F9A8D4"/>
  <circle cx="168" cy="145" r="5" fill="#7DD3FC"/>
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
  <circle cx="76" cy="96" r="6.5" fill="#3D2C2E"/>
  <circle cx="78" cy="94" r="2.2" fill="#FFFFFF"/>
  <circle cx="124" cy="96" r="6.5" fill="#3D2C2E"/>
  <circle cx="126" cy="94" r="2.2" fill="#FFFFFF"/>
  <ellipse cx="63" cy="112" rx="10" ry="5.5" fill="#FF9AA2" opacity="0.8"/>
  <ellipse cx="137" cy="112" rx="10" ry="5.5" fill="#FF9AA2" opacity="0.8"/>
  <!-- Hidung & Senyum -->
  <ellipse cx="100" cy="114" rx="8" ry="5" fill="#5C4033"/>
  <path d="M 90 126 Q 100 136 110 126" stroke="#5C4033" stroke-width="3.5" fill="none" stroke-linecap="round"/>
  <!-- Syal Biru Pastel -->
  <path d="M 55 158 Q 100 175 145 158 L 138 180 Q 100 192 62 180 Z" fill="#7DD3FC"/>
</svg>
</div>
"""

# Ilustrasi Beruang Pastel 2 (Edukasi Pakai Kacamata & Topi Mini)
BEAR_EDU_SVG = """
<div style="text-align:center; margin: 0.2rem 0;">
<svg width="115" height="115" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <!-- Doodle Hati & Bintang -->
  <path d="M172 42c-2.5-3.2-7.5-2.5-9 1-1.5-3.5-6.5-4.2-9-1-2.5 3.2-.8 7.8 9 12.8 9.8-5 11.5-9.6 9-12.8z" fill="#FDA4AF"/>
  <path d="M28 135l2.5 5 5.5.8-4 3.9.9 5.5-4.9-2.6-4.9 2.6.9-5.5-4-3.9 5.5-.8z" fill="#6EE7B7"/>
  <!-- Telinga -->
  <circle cx="52" cy="58" r="23" fill="#EDE0D4"/>
  <circle cx="52" cy="58" r="12" fill="#FFB5A7"/>
  <circle cx="148" cy="58" r="23" fill="#EDE0D4"/>
  <circle cx="148" cy="58" r="12" fill="#FFB5A7"/>
  <!-- Kepala -->
  <circle cx="100" cy="108" r="60" fill="#F5EBE0"/>
  <!-- Moncong -->
  <ellipse cx="100" cy="124" rx="24" ry="18" fill="#FFFFFF"/>
  <!-- Mata & Kacamata Bulat Pastel -->
  <circle cx="76" cy="98" r="5.5" fill="#334155"/>
  <circle cx="124" cy="98" r="5.5" fill="#334155"/>
  <circle cx="76" cy="98" r="16" stroke="#38BDF8" stroke-width="3.5" fill="none"/>
  <circle cx="124" cy="98" r="16" stroke="#38BDF8" stroke-width="3.5" fill="none"/>
  <line x1="92" y1="98" x2="108" y2="98" stroke="#38BDF8" stroke-width="3.5"/>
  <!-- Pipi & Hidung -->
  <ellipse cx="62" cy="115" rx="9" ry="5" fill="#FFCAD4"/>
  <ellipse cx="138" cy="115" rx="9" ry="5" fill="#FFCAD4"/>
  <ellipse cx="100" cy="116" rx="7" ry="4.5" fill="#7F5539"/>
  <path d="M 92 127 Q 100 135 108 127" stroke="#7F5539" stroke-width="3" fill="none" stroke-linecap="round"/>
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
    st.markdown(TOP_DOODLE_SVG, unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;'><span class='step-indicator'>Selamat Datang • 1/5</span></div>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([0.8, 3.4, 0.8])
    with c2:
        if os.path.exists("mascot.png"):
            st.image("mascot.png", use_container_width=True)
        else:
            st.markdown("<h1 style='text-align:center;'>RespiKidd</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class="mobile-card card-yellow" style="text-align:center; margin-top:0.5rem;">
        <h2 style="color:#0369A1; margin:0 0 0.4rem 0; font-size:1.5rem;">Halo Ayah & Bunda!</h2>
        <p style="font-size:0.95rem; color:#475569; line-height:1.5; margin:0;">
            <b>RespiKidd</b> membantu skrining dini gejala bronkitis pada batita lewat suara batuk dengan cara yang lembut, ceria, dan <b>bebas rasa takut</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.button("Mulai Sekarang ➔", on_click=go_to, args=(2,))

# ==============================================================================
# LAYAR 2: PANDUAN CARA PENGGUNAAN (DENGAN BERUANG LUCU)
# ==============================================================================
elif st.session_state.page == 2:
    st.markdown(TOP_DOODLE_SVG, unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;'><span class='step-indicator'>Cara Pakai • 2/5</span></div>", unsafe_allow_html=True)
    st.markdown(BEAR_GUIDE_SVG, unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; margin-top:0; font-size:1.35rem;'>Cara Menggunakan Aplikasi</h3>", unsafe_allow_html=True)

    st.markdown("""
    <div class="mobile-card card-blue">
        <b style="color:#0369A1;">1. Tenangkan Si Kecil</b><br>
        <span style="font-size:0.88rem;">Pastikan suasana sekitar cukup tenang agar suara batuk terekam jernih.</span>
    </div>
    <div class="mobile-card card-peach">
        <b style="color:#C2410C;">2. Rekam / Unggah Suara</b><br>
        <span style="font-size:0.88rem;">Dekatkan HP sekitar 15–20 cm saat anak batuk, atau pilih file audio.</span>
    </div>
    <div class="mobile-card card-mint">
        <b style="color:#047857;">3. Lihat Hasil & Edukasi</b><br>
        <span style="font-size:0.88rem;">AI menganalisis pola suara dalam hitungan detik beserta panduan medis.</span>
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
    st.markdown(TOP_DOODLE_SVG, unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;'><span class='step-indicator'>Cek Suara • 3/5</span></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; font-size:1.35rem;'>Masukkan Suara Batuk</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="mobile-card card-blue" style="text-align:center; padding:0.8rem;">
        <span style="font-size:0.88rem;">Pilih rekam langsung lewat mikrofon HP atau unggah file rekaman (.wav / .mp3).</span>
    </div>
    """, unsafe_allow_html=True)

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
                    
                    # Buang bagian hening di awal & akhir agar fokus ke suara batuk saja
                    y_trim, _ = librosa.effects.trim(y, top_db=25)
                    if len(y_trim) > 0:
                        y = y_trim

                    # Normalisasi audio
                    y_norm = librosa.util.normalize(y)

                    # Mel-Spectrogram (128x128)
                    mel_spec = librosa.feature.melspectrogram(y=y_norm, sr=sr, n_mels=128)
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    
                    # Normalisasi standar dB (-80 dB s/d 0 dB -> 0 s/d 1)
                    mel_norm = np.clip((mel_spec_db + 80.0) / 80.0, 0.0, 1.0)

                    mel_resized = tf.image.resize(mel_norm[..., np.newaxis], (128, 128)).numpy()
                    img_input = np.expand_dims(mel_resized, axis=0)

                    pred_prob = float(model.predict(img_input, verbose=0)[0][0])
                    response_time = round(time.time() - start_time, 2)

                    st.session_state.result_data = {
                        "prob": pred_prob,
                        "time": response_time,
                        "mel_db": mel_spec_db,
                        "sr": sr
                    }
                    st.session_state.page = 4
                    st.rerun()
                except Exception as e:
                    st.error(f"Terjadi kendala saat memproses audio: {e}")

    st.write("")
    st.button("⬅ Kembali ke Panduan", on_click=go_to, args=(2,))

# ==============================================================================
# LAYAR 4: HASIL ANALISIS AI & MEL-SPECTROGRAM
# ==============================================================================
elif st.session_state.page == 4:
    st.markdown(TOP_DOODLE_SVG, unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;'><span class='step-indicator'>Hasil Skrining • 4/5</span></div>", unsafe_allow_html=True)
    res = st.session_state.result_data

    if res is not None:
        pred_prob = res["prob"]
# Since alphabetical order sets 0 = bronkitis and 1 = non_bronkitis:
        if pred_prob <= 0.5:
            conf = (1 - pred_prob) * 100
            st.markdown(f"""
            <div class="mobile-card card-pink" style="text-align:center;">
                <h3 style="color:#BE123C; margin:0;">Indikasi Gejala Bronkitis</h3>
                <p style="font-size:0.88rem; margin-top:0.4rem;">
                    Pola suara mengarah pada batuk berlendir/menyempit. Tetap tenang dan segera konsultasikan ke dokter atau Puskesmas terdekat.
                </p>
                <b style="color:#9F1239; font-size:0.85rem;">Keyakinan AI: {conf:.1f}% • Waktu: {res['time']} dtk</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            conf = pred_prob * 100
            st.markdown(f"""
            <div class="mobile-card card-mint" style="text-align:center;">
                <h3 style="color:#047857; margin:0;">Pola Batuk Non-Bronkitis</h3>
                <p style="font-size:0.88rem; margin-top:0.4rem;">
                    Tidak terdeteksi ciri khas bronkitis pada sampel suara ini. Terus pantau kondisi dan kecukupan cairan si kecil.
                </p>
                <b style="color:#065F46; font-size:0.85rem;">Keyakinan AI: {conf:.1f}% • Waktu: {res['time']} dtk</b>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<p style='font-size:0.85rem; font-weight:800; color:#0369A1; margin-bottom:0.2rem;'>Grafik Mel-Spectrogram:</p>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 2.3))
        librosa.display.specshow(res["mel_db"], sr=res["sr"], x_axis='time', y_axis='mel', ax=ax, cmap='magma')
        ax.set_xlabel("Waktu (detik)", fontsize=8)
        ax.set_ylabel("Frekuensi", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

    col_r1, col_r2 = st.columns([1, 1.4])
    with col_r1:
        st.button("Cek Ulang", on_click=go_to, args=(3,))
    with col_r2:
        st.button("Baca Edukasi ➔", on_click=go_to, args=(5,))

# ==============================================================================
# LAYAR 5: EDUKASI SINGKAT (DENGAN BERUANG PINTAR)
# ==============================================================================
elif st.session_state.page == 5:
    st.markdown(TOP_DOODLE_SVG, unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;'><span class='step-indicator'>Edukasi Singkat • 5/5</span></div>", unsafe_allow_html=True)
    st.markdown(BEAR_EDU_SVG, unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; margin-top:0; font-size:1.35rem;'>Info Singkat untuk Orang Tua</h3>", unsafe_allow_html=True)

    st.markdown("""
    <div class="mobile-card card-blue">
        <b style="color:#0369A1;">Kenapa Batita Rentan?</b><br>
        <span style="font-size:0.86rem;">Saluran napas anak di bawah 3 tahun masih sempit, sehingga peradangan mudah memicu lendir berlebih dan batuk berat.</span>
    </div>
    <div class="mobile-card card-pink">
        <b style="color:#BE123C;">Tanda Bahaya (Segera ke Dokter)</b><br>
        <span style="font-size:0.86rem;">Napas berbunyi ngik-ngik (<i>wheezing</i>), napas sangat cepat saat tenang, atau anak tampak lemas dan demam tinggi.</span>
    </div>
    <div class="mobile-card card-peach">
        <b style="color:#C2410C;">Jauhkan Asap Rokok</b><br>
        <span style="font-size:0.86rem;">Asap rokok di rumah memperparah iritasi saluran pernapasan batita. Jaga udara rumah tetap bersih.</span>
    </div>
    """, unsafe_allow_html=True)

    col_e1, col_e2 = st.columns([1, 1.3])
    with col_e1:
        st.button("⬅ Hasil", on_click=go_to, args=(4,))
    with col_e2:
        st.button("Kembali ke Awal", on_click=go_to, args=(1,))
