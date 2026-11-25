import streamlit as st
import librosa
import librosa.display
import noisereduce as nr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import os
import urllib.parse
from datetime import datetime
from streamlit_js_eval import get_geolocation

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="RespiKidd Mobile App",
    page_icon="mascot.png.png" if os.path.exists("mascot.png.png") else "🐾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Custom CSS: Background Cerah + Pola Doodle Gemoy + Layar Mobile Pastel
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600;700&family=Nunito:wght@600;700;800&display=swap');

    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}

    .stApp {
        background-color: #E0F7FA;
        background-image: 
            url("data:image/svg+xml,%3Csvg width='180' height='180' viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cpath d='M35 55c-6.6 0-12-5.4-12-12 0-5.8 4.1-10.6 9.6-11.8C34.5 24.8 40.4 20 47.5 20c8.6 0 15.5 6.9 15.5 15.5 0 .6 0 1.2-.1 1.8 4.6 1.5 8.1 5.8 8.1 11 0 6.5-5.2 11.7-11.7 11.7H35z' fill='%23FFFFFF' fill-opacity='0.85'/%3E%3Cpath d='M135 32l3.2 6.5 7.2 1-5.2 5.1 1.2 7.2-6.4-3.4-6.4 3.4 1.2-7.2-5.2-5.1 7.2-1z' fill='%23FDE047' fill-opacity='0.9'/%3E%3Cpath d='M45 130c-3.5-4.5-10.5-3.5-12.5 1.5-2-5-9-6-12.5-1.5-3.5 4.5-1 11 12.5 18 13.5-7 16-13.5 12.5-18z' fill='%23F9A8D4' fill-opacity='0.75'/%3E%3Cpath d='M130 125q5 0 5-5 0 5 5 5-5 0-5 5 0-5-5-5z' stroke='%235EEAD4' stroke-width='3' stroke-linecap='round'/%3E%3Ccircle cx='90' cy='85' r='4' fill='%23FDBA74' fill-opacity='0.7'/%3E%3Ccircle cx='160' cy='95' r='3.5' fill='%2393C5FD' fill-opacity='0.8'/%3E%3Ccircle cx='18' cy='90' r='3.5' fill='%23C4B5FD' fill-opacity='0.7'/%3E%3C/g%3E%3C/svg%3E"),
            linear-gradient(135deg, #E0F2FE 0%, #FEF9C3 50%, #FCE7F3 100%);
        font-family: 'Nunito', sans-serif;
    }

    [data-testid="stMainBlockContainer"], .block-container {
        max-width: 440px !important;
        width: 92% !important;
        background-color: #FFFDF7 !important;
        background-image: radial-gradient(#FDE68A 1.2px, transparent 1.2px), radial-gradient(#BAE6FD 1.2px, #FFFDF7 1.2px);
        background-size: 36px 36px;
        background-position: 0 0, 18px 18px;
        margin-top: 2rem !important;
        margin-bottom: 2rem !important;
        padding: 1.8rem 1.4rem 2rem 1.4rem !important;
        border-radius: 36px !important;
        border: 7px solid #FFFFFF !important;
        box-shadow: 0 18px 40px rgba(56, 189, 248, 0.22), 0 0 0 3px #BAE6FD !important;
    }

    h1, h2, h3 {
        font-family: 'Fredoka', sans-serif !important;
        color: #0369A1 !important;
        letter-spacing: 0.3px;
    }

    p, div, span, label {
        font-family: 'Nunito', sans-serif !important;
        color: #334155;
    }

    .step-indicator {
        text-align: center;
        font-size: 0.82rem;
        font-weight: 800;
        color: #9A3412 !important;
        background: linear-gradient(90deg, #FEF08A 0%, #FED7AA 100%);
        padding: 0.35rem 1rem;
        border-radius: 999px;
        border: 2px solid #FDE047;
        display: inline-block;
        margin-bottom: 0.6rem;
        box-shadow: 0 3px 8px rgba(250, 204, 21, 0.25);
    }

    .mobile-card {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 24px;
        padding: 1.15rem;
        margin-bottom: 0.85rem;
        border: 2.5px solid #F1F5F9;
        box-shadow: 0 6px 14px rgba(148, 163, 184, 0.1);
    }

    .card-blue { background-color: #F0F9FF !important; border-color: #7DD3FC !important; }
    .card-peach { background-color: #FFF7ED !important; border-color: #FDBA74 !important; }
    .card-mint { background-color: #ECFDF5 !important; border-color: #6EE7B7 !important; }
    .card-pink { background-color: #FFF1F2 !important; border-color: #FDA4AF !important; }
    .card-yellow { background-color: #FEFCE8 !important; border-color: #FDE047 !important; }

    .stButton>button, .stDownloadButton>button {
        width: 100%;
        border-radius: 20px !important;
        padding: 0.75rem 1rem !important;
        font-family: 'Fredoka', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.02rem !important;
        border: 2px solid #FFFFFF !important;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%) !important;
        color: white !important;
        box-shadow: 0 6px 16px rgba(56, 189, 248, 0.35);
        transition: all 0.2s ease;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        transform: translateY(-2px) scale(1.02);
        background: linear-gradient(135deg, #0EA5E9 0%, #6366F1 100%) !important;
    }
    .stButton>button p, .stDownloadButton>button p {
        color: white !important;
    }

    .action-btn {
        display: block;
        width: 100%;
        text-align: center;
        padding: 0.72rem 1rem;
        border-radius: 18px;
        color: #FFFFFF !important;
        font-family: 'Fredoka', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        text-decoration: none;
        border: 2px solid #FFFFFF;
        margin-top: 0.45rem;
        margin-bottom: 0.45rem;
        box-sizing: border-box;
    }
    .action-btn:hover {
        opacity: 0.94;
        color: #FFFFFF !important;
    }
    .btn-green { background: linear-gradient(135deg, #10B981 0%, #059669 100%); box-shadow: 0 5px 12px rgba(16, 185, 129, 0.28); }
    .btn-blue { background: linear-gradient(135deg, #0284C7 0%, #2563EB 100%); box-shadow: 0 5px 12px rgba(37, 99, 235, 0.28); }
    .btn-red { background: linear-gradient(135deg, #F43F5E 0%, #E11D48 100%); box-shadow: 0 5px 12px rgba(225, 29, 72, 0.28); }
    .btn-wa { background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%); box-shadow: 0 5px 12px rgba(22, 163, 74, 0.28); }
</style>
""", unsafe_allow_html=True)

TOP_DOODLE_SVG = """
<div style="text-align:center; margin-bottom:-0.3rem;">
<svg width="240" height="42" viewBox="0 0 240 42" xmlns="http://www.w3.org/2000/svg">
  <path d="M28 10l2.5 5 5.5.8-4 3.9.9 5.5-4.9-2.6-4.9 2.6.9-5.5-4-3.9 5.5-.8z" fill="#FACC15"/>
  <path d="M58 32c-4.5 0-8-3.5-8-8 0-3.8 2.7-7 6.3-7.8 1.2-4.2 5.1-7.2 9.7-7.2 5.6 0 10.2 4.5 10.2 10.1 0 .4 0 .8-.1 1.2 3 1 5.3 3.8 5.3 7.2 0 4.2-3.4 7.5-7.6 7.5H58z" fill="#BAE6FD"/>
  <circle cx="63" cy="24" r="1.4" fill="#0369A1"/>
  <circle cx="71" cy="24" r="1.4" fill="#0369A1"/>
  <path d="M65.5 26.5q1.5 2 3 0" stroke="#0369A1" stroke-width="1.4" fill="none" stroke-linecap="round"/>
  <path d="M124 14c-2.5-3.2-7.5-2.5-9 1-1.5-3.5-6.5-4.2-9-1-2.5 3.2-.8 7.8 9 12.8 9.8-5 11.5-9.6 9-12.8z" fill="#F472B6"/>
  <path d="M162 32c-4.5 0-8-3.5-8-8 0-3.8 2.7-7 6.3-7.8 1.2-4.2 5.1-7.2 9.7-7.2 5.6 0 10.2 4.5 10.2 10.1 0 .4 0 .8-.1 1.2 3 1 5.3 3.8 5.3 7.2 0 4.2-3.4 7.5-7.6 7.5h-15.8z" fill="#FBCFE8"/>
  <circle cx="167" cy="24" r="1.4" fill="#9D174D"/>
  <circle cx="175" cy="24" r="1.4" fill="#9D174D"/>
  <path d="M169.5 26.5q1.5 2 3 0" stroke="#9D174D" stroke-width="1.4" fill="none" stroke-linecap="round"/>
  <path d="M212 10l2.5 5 5.5.8-4 3.9.9 5.5-4.9-2.6-4.9 2.6.9-5.5-4-3.9 5.5-.8z" fill="#4ADE80"/>
</svg>
</div>
"""

BEAR_GUIDE_SVG = """
<div style="text-align:center; margin: 0.2rem 0;">
<svg width="115" height="115" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <circle cx="52" cy="55" r="24" fill="#D4A373"/>
  <circle cx="148" cy="55" r="24" fill="#D4A373"/>
  <circle cx="100" cy="105" r="62" fill="#E6B88A"/>
  <ellipse cx="100" cy="122" rx="26" ry="20" fill="#FEFAE0"/>
  <circle cx="76" cy="96" r="6.5" fill="#3D2C2E"/>
  <circle cx="124" cy="96" r="6.5" fill="#3D2C2E"/>
  <ellipse cx="100" cy="114" rx="8" ry="5" fill="#5C4033"/>
  <path d="M 90 126 Q 100 136 110 126" stroke="#5C4033" stroke-width="3.5" fill="none" stroke-linecap="round"/>
</svg>
</div>
"""

BEAR_EDU_SVG = """
<div style="text-align:center; margin: 0.2rem 0;">
<svg width="115" height="115" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <circle cx="52" cy="58" r="23" fill="#EDE0D4"/>
  <circle cx="148" cy="58" r="23" fill="#EDE0D4"/>
  <circle cx="100" cy="108" r="60" fill="#F5EBE0"/>
  <ellipse cx="100" cy="124" rx="24" ry="18" fill="#FFFFFF"/>
  <circle cx="76" cy="98" r="5.5" fill="#334155"/>
  <circle cx="124" cy="98" r="5.5" fill="#334155"/>
  <circle cx="76" cy="98" r="16" stroke="#38BDF8" stroke-width="3.5" fill="none"/>
  <circle cx="124" cy="98" r="16" stroke="#38BDF8" stroke-width="3.5" fill="none"/>
  <line x1="92" y1="98" x2="108" y2="98" stroke="#38BDF8" stroke-width="3.5"/>
  <ellipse cx="100" cy="116" rx="7" ry="4.5" fill="#7F5539"/>
  <path d="M 92 127 Q 100 135 108 127" stroke="#7F5539" stroke-width="3" fill="none" stroke-linecap="round"/>
</svg>
</div>
"""

if "page" not in st.session_state:
    st.session_state.page = 1
if "result_data" not in st.session_state:
    st.session_state.result_data = None
if "ask_gps" not in st.session_state:
    st.session_state.ask_gps = False
if "user_coords" not in st.session_state:
    st.session_state.user_coords = None

def go_to(page_num):
    st.session_state.page = page_num

def render_mascot(width_ratio=2.4, fallback_svg=BEAR_GUIDE_SVG):
    if os.path.exists("mascot.png.png"):
        side = (5.0 - width_ratio) / 2
        c1, c2, c3 = st.columns([side, width_ratio, side])
        with c2:
            st.image("mascot.png.png", use_container_width=True)
    elif fallback_svg:
        st.markdown(fallback_svg, unsafe_allow_html=True)

@st.cache_resource
def load_respikidd_model():
    return tf.keras.models.load_model('respikidd_model.keras')

def preprocess_and_extract_mel(audio_file, use_noise_reduce=True):
    y, sr = librosa.load(audio_file, sr=22050, duration=3.0)

    rms = np.mean(librosa.feature.rms(y=y))
    if rms < 0.005:
        return None, None, "Suara terlalu pelan atau tidak terdeteksi batuk. Dekatkan HP 15–20 cm dan coba rekam ulang."

    if use_noise_reduce:
        y = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.75, stationary=True)

    y_trim, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trim) > int(0.3 * sr):
        y = y_trim

    y_norm = librosa.util.normalize(y)

    mel_spec = librosa.feature.melspectrogram(y=y_norm, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    mel_norm = np.clip((mel_spec_db + 80.0) / 80.0, 0.0, 1.0)
    mel_resized = tf.image.resize(mel_norm[..., np.newaxis], (128, 128)).numpy()
    img_input = np.expand_dims(mel_resized, axis=0)

    return img_input, (mel_spec_db, sr), None


# ==============================================================================
# LAYAR 1: SPLASH & SAMBUTAN MASKOT UTAMA
# ==============================================================================
if st.session_state.page == 1:
    st.markdown(TOP_DOODLE_SVG, unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;'><span class='step-indicator'>Selamat Datang • 1/5</span></div>", unsafe_allow_html=True)
    
    render_mascot(width_ratio=3.4, fallback_svg=BEAR_GUIDE_SVG)

    st.markdown("""
    <div class="mobile-card card-yellow" style="text-align:center; margin-top:0.5rem;">
        <h2 style="color:#0369A1; margin:0 0 0.4rem 0; font-size:1.5rem;">Halo Ayah & Bunda!</h2>
        <p style="font-size:0.95rem; color:#475569; line-height:1.5; margin:0;">
            <b>RespiKidd</b> membantu skrining dini gejala bronkitis pada batita lewat suara batuk serta menghubungkan ke kontak medis & Puskesmas terdekat di seluruh Indonesia.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.button("Mulai Sekarang ➔", on_click=go_to, args=(2,))

# ==============================================================================
# LAYAR 2: PANDUAN CARA PENGGUNAAN
# ==============================================================================
elif st.session_state.page == 2:
    st.markdown(TOP_DOODLE_SVG, unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;'><span class='step-indicator'>Cara Pakai • 2/5</span></div>", unsafe_allow_html=True)
    
    render_mascot(width_ratio=1.8, fallback_svg=BEAR_GUIDE_SVG)
    
    st.markdown("<h3 style='text-align:center; margin-top:0; font-size:1.35rem;'>Cara Menggunakan Aplikasi</h3>", unsafe_allow_html=True)

    st.markdown("""
    <div class="mobile-card card-blue">
        <b style="color:#0369A1;">1. Isi Data Singkat & Tenangkan Anak</b><br>
        <span style="font-size:0.88rem;">Data nama & usia membantu tenaga medis di Puskesmas/Posyandu mencatat riwayat skrining dengan cepat.</span>
    </div>
    <div class="mobile-card card-peach">
        <b style="color:#C2410C;">2. Rekam / Unggah Suara Batuk</b><br>
        <span style="font-size:0.88rem;">Dekatkan HP sekitar 15–20 cm saat anak batuk, atau pilih file rekaman audio.</span>
    </div>
    <div class="mobile-card card-mint">
        <b style="color:#047857;">3. Lihat Hasil & Kontak Faskes Terdekat</b><br>
        <span style="font-size:0.88rem;">Dapatkan hasil skrining AI, kontak darurat medis, nomor telepon Puskesmas/RS sekitar, serta laporan siap kirim.</span>
    </div>
    """, unsafe_allow_html=True)

    col_b1, col_b2 = st.columns([1, 2])
    with col_b1:
        st.button("⬅ Kembali", on_click=go_to, args=(1,))
    with col_b2:
        st.button("Lanjut Cek Batuk ➔", on_click=go_to, args=(3,))

# ==============================================================================
# LAYAR 3: IDENTITAS PASIEN (3T) + REKAM / UNGGAH SUARA BATUK
# ==============================================================================
elif st.session_state.page == 3:
    st.markdown(TOP_DOODLE_SVG, unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;'><span class='step-indicator'>Data & Suara • 3/5</span></div>", unsafe_allow_html=True)
    
    render_mascot(width_ratio=1.4, fallback_svg=BEAR_GUIDE_SVG)
    
    st.markdown("<h3 style='text-align:center; margin-top:0; font-size:1.3rem;'>Data Anak & Suara Batuk</h3>", unsafe_allow_html=True)

    st.markdown("""
    <div class="mobile-card card-yellow" style="padding:0.75rem 1rem; margin-bottom:0.6rem;">
        <b style="color:#9A3412; font-size:0.88rem;">📋 Identitas Pasien (Untuk Pendataan Medis)</b><br>
        <span style="font-size:0.8rem; color:#78350F;">Membantu bidan, perawat, atau dokter Puskesmas mendata pasien dengan cepat.</span>
    </div>
    """, unsafe_allow_html=True)

    child_name = st.text_input("Nama Lengkap Anak *", placeholder="Contoh: Mahirah")
    c_id1, c_id2 = st.columns([1.4, 1])
    with c_id1:
        parent_name = st.text_input("Nama Orang Tua / Wali *", placeholder="Contoh: Elita")
    with c_id2:
        child_age = st.number_input("Usia (Bulan)", min_value=1, max_value=60, value=18)

    region_manual = st.text_input("Desa / Kecamatan / Kabupaten", placeholder="Contoh: Kec. Wamena, Kab. Jayawijaya")

    st.markdown("<p style='font-size:0.9rem; font-weight:800; color:#0369A1; margin-top:0.6rem; margin-bottom:0.2rem;'>🎙️ Rekam atau Unggah Suara Batuk:</p>", unsafe_allow_html=True)

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

    st.markdown("<p style='font-size:0.88rem; font-weight:700; color:#0369A1; margin-top:0.5rem; margin-bottom:0.2rem;'>Gejala Tambahan (Opsional):</p>", unsafe_allow_html=True)
    c_g1, c_g2 = st.columns(2)
    with c_g1:
        has_wheezing = st.checkbox("Napas ngik-ngik / berlendir")
    with c_g2:
        has_fever = st.checkbox("Demam / batuk > 3 hari")

    if audio_data is not None:
        st.audio(audio_data)
        if st.button("Analisis Suara Sekarang ✨"):
            if not child_name.strip() or not parent_name.strip():
                st.warning("Mohon isi **Nama Anak** dan **Nama Orang Tua / Wali** terlebih dahulu untuk keperluan pendataan medis.")
            else:
                with st.spinner("Membersihkan noise & menganalisis suara batuk..."):
                    try:
                        model = load_respikidd_model()
                        start_time = time.time()

                        img_input, spec_info, err_msg = preprocess_and_extract_mel(audio_data, use_noise_reduce=True)

                        if err_msg:
                            st.warning(err_msg)
                        else:
                            mel_spec_db, sr = spec_info
                            raw_prob = float(model.predict(img_input, verbose=0)[0][0])
                            
                            bronchitis_score = 1.0 - raw_prob

                            if has_wheezing:
                                bronchitis_score = min(1.0, bronchitis_score + 0.08)
                            if has_fever:
                                bronchitis_score = min(1.0, bronchitis_score + 0.05)

                            response_time = round(time.time() - start_time, 2)

                            st.session_state.result_data = {
                                "child_name": child_name.strip(),
                                "parent_name": parent_name.strip(),
                                "child_age": child_age,
                                "region_manual": region_manual.strip(),
                                "has_wheezing": has_wheezing,
                                "has_fever": has_fever,
                                "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M"),
                                "bronchitis_score": bronchitis_score,
                                "raw_prob": raw_prob,
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
# LAYAR 4: HASIL ANALISIS AI, KONTAK MEDIS TERDEKAT & LAPORAN PARAMEDIS
# ==============================================================================
elif st.session_state.page == 4:
    st.markdown(TOP_DOODLE_SVG, unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;'><span class='step-indicator'>Hasil Skrining • 4/5</span></div>", unsafe_allow_html=True)
    
    render_mascot(width_ratio=1.4, fallback_svg=BEAR_EDU_SVG)
    
    res = st.session_state.result_data

    if res is not None:
        # 1. Kartu Identitas Pasien
        st.markdown(f"""
        <div class="mobile-card card-blue" style="padding:0.85rem 1rem; margin-bottom:0.7rem;">
            <div style="font-size:0.82rem; color:#0369A1; font-weight:800;">🪪 REKAM SKRINING PASIEN</div>
            <div style="font-size:0.88rem; margin-top:0.25rem;">
                <b>Anak:</b> {res['child_name']} ({res['child_age']} bln)<br>
                <b>Orang Tua/Wali:</b> {res['parent_name']}<br>
                <b>Wilayah:</b> {res['region_manual'] if res['region_manual'] else '-'}<br>
                <b>Waktu:</b> {res['timestamp']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Kartu Hasil Analisis AI
        b_score = res["bronchitis_score"]
        if b_score >= 0.5:
            conf = b_score * 100
            status_label = "INDIKASI GEJALA BRONKITIS"
            st.markdown(f"""
            <div class="mobile-card card-pink" style="text-align:center;">
                <h3 style="color:#BE123C; margin:0;">Indikasi Gejala Bronkitis</h3>
                <p style="font-size:0.88rem; margin-top:0.4rem;">
                    Pola suara mengarah pada batuk berlendir/menyempit. Segera hubungi atau bawa <b>{res['child_name']}</b> ke dokter atau Puskesmas terdekat.
                </p>
                <b style="color:#9F1239; font-size:0.85rem;">Keyakinan AI: {conf:.1f}% • Waktu: {res['time']} dtk</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            conf = (1.0 - b_score) * 100
            status_label = "POLA BATUK NON-BRONKITIS"
            st.markdown(f"""
            <div class="mobile-card card-mint" style="text-align:center;">
                <h3 style="color:#047857; margin:0;">Pola Batuk Non-Bronkitis</h3>
                <p style="font-size:0.88rem; margin-top:0.4rem;">
                    Tidak terdeteksi ciri khas bronkitis pada sampel suara ini. Terus pantau kondisi dan kecukupan cairan si kecil.
                </p>
                <b style="color:#065F46; font-size:0.85rem;">Keyakinan AI: {conf:.1f}% • Waktu: {res['time']} dtk</b>
            </div>
            """, unsafe_allow_html=True)

        # ======================================================================
        # 3. KONTAK MEDIS, DARURAT & DOKTER / PUSKESMAS TERDEKAT
        # ======================================================================
        st.markdown("""
        <div class="mobile-card card-peach" style="padding:0.95rem 1rem;">
            <b style="color:#C2410C; font-size:0.95rem;">📞 Kontak Medis & Faskes Terdekat</b><br>
            <span style="font-size:0.82rem; color:#7C2D12;">
                Hubungi layanan darurat nasional atau cari nomor telepon & kontak Puskesmas/RS terdekat dari lokasi Anda:
            </span>
            <a href="tel:119" class="action-btn btn-red">🚑 Telepon Darurat Kemenkes (119)</a>
            <a href="tel:112" class="action-btn btn-red" style="background:linear-gradient(135deg, #EA580C 0%, #C2410C 100%);">🆘 Panggilan Darurat Nasional (112)</a>
        </div>
        """, unsafe_allow_html=True)

        if st.button("📍 Izinkan Lokasi GPS untuk Kontak RS & Dokter Terdekat"):
            st.session_state.ask_gps = True

        if st.session_state.ask_gps and st.session_state.user_coords is None:
            loc = get_geolocation(component_key="get_user_gps")
            if loc and "coords" in loc:
                st.session_state.user_coords = {
                    "lat": loc["coords"]["latitude"],
                    "lon": loc["coords"]["longitude"]
                }
                st.rerun()
            else:
                st.info("Klik **Allow / Izinkan** pada popup izin lokasi di browser/HP Anda...")

        coords_text = res["region_manual"] if res["region_manual"] else "Tidak dicantumkan"

        if st.session_state.user_coords is not None:
            lat = st.session_state.user_coords["lat"]
            lon = st.session_state.user_coords["lon"]
            coords_text = f"{lat:.5f}, {lon:.5f}" + (f" ({res['region_manual']})" if res["region_manual"] else "")
            
            st.success(f"Lokasi GPS aktif: `{lat:.5f}, {lon:.5f}`")
            
            # Link Maps berisi pin + nomor telepon faskes terdekat
            maps_puskesmas_url = f"https://www.google.com/maps/search/nomor+telepon+Puskesmas+Klinik+terdekat/@{lat},{lon},14z"
            maps_dokter_url = f"https://www.google.com/maps/search/nomor+telepon+Dokter+Anak+Rumah+Sakit+terdekat/@{lat},{lon},13z"
            
            st.markdown(f"""
            <a href="{maps_puskesmas_url}" target="_blank" class="action-btn btn-green">🏥 Lihat Kontak & Lokasi Puskesmas Terdekat</a>
            <a href="{maps_dokter_url}" target="_blank" class="action-btn btn-blue">🩺 Lihat Kontak & Lokasi Dokter Anak / RS</a>
            """, unsafe_allow_html=True)

        # Pencarian kontak berdasarkan nama wilayah yang diketik (sangat berguna di 3T jika GPS tidak akurat)
        if res["region_manual"]:
            q_puskesmas = urllib.parse.quote(f"nomor telepon Puskesmas {res['region_manual']}")
            q_rs = urllib.parse.quote(f"nomor telepon Rumah Sakit Dokter Anak {res['region_manual']}")
            st.markdown(f"""
            <a href="https://www.google.com/search?q={q_puskesmas}" target="_blank" class="action-btn btn-green">☎️ Cari Nomor Telepon Puskesmas {res['region_manual']}</a>
            <a href="https://www.google.com/search?q={q_rs}" target="_blank" class="action-btn btn-blue">☎️ Cari Nomor RS / Dokter di {res['region_manual']}</a>
            """, unsafe_allow_html=True)

        # ======================================================================
        # 4. KIRIM LAPORAN KE WHATSAPP BIDAN / DOKTER & UNDUH FILE
        # ======================================================================
        gejala_list = []
        if res["has_wheezing"]:
            gejala_list.append("Napas ngik-ngik/berlendir")
        if res["has_fever"]:
            gejala_list.append("Demam/batuk > 3 hari")
        gejala_str = ", ".join(gejala_list) if gejala_list else "Tidak ada gejala tambahan"

        laporan_teks = f"""LAPORAN SKRINING AWAL RESPIKIDD
========================================
Waktu Pemeriksaan : {res['timestamp']}
Nama Anak         : {res['child_name']}
Usia Anak         : {res['child_age']} bulan
Nama Orang Tua    : {res['parent_name']}
Lokasi / Wilayah  : {coords_text}
Gejala Tambahan   : {gejala_str}
----------------------------------------
Hasil Skrining AI : {status_label}
Tingkat Keyakinan : {conf:.1f}%
----------------------------------------
Catatan: Hasil skrining awal suara batuk untuk membantu pemeriksaan tenaga medis."""

        wa_encoded = urllib.parse.quote(laporan_teks)
        st.markdown(f"""
        <a href="https://wa.me/?text={wa_encoded}" target="_blank" class="action-btn btn-wa">💬 Kirim Hasil Skrining via WhatsApp ke Bidan / Dokter</a>
        """, unsafe_allow_html=True)

        st.download_button(
            label="📄 Unduh Laporan Rekam Medis (.txt)",
            data=laporan_teks,
            file_name=f"Skrining_RespiKidd_{res['child_name'].replace(' ', '_')}.txt",
            mime="text/plain"
        )

        # 5. Grafik Mel-Spectrogram
        st.markdown("<p style='font-size:0.85rem; font-weight:800; color:#0369A1; margin-top:0.6rem; margin-bottom:0.2rem;'>Grafik Mel-Spectrogram:</p>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 2.3))
        librosa.display.specshow(res["mel_db"], sr=res["sr"], x_axis='time', y_axis='mel', ax=ax, cmap='magma')
        ax.set_xlabel("Waktu (detik)", fontsize=8)
        ax.set_ylabel("Frekuensi", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    col_r1, col_r2 = st.columns([1, 1.4])
    with col_r1:
        st.button("Cek Ulang", on_click=go_to, args=(3,))
    with col_r2:
        st.button("Baca Edukasi ➔", on_click=go_to, args=(5,))

# ==============================================================================
# LAYAR 5: EDUKASI SINGKAT
# ==============================================================================
elif st.session_state.page == 5:
    st.markdown(TOP_DOODLE_SVG, unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;'><span class='step-indicator'>Edukasi Singkat • 5/5</span></div>", unsafe_allow_html=True)
    
    render_mascot(width_ratio=1.8, fallback_svg=BEAR_EDU_SVG)
    
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
