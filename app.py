import streamlit as st
import librosa
import librosa.display
import noisereduce as nr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import os

# 1. Konfigurasi Halaman (Hanya dipanggil 1 kali di paling atas)
st.set_page_config(
    page_title="RespiKidd Mobile App",
    page_icon="assets/logo.png" if os.path.exists("assets/logo.png") else "🐾",
    layout="centered",
    initial_sidebar_state="collapsed"
# 2. Custom CSS: Background Cerah + Pola Doodle Gemoy + Layar Mobile Pastel (FIXED)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600;700&family=Nunito:wght@600;700;800&display=swap');

    /* Sembunyikan elemen bawaan Streamlit */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}

    /* Background Luar Cerah dengan Pola Doodle SVG */
    .stApp {
        background-color: #E0F7FA;
        background-image: 
            url("data:image/svg+xml,%3Csvg width='180' height='180' viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cpath d='M35 55c-6.6 0-12-5.4-12-12 0-5.8 4.1-10.6 9.6-11.8C34.5 24.8 40.4 20 47.5 20c8.6 0 15.5 6.9 15.5 15.5 0 .6 0 1.2-.1 1.8 4.6 1.5 8.1 5.8 8.1 11 0 6.5-5.2 11.7-11.7 11.7H35z' fill='%23FFFFFF' fill-opacity='0.85'/%3E%3Cpath d='M135 32l3.2 6.5 7.2 1-5.2 5.1 1.2 7.2-6.4-3.4-6.4 3.4 1.2-7.2-5.2-5.1 7.2-1z' fill='%23FDE047' fill-opacity='0.9'/%3E%3Cpath d='M45 130c-3.5-4.5-10.5-3.5-12.5 1.5-2-5-9-6-12.5-1.5-3.5 4.5-1 11 12.5 18 13.5-7 16-13.5 12.5-18z' fill='%23F9A8D4' fill-opacity='0.75'/%3E%3Cpath d='M130 125q5 0 5-5 0 5 5 5-5 0-5 5 0-5-5-5z' stroke='%235EEAD4' stroke-width='3' stroke-linecap='round'/%3E%3Ccircle cx='90' cy='85' r='4' fill='%23FDBA74' fill-opacity='0.7'/%3E%3Ccircle cx='160' cy='95' r='3.5' fill='%2393C5FD' fill-opacity='0.8'/%3E%3Ccircle cx='18' cy='90' r='3.5' fill='%23C4B5FD' fill-opacity='0.7'/%3E%3C/g%3E%3C/svg%3E"),
            linear-gradient(135deg, #E0F2FE 0%, #FEF9C3 50%, #FCE7F3 100%);
        font-family: 'Nunito', sans-serif;
    }

    /* Bingkai Layar Mobile App (Tanpa merusak flex container bawaan Streamlit) */
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

    /* Indikator Halaman Pill Ceria */
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

    /* Kartu Pastel Mobile Gemoy */
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
    .stButton>button p {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)
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

# Ilustrasi Beruang Pastel 1 (Panduan)
BEAR_GUIDE_SVG = """
<div style="text-align:center; margin: 0.2rem 0;">
<svg width="115" height="115" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <path d="M28 45l3 6 6.5 1-4.7 4.5 1.1 6.5-5.9-3.1-5.9 3.1 1.1-6.5-4.7-4.5 6.5-1z" fill="#FDE047"/>
  <circle cx="175" cy="50" r="6" fill="#F9A8D4"/>
  <circle cx="168" cy="145" r="5" fill="#7DD3FC"/>
  <circle cx="52" cy="55" r="24" fill="#D4A373"/>
  <circle cx="52" cy="55" r="13" fill="#FAEDCD"/>
  <circle cx="148" cy="55" r="24" fill="#D4A373"/>
  <circle cx="148" cy="55" r="13" fill="#FAEDCD"/>
  <circle cx="100" cy="105" r="62" fill="#E6B88A"/>
  <ellipse cx="100" cy="122" rx="26" ry="20" fill="#FEFAE0"/>
  <circle cx="76" cy="96" r="6.5" fill="#3D2C2E"/>
  <circle cx="78" cy="94" r="2.2" fill="#FFFFFF"/>
  <circle cx="124" cy="96" r="6.5" fill="#3D2C2E"/>
  <circle cx="126" cy="94" r="2.2" fill="#FFFFFF"/>
  <ellipse cx="63" cy="112" rx="10" ry="5.5" fill="#FF9AA2" opacity="0.8"/>
  <ellipse cx="137" cy="112" rx="10" ry="5.5" fill="#FF9AA2" opacity="0.8"/>
  <ellipse cx="100" cy="114" rx="8" ry="5" fill="#5C4033"/>
  <path d="M 90 126 Q 100 136 110 126" stroke="#5C4033" stroke-width="3.5" fill="none" stroke-linecap="round"/>
  <path d="M 55 158 Q 100 175 145 158 L 138 180 Q 100 192 62 180 Z" fill="#7DD3FC"/>
</svg>
</div>
"""

# Ilustrasi Beruang Pastel 2 (Edukasi)
BEAR_EDU_SVG = """
<div style="text-align:center; margin: 0.2rem 0;">
<svg width="115" height="115" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <path d="M172 42c-2.5-3.2-7.5-2.5-9 1-1.5-3.5-6.5-4.2-9-1-2.5 3.2-.8 7.8 9 12.8 9.8-5 11.5-9.6 9-12.8z" fill="#FDA4AF"/>
  <path d="M28 135l2.5 5 5.5.8-4 3.9.9 5.5-4.9-2.6-4.9 2.6.9-5.5-4-3.9 5.5-.8z" fill="#6EE7B7"/>
  <circle cx="52" cy="58" r="23" fill="#EDE0D4"/>
  <circle cx="52" cy="58" r="12" fill="#FFB5A7"/>
  <circle cx="148" cy="58" r="23" fill="#EDE0D4"/>
  <circle cx="148" cy="58" r="12" fill="#FFB5A7"/>
  <circle cx="100" cy="108" r="60" fill="#F5EBE0"/>
  <ellipse cx="100" cy="124" rx="24" ry="18" fill="#FFFFFF"/>
  <circle cx="76" cy="98" r="5.5" fill="#334155"/>
  <circle cx="124" cy="98" r="5.5" fill="#334155"/>
  <circle cx="76" cy="98" r="16" stroke="#38BDF8" stroke-width="3.5" fill="none"/>
  <circle cx="124" cy="98" r="16" stroke="#38BDF8" stroke-width="3.5" fill="none"/>
  <line x1="92" y1="98" x2="108" y2="98" stroke="#38BDF8" stroke-width="3.5"/>
  <ellipse cx="62" cy="115" rx="9" ry="5" fill="#FFCAD4"/>
  <ellipse cx="138" cy="115" rx="9" ry="5" fill="#FFCAD4"/>
  <ellipse cx="100" cy="116" rx="7" ry="4.5" fill="#7F5539"/>
  <path d="M 92 127 Q 100 135 108 127" stroke="#7F5539" stroke-width="3" fill="none" stroke-linecap="round"/>
</svg>
</div>
"""

# Inisialisasi State Halaman
if "page" not in st.session_state:
    st.session_state.page = 1
if "result_data" not in st.session_state:
    st.session_state.result_data = None

def go_to(page_num):
    st.session_state.page = page_num

@st.cache_resource
def load_respikidd_model():
    return tf.keras.models.load_model('respikidd_model.keras')

def preprocess_and_extract_mel(audio_file, use_noise_reduce=True):
    """Membersihkan noise, memvalidasi volume, dan membuat Mel-Spectrogram 128x128."""
    y, sr = librosa.load(audio_file, sr=22050, duration=3.0)

    # 1. Validasi kualitas suara (cegah prediksi palsu dari rekaman hening)
