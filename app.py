import streamlit as st
import streamlit.components.v1 as components
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

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="RespiKidd Offline App",
    page_icon="mascot.png.png" if os.path.exists("mascot.png.png") else "🐾",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Custom CSS: 100% OFFLINE (Tanpa Google Fonts eksternal agar instan tanpa sinyal)
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}

    .stApp {
        background-color: #E0F7FA;
        background-image: 
            url("data:image/svg+xml,%3Csvg width='180' height='180' viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cpath d='M35 55c-6.6 0-12-5.4-12-12 0-5.8 4.1-10.6 9.6-11.8C34.5 24.8 40.4 20 47.5 20c8.6 0 15.5 6.9 15.5 15.5 0 .6 0 1.2-.1 1.8 4.6 1.5 8.1 5.8 8.1 11 0 6.5-5.2 11.7-11.7 11.7H35z' fill='%23FFFFFF' fill-opacity='0.85'/%3E%3Cpath d='M135 32l3.2 6.5 7.2 1-5.2 5.1 1.2 7.2-6.4-3.4-6.4 3.4 1.2-7.2-5.2-5.1 7.2-1z' fill='%23FDE047' fill-opacity='0.9'/%3E%3Cpath d='M45 130c-3.5-4.5-10.5-3.5-12.5 1.5-2-5-9-6-12.5-1.5-3.5 4.5-1 11 12.5 18 13.5-7 16-13.5 12.5-18z' fill='%23F9A8D4' fill-opacity='0.75'/%3E%3Cpath d='M130 125q5 0 5-5 0 5 5 5-5 0-5 5 0-5-5-5z' stroke='%235EEAD4' stroke-width='3' stroke-linecap='round'/%3E%3Ccircle cx='90' cy='85' r='4' fill='%23FDBA74' fill-opacity='0.7'/%3E%3Ccircle cx='160' cy='95' r='3.5' fill='%2393C5FD' fill-opacity='0.8'/%3E%3Ccircle cx='18' cy='90' r='3.5' fill='%23C4B5FD' fill-opacity='0.7'/%3E%3C/g%3E%3C/svg%3E"),
            linear-gradient(135deg, #E0F2FE 0%, #FEF9C3 50%, #FCE7F3 100%);
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
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
        font-family: 'Arial Rounded MT Bold', 'Quicksand', 'Segoe UI', sans-serif !important;
        font-weight: 800 !important;
        color: #0369A1 !important;
        letter-spacing: 0.3px;
    }

    p, div, span, label {
        font-family: 'Segoe UI', 'Roboto', Arial, sans-serif !important;
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

    .offline-badge {
        display: inline-block;
        background-color: #DCFCE7;
        color: #166534 !important;
        border: 1.5px solid #86EFAC;
        font-size: 0.72rem;
        font-weight: 800;
        padding: 0.2rem 0.65rem;
        border-radius: 999px;
        margin-bottom: 0.4rem;
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
        font-weight: 700 !important;
        font-size: 1.0rem !important;
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
        font-weight: 700;
        font-size: 0.9rem;
        text-decoration: none;
        border: 2px solid #FFFFFF;
        margin-top: 0.45rem;
        margin-bottom: 0.45rem;
        box-sizing: border-box;
    }
    .btn-green { background: linear-gradient(135deg, #10B981 0%, #059669 100%); }
    .btn-blue { background: linear-gradient(135deg, #0284C7 0%, #2563EB 100%); }
    .btn-red { background: linear-gradient(135deg, #F43F5E 0%, #E11D48 100%); }
    .btn-amber { background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%); }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATABASE OFFLINE KONTAK MEDIS NASIONAL, PUSKESMAS & RS RUJUKAN SELURUH INDONESIA
# ==============================================================================
OFFLINE_FASKES_DB = {
    "Pilih Wilayah Rujukan (Seluruh Indonesia)...": [],
    "🌐 Layanan Kesehatan & Darurat Nasional (24 Jam)": [
        {"nama": "PSC 119 Kemenkes RI (Ambulans & Gawat Darurat)", "telp": "119", "ket": "Bebas Pulsa / Layanan Darurat Nasional"},
        {"nama": "Panggilan Darurat Terpadu Nasional", "telp": "112", "ket": "Aktif Tanpa Kuota / Sinyal Darurat"},
        {"nama": "Halo Kemenkes RI (Informasi Puskesmas & Rujukan)", "telp": "1500567", "ket": "Pusat Layanan Informasi Kesehatan RI"},
        {"nama": "Palang Merah Indonesia (PMI Pusat)", "telp": "0217992325", "ket": "Layanan Darurat & Kemanusiaan"},
        {"nama": "RSAB Harapan Kita (Rujukan Nasional Anak)", "telp": "0215668284", "ket": "Spesialis Anak & Pernapasan Anak Nasional"},
    ],
    "Aceh & Sumatera Utara (Banda Aceh, Medan, Nias, Simeulue)": [
        {"nama": "RSUD Dr. Zainoel Abidin Banda Aceh", "telp": "065134565", "ket": "RS Rujukan Utama Aceh"},
        {"nama": "RSUD Simeulue (Kepulauan 3T Aceh)", "telp": "065021018", "ket": "IGD & Poli Anak Simeulue"},
        {"nama": "RSUP H. Adam Malik Medan", "telp": "0618360381", "ket": "RS Rujukan Utama Sumatera Utara"},
        {"nama": "RSUD dr. M. Thomsen Nias (Gunungsitoli)", "telp": "063921018", "ket": "RS Rujukan Kepulauan Nias (3T)"},
        {"nama": "Puskesmas Teluk Dalam (Nias Selatan)", "telp": "063021018", "ket": "Layanan Kesehatan Dasar Nias Selatan"},
    ],
    "Sumatera Barat, Riau & Kep. Riau (Padang, Mentawai, Pekanbaru, Natuna)": [
        {"nama": "RSUP Dr. M. Djamil Padang", "telp": "075132371", "ket": "RS Rujukan Utama Sumatera Barat"},
        {"nama": "RSUD Kabupaten Kepulauan Mentawai (Tuapejat)", "telp": "0759320018", "ket": "RS Rujukan Kepulauan Mentawai (3T)"},
        {"nama": "RSUD Arifin Achmad Pekanbaru", "telp": "076121618", "ket": "RS Rujukan Utama Riau"},
        {"nama": "RSUD Kabupaten Natuna (Ranai)", "telp": "077331018", "ket": "RS Perbatasan Kepulauan Natuna (3T)"},
        {"nama": "RSUD Tarempa (Kepulauan Anambas)", "telp": "077231018", "ket": "RS Perbatasan Anambas (3T)"},
    ],
    "Jambi, Sumsel, Bengkulu, Babel & Lampung": [
        {"nama": "RSUD Raden Mattaher Jambi", "telp": "074161692", "ket": "RS Rujukan Utama Jambi"},
        {"nama": "RSUP Dr. Mohammad Hoesin Palembang", "telp": "0711354088", "ket": "RS Rujukan Utama Sumatera Selatan"},
        {"nama": "RSUD Dr. M. Yunus Bengkulu", "telp": "073652004", "ket": "RS Rujukan Utama Bengkulu"},
        {"nama": "RSUD Dr. (H.C.) Ir. Soekarno Bangka Belitung", "telp": "07179106756", "ket": "RS Rujukan Provinsi Babel"},
        {"nama": "RSUD Dr. H. Abdul Moeloek Bandar Lampung", "telp": "0721703312", "ket": "RS Rujukan Utama Lampung"},
    ],
    "DKI Jakarta, Jawa Barat & Banten (Bandung, Cimenyan, Bogor, Serang)": [
        {"nama": "RSUPN Dr. Cipto Mangunkusumo (RSCM) Jakarta", "telp": "0211500135", "ket": "RS Rujukan Nasional & Anak"},
        {"nama": "RSUP Dr. Hasan Sadikin (RSHS) Bandung", "telp": "0222034953", "ket": "RS Rujukan Utama Jawa Barat"},
        {"nama": "RSUD Bandung Kiwari (Khusus Ibu & Anak)", "telp": "0225200505", "ket": "IGD & Poli Spesialis Anak Kota Bandung"},
        {"nama": "Puskesmas Cimenyan (Kab. Bandung)", "telp": "0222503619", "ket": "Layanan Kesehatan Kec. Cimenyan"},
        {"nama": "RSUD Dr. Drajat Prawiranegara Serang", "telp": "0254200528", "ket": "RS Rujukan Utama Banten"},
    ],
    "Jawa Tengah, DI Yogyakarta & Jawa Timur (Semarang, Solo, Jogja, Surabaya)": [
        {"nama": "RSUP Dr. Kariadi Semarang", "telp": "0248413476", "ket": "RS Rujukan Utama Jawa Tengah"},
        {"nama": "RSUP Dr. Sardjito Yogyakarta", "telp": "0274587333", "ket": "RS Rujukan Utama DIY"},
        {"nama": "RSUD Dr. Soetomo Surabaya", "telp": "0315501078", "ket": "RS Rujukan Utama Jawa Timur"},
        {"nama": "RSUD Dr. Saiful Anwar Malang", "telp": "0341362101", "ket": "RS Rujukan Jawa Timur Selatan"},
        {"nama": "RSUD Moh. Anwar Sumenep (Kepulauan Madura)", "telp": "0328662494", "ket": "Rujukan Wilayah Kepulauan Sumenep"},
    ],
    "Bali, NTB & NTT (Denpasar, Mataram, Bima, Kupang, Sumba, Flores, Alor)": [
        {"nama": "RSUP Prof. Dr. I.G.N.G. Ngoerah (Sanglah) Denpasar", "telp": "0361227911", "ket": "RS Rujukan Utama Bali"},
        {"nama": "RSUD Provinsi NTB (Mataram)", "telp": "03707502424", "ket": "RS Rujukan Utama NTB"},
        {"nama": "RSUD Bima (Pulau Sumbawa)", "telp": "037443142", "ket": "IGD & Anak Wilayah Bima/Dompu"},
        {"nama": "RSUD Prof. Dr. W.Z. Johannes Kupang", "telp": "0380833614", "ket": "RS Rujukan Utama NTT"},
        {"nama": "RSUD Umbu Rara Meha Waingapu (Sumba Timur)", "telp": "038761302", "ket": "RS Rujukan Pulau Sumba (3T)"},
        {"nama": "RSUD Kalabahi (Kab. Alor)", "telp": "038621008", "ket": "RS Kepulauan Alor (3T)"},
        {"nama": "RSUD Ba'a (Kab. Rote Ndao)", "telp": "0380871014", "ket": "RS Wilayah Terselatan RI (3T)"},
    ],
    "Kalimantan (Pontianak, Kapuas Hulu, Banjarmasin, Samarinda, Nunukan, Malinau)": [
        {"nama": "RSUD Dr. Soedarso Pontianak", "telp": "0561737701", "ket": "RS Rujukan Utama Kalbar"},
        {"nama": "RSUD dr. Achmad Diponegoro Putussibau", "telp": "056721018", "ket": "Kapuas Hulu / Perbatasan Kalbar (3T)"},
        {"nama": "RSUD Ulin Banjarmasin", "telp": "05113252180", "ket": "RS Rujukan Utama Kalsel"},
        {"nama": "RSUD Dr. Doris Sylvanus Palangka Raya", "telp": "05363224695", "ket": "RS Rujukan Utama Kalteng"},
        {"nama": "RSUD Abdul Wahab Sjahranie Samarinda", "telp": "0541738118", "ket": "RS Rujukan Utama Kaltim"},
        {"nama": "RSUD Kabupaten Nunukan (Perbatasan)", "telp": "055621118", "ket": "RS Perbatasan Kaltara (3T)"},
        {"nama": "RSUD Kabupaten Malinau", "telp": "055321118", "ket": "RS Pedalaman Kaltara (3T)"},
    ],
    "Sulawesi (Makassar, Manado, Talaud, Sangihe, Palu, Kendari, Wakatobi)": [
        {"nama": "RSUP Dr. Wahidin Sudirohusodo Makassar", "telp": "0411584677", "ket": "RS Rujukan Utama Indonesia Timur"},
        {"nama": "RSUP Prof. Dr. R.D. Kandou Manado", "telp": "0431838305", "ket": "RS Rujukan Utama Sulut"},
        {"nama": "RSUD Mala Melonguane (Kep. Talaud)", "telp": "0433311018", "ket": "RS Perbatasan Utara RI (3T)"},
        {"nama": "RSUD Liun Kendage Tahuna (Sangihe)", "telp": "043221018", "ket": "RS Kepulauan Sangihe (3T)"},
        {"nama": "RSUD Undata Palu", "telp": "04514908020", "ket": "RS Rujukan Utama Sulteng"},
        {"nama": "RSUD Bahteramas Kendari", "telp": "04013195611", "ket": "RS Rujukan Utama Sultra"},
        {"nama": "RSUD Kabupaten Wakatobi", "telp": "040421018", "ket": "RS Kepulauan Wakatobi (3T)"},
    ],
    "Maluku & Maluku Utara (Ambon, Tual, Saumlaki, Ternate, Halmahera)": [
        {"nama": "RSUD Dr. M. Haulussy Ambon", "telp": "0911344871", "ket": "RS Rujukan Utama Maluku"},
        {"nama": "RSUD Karel Sadsuitubun Tual (Maluku Tenggara)", "telp": "091621023", "ket": "RS Kepulauan Kei / Tual (3T)"},
        {"nama": "RSUD Dr. P.P. Magretti Saumlaki (Tanimbar)", "telp": "091821018", "ket": "RS Kepulauan Tanimbar (3T)"},
        {"nama": "RSUD Dr. H. Chasan Boesoirie Ternate", "telp": "09213121281", "ket": "RS Rujukan Utama Maluku Utara"},
    ],
    "Tanah Papua (Jayapura, Wamena, Asmat, Timika, Merauke, Sorong, Manokwari)": [
        {"nama": "RSUD Dok II Jayapura", "telp": "0967533616", "ket": "RS Rujukan Utama Papua"},
        {"nama": "RSUD Wamena (Kab. Jayawijaya)", "telp": "096931066", "ket": "RS Rujukan Papua Pegunungan (3T)"},
        {"nama": "RSUD Agats (Kab. Asmat)", "telp": "090231018", "ket": "RS Wilayah Pedalaman Asmat (3T)"},
        {"nama": "RSUD Mimika (Timika)", "telp": "0901322222", "ket": "RS Rujukan Papua Tengah"},
        {"nama": "RSUD Merauke", "telp": "0971321124", "ket": "RS Rujukan Papua Selatan"},
        {"nama": "RSUD Sele Be Solu Kota Sorong", "telp": "0951321850", "ket": "RS Rujukan Papua Barat Daya"},
        {"nama": "RSUD Manokwari", "telp": "0986211440", "ket": "RS Rujukan Papua Barat"},
    ]
}

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
if "custom_faskes_list" not in st.session_state:
    st.session_state.custom_faskes_list = []

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
    st.markdown("<div style='text-align:center;'><span class='step-indicator'>Selamat Datang • 1/5</span><br><span class='offline-badge'>⚡ Mode 100% Offline Siap Pakai (3T)</span></div>", unsafe_allow_html=True)
    
    render_mascot(width_ratio=3.4, fallback_svg=BEAR_GUIDE_SVG)

    st.markdown("""
    <div class="mobile-card card-yellow" style="text-align:center; margin-top:0.5rem;">
        <h2 style="color:#0369A1; margin:0 0 0.4rem 0; font-size:1.5rem;">Halo Ayah & Bunda!</h2>
        <p style="font-size:0.95rem; color:#475569; line-height:1.5; margin:0;">
            <b>RespiKidd</b> bekerja <b>tanpa koneksi internet</b> untuk skrining dini gejala bronkitis pada batita lewat suara batuk, dilengkapi database kontak Puskesmas & RS rujukan seluruh Indonesia.
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
        <b style="color:#0369A1;">1. Isi Identitas & Wilayah</b><br>
        <span style="font-size:0.88rem;">Data nama anak, orang tua, dan wilayah membantu paramedis Puskesmas/Posyandu mencatat pasien secara offline.</span>
    </div>
    <div class="mobile-card card-peach">
        <b style="color:#C2410C;">2. Rekam / Unggah Suara Batuk</b><br>
        <span style="font-size:0.88rem;">Dekatkan HP 15–20 cm saat anak batuk. AI langsung menganalisis di perangkat tanpa butuh kuota data.</span>
    </div>
    <div class="mobile-card card-mint">
        <b style="color:#047857;">3. Hubungi Kontak Faskes / Kirim SMS</b><br>
        <span style="font-size:0.88rem;">Akses direktori offline nomor RSUD, Puskesmas, darurat 119/112, serta kirim hasil via SMS atau file rekam medis.</span>
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
        <b style="color:#9A3412; font-size:0.88rem;">📋 Identitas Pasien (Pendataan Offline 3T)</b><br>
        <span style="font-size:0.8rem; color:#78350F;">Membantu bidan, perawat, atau dokter Puskesmas mendata pasien dengan cepat.</span>
    </div>
    """, unsafe_allow_html=True)

    child_name = st.text_input("Nama Lengkap Anak *", placeholder="Contoh: Budi Santoso")
    c_id1, c_id2 = st.columns([1.4, 1])
    with c_id1:
        parent_name = st.text_input("Nama Orang Tua / Wali *", placeholder="Contoh: Ibu Siti")
    with c_id2:
        child_age = st.number_input("Usia (Bulan)", min_value=1, max_value=60, value=18)

    selected_region = st.selectbox("Pilih Provinsi / Wilayah Rujukan *", list(OFFLINE_FASKES_DB.keys()))
    region_detail = st.text_input("Desa / Kecamatan / Puskesmas Setempat", placeholder="Contoh: Desa Wouma, Kec. Wamena")

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
        if st.button("Analisis Suara Offline ✨"):
            if not child_name.strip() or not parent_name.strip():
                st.warning("Mohon isi **Nama Anak** dan **Nama Orang Tua / Wali** terlebih dahulu.")
            else:
                with st.spinner("Membersihkan noise & menganalisis suara secara offline..."):
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
                                "selected_region": selected_region,
                                "region_detail": region_detail.strip(),
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
# LAYAR 4: HASIL ANALISIS AI, DIREKTORI KONTAK OFFLINE & LAPORAN PARAMEDIS
# ==============================================================================
elif st.session_state.page == 4:
    st.markdown(TOP_DOODLE_SVG, unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;'><span class='step-indicator'>Hasil Skrining • 4/5</span></div>", unsafe_allow_html=True)
    
    render_mascot(width_ratio=1.4, fallback_svg=BEAR_EDU_SVG)
    
    res = st.session_state.result_data

    if res is not None:
        wilayah_tampil = res["region_detail"] if res["region_detail"] else (
            res["selected_region"] if "Pilih Wilayah" not in res["selected_region"] else "-"
        )

        # 1. Kartu Identitas Pasien
        st.markdown(f"""
        <div class="mobile-card card-blue" style="padding:0.85rem 1rem; margin-bottom:0.7rem;">
            <div style="font-size:0.82rem; color:#0369A1; font-weight:800;">🪪 REKAM SKRINING PASIEN (OFFLINE)</div>
            <div style="font-size:0.88rem; margin-top:0.25rem;">
                <b>Anak:</b> {res['child_name']} ({res['child_age']} bln)<br>
                <b>Orang Tua/Wali:</b> {res['parent_name']}<br>
                <b>Wilayah:</b> {wilayah_tampil}<br>
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
                    Pola suara mengarah pada batuk berlendir/menyempit. Segera hubungi atau bawa <b>{res['child_name']}</b> ke Puskesmas/RS rujukan di bawah ini.
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
        # 3. DIREKTORI KONTAK OFFLINE: DARURAT NASIONAL, RSUD & PUSKESMAS
        # ======================================================================
        st.markdown("""
        <div class="mobile-card card-peach" style="padding:0.95rem 1rem; margin-bottom:0.6rem;">
            <b style="color:#C2410C; font-size:0.95rem;">📞 Direktori Kontak Medis Offline</b><br>
            <span style="font-size:0.82rem; color:#7C2D12;">
                Tekan tombol untuk menelepon langsung via jaringan seluler (tanpa kuota internet):
            </span>
            <a href="tel:119" class="action-btn btn-red">🚑 Darurat Medis Kemenkes (119)</a>
            <a href="tel:112" class="action-btn btn-red" style="background:linear-gradient(135deg, #EA580C 0%, #C2410C 100%);">🆘 Panggilan Darurat Nasional (112)</a>
            <a href="tel:1500567" class="action-btn btn-amber">☎️ Halo Kemenkes RI (1500-567)</a>
        </div>
        """, unsafe_allow_html=True)

        # Tampilkan daftar RSUD & Puskesmas sesuai provinsi yang dipilih di Layar 3 (atau ubah di sini)
        wilayah_pilih = st.selectbox(
            "Lihat Kontak RSUD / Puskesmas Wilayah:",
            list(OFFLINE_FASKES_DB.keys()),
            index=list(OFFLINE_FASKES_DB.keys()).index(res["selected_region"])
        )

        faskes_list = OFFLINE_FASKES_DB.get(wilayah_pilih, [])
        if faskes_list:
            for f in faskes_list:
                st.markdown(f"""
                <div class="mobile-card" style="padding:0.8rem 1rem; margin-bottom:0.5rem; border-color:#BAE6FD;">
                    <b style="color:#0369A1; font-size:0.9rem;">🏥 {f['nama']}</b><br>
                    <span style="font-size:0.8rem; color:#64748B;">{f['ket']} • Telp: <b>{f['telp']}</b></span>
                    <a href="tel:{f['telp']}" class="action-btn btn-green" style="margin-top:0.4rem; margin-bottom:0;">📞 Telepon {f['nama']}</a>
                </div>
                """, unsafe_allow_html=True)

        # Fitur Tambah Kontak Bidan / Dokter / Puskesmas Desa Setempat (Offline)
        with st.expander("➕ Tambah Kontak Bidan / Dokter / Puskesmas Desa Lokal"):
            new_faskes_name = st.text_input("Nama Bidan / Dokter / Puskesmas:", placeholder="Contoh: Bidan Ani (Pustu Desa)")
            new_faskes_phone = st.text_input("Nomor HP / Telepon:", placeholder="Contoh: 081234567890")
            if st.button("Simpan Kontak Lokal"):
                if new_faskes_name.strip() and new_faskes_phone.strip():
                    st.session_state.custom_faskes_list.append({
                        "nama": new_faskes_name.strip(),
                        "telp": new_faskes_phone.strip()
                    })
                    st.success("Kontak lokal tersimpan!")

        for cf in st.session_state.custom_faskes_list:
            st.markdown(f"""
            <div class="mobile-card card-mint" style="padding:0.8rem 1rem; margin-bottom:0.5rem;">
                <b style="color:#047857; font-size:0.9rem;">🩺 {cf['nama']} (Kontak Lokal)</b><br>
                <span style="font-size:0.82rem;">Nomor: <b>{cf['telp']}</b></span>
                <a href="tel:{cf['telp']}" class="action-btn btn-blue" style="margin-top:0.4rem; margin-bottom:0;">📞 Hubungi {cf['nama']}</a>
            </div>
            """, unsafe_allow_html=True)

        # ======================================================================
        # 4. GPS SATELIT OFFLINE (TANPA INTERNET)
        # ======================================================================
        components.html("""
        <div style="font-family: 'Segoe UI', sans-serif; background: #F0F9FF; border: 2px solid #7DD3FC; border-radius: 20px; padding: 12px; text-align: center;">
            <div style="color: #0369A1; font-weight: 800; font-size: 13.5px; margin-bottom: 4px;">
                🛰️ Ambil Koordinat GPS Satelit (Offline)
            </div>
            <div style="color: #475569; font-size: 12px; margin-bottom: 8px;">
                Mengambil titik koordinat lokasi pasien untuk dicantumkan pada laporan rujukan medis.
            </div>
            <button id="gpsBtn" onclick="getOfflineGPS()" style="
                width: 100%; padding: 9px 12px; border-radius: 14px; border: 2px solid #FFFFFF;
                background: linear-gradient(135deg, #0284C7 0%, #2563EB 100%);
                color: white; font-weight: 700; font-size: 13px; cursor: pointer;
            ">📍 Izinkan & Baca Koordinat GPS</button>
            <div id="gpsOut" style="font-size: 12.5px; font-weight: 700; color: #0F172A; margin-top: 8px;"></div>
        </div>
        <script>
        function getOfflineGPS() {
            const out = document.getElementById("gpsOut");
            const btn = document.getElementById("gpsBtn");
            if (!navigator.geolocation) {
                out.innerHTML = "❌ Perangkat tidak mendukung GPS.";
                return;
            }
            btn.innerHTML = "🛰️ Mengunci sinyal satelit GPS...";
            navigator.geolocation.getCurrentPosition(
                function(pos) {
                    btn.style.display = "none";
                    out.innerHTML = "✅ Koordinat: " + pos.coords.latitude.toFixed(5) + ", " + pos.coords.longitude.toFixed(5) + "<br><span style='font-weight:400; font-size:11px; color:#475569;'>Catat/screenshot koordinat ini untuk tim evakuasi medis.</span>";
                },
                function(err) {
                    btn.innerHTML = "📍 Coba Baca GPS Lagi";
                    out.innerHTML = "⚠️ Izin lokasi ditolak atau GPS perangkat belum aktif.";
                },
                { enableHighAccuracy: true, timeout: 10000 }
            );
        }
        </script>
        """, height=145)

        # ======================================================================
        # 5. KIRIM VIA SMS (TANPA KUOTA) & UNDUH REKAM MEDIS OFFLINE (.TXT)
        # ======================================================================
        gejala_list = []
        if res["has_wheezing"]:
            gejala_list.append("Napas ngik-ngik/berlendir")
        if res["has_fever"]:
            gejala_list.append("Demam/batuk > 3 hari")
        gejala_str = ", ".join(gejala_list) if gejala_list else "Tidak ada gejala tambahan"

        laporan_teks = f"""LAPORAN SKRINING AWAL RESPIKIDD (OFFLINE)
========================================
Waktu Pemeriksaan : {res['timestamp']}
Nama Anak         : {res['child_name']}
Usia Anak         : {res['child_age']} bulan
Nama Orang Tua    : {res['parent_name']}
Wilayah           : {wilayah_tampil}
Gejala Tambahan   : {gejala_str}
----------------------------------------
Hasil Skrining AI : {status_label}
Tingkat Keyakinan : {conf:.1f}%
----------------------------------------
Catatan: Hasil skrining awal suara batuk untuk membantu pemeriksaan bidan/dokter."""

        sms_encoded = urllib.parse.quote(laporan_teks)
        st.markdown(f"""
        <a href="sms:?body={sms_encoded}" class="action-btn btn-blue">✉️ Kirim Laporan via SMS (Tanpa Kuota Data)</a>
        """, unsafe_allow_html=True)

        st.download_button(
            label="📄 Simpan Laporan Rekam Medis ke HP (.txt)",
            data=laporan_teks,
            file_name=f"Skrining_RespiKidd_{res['child_name'].replace(' ', '_')}.txt",
            mime="text/plain"
        )

        # 6. Grafik Mel-Spectrogram
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
