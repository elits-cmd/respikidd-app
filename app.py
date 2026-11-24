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

    /* Background Luar Cerah dengan Pola Doodle SVG */
    .stApp {
        background-color: #E0F7FA;
        background-image: 
            url("data:image/svg+xml,%3Csvg width='180' height='180' viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cpath d='M35 55c-6.6 0-12-5.4-12-12 0-5.8 4.1-10.6 9.6-11.8C34.5 24.8 40.4 20 47.5 20c8.6 0 15.5 6.9 15.5 15.5 0 .6 0 1.2-.1 1.8 4.6 1.5 8.1 5.8 8.1 11 0 6.5-5.2 11.7-11.7 11.7H35z' fill='%23FFFFFF' fill-opacity='0.85'/%3E%3Cpath d='M135 32l3.2 6.5 7.2 1-5.2 5.1 1.2 7.2-6.4-3.4-6.4 3.4 1.2-7.2-5.2-5.1 7.2-1z' fill='%23FDE047' fill-opacity='0.9'/%3E%3Cpath d='M45 130c-3.5-4.5-10.5-3.5-12.5 1.5-2-5-9-6-12.5-1.5-3.5 4.5-1 11 12.5 18 13.5-7 16-13.5 12.5-18z' fill='%23F9A8D4' fill-opacity='0.75'/%3E%3Cpath d='M130 125q5 0 5-5 0 5 5 5-5 0-5 5 0-5-5-5z' stroke='%235EEAD4' stroke-width='3' stroke-linecap='round'/%3E%3Ccircle cx='90' cy='85' r='4' fill='%23FDBA74' fill-opacity='0.7'/%3E%3Ccircle cx='160' cy='95' r='3.5' fill='%2393C5FD' fill-opacity='0.8'/%3E%3Ccircle cx='18' cy='90' r='3.5' fill='%23C4B5FD' fill-opacity='0.7'/%3E%3C/g%3E%3C/svg%3E"),
            linear-gradient(135deg, #E0F2FE 0%, #FEF9C3 50%, #FCE7F3 100%);
        font-family: 'Nunito', sans-serif;
    }

    /* Bingkai Layar Mobile App */
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

# Hiasan Doodle Awan & Bintang Gemoy untuk Bagian Atas Layar
TOP_DOODLE_SVG = """
<div style="text-align:center; margin-bottom:-0.3rem
