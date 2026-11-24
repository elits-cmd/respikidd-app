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
    page_title="RespiKidd | Sahabat Pernapasan Si Kecil",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS Tema Pastel Ramah Anak (Anti-Trauma)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500;600;700&display=swap');
    
    /* Latar Belakang Utama Krem Lembut */
    .stApp {
        background-color: #FFFDF9;
        font-family: 'Quicksand', sans-serif;
        color: #334155;
    }

    h1, h2, h3, h4, p, div, span, label {
        font-family: 'Quicksand', sans-serif !important;
    }

    /* Sidebar Pastel */
    [data-testid="stSidebar"] {
        background-color: #F0F9FF;
        border-right: 2px dashed #BAE6FD;
    }

    /* Kartu Sambutan Utama */
    .pastel-hero {
        background: linear-gradient(135deg, #E0F2FE 0%, #FFEDD5 100%);
        padding: 2rem;
        border-radius: 24px;
        border: 2px solid #FDE68A;
        box-shadow: 0 8px 20px rgba(251, 191, 36, 0.12);
        margin-bottom: 1.5rem;
    }

    /* Kartu Langkah Panduan */
    .step-card {
        background-color: #FFFFFF;
        padding: 1.4rem;
        border-radius: 20px;
        border: 2px solid #E2E8F0;
        box-shadow: 0 4px 12px rgba(148, 163, 184, 0.08);
        height: 100%;
    }
    .step-badge {
        display: inline-block;
        background-color: #FDBA74;
        color: #7C2D12;
        font-weight: 700;
        padding: 0.25rem 0.8rem;
        border-radius: 999px;
        font-size: 0.85rem;
        margin-bottom: 0.6rem;
    }

    /* Kartu Edukasi */
    .edu-card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 20px;
        border-left: 6px solid #7DD3FC;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
    }

    /* Kartu Hasil Deteksi Ramah Orang Tua */
    .result-pastel-warn {
        background-color: #FFF1F2;
        border: 2px solid #FDA4AF;
        border-left: 8px solid #FB7185;
        padding: 1.5rem;
        border-radius: 20px;
        color: #881337;
    }
    .result-pastel-safe {
        background-color: #ECFDF5;
        border: 2px solid #6EE7B7;
        border-left: 8px solid #34D399;
        padding: 1.5rem;
        border-radius: 20px;
        color: #065F46;
    }

    /* Tombol Pastel Lembut */
    .stButton>button {
        background: linear-gradient(90deg, #38BDF8 0%, #60A5FA 100%);
        color: white !important;
        border: none;
        border-radius: 16px;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
        font-size: 1.05rem;
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.3);
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        background: linear-gradient(90deg, #0284C7 0%, #3B82F6 100%);
    }
</style>
""", unsafe_allow_html=True)

# Inisialisasi Navigasi Halaman (Default: Halaman Panduan Pertama)
if "menu" not in st.session_state:
    st.session_state.menu = "🏠 Panduan & Cara Pakai"

def pindah_menu(nama_menu):
    st.session_state.menu = nama_menu

# 3. Sidebar dengan Maskot & Navigasi
with st.sidebar:
    col_img1, col_img2, col_img3 = st.columns([1, 2.5, 1])
    with col_img2:
        if os.path.exists("mascot.png"):
            st.image("mascot.png", use_container_width=True)
        else:
            st.markdown("<h1 style='text-align:center; font-size:4rem;'>🐱🩺</h1>", unsafe_allow_html=True)
            
    st.markdown("<h2 style='text-align:center; color:#0369A1; margin-top:0;'>RespiKidd</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:0.9rem; color:#64748B;'>Sahabat Skrining Pernapasan Batita yang Ramah & Bebas Trauma</p>", unsafe_allow_html=True)
    st.divider()
    
    # Pilihan Halaman
    pilihan = st.radio(
        "Pilih Halaman:",
        ["🏠 Panduan & Cara Pakai", "🩺 Cek Suara Batuk (AI)", "📚 Edukasi Bronkitis Anak"],
        index=["🏠 Panduan & Cara Pakai", "🩺 Cek Suara Batuk (AI)", "📚 Edukasi Bronkitis Anak"].index(st.session_state.menu)
    )
    st.session_state.menu = pilihan
    
    st.divider()
    st.caption("🏆 **ISPO 2026 — Bidang Komputer**")
    st.caption("Keyza Elita K. E. & Mahirah Alya S.\nPakar Belia Islamic Boarding School")

# Load Model CNN
@st.cache_resource
def load_respikidd_model():
    return tf.keras.models.load_model('respikidd_model.keras')

# ==============================================================================
# HALAMAN 1: SAMBUTAN & CARA PENGGUNAAN APLIKASI (HALAMAN PERTAMA)
# ==============================================================================
if st.session_state.menu == "🏠 Panduan & Cara Pakai":
    
    # Banner Sambutan dengan Maskot
    st.markdown('<div class="pastel-hero">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 3], vertical_alignment="center")
    with c1:
        if os.path.exists("mascot.png"):
            st.image("mascot.png", use_container_width=True)
        else:
            st.markdown("<div style='font-size:5rem; text-align:center;'>🐱🩺</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<h1 style='color:#0C4A6E; margin-bottom:0.3rem;'>Halo Ayah, Bunda & Si Kecil! 👋</h1>", unsafe_allow_html=True)
        st.markdown("""
        <p style='font-size:1.1rem; color:#334155; line-height:1.6;'>
        Selamat datang di <b>RespiKidd</b>! Kami dirancang khusus agar pemeriksaan awal batuk si kecil terasa nyaman, menyenangkan, dan <b>tanpa alat medis yang menakutkan</b>. Cukup rekam suara batuk lewat HP, AI kami akan membantu mendeteksi apakah ada tanda-tanda mengarah ke bronkitis.
        </p>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 📖 Cara Penggunaan Aplikasi RespiKidd")
    st.write("Ikuti 4 langkah mudah berikut sebelum melakukan pengecekan:")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="step-card">
            <span class="step-badge">Langkah 1</span>
            <h4>🧸 Buat Anak Nyaman</h4>
            <p style="font-size:0.95rem;">Pastikan si kecil dalam posisi tenang dan nyaman. Jauhkan dari suara TV atau keramaian agar rekaman suara batuk terdengar jernih.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="step-card">
            <span class="step-badge">Langkah 2</span>
            <h4>🎤 Rekam / Unggah</h4>
            <p style="font-size:0.95rem;">Buka menu <b>Cek Suara Batuk</b>, dekatkan mikrofon HP sekitar 15–20 cm dari anak saat batuk, atau unggah file rekaman (.wav/.mp3).</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="step-card">
            <span class="step-badge">Langkah 3</span>
            <h4>✨ Analisis Cerdas AI</h4>
            <p style="font-size:0.95rem;">Sistem otomatis membersihkan bising latar, mengubah suara menjadi grafik warna <i>Mel-Spectrogram</i>, lalu membacanya dengan Deep Learning.</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="step-card">
            <span class="step-badge">Langkah 4</span>
            <h4>🩺 Langkah Lanjutan</h4>
            <p style="font-size:0.95rem;">Lihat hasil skrining dalam hitungan detik. Gunakan hasil ini sebagai panduan awal sebelum berkonsultasi dengan dokter atau Puskesmas.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        st.button("🚀 Mulai Cek Suara Batuk Sekarang", on_click=pindah_menu, args=("🩺 Cek Suara Batuk (AI)",), use_container_width=True)

# ==============================================================================
# HALAMAN 2: DETEKSI SUARA BATUK (AI & MEL-SPECTROGRAM)
# ==============================================================================
elif st.session_state.menu == "🩺 Cek Suara Batuk (AI)":
    st.markdown("<h2 style='color:#0369A1;'>🩺 Pemeriksaan Suara Batuk Si Kecil</h2>", unsafe_allow_html=True)
    st.write("Rekam suara batuk secara langsung atau unggah rekaman suara yang sudah tersimpan.")

    try:
        model = load_respikidd_model()
    except Exception as e:
        st.error(f"Model AI belum ditemukan (`respikidd_model.keras`). Detail: {e}")
        st.stop()

    col_left, col_right = st.columns([1, 1.3], gap="large")

    with col_left:
        st.markdown("#### 1. Masukkan Sampel Suara")
        tab1, tab2 = st.tabs(["🎤 Rekam Langsung", "📂 Unggah File Audio"])
        audio_data = None
        
        with tab1:
            recorded_audio = st.audio_input("Tekan ikon mikrofon untuk merekam batuk:")
            if recorded_audio:
                audio_data = recorded_audio
                
        with tab2:
            uploaded_audio = st.file_uploader("Pilih file rekaman (.wav / .mp3 / .ogg):", type=["wav", "mp3", "ogg"])
            if uploaded_audio:
                audio_data = uploaded_audio

        if audio_data is not None:
            st.audio(audio_data)
            analyze_btn = st.button("🔍 Analisis Suara Batuk", use_container_width=True)
        else:
            st.info("🐱 **Kata Dokter Kucing:** Pastikan rekaman berisi suara batuk si kecil selama 1–3 detik ya!")
            analyze_btn = False

    with col_right:
        st.markdown("#### 2. Hasil Skrining & Grafik Suara")
        if audio_data is not None and analyze_btn:
            with st.spinner("Dokter Kucing sedang mendengarkan dan menganalisis gelombang suara..."):
                start_time = time.time()
                
                # Preprocessing Audio
                y, sr = librosa.load(audio_data, sr=22050, duration=3.0)
                y_clean = nr.reduce_noise(y=y, sr=sr)
                y_norm = librosa.util.normalize(y_clean)
                
                # Mel-Spectrogram
                mel_spec = librosa.feature.melspectrogram(y=y_norm, sr=sr, n_mels=128)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                mel_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
                
                # Resize ke (1, 128, 128, 1)
                mel_resized = tf.image.resize(mel_norm[..., np.newaxis], (128, 128)).numpy()
                img_input = np.expand_dims(mel_resized, axis=0)
                
                # Prediksi
                pred_prob = float(model.predict(img_input, verbose=0)[0][0])
                response_time = round(time.time() - start_time, 2)

            if pred_prob >= 0.5:
                confidence = pred_prob * 100
                st.markdown(f"""
                <div class="result-pastel-warn">
                    <h3 style="margin:0; color:#9F1239;">🧡 Terdeteksi Indikasi Gejala Bronkitis</h3>
                    <p style="margin-top:0.5rem; margin-bottom:0;">
                        Ayah & Bunda tidak perlu panik, namun pola suara batuk si kecil memiliki karakteristik berlendir/menyempit yang mengarah pada bronkitis. Sebaiknya segera periksakan ke dokter anak atau Puskesmas terdekat untuk penanganan tepat.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                confidence = (1 - pred_prob) * 100
                st.markdown(f"""
                <div class="result-pastel-safe">
                    <h3 style="margin:0; color:#047857;">💚 Pola Suara Batuk Normal (Non-Bronkitis)</h3>
                    <p style="margin-top:0.5rem; margin-bottom:0;">
                        Kabar baik! Tidak terdeteksi penanda khas bronkitis pada rekaman suara ini. Tetap berikan ASI/air putih hangat yang cukup dan pantau suhu tubuh si kecil.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            m1, m2, m3 = st.columns(3)
            m1.metric("Tingkat Keyakinan AI", f"{confidence:.1f}%")
            m2.metric("Kecepatan Proses", f"{response_time} detik")
            m3.metric("Dimensi Citra CNN", "128 × 128 px")

            st.markdown("**Grafik Visualisasi Mel-Spectrogram (Setelah Denoising):**")
            fig, ax = plt.subplots(figsize=(8, 3))
            img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set_title("Pola Frekuensi Suara Batuk Balita", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.markdown("""
            <div style="background-color:#FFFFFF; border: 2px dashed #BAE6FD; border-radius: 20px; padding: 3rem; text-align: center; color: #64748B;">
                <h4>Belum Ada Rekaman yang Dianalisis</h4>
                <p>Rekam atau unggah sampel suara di sebelah kiri, lalu tekan tombol <b>Analisis Suara Batuk</b>.</p>
            </div>
            """, unsafe_allow_html=True)

# ==============================================================================
# HALAMAN 3: EDUKASI KESEHATAN PERNAPASAN ANAK
# ==============================================================================
elif st.session_state.menu == "📚 Edukasi Bronkitis Anak":
    st.markdown("<h2 style='color:#0369A1;'>📚 Pojok Edukasi Ayah & Bunda</h2>", unsafe_allow_html=True)
    st.write("Informasi ringkas mengenai kesehatan saluran napas batita (usia di bawah 3 tahun).")

    col_ed1, col_ed2 = st.columns(2, gap="medium")

    with col_ed1:
        st.markdown("""
        <div class="edu-card">
            <h4 style="color:#0284C7; margin-top:0;">🫁 Apa Itu Bronkitis pada Batita?</h4>
            <p>Bronkitis adalah peradangan pada saluran udara utama (bronkus) yang umumnya dipicu oleh infeksi virus. Karena saluran napas anak di bawah usia 3 tahun masih sangat sempit dan sistem imunnya belum sempurna, pembengkakan dan lendir berlebih membuat napas serta batuk anak terdengar lebih berat dan basah.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="edu-card" style="border-left-color: #FDBA74;">
            <h4 style="color:#C2410C; margin-top:0;">🚭 Waspada Bahaya Asap Rokok!</h4>
            <p>Paparan asap rokok di lingkungan rumah terbukti memperparah kerentanan batita terhadap infeksi saluran pernapasan bawah. Pastikan rumah bebas dari asap rokok dan polusi debu agar saluran napas anak tetap sehat.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_ed2:
        st.markdown("""
        <div class="edu-card" style="border-left-color: #FDA4AF;">
            <h4 style="color:#BE123C; margin-top:0;">🚨 Tanda Bahaya yang Wajib Segera ke Dokter</h4>
            <ul>
                <li><b>Wheezing (Mengi):</b> Terdengar suara siulan nyaring saat anak menghembuskan napas.</li>
                <li><b>Takipnea (Napas Cepat):</b> Napas anak tampak sangat cepat dan terengah-engah meski sedang tenang/tidur.</li>
                <li><b>Tarikan Dinding Dada:</b> Kulit di sela tulang iga atau leher tampak tertarik ke dalam saat menarik napas.</li>
                <li>Anak tampak sangat lemas, demam tinggi, atau menolak menyusu/minum.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="edu-card" style="border-left-color: #6EE7B7;">
            <h4 style="color:#047857; margin-top:0;">💧 Perawatan Awal di Rumah</h4>
            <p>Jaga cairan tubuh anak dengan memberikan lebih banyak ASI atau air hangat untuk membantu mengencerkan lendir, gunakan pelembap udara ruangan bila perlu, dan selalu gunakan <b>RespiKidd</b> hanya sebagai alat skrining awal—bukan pengganti diagnosis dokter.</p>
        </div>
        """, unsafe_allow_html=True)
