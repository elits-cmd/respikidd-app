import streamlit as st
import librosa
import librosa.display
import noisereduce as nr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

# Pengaturan Halaman Web
st.set_page_config(page_title="RespiKidd AI", page_icon="🫁", layout="centered")

st.title("🫁 RespiKidd AI")
st.markdown("**Deteksi Dini Penyakit Bronkitis Anak Lewat Suara Batuk Berbasis Deep Learning**")
st.divider()

# 1. Load Model CNN yang sudah dilatih
@st.cache_resource
def load_respikidd_model():
    return tf.keras.models.load_model('respikidd_model.keras')

try:
    model = load_respikidd_model()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file 'respikidd_model.keras' ada di folder yang sama. Error: {e}")
    st.stop()

# 2. Input Audio (Bisa rekam langsung lewat mic atau unggah file)
st.subheader("1. Input Sampel Suara Batuk Balita")
tab1, tab2 = st.tabs(["🎙️ Rekam Langsung", "📁 Unggah File Audio"])

audio_data = None
with tab1:
    recorded_audio = st.audio_input("Tekan ikon mikrofon untuk mulai merekam suara batuk:")
    if recorded_audio:
        audio_data = recorded_audio

with tab2:
    uploaded_audio = st.file_uploader("Pilih file rekaman suara batuk (.wav / .mp3):", type=["wav", "mp3", "ogg"])
    if uploaded_audio:
        audio_data = uploaded_audio

# 3. Proses Analisis Audio & Prediksi
if audio_data is not None:
    st.audio(audio_data)
    
    if st.button("🔍 Analisis Suara Batuk Sekarang", type="primary", use_container_width=True):
        with st.spinner("Membersihkan bising (denoising) dan mengekstrak Mel-Spectrogram..."):
            start_time = time.time()
            
            # Load audio (sample rate standar 22050 Hz)
            y, sr = librosa.load(audio_data, sr=22050, duration=3.0)
            
            # Denoising menggunakan noisereduce (Bab 2.2)
            y_clean = nr.reduce_noise(y=y, sr=sr)
            
            # Normalisasi audio
            y_norm = librosa.util.normalize(y_clean)
            
            # Ekstraksi Mel-Spectrogram (tinggi = 128 sesuai input_shape=(128, 128, 1))
            mel_spec = librosa.feature.melspectrogram(y=y_norm, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalisasi nilai matriks ke rentang 0 - 1
            mel_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
            
            # Ubah ukuran (resize) agar lebarnya tepat 128 -> menjadi (128, 128, 1)
            mel_resized = tf.image.resize(mel_norm[..., np.newaxis], (128, 128)).numpy()
            img_input = np.expand_dims(mel_resized, axis=0)  # Shape akhir: (1, 128, 128, 1)
            
            # Prediksi menggunakan model CNN
            pred_prob = float(model.predict(img_input, verbose=0)[0][0])
            response_time = round(time.time() - start_time, 2)
            
        st.divider()
        st.subheader("2. Visualisasi Mel-Spectrogram (Rumusan Masalah #1)")
        fig, ax = plt.subplots(figsize=(7, 3))
        img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title("Grafik Frekuensi Suara Batuk (Setelah Denoising)")
        st.pyplot(fig)
        
        st.subheader("3. Hasil Skrining RespiKidd")
        col1, col2 = st.columns(2)
        
        # Interpretasi hasil (0 = non-bronkitis, 1 = bronchitis)
        if pred_prob >= 0.5:
            confidence = pred_prob * 100
            col1.metric("Status Deteksi", "Indikasi Bronkitis", delta="Perlu Pemeriksaan", delta_color="inverse")
            col2.metric("Tingkat Keyakinan AI", f"{confidence:.1f}%")
            st.error("⚠️ **Terdeteksi pola akustik batuk yang mengarah pada gejala Bronkitis.** Segera konsultasikan dengan dokter atau tenaga kesehatan di Puskesmas/RS terdekat untuk pemeriksaan fisik lebih lanjut.")
        else:
            confidence = (1 - pred_prob) * 100
            col1.metric("Status Deteksi", "Batuk Normal", delta="Aman / Non-Bronkitis")
            col2.metric("Tingkat Keyakinan AI", f"{confidence:.1f}%")
            st.success("✅ **Pola suara batuk cenderung normal (tidak terdeteksi ciri khas bronkitis).** Tetap pantau kondisi anak dan pastikan hidrasi tercukupi.")
            
        st.caption(f"⏱️ Waktu Respon Pemrosesan Cloud: **{response_time} detik** (Rumusan Masalah #3)")