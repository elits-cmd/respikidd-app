import os
import time
import wave
from datetime import datetime
import numpy as np

from kivy.app import App
from kivy.core.window import Window
from kivy.utils import get_color_from_hex
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.spinner import Spinner
from kivy.uix.checkbox import CheckBox
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.graphics import Color, RoundedRectangle

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

try:
    from plyer import audio
    HAS_PLYER_AUDIO = True
except Exception:
    HAS_PLYER_AUDIO = False

Window.clearcolor = get_color_from_hex("#E0F7FA")

# ==============================================================================
# DATABASE OFFLINE KONTAK MEDIS NASIONAL & RS RUJUKAN SELURUH INDONESIA
# ==============================================================================
OFFLINE_FASKES_DB = {
    "Layanan Darurat Nasional (24 Jam)": [
        ("PSC 119 Kemenkes RI", "119", "Ambulans & Gawat Darurat Nasional"),
        ("Panggilan Darurat Terpadu", "112", "Aktif Tanpa Kuota / Sinyal Darurat"),
        ("Halo Kemenkes RI", "1500567", "Pusat Informasi Kesehatan RI"),
        ("RSAB Harapan Kita Jakarta", "0215668284", "Rujukan Nasional Pernapasan Anak"),
    ],
    "Jawa Barat & DKI Jakarta": [
        ("RSCM Jakarta", "0211500135", "RS Rujukan Nasional & Anak"),
        ("RSUP Dr. Hasan Sadikin Bandung", "0222034953", "RS Rujukan Utama Jawa Barat"),
        ("RSUD Bandung Kiwari", "0225200505", "Khusus Ibu & Anak Kota Bandung"),
        ("Puskesmas Cimenyan", "0222503619", "Layanan Kesehatan Kec. Cimenyan"),
    ],
    "Aceh & Sumatera Utara": [
        ("RSUD Dr. Zainoel Abidin", "065134565", "RS Rujukan Utama Aceh"),
        ("RSUD Simeulue (3T)", "065021018", "IGD & Poli Anak Simeulue"),
        ("RSUP H. Adam Malik Medan", "0618360381", "RS Rujukan Utama Sumut"),
        ("RSUD dr. M. Thomsen Nias", "063921018", "Rujukan Kepulauan Nias (3T)"),
    ],
    "Sumatera Barat, Riau & Kepri": [
        ("RSUP Dr. M. Djamil Padang", "075132371", "RS Rujukan Utama Sumbar"),
        ("RSUD Mentawai (Tuapejat)", "0759320018", "Rujukan Kepulauan Mentawai (3T)"),
        ("RSUD Arifin Achmad Pekanbaru", "076121618", "RS Rujukan Utama Riau"),
        ("RSUD Kabupaten Natuna", "077331018", "RS Perbatasan Natuna (3T)"),
    ],
    "Jawa Tengah, DIY & Jawa Timur": [
        ("RSUP Dr. Kariadi Semarang", "0248413476", "RS Rujukan Utama Jateng"),
        ("RSUP Dr. Sardjito Yogyakarta", "0274587333", "RS Rujukan Utama DIY"),
        ("RSUD Dr. Soetomo Surabaya", "0315501078", "RS Rujukan Utama Jatim"),
        ("RSUD Moh. Anwar Sumenep", "0328662494", "Rujukan Kepulauan Madura"),
    ],
    "Bali, NTB & NTT": [
        ("RSUP Prof. Ngoerah Denpasar", "0361227911", "RS Rujukan Utama Bali"),
        ("RSUD Provinsi NTB Mataram", "03707502424", "RS Rujukan Utama NTB"),
        ("RSUD Prof. W.Z. Johannes Kupang", "0380833614", "RS Rujukan Utama NTT"),
        ("RSUD Umbu Rara Meha Sumba", "038761302", "RS Rujukan Pulau Sumba (3T)"),
    ],
    "Kalimantan": [
        ("RSUD Dr. Soedarso Pontianak", "0561737701", "RS Rujukan Utama Kalbar"),
        ("RSUD Putussibau (Kapuas Hulu)", "056721018", "Perbatasan Kalbar (3T)"),
        ("RSUD Ulin Banjarmasin", "05113252180", "RS Rujukan Utama Kalsel"),
        ("RSUD Kabupaten Nunukan", "055621118", "RS Perbatasan Kaltara (3T)"),
    ],
    "Sulawesi, Maluku & Papua": [
        ("RSUP Dr. Wahidin Makassar", "0411584677", "Rujukan Utama Indonesia Timur"),
        ("RSUP Prof. R.D. Kandou Manado", "0431838305", "RS Rujukan Utama Sulut"),
        ("RSUD Dr. M. Haulussy Ambon", "0911344871", "RS Rujukan Utama Maluku"),
        ("RSUD Dok II Jayapura", "0967533616", "RS Rujukan Utama Papua"),
        ("RSUD Wamena (Jayawijaya)", "096931066", "Papua Pegunungan (3T)"),
    ],
}

# ==============================================================================
# PIPELINE AUDIO & MEL-SPECTROGRAM 100% NUMPY MURNI (RINGAN UNTUK ANDROID)
# ==============================================================================
def read_wav_numpy(wav_path):
    with wave.open(wav_path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    if sampwidth == 2:
        data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        data = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        data = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError("Format bit WAV tidak didukung. Gunakan 16-bit PCM WAV.")

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return sr, data


def resize_2d_nearest(arr, target_h=128, target_w=128):
    row_idx = (np.linspace(0, arr.shape[0] - 1, target_h)).astype(int)
    col_idx = (np.linspace(0, arr.shape[1] - 1, target_w)).astype(int)
    return arr[np.ix_(row_idx, col_idx)]


def extract_mel_spectrogram_offline(wav_path, target_sr=22050, n_mels=128):
    sr, y = read_wav_numpy(wav_path)

    # Resample sederhana menggunakan interpolasi linear NumPy
    if sr != target_sr and len(y) > 0:
        duration = len(y) / float(sr)
        target_len = int(duration * target_sr)
        x_old = np.linspace(0, duration, len(y))
        x_new = np.linspace(0, duration, target_len)
        y = np.interp(x_new, x_old, y).astype(np.float32)
        sr = target_sr

    # Potong maksimal 3 detik
    max_samples = target_sr * 3
    if len(y) > max_samples:
        y = y[:max_samples]

    # Cek RMS (validasi suara hening)
    rms = np.sqrt(np.mean(y ** 2)) if len(y) > 0 else 0.0
    if rms < 0.004:
        return None, "Suara terlalu pelan atau hening. Dekatkan HP 15-20 cm saat anak batuk."

    # Normalisasi amplitudo
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    # Hitung STFT menggunakan NumPy rfft
    n_fft = 2048
    hop_length = 512
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))

    window = np.hanning(n_fft).astype(np.float32)
    num_frames = 1 + (len(y) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(num_frames, n_fft),
        strides=(y.strides[0] * hop_length, y.strides[0]),
    )
    stft_mag = np.abs(np.fft.rfft(frames * window, axis=1)).T
    power_spec = stft_mag ** 2

    # Buat Mel Filterbank
    f_min, f_max = 0.0, sr / 2.0
    mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbank = np.zeros((n_mels, int(np.floor(n_fft / 2 + 1))), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus, f_m, f_m_plus = bins[m - 1], bins[m], bins[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    mel_spec = np.dot(fbank, power_spec)
    mel_spec_db = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
    mel_spec_db -= np.max(mel_spec_db)

    # Normalisasi Min-Max
    mel_min_val, mel_max_val = mel_spec_db.min(), mel_spec_db.max()
    if mel_max_val - mel_min_val > 1e-6:
        mel_norm = (mel_spec_db - mel_min_val) / (mel_max_val - mel_min_val)
    else:
        mel_norm = np.clip((mel_spec_db + 80.0) / 80.0, 0.0, 1.0)

    mel_resized = resize_2d_nearest(mel_norm, 128, 128).astype(np.float32)
    img_input = np.expand_dims(mel_resized[..., np.newaxis], axis=0)
    return img_input, None


def predict_with_tflite(img_input, model_path="respikidd_model.tflite
