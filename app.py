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
# MESIN PREDIKSI CNN 100% NUMPY MURNI (EXACT MATCH DENGAN MODEL KERAS)
# ==============================================================================
def conv2d_same_relu(x, kernel, bias):
    # x: (H, W, C_in), kernel: (3, 3, C_in, C_out), bias: (C_out,)
    H, W, C_in = x.shape
    kh, kw, _, C_out = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    x_pad = np.pad(x, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode="constant")

    # Ekstrak patch 3x3 secara vektorial menggunakan stride_tricks (sangat cepat)
    s0, s1, s2 = x_pad.strides
    patches = np.lib.stride_tricks.as_strided(
        x_pad,
        shape=(H, W, kh, kw, C_in),
        strides=(s0, s1, s0, s1, s2),
    )
    out = np.tensordot(patches, kernel, axes=([2, 3, 4], [0, 1, 2])) + bias
    return np.maximum(0.0, out)


def batch_norm(x, gamma, beta, moving_mean, moving_var, eps=1e-3):
    return gamma * (x - moving_mean) / np.sqrt(moving_var + eps) + beta


def max_pool2x2(x):
    H, W, C = x.shape
    return x.reshape(H // 2, 2, W // 2, 2, C).max(axis=(1, 3))


def predict_with_numpy_cnn(img_input, weights_path="respikidd_weights.npz"):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"File bobot '{weights_path}' tidak ditemukan.")
    w = np.load(weights_path)
    x = img_input[0]  # (128, 128, 1)

    # Blok 1: Conv2D (0) -> BatchNorm (1) -> MaxPool (2)
    x = conv2d_same_relu(x, w["layer_0_Conv2D_w0"], w["layer_0_Conv2D_w1"])
    x = batch_norm(x, w["layer_1_BatchNormalization_w0"], w["layer_1_BatchNormalization_w1"],
                   w["layer_1_BatchNormalization_w2"], w["layer_1_BatchNormalization_w3"])
    x = max_pool2x2(x)

    # Blok 2: Conv2D (3) -> BatchNorm (4) -> MaxPool (5)
    x = conv2d_same_relu(x, w["layer_3_Conv2D_w0"], w["layer_3_Conv2D_w1"])
    x = batch_norm(x, w["layer_4_BatchNormalization_w0"], w["layer_4_BatchNormalization_w1"],
                   w["layer_4_BatchNormalization_w2"], w["layer_4_BatchNormalization_w3"])
    x = max_pool2x2(x)

    # Blok 3: Conv2D (6) -> BatchNorm (7) -> MaxPool (8)
    x = conv2d_same_relu(x, w["layer_6_Conv2D_w0"], w["layer_6_Conv2D_w1"])
    x = batch_norm(x, w["layer_7_BatchNormalization_w0"], w["layer_7_BatchNormalization_w1"],
                   w["layer_7_BatchNormalization_w2"], w["layer_7_BatchNormalization_w3"])
    x = max_pool2x2(x)

    # GlobalAveragePooling2D (9) -> (64,)
    x = np.mean(x, axis=(0, 1))

    # Dense 1 (11) + ReLU
    x = np.maximum(0.0, np.dot(x, w["layer_11_Dense_w0"]) + w["layer_11_Dense_w1"])

    # Dense 2 (12) + Sigmoid
    logit = float(np.dot(x, w["layer_12_Dense_w0"])[0] + w["layer_12_Dense_w1"][0])
    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -50.0, 50.0)))
    return float(prob)


# ==============================================================================
# PIPELINE AUDIO & MEL-SPECTROGRAM 100% NUMPY MURNI
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

    if sr != target_sr and len(y) > 0:
        duration = len(y) / float(sr)
        target_len = int(duration * target_sr)
        x_old = np.linspace(0, duration, len(y))
        x_new = np.linspace(0, duration, target_len)
        y = np.interp(x_new, x_old, y).astype(np.float32)
        sr = target_sr

    max_samples = target_sr * 3
    if len(y) > max_samples:
        y = y[:max_samples]

    rms = np.sqrt(np.mean(y ** 2)) if len(y) > 0 else 0.0
    if rms < 0.004:
        return None, "Suara terlalu pelan atau hening. Dekatkan HP 15-20 cm saat anak batuk."

    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

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

    mel_min_val, mel_max_val = mel_spec_db.min(), mel_spec_db.max()
    if mel_max_val - mel_min_val > 1e-6:
        mel_norm = (mel_spec_db - mel_min_val) / (mel_max_val - mel_min_val)
    else:
        mel_norm = np.clip((mel_spec_db + 80.0) / 80.0, 0.0, 1.0)

    mel_resized = resize_2d_nearest(mel_norm, 128, 128).astype(np.float32)
    img_input = np.expand_dims(mel_resized[..., np.newaxis], axis=0)
    return img_input, None


# ==============================================================================
# KOMPONEN UI KARTU & TOMBOL PASTEL KIVY
# ==============================================================================
class PastelCard(BoxLayout):
    def __init__(self, bg_hex="#FFFFFF", **kwargs):
        super().__init__(orientation="vertical", padding=16, spacing=8, size_hint_y=None, **kwargs)
        self.bind(minimum_height=self.setter("height"))
        with self.canvas.before:
            Color(*get_color_from_hex(bg_hex))
            self.rect = RoundedRectangle(pos=self.pos, size=self.size, radius=[20])
        self.bind(pos=self._update_rect, size=self._update_rect)

    def _update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size


def make_label(text, font_size=15, bold=False, color_hex="#334155", halign="left"):
    lbl = Label(
        text=text,
        font_size=f"{font_size}sp",
        bold=bold,
        color=get_color_from_hex(color_hex),
        halign=halign,
        valign="middle",
        size_hint_y=None,
    )
    lbl.bind(width=lambda instance, value: setattr(instance, "text_size", (value, None)))
    lbl.bind(texture_size=lambda instance, value: setattr(instance, "height", value[1] + 6))
    return lbl


def make_button(text, bg_hex="#0284C7", on_press=None):
    btn = Button(
        text=text,
        font_size="16sp",
        bold=True,
        size_hint_y=None,
        height=50,
        background_normal="",
        background_color=get_color_from_hex(bg_hex),
        color=(1, 1, 1, 1),
    )
    if on_press:
        btn.bind(on_press=on_press)
    return btn


class WelcomeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        root = BoxLayout(orientation="vertical", padding=20, spacing=14)
        badge = make_label("100% MODE OFFLINE SIAP PAKAI (1/4)", 13, True, "#166534", "center")
        title = make_label("RespiKidd Android", 26, True, "#0369A1", "center")

        card = PastelCard("#FEFCE8")
        card.add_widget(make_label("Halo Ayah & Bunda!", 20, True, "#0369A1", "center"))
        card.add_widget(
            make_label(
                "RespiKidd bekerja 100% tanpa koneksi internet untuk skrining dini gejala bronkitis pada batita lewat suara batuk, dilengkapi kontak Puskesmas & RS rujukan seluruh Indonesia.",
                15,
                False,
                "#475569",
                "center",
            )
        )

        btn_start = make_button("Mulai Sekarang  >>", "#0284C7", lambda x: setattr(self.manager, "current", "guide"))
        root.add_widget(badge)
        root.add_widget(title)
        root.add_widget(card)
        root.add_widget(btn_start)
        self.add_widget(root)


class GuideScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        scroll = ScrollView()
        box = BoxLayout(orientation="vertical", padding=20, spacing=12, size_hint_y=None)
        box.bind(minimum_height=box.setter("height"))

        box.add_widget(make_label("Cara Pakai Aplikasi (2/4)", 18, True, "#0369A1", "center"))

        c1 = PastelCard("#F0F9FF")
        c1.add_widget(make_label("1. Isi Identitas & Wilayah", 16, True, "#0369A1"))
        c1.add_widget(make_label("Data anak & wilayah membantu paramedis Puskesmas mencatat pasien secara offline.", 14))
        box.add_widget(c1)

        c2 = PastelCard("#FFF7ED")
        c2.add_widget(make_label("2. Rekam / Pilih File Audio Batuk", 16, True, "#C2410C"))
        c2.add_widget(make_label("Dekatkan HP 15-20 cm saat anak batuk. AI memproses langsung di HP tanpa kuota.", 14))
        box.add_widget(c2)

        c3 = PastelCard("#ECFDF5")
        c3.add_widget(make_label("3. Lihat Hasil & Kontak Rujukan", 16, True, "#047857"))
        c3.add_widget(make_label("Akses nomor darurat 119/112, RSUD rujukan, dan simpan laporan teks di HP.", 14))
        box.add_widget(c3)

        box.add_widget(make_button("Lanjut Cek Batuk  >>", "#10B981", lambda x: setattr(self.manager, "current", "input")))
        box.add_widget(make_button("<<  Kembali", "#64748B", lambda x: setattr(self.manager, "current", "welcome")))

        scroll.add_widget(box)
        self.add_widget(scroll)


class InputScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.selected_audio_path = ""
        self.is_recording = False

        scroll = ScrollView()
        box = BoxLayout(orientation="vertical", padding=20, spacing=10, size_hint_y=None)
        box.bind(minimum_height=box.setter("height"))

        box.add_widget(make_label("Data Anak & Suara Batuk (3/4)", 18, True, "#0369A1", "center"))

        self.in_child = TextInput(hint_text="Nama Lengkap Anak *", multiline=False, size_hint_y=None, height=44)
        self.in_parent = TextInput(hint_text="Nama Orang Tua / Wali *", multiline=False, size_hint_y=None, height=44)
        self.in_age = TextInput(hint_text="Usia Anak (Bulan), contoh: 18", input_filter="int", multiline=False, size_hint_y=None, height=44)
        self.in_region = Spinner(
            text="Pilih Wilayah Rujukan...",
            values=list(OFFLINE_FASKES_DB.keys()),
            size_hint_y=None,
            height=46,
        )
        self.in_village = TextInput(hint_text="Desa / Kecamatan / Puskesmas", multiline=False, size_hint_y=None, height=44)

        box.add_widget(self.in_child)
        box.add_widget(self.in_parent)
        box.add_widget(self.in_age)
        box.add_widget(self.in_region)
        box.add_widget(self.in_village)

        row_g1 = BoxLayout(orientation="horizontal", size_hint_y=None, height=36)
        self.chk_wheezing = CheckBox(size_hint_x=None, width=40)
        row_g1.add_widget(self.chk_wheezing)
        row_g1.add_widget(make_label("Napas ngik-ngik / berlendir", 14))
        box.add_widget(row_g1)

        row_g2 = BoxLayout(orientation="horizontal", size_hint_y=None, height=36)
        self.chk_fever = CheckBox(size_hint_x=None, width=40)
        row_g2.add_widget(self.chk_fever)
        row_g2.add_widget(make_label("Demam / batuk > 3 hari", 14))
        box.add_widget(row_g2)

        self.lbl_audio = make_label("Belum ada file audio (.wav) dipilih.", 13, True, "#B45309")
        box.add_widget(self.lbl_audio)

        self.btn_rec = make_button("Rekam Suara Batuk (Mic HP)", "#F59E0B", self.toggle_record)
        btn_pick = make_button("Pilih File Audio (.wav) dari HP", "#6366F1", self.open_file_picker)
        btn_analyze = make_button("Analisis Suara Offline Sekarang", "#0284C7", self.run_analysis)

        box.add_widget(self.btn_rec)
        box.add_widget(btn_pick)
        box.add_widget(btn_analyze)
        box.add_widget(make_button("<<  Kembali", "#64748B", lambda x: setattr(self.manager, "current", "guide")))

        scroll.add_widget(box)
        self.add_widget(scroll)

    def show_msg(self, title, msg):
        content = BoxLayout(orientation="vertical", padding=12, spacing=10)
        content.add_widget(make_label(msg, 14))
        btn_ok = make_button("Tutup", "#0284C7")
        content.add_widget(btn_ok)
        popup = Popup(title=title, content=content, size_hint=(0.88, 0.4))
        btn_ok.bind(on_press=popup.dismiss)
        popup.open()

    def toggle_record(self, instance):
        if not HAS_PLYER_AUDIO:
            self.show_msg("Info Mic", "Fitur rekam langsung hanya aktif di HP Android. Silakan pilih file .wav.")
            return
        try:
            rec_path = os.path.join(App.get_running_app().user_data_dir, "rekaman_batuk.wav")
            if not self.is_recording:
                audio.file_path = rec_path
                audio.start()
                self.is_recording = True
                self.btn_rec.text = "Hentikan Rekaman (Sedang Merekam...)"
                self.btn_rec.background_color = get_color_from_hex("#E11D48")
            else:
                audio.stop()
                self.is_recording = False
                self.selected_audio_path = rec_path
                self.lbl_audio.text = f"Terekam: {os.path.basename(rec_path)}"
                self.btn_rec.text = "Rekam Ulang Suara Batuk"
                self.btn_rec.background_color = get_color_from_hex("#F59E0B")
        except Exception as e:
            self.show_msg("Kendala Mic", f"Gagal merekam audio: {e}")

    def open_file_picker(self, instance):
        content = BoxLayout(orientation="vertical", spacing=8, padding=8)
        chooser = FileChooserListView(path="/sdcard" if os.path.exists("/sdcard") else ".", filters=["*.wav"])
        btn_select = make_button("Gunakan File Ini", "#10B981")
        content.add_widget(chooser)
        content.add_widget(btn_select)

        popup = Popup(title="Pilih File Audio (.wav)", content=content, size_hint=(0.92, 0.8))

        def on_select(_):
            if chooser.selection:
                self.selected_audio_path = chooser.selection[0]
                self.lbl_audio.text = f"File dipilih: {os.path.basename(self.selected_audio_path)}"
                popup.dismiss()

        btn_select.bind(on_press=on_select)
        popup.open()

    def run_analysis(self, instance):
        child = self.in_child.text.strip()
        parent = self.in_parent.text.strip()
        if not child or not parent:
            self.show_msg("Data Belum Lengkap", "Mohon isi Nama Anak dan Nama Orang Tua terlebih dahulu.")
            return
        if not self.selected_audio_path or not os.path.exists(self.selected_audio_path):
            self.show_msg("Audio Belum Ada", "Silakan rekam suara batuk atau pilih file .wav terlebih dahulu.")
            return

        try:
            start_t = time.time()
            img_input, err_msg = extract_mel_spectrogram_offline(self.selected_audio_path)
            if err_msg:
                self.show_msg("Kualitas Suara", err_msg)
                return

            raw_prob = predict_with_numpy_cnn(img_input, "respikidd_weights.npz")
            bronchitis_score = 1.0 - raw_prob
            if self.chk_wheezing.active:
                bronchitis_score = min(1.0, bronchitis_score + 0.05)
            if self.chk_fever.active:
                bronchitis_score = min(1.0, bronchitis_score + 0.03)

            elapsed = round(time.time() - start_t, 2)
            result_screen = self.manager.get_screen("result")
            result_screen.display_result(
                {
                    "child_name": child,
                    "parent_name": parent,
                    "child_age": self.in_age.text.strip() or "-",
                    "region": self.in_region.text,
                    "village": self.in_village.text.strip() or "-",
                    "wheezing": "Ya" if self.chk_wheezing.active else "Tidak",
                    "fever": "Ya" if self.chk_fever.active else "Tidak",
                    "score": bronchitis_score,
                    "time": elapsed,
                    "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M"),
                }
            )
            self.manager.current = "result"
        except Exception as e:
            self.show_msg("Error Analisis", f"Terjadi kendala: {e}")


class ResultScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.latest_report_text = ""
        self.scroll = ScrollView()
        self.box = BoxLayout(orientation="vertical", padding=20, spacing=12, size_hint_y=None)
        self.box.bind(minimum_height=self.box.setter("height"))
        self.scroll.add_widget(self.box)
        self.add_widget(self.scroll)

    def display_result(self, data):
        self.box.clear_widgets()
        self.box.add_widget(make_label("Hasil Skrining Offline (4/4)", 18, True, "#0369A1", "center"))

        pct = round(data["score"] * 100, 1)
        is_high = data["score"] >= 0.50

        card_res = PastelCard("#FFF1F2" if is_high else "#ECFDF5")
        status_txt = "TERDETEKSI INDIKASI GEJALA BRONKITIS" if is_high else "TIDAK TERDETEKSI GEJALA BRONKITIS"
        status_col = "#BE123C" if is_high else "#047857"

        card_res.add_widget(make_label(status_txt, 17, True, status_col, "center"))
        card_res.add_widget(make_label(f"Skor Risiko AI: {pct}% (Waktu proses: {data['time']} dtk)", 15, True, "#1E293B", "center"))
        card_res.add_widget(
            make_label(
                "Catatan: Hasil ini merupakan skrining awal berbasis AI offline, bukan diagnosis medis final. Segera konsultasikan ke dokter/Puskesmas.",
                13,
                False,
                "#475569",
            )
        )
        self.box.add_widget(card_res)

        card_info = PastelCard("#F0F9FF")
        info_str = (
            f"Waktu: {data['timestamp']}\n"
            f"Nama Anak: {data['child_name']} ({data['child_age']} bln)\n"
            f"Orang Tua: {data['parent_name']}\n"
            f"Wilayah: {data['region']} ({data['village']})\n"
            f"Napas Ngik-ngik: {data['wheezing']} | Demam >3 hari: {data['fever']}"
        )
        card_info.add_widget(make_label("Ringkasan Data Pasien", 15, True, "#0369A1"))
        card_info.add_widget(make_label(info_str, 14))
        self.box.add_widget(card_info)

        self.latest_report_text = f"=== LAPORAN SKRINING RESPIKIDD (OFFLINE) ===\n{info_str}\nHasil AI: {status_txt} ({pct}%)\n"

        region_key = data["region"] if data["region"] in OFFLINE_FASKES_DB else "Layanan Darurat Nasional (24 Jam)"
        faskes_list = OFFLINE_FASKES_DB.get(region_key, []) + OFFLINE_FASKES_DB["Layanan Darurat Nasional (24 Jam)"]

        card_faskes = PastelCard("#FEFCE8")
        card_faskes.add_widget(make_label(f"Kontak Faskes Offline ({region_key})", 15, True, "#9A3412"))
        for nama, telp, ket in faskes_list[:5]:
            card_faskes.add_widget(make_label(f"• {nama}\n  Telp: {telp} ({ket})", 13, False, "#334155"))
        self.box.add_widget(card_faskes)

        self.box.add_widget(make_button("Simpan Laporan Rekam Medis (.txt)", "#10B981", self.save_report))
        self.box.add_widget(make_button("Periksa Anak Lainnya", "#0284C7", lambda x: setattr(self.manager, "current", "input")))

    def save_report(self, instance):
        save_dir = "/sdcard/Download" if os.path.exists("/sdcard/Download") else App.get_running_app().user_data_dir
        fname = f"Laporan_RespiKidd_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        full_path = os.path.join(save_dir, fname)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(self.latest_report_text)
        instance.text = f"Tersimpan di: {fname}"


class RespiKiddApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(WelcomeScreen(name="welcome"))
        sm.add_widget(GuideScreen(name="guide"))
        sm.add_widget(InputScreen(name="input"))
        sm.add_widget(ResultScreen(name="result"))
        return sm


if __name__ == "__main__":
    RespiKiddApp().run()
