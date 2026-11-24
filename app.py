import os
import numpy as np
import librosa
import noisereduce as nr
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ==============================================================================
# 1. KONFIGURASI FOLDER DATASET
# Pastikan struktur folder datasetmu seperti ini:
# dataset/
#   ├── bronkitis/       (isi file .wav / .mp3 batuk bronkitis)
#   └── non_bronkitis/   (isi file .wav / .mp3 batuk sehat / non-bronkitis)
# ==============================================================================
DATASET_DIR = "dataset"  # Sesuaikan dengan path folder datasetmu di Colab/Drive
CLASSES = ["bronkitis", "non_bronkitis"]  # 0 = bronkitis, 1 = non_bronkitis

def audio_to_mel(y, sr=22050):
    """Fungsi ekstraksi Mel-Spectrogram yang IDENTIK dengan app.py"""
    y_trim, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trim) > int(0.3 * sr):
        y = y_trim
    y_norm = librosa.util.normalize(y)
    
    mel_spec = librosa.feature.melspectrogram(y=y_norm, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_norm = np.clip((mel_spec_db + 80.0) / 80.0, 0.0, 1.0)
    
    mel_resized = tf.image.resize(mel_norm[..., np.newaxis], (128, 128)).numpy()
    return mel_resized

def augment_audio(y, sr=22050):
    """Menghasilkan variasi audio agar dataset lebih kaya & tahan banting"""
    augmented = [y]
    # 1. Tambah sedikit noise latar
    noise = y + 0.005 * np.random.randn(len(y))
    augmented.append(noise)
    # 2. Geser nada (pitch shift) naik & turun
    augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=1.5))
    augmented.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=-1.5))
    # 3. Ubah tempo sedikit (time stretch)
    augmented.append(librosa.effects.time_stretch(y, rate=0.9))
    augmented.append(librosa.effects.time_stretch(y, rate=1.1))
    return augmented

# ==============================================================================
# 2. LOAD & EKSTRAKSI FITUR DATASET
# ==============================================================================
X_raw, y_raw = [], []

for label_idx, class_name in enumerate(CLASSES):
    folder_path = os.path.join(DATASET_DIR, class_name)
    for fname in os.listdir(folder_path):
        if fname.lower().endswith((".wav", ".mp3", ".ogg")):
            fpath = os.path.join(folder_path, fname)
            try:
                audio, sr = librosa.load(fpath, sr=22050, duration=3.0)
                audio = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.75, stationary=True)
                X_raw.append(audio)
                y_raw.append(label_idx)
            except Exception as e:
                print(f"Gagal memuat {fname}: {e}")

# Bagi data latih & validasi SEBELUM augmentasi (agar tidak terjadi data leakage)
X_train_aud, X_val_aud, y_train_raw, y_val_raw = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

X_train, y_train = [], []
for audio, label in zip(X_train_aud, y_train_raw):
    for aug_aud in augment_audio(audio):
        X_train.append(audio_to_mel(aug_aud))
        y_train.append(label)

X_val, y_val = [], []
for audio, label in zip(X_val_aud, y_val_raw):
    X_val.append(audio_to_mel(audio))
    y_val.append(label)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
X_val = np.array(X_val, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)

print(f"Total data latih (setelah augmentasi): {len(X_train)}, Validasi: {len(X_val)}")

# Hitung bobot kelas otomatis agar seimbang
weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = {0: weights[0], 1: weights[1]}
print("Class weights:", class_weights)

# ==============================================================================
# 3. ARSITEKTUR TRANSFER LEARNING (MobileNetV2)
# Tetap menerima input (128, 128, 1) agar langsung cocok dengan app.py
# ==============================================================================
inputs = tf.keras.Input(shape=(128, 128, 1))
# Ubah 1 channel grayscale menjadi 3 channel RGB untuk MobileNetV2
x = tf.keras.layers.Concatenate()([inputs, inputs, inputs])

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),
    include_top=False,
    weights="imagenet"
)
# Buka 30 layer terakhir agar menyesuaikan ciri spektrogram batuk
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Recall(name="sensitivity")]
)

# ==============================================================================
# 4. TRAINING & SIMPAN MODEL BARU
# ==============================================================================
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=16,
    class_weight=class_weights,
    callbacks=callbacks
)

model.save("respikidd_model.keras")
print("Model baru berhasil disimpan sebagai respikidd_model.keras!")
