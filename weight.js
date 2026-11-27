import os
import json
import numpy as np
import tensorflow as tf

# Pastikan berada di dalam folder respikidd-app
if not os.path.exists("respikidd_model.keras"):
    if os.path.exists("respikidd-app/respikidd_model.keras"):
        %cd respikidd-app
    else:
        !git clone https://github.com/elits-cmd/respikidd-app.git
        %cd respikidd-app

# Muat model Keras dan ekspor langsung ke weights.js
model = tf.keras.models.load_model("respikidd_model.keras")
weights_json = {}
for i, layer in enumerate(model.layers):
    for j, arr in enumerate(layer.get_weights()):
        weights_json[f"layer_{i}_{layer.__class__.__name__}_w{j}"] = np.round(arr, 6).tolist()

with open("weights.js", "w", encoding="utf-8") as f:
    f.write("const RESPIKIDD_WEIGHTS = " + json.dumps(weights_json) + ";\n")

print("Berhasil! Ukuran weights.js:", round(os.path.getsize("weights.js") / 1024, 1), "KB")

# Otomatis memicu download file weights.js ke laptop kamu
from google.colab import files
files.download("weights.js")
