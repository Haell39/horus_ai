"""Validate an audio Keras model for Horus.

This script locates the `model_audio_keras` package in the repo, loads
`models/audio_model_finetune.keras` and its metadata, runs a 1 kHz test tone
through the model (using the provided `preprocess.py`) and prints results.

Usage (from repo root):
  python backend/app/ml/validate_audio_model_for_horus.py

Notes:
- This is a safe, read-only validator (does not copy files by default).
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

# locate model_audio_keras folder by walking up to repo root
HERE = Path(__file__).resolve()
model_pkg = None
for up in range(6):
    candidate = HERE.parents[up] / "model_audio_keras"
    if candidate.exists():
        model_pkg = candidate
        break

if model_pkg is None:
    print("ERROR: could not find 'model_audio_keras' directory in repo tree.")
    sys.exit(2)

# ensure we can import the provided preprocess utilities
sys.path.insert(0, str(model_pkg))
try:
    import preprocess
except Exception as e:
    print("ERROR: failed to import preprocess from model_audio_keras:", e)
    sys.exit(3)

# paths
models_dir = model_pkg / "models"
model_path = models_dir / "audio_model_finetune.keras"
metadata_path = models_dir / "audio_model_finetune.metadata.json"
labels_path = models_dir / "training_files" / "labels.csv"

if not model_path.exists():
    print(f"ERROR: model file not found at {model_path}")
    sys.exit(4)
if not metadata_path.exists():
    print(f"ERROR: metadata file not found at {metadata_path}")
    sys.exit(5)
if not labels_path.exists():
    print(f"WARNING: labels file not found at {labels_path} (will still try to run)")

# load metadata
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Loaded metadata:")
print(json.dumps(metadata, indent=2, ensure_ascii=False))

# load model
print("Loading Keras model (this may take a few seconds)...")
try:
    model = tf.keras.models.load_model(str(model_path))
except Exception as e:
    print("ERROR: failed to load model:", e)
    sys.exit(6)

print("Model loaded. Model summary:")
try:
    model.summary()
except Exception:
    print("(model.summary() failed to print — likely fine)")

# check input shape from metadata vs model
meta_input_shape = metadata.get("input_shape")
print("Metadata input_shape:", meta_input_shape)
try:
    print("Model input shape:", model.input_shape)
except Exception:
    pass

# load labels if present
labels = []
if labels_path.exists():
    import csv
    with open(labels_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if r]
    if rows:
        if rows[0][0].lower() == "class":
            labels = [r[0] for r in rows[1:]]
        else:
            labels = [r[0] for r in rows]

if labels:
    print("Loaded labels:", labels)
else:
    print("No labels loaded (labels.csv missing or empty). Results will print indices.")

# generate a 1kHz test tone according to metadata sample_rate and duration
sr = int(metadata.get("sample_rate", 16000))
duration = float(metadata.get("segment_duration_s", preprocess.SEGMENT_DURATION))
print(f"Generating 1 kHz tone: sr={sr}, duration={duration}s")

t = np.linspace(0, duration, int(sr * duration), endpoint=False)
# 0.5 amplitude to avoid clipping in float32
y = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)

# preprocess to model input using provided util
try:
    x = preprocess.audio_segment_to_input(y, sr=sr)
except Exception as e:
    print("ERROR: preprocess.audio_segment_to_input failed:", e)
    sys.exit(7)

print("Prepared input shape:", x.shape, "dtype=", x.dtype)

# run inference
try:
    preds = model(x, training=False)
except Exception as e:
    print("ERROR: model inference failed:", e)
    sys.exit(8)

import tensorflow as _tf
probs = _tf.nn.softmax(preds, axis=-1).numpy()[0]

# print top-3
top_k = min(3, probs.shape[0])
idxs = probs.argsort()[::-1][:top_k]
print("Top predictions (softmax):")
for i in idxs:
    label = labels[i] if i < len(labels) else str(i)
    print(f" - {label}: {probs[i]:.4f}")

print("Validation finished — model is loadable and runs a sample inference.")
print("Next steps: copy the `.keras`, `labels.csv` and metadata into `backend/app/ml/models/` and restart the backend. If you want, I can automate the copy.")

sys.exit(0)
