#!/usr/bin/env python3
"""
Inference script for Heimdall Audio Model Ultra V1.
Compatible with Horus AI integration.
"""
import argparse
import json
import numpy as np
import librosa
import cv2
import tensorflow as tf
from pathlib import Path

# ============ CONFIG ============
# These values match the training of Ultra V1
SAMPLE_RATE = 22050
DURATION = 3.0
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMAX = 8000
IMG_SIZE = 128

# Normalization params (Fixed Reference)
REF_DB = 1.0
MIN_DB = -80.0
MAX_DB = 80.0

CLASSES = ['ausencia_audio', 'eco_reverb', 'ruido_hiss', 'sinal_teste', 'normal']

def load_model(model_path):
    """Load the Keras model."""
    return tf.keras.models.load_model(model_path)

def preprocess_audio_segment(y):
    """
    Preprocess a 3s audio segment into the model input format.
    Input: np.array of shape (N,) (float32)
    Output: np.array of shape (1, 128, 128, 1) (float32)
    """
    # 1. Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=FMAX
    )
    
    # 2. Power to DB (Fixed Reference)
    mel_db = librosa.power_to_db(mel, ref=REF_DB)
    
    # 3. Fixed Normalization [-80, 80] -> [0, 1]
    mel_db = np.clip(mel_db, MIN_DB, MAX_DB)
    mel_norm = (mel_db - MIN_DB) / (MAX_DB - MIN_DB)
    
    # 4. Resize to 128x128
    mel_resized = cv2.resize(mel_norm, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    
    # 5. Add batch and channel dims
    return mel_resized.astype(np.float32)[np.newaxis, ..., np.newaxis]

def run_inference(model, audio_path):
    """
    Run inference on a full audio file.
    Returns a list of predictions per 3s segment.
    """
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    
    seg_len = int(DURATION * SAMPLE_RATE)
    results = []
    
    for i in range(0, len(y), seg_len):
        segment = y[i:i+seg_len]
        if len(segment) < seg_len:
            segment = np.pad(segment, (0, seg_len - len(segment)))
            
        input_tensor = preprocess_audio_segment(segment)
        probs = model.predict(input_tensor, verbose=0)[0]
        pred_idx = np.argmax(probs)
        
        results.append({
            "start": i / SAMPLE_RATE,
            "end": (i + seg_len) / SAMPLE_RATE,
            "class": CLASSES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {k: float(v) for k, v in zip(CLASSES, probs)}
        })
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", help="Path to wav/mp3 file")
    parser.add_argument("--model", default="audio_model.keras", help="Path to model file")
    args = parser.parse_args()
    
    model = load_model(args.model)
    predictions = run_inference(model, args.audio_file)
    
    print(json.dumps(predictions, indent=2))
