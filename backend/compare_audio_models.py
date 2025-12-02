#!/usr/bin/env python3
"""
Compara modelo original vs modelo fixed.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import librosa
import cv2
import warnings
warnings.filterwarnings('ignore')

def preprocess(y, sr=22050):
    target_len = int(3.0 * 22050)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        y = y[:target_len]
    
    mel = librosa.feature.melspectrogram(y=y, sr=22050, n_mels=128, n_fft=2048, hop_length=512, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=1.0)
    mel_db = np.clip(mel_db, -80.0, 80.0)
    mel_norm = (mel_db - (-80.0)) / (80.0 - (-80.0))
    mel_resized = cv2.resize(mel_norm, (128, 128))
    return mel_resized.astype(np.float32)[np.newaxis, ..., np.newaxis]

labels = ['ausencia_audio', 'eco_reverb', 'ruido_hiss', 'sinal_teste', 'normal']

# Arquivos de teste
test_files = [
    '../validade_model_audio/hiss1.mp4',
    '../validade_model_audio/hiss2.mp4', 
    '../validade_model_audio/falha_audio_eco_02.mp4',
    '../validade_model_audio/sinalerro1.mp4',
]

print("=" * 70)
print("COMPARAÇÃO: Modelo Original vs Modelo Fixed")
print("=" * 70)

# Testa modelo ORIGINAL
print("\n[1] Modelo ORIGINAL (audio_model.keras):")
try:
    model_orig = tf.keras.models.load_model('app/ml/models/audio/heimdall_audio_model_ultra_v1/audio_model.keras')
    print(f"    ✓ Carregado. Output shape: {model_orig.output_shape}")
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"    Arquivo não encontrado: {test_file}")
            continue
        y, sr = librosa.load(test_file, sr=22050, mono=True)
        input_tensor = preprocess(y)
        probs = model_orig.predict(input_tensor, verbose=0)[0]
        pred = labels[np.argmax(probs)]
        conf = probs[np.argmax(probs)]
        print(f"    {os.path.basename(test_file)}: {pred} ({conf:.2%})")
except Exception as e:
    print(f"    ERRO ao carregar modelo original: {e}")

# Testa modelo FIXED
print("\n[2] Modelo FIXED (audio_model_fixed.keras):")
try:
    model_fixed = tf.keras.models.load_model('app/ml/models/audio/heimdall_audio_model_ultra_v1/audio_model_fixed.keras')
    print(f"    ✓ Carregado. Output shape: {model_fixed.output_shape}")
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            continue
        y, sr = librosa.load(test_file, sr=22050, mono=True)
        input_tensor = preprocess(y)
        probs = model_fixed.predict(input_tensor, verbose=0)[0]
        pred = labels[np.argmax(probs)]
        conf = probs[np.argmax(probs)]
        print(f"    {os.path.basename(test_file)}: {pred} ({conf:.2%})")
except Exception as e:
    print(f"    ERRO ao carregar modelo fixed: {e}")

# Compara pesos
print("\n[3] Comparando pesos dos modelos:")
try:
    if 'model_orig' in dir() and 'model_fixed' in dir():
        orig_weights = model_orig.get_weights()
        fixed_weights = model_fixed.get_weights()
        
        print(f"    Número de arrays de pesos - Original: {len(orig_weights)}, Fixed: {len(fixed_weights)}")
        
        if len(orig_weights) == len(fixed_weights):
            all_equal = True
            for i, (w1, w2) in enumerate(zip(orig_weights, fixed_weights)):
                if w1.shape != w2.shape:
                    print(f"    Layer {i}: Shape diferente! {w1.shape} vs {w2.shape}")
                    all_equal = False
                elif not np.allclose(w1, w2, rtol=1e-5, atol=1e-5):
                    diff = np.abs(w1 - w2).max()
                    print(f"    Layer {i}: Valores diferentes! Max diff: {diff:.2e}")
                    all_equal = False
            
            if all_equal:
                print("    ✓ TODOS os pesos são IDÊNTICOS!")
            else:
                print("    ✗ Alguns pesos são diferentes.")
        else:
            print("    ✗ Número diferente de layers!")
except Exception as e:
    print(f"    ERRO na comparação: {e}")
