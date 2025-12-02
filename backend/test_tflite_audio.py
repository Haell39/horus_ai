#!/usr/bin/env python3
"""
Testa modelo TFLite original para comparar com Keras fixed.
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
    '../validade_model_audio/falha_audio_eco_04.mp4',
    '../validade_model_audio/sinalerro1.mp4',
    '../validade_model_audio/sinalerro2.mp4',
    '../validade_model_audio/falha_audio_mudo_jornal_07.mp4',
]

print("=" * 70)
print("TESTE DO MODELO TFLite ORIGINAL")
print("=" * 70)

# Carrega TFLite
tflite_path = 'app/ml/models/audio/heimdall_audio_model_ultra_v1/audio_model.tflite'
print(f"\n[1] Carregando TFLite: {tflite_path}")

interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print(f"    Input shape: {input_details['shape']}")
print(f"    Output shape: {output_details['shape']}")

print("\n[2] Testando arquivos:")
print("-" * 70)

for test_file in test_files:
    if not os.path.exists(test_file):
        print(f"    Arquivo não encontrado: {test_file}")
        continue
    
    y, sr = librosa.load(test_file, sr=22050, mono=True)
    input_tensor = preprocess(y)
    
    interpreter.set_tensor(input_details['index'], input_tensor)
    interpreter.invoke()
    probs = interpreter.get_tensor(output_details['index'])[0]
    
    pred_idx = np.argmax(probs)
    pred = labels[pred_idx]
    conf = probs[pred_idx]
    
    # Mostra top 3
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [f"{labels[i]}:{probs[i]:.2%}" for i in top3_idx]
    
    print(f"  {os.path.basename(test_file):35} -> {pred:15} ({conf:.2%})")
    print(f"    Top3: {', '.join(top3)}")

print("\n" + "=" * 70)
print("COMPARAÇÃO COM KERAS FIXED")
print("=" * 70)

# Carrega Keras fixed
keras_model = tf.keras.models.load_model('app/ml/models/audio/heimdall_audio_model_ultra_v1/audio_model_fixed.keras')

print("\n[3] Comparando predições TFLite vs Keras Fixed:")
print("-" * 70)

for test_file in test_files[:3]:  # Apenas 3 para ser breve
    if not os.path.exists(test_file):
        continue
    
    y, sr = librosa.load(test_file, sr=22050, mono=True)
    input_tensor = preprocess(y)
    
    # TFLite
    interpreter.set_tensor(input_details['index'], input_tensor)
    interpreter.invoke()
    tflite_probs = interpreter.get_tensor(output_details['index'])[0]
    
    # Keras
    keras_probs = keras_model.predict(input_tensor, verbose=0)[0]
    
    print(f"\n  {os.path.basename(test_file)}:")
    print(f"    TFLite: {labels[np.argmax(tflite_probs)]} ({tflite_probs[np.argmax(tflite_probs)]:.2%})")
    print(f"    Keras:  {labels[np.argmax(keras_probs)]} ({keras_probs[np.argmax(keras_probs)]:.2%})")
    
    # Diferença máxima
    diff = np.abs(tflite_probs - keras_probs).max()
    print(f"    Max prob diff: {diff:.4f}")
