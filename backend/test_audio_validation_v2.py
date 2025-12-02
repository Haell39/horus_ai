#!/usr/bin/env python3
"""
Script para validar o modelo de áudio usando os TIMESTAMPS do CSV.
Testa o modelo Heimdall Audio Ultra V1 apenas nos trechos marcados.
"""

import os
import sys
import csv
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import librosa
import warnings
warnings.filterwarnings('ignore')

# === Configurações do modelo Heimdall Audio Ultra V1 ===
SAMPLE_RATE = 22050
DURATION = 3.0
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMAX = 8000
IMG_SIZE = 128
REF_DB = 1.0
MIN_DB = -80.0
MAX_DB = 80.0

MODEL_LABELS = ['ausencia_audio', 'eco_reverb', 'ruido_hiss', 'sinal_teste', 'normal']

LABEL_MAP = {
    'ausencia_audio': 'ausencia_audio',
    'eco': 'eco_reverb',
    'hiss': 'ruido_hiss',
    'sinal_erro': 'sinal_teste',
    'normal': 'normal',
    'eco_reverb': 'eco_reverb',
    'ruido_hiss': 'ruido_hiss',
    'sinal_teste': 'sinal_teste',
}


def parse_timestamp(ts_str):
    """Converte HH:MM:SS para segundos."""
    parts = ts_str.split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(ts_str)


def preprocess_audio_segment(y, sr=SAMPLE_RATE):
    """Preprocessa um segmento de áudio para o modelo."""
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    
    target_len = int(DURATION * SAMPLE_RATE)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        y = y[:target_len]
    
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=FMAX
    )
    mel_db = librosa.power_to_db(mel, ref=REF_DB)
    mel_db = np.clip(mel_db, MIN_DB, MAX_DB)
    mel_norm = (mel_db - MIN_DB) / (MAX_DB - MIN_DB)
    mel_resized = cv2.resize(mel_norm, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return mel_resized.astype(np.float32)[np.newaxis, ..., np.newaxis]


def run_inference_on_segment(model, y_full, sr, t_start, t_end):
    """
    Roda inferência apenas no trecho especificado.
    Retorna a classe com maior confiança.
    """
    start_sample = int(t_start * sr)
    end_sample = int(t_end * sr)
    
    # Garante que não ultrapassa o tamanho do áudio
    end_sample = min(end_sample, len(y_full))
    
    segment = y_full[start_sample:end_sample]
    
    if len(segment) == 0:
        return None, 0.0, {}
    
    seg_len = int(DURATION * sr)
    all_probs = []
    
    # Processa em segmentos de 3s dentro do trecho
    for i in range(0, len(segment), seg_len):
        sub_segment = segment[i:i+seg_len]
        if len(sub_segment) < seg_len:
            sub_segment = np.pad(sub_segment, (0, seg_len - len(sub_segment)))
        
        input_tensor = preprocess_audio_segment(sub_segment, sr)
        probs = model.predict(input_tensor, verbose=0)[0]
        all_probs.append(probs)
    
    if not all_probs:
        return None, 0.0, {}
    
    # Estratégia: pega a MAIOR confiança de qualquer classe não-normal
    # dentro do intervalo (ao invés de média)
    best_fault_class = 'normal'
    max_fault_conf = 0.0
    
    for probs in all_probs:
        for i, label in enumerate(MODEL_LABELS):
            if label != 'normal' and probs[i] > max_fault_conf:
                max_fault_conf = probs[i]
                best_fault_class = label
    
    # Se nenhuma falha foi detectada com confiança razoável, retorna normal
    avg_probs = np.mean(all_probs, axis=0)
    if max_fault_conf < 0.3:  # threshold baixo para detecção
        pred_idx = np.argmax(avg_probs)
        best_fault_class = MODEL_LABELS[pred_idx]
        max_fault_conf = float(avg_probs[pred_idx])
    
    prob_dict = {MODEL_LABELS[i]: float(avg_probs[i]) for i in range(len(MODEL_LABELS))}
    
    return best_fault_class, max_fault_conf, prob_dict


def main():
    print("=" * 70)
    print("VALIDAÇÃO DO MODELO DE ÁUDIO (COM TIMESTAMPS)")
    print("=" * 70)
    
    # Carrega o modelo
    model_path = os.path.join(
        os.path.dirname(__file__),
        'app', 'ml', 'models', 'audio', 'heimdall_audio_model_ultra_v1', 'audio_model_fixed.keras'
    )
    
    print(f"\n[1] Carregando modelo: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"    ✓ Modelo carregado. Output shape: {model.output_shape}")
    print(f"    ✓ Labels do modelo: {MODEL_LABELS}")
    
    # Carrega dados de validação
    validation_dir = os.path.join(os.path.dirname(__file__), '..', 'validade_model_audio')
    csv_path = os.path.join(validation_dir, 'timestamp.csv')
    
    print(f"\n[2] Carregando dados de validação: {csv_path}")
    
    # Estrutura do CSV
    print("\n[2.1] Estrutura do CSV:")
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:5]
        for line in lines:
            print(f"    {line.strip()}")
    
    # Processa cada arquivo
    print(f"\n[3] Executando inferência nos trechos especificados...")
    print("-" * 70)
    
    results = {'correct': 0, 'incorrect': 0, 'errors': 0, 'by_class': {}}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get('video_filename')
            t_start_str = row.get('t_start')
            t_end_str = row.get('t_end')
            label_orig = row.get('class')
            
            if not all([filename, t_start_str, t_end_str, label_orig]):
                continue
            
            expected_label = LABEL_MAP.get(label_orig.lower().strip(), label_orig.lower().strip())
            
            # Constrói caminho do arquivo
            filepath = os.path.join(validation_dir, filename)
            if not os.path.exists(filepath):
                filepath = os.path.join(validation_dir, filename + '.mp4')
            
            if not os.path.exists(filepath):
                print(f"  ⚠ Arquivo não encontrado: {filename}")
                results['errors'] += 1
                continue
            
            # Parse timestamps
            t_start = parse_timestamp(t_start_str)
            t_end = parse_timestamp(t_end_str)
            
            # Carrega áudio
            try:
                y_full, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
            except Exception as e:
                print(f"  ⚠ Erro ao carregar: {filename} - {e}")
                results['errors'] += 1
                continue
            
            # Roda inferência no trecho
            pred_class, confidence, probs = run_inference_on_segment(model, y_full, sr, t_start, t_end)
            
            if pred_class is None:
                print(f"  ⚠ Erro na inferência: {filename}")
                results['errors'] += 1
                continue
            
            is_correct = (pred_class == expected_label)
            
            if expected_label not in results['by_class']:
                results['by_class'][expected_label] = {'correct': 0, 'total': 0}
            results['by_class'][expected_label]['total'] += 1
            
            if is_correct:
                results['correct'] += 1
                results['by_class'][expected_label]['correct'] += 1
                status = "✓"
            else:
                results['incorrect'] += 1
                status = "✗"
            
            print(f"\n  {status} {filename} [{t_start_str} - {t_end_str}]")
            print(f"    Esperado: {expected_label} (original: {label_orig})")
            print(f"    Predito:  {pred_class} ({confidence:.2%})")
            top3 = sorted(probs.items(), key=lambda x: -x[1])[:3]
            print(f"    Top3:     {', '.join([f'{k}:{v:.2%}' for k,v in top3])}")
    
    # Adiciona testes para clips normais
    print("\n[4] Testando clips normais...")
    print("-" * 70)
    
    normal_clips_dir = os.path.join(validation_dir, 'normal_clips')
    if os.path.exists(normal_clips_dir):
        if 'normal' not in results['by_class']:
            results['by_class']['normal'] = {'correct': 0, 'total': 0}
        
        for f in os.listdir(normal_clips_dir):
            if not f.endswith(('.mp4', '.wav', '.mp3')):
                continue
            
            filepath = os.path.join(normal_clips_dir, f)
            
            try:
                y_full, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
            except Exception as e:
                results['errors'] += 1
                continue
            
            # Para clips normais, analisa o arquivo inteiro
            seg_len = int(DURATION * SAMPLE_RATE)
            all_probs = []
            
            for i in range(0, len(y_full), seg_len):
                segment = y_full[i:i+seg_len]
                if len(segment) < seg_len:
                    segment = np.pad(segment, (0, seg_len - len(segment)))
                input_tensor = preprocess_audio_segment(segment, sr)
                probs_arr = model.predict(input_tensor, verbose=0)[0]
                all_probs.append(probs_arr)
            
            if all_probs:
                avg_probs = np.mean(all_probs, axis=0)
                pred_idx = np.argmax(avg_probs)
                pred_class = MODEL_LABELS[pred_idx]
                confidence = float(avg_probs[pred_idx])
                
                results['by_class']['normal']['total'] += 1
                is_correct = (pred_class == 'normal')
                
                if is_correct:
                    results['correct'] += 1
                    results['by_class']['normal']['correct'] += 1
                    status = "✓"
                else:
                    results['incorrect'] += 1
                    status = "✗"
                
                print(f"  {status} {f}: {pred_class} ({confidence:.2%})")
    
    # Resumo
    total = results['correct'] + results['incorrect']
    accuracy = results['correct'] / total if total > 0 else 0
    
    print("\n" + "=" * 70)
    print("RESUMO")
    print("=" * 70)
    print(f"  Total testado:  {total}")
    print(f"  Corretos:       {results['correct']}")
    print(f"  Incorretos:     {results['incorrect']}")
    print(f"  Erros:          {results['errors']}")
    print(f"  ACCURACY:       {accuracy:.2%}")
    
    print("\n  Por classe:")
    for cls, data in results['by_class'].items():
        cls_acc = data['correct'] / data['total'] if data['total'] > 0 else 0
        print(f"    {cls}: {data['correct']}/{data['total']} ({cls_acc:.2%})")


if __name__ == '__main__':
    main()
