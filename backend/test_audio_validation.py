#!/usr/bin/env python3
"""
Script para validar o modelo de áudio contra os clips de validação.
Testa o modelo Heimdall Audio Ultra V1 com os arquivos de validação.
"""

import os
import sys
import csv
import numpy as np
import cv2

# Adiciona o diretório pai ao path para importar módulos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suprime logs do TF

import tensorflow as tf
import librosa

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

# Labels conforme treinados no modelo
MODEL_LABELS = ['ausencia_audio', 'eco_reverb', 'ruido_hiss', 'sinal_teste', 'normal']

# Mapeamento dos labels do CSV de validação para os labels do modelo
LABEL_MAP = {
    'ausencia_audio': 'ausencia_audio',
    'eco': 'eco_reverb',
    'hiss': 'ruido_hiss',
    'sinal_erro': 'sinal_teste',
    'normal': 'normal',
    # Variantes possíveis
    'eco_reverb': 'eco_reverb',
    'ruido_hiss': 'ruido_hiss',
    'sinal_teste': 'sinal_teste',
}


def preprocess_audio_segment(y, sr=SAMPLE_RATE):
    """
    Preprocessa um segmento de áudio para o modelo.
    """
    # Resample se necessário
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    
    # Garante comprimento correto (3.0s)
    target_len = int(DURATION * SAMPLE_RATE)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        y = y[:target_len]
    
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
    
    # 5. Add batch and channel dims -> (1, 128, 128, 1)
    return mel_resized.astype(np.float32)[np.newaxis, ..., np.newaxis]


def run_inference(model, audio_path):
    """
    Roda inferência em um arquivo de áudio completo.
    Retorna a classe com maior confiança média.
    """
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"  ERRO ao carregar áudio: {e}")
        return None, 0.0, {}
    
    if len(y) == 0:
        return 'ausencia_audio', 1.0, {'ausencia_audio': 1.0}
    
    seg_len = int(DURATION * SAMPLE_RATE)
    all_probs = []
    
    # Processa em segmentos de 3s
    for i in range(0, len(y), seg_len):
        segment = y[i:i+seg_len]
        if len(segment) < seg_len:
            segment = np.pad(segment, (0, seg_len - len(segment)))
        
        input_tensor = preprocess_audio_segment(segment)
        probs = model.predict(input_tensor, verbose=0)[0]
        all_probs.append(probs)
    
    if not all_probs:
        return None, 0.0, {}
    
    # Média das probabilidades de todos os segmentos
    avg_probs = np.mean(all_probs, axis=0)
    pred_idx = np.argmax(avg_probs)
    pred_class = MODEL_LABELS[pred_idx]
    confidence = float(avg_probs[pred_idx])
    
    prob_dict = {MODEL_LABELS[i]: float(avg_probs[i]) for i in range(len(MODEL_LABELS))}
    
    return pred_class, confidence, prob_dict


def load_validation_data(csv_path, clips_dir):
    """
    Carrega os dados de validação do CSV.
    Retorna lista de (arquivo, label_esperado).
    """
    data = []
    # A pasta com os arquivos de falha é a pasta pai do CSV
    base_dir = os.path.dirname(csv_path)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Suporta diferentes nomes de coluna
            filename = row.get('video_filename') or row.get('filename') or row.get('file') or row.get('clip')
            label = row.get('class') or row.get('label') or row.get('tipo')
            
            if not filename or not label:
                continue
            
            # Normaliza o label usando o mapa
            label_normalized = LABEL_MAP.get(label.lower().strip(), label.lower().strip())
            
            # Constrói o caminho completo - tenta na pasta base primeiro
            filepath = os.path.join(base_dir, filename)
            
            if not os.path.exists(filepath):
                # Tenta com diferentes extensões
                for ext in ['.mp4', '.wav', '.mp3', '']:
                    test_path = os.path.join(base_dir, filename + ext)
                    if os.path.exists(test_path):
                        filepath = test_path
                        break
            
            data.append((filepath, label_normalized, label))  # (path, label_normalizado, label_original)
    
    # Adiciona clips "normal" da pasta normal_clips
    normal_clips_dir = os.path.join(base_dir, 'normal_clips')
    if os.path.exists(normal_clips_dir):
        for f in os.listdir(normal_clips_dir):
            if f.endswith(('.mp4', '.wav', '.mp3')):
                filepath = os.path.join(normal_clips_dir, f)
                data.append((filepath, 'normal', 'normal'))
    
    return data


def main():
    print("=" * 70)
    print("VALIDAÇÃO DO MODELO DE ÁUDIO - Heimdall Audio Ultra V1")
    print("=" * 70)
    
    # Carrega o modelo
    model_path = os.path.join(
        os.path.dirname(__file__),
        'app', 'ml', 'models', 'audio', 'heimdall_audio_model_ultra_v1', 'audio_model_fixed.keras'
    )
    
    print(f"\n[1] Carregando modelo: {model_path}")
    if not os.path.exists(model_path):
        print(f"ERRO: Modelo não encontrado em {model_path}")
        return
    
    model = tf.keras.models.load_model(model_path)
    print(f"    ✓ Modelo carregado. Output shape: {model.output_shape}")
    print(f"    ✓ Labels do modelo: {MODEL_LABELS}")
    
    # Carrega dados de validação
    validation_dir = os.path.join(os.path.dirname(__file__), '..', 'validade_model_audio')
    csv_path = os.path.join(validation_dir, 'timestamp.csv')
    
    # Descobre onde estão os clips
    clips_dirs = [
        os.path.join(validation_dir, 'normal_clips'),
        validation_dir,
    ]
    
    print(f"\n[2] Carregando dados de validação: {csv_path}")
    
    if not os.path.exists(csv_path):
        print(f"ERRO: CSV não encontrado em {csv_path}")
        # Lista arquivos disponíveis
        print(f"\nArquivos em {validation_dir}:")
        for f in os.listdir(validation_dir):
            print(f"  - {f}")
        return
    
    # Primeiro, vamos ver a estrutura do CSV
    print("\n[2.1] Estrutura do CSV:")
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[:5]
        for line in lines:
            print(f"    {line.strip()}")
    
    # Carrega os dados
    clips_dir = clips_dirs[0] if os.path.exists(clips_dirs[0]) else validation_dir
    validation_data = load_validation_data(csv_path, clips_dir)
    
    print(f"\n    ✓ {len(validation_data)} arquivos encontrados no CSV")
    
    # Lista arquivos reais na pasta
    print(f"\n[2.2] Arquivos na pasta de clips ({clips_dir}):")
    if os.path.exists(clips_dir):
        files = os.listdir(clips_dir)[:10]
        for f in files:
            print(f"    - {f}")
        if len(os.listdir(clips_dir)) > 10:
            print(f"    ... e mais {len(os.listdir(clips_dir)) - 10} arquivos")
    
    # Roda validação
    print(f"\n[3] Executando inferência...")
    print("-" * 70)
    
    results = {
        'correct': 0,
        'incorrect': 0,
        'errors': 0,
        'by_class': {}
    }
    
    for filepath, expected_label, original_label in validation_data:
        filename = os.path.basename(filepath)
        
        if not os.path.exists(filepath):
            print(f"  ⚠ Arquivo não encontrado: {filename}")
            results['errors'] += 1
            continue
        
        pred_class, confidence, probs = run_inference(model, filepath)
        
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
        
        # Mostra detalhes para erros ou primeiros resultados
        if not is_correct or results['correct'] + results['incorrect'] <= 5:
            print(f"\n  {status} {filename}")
            print(f"    Esperado: {expected_label} (original: {original_label})")
            print(f"    Predito:  {pred_class} ({confidence:.2%})")
            print(f"    Probs:    {', '.join([f'{k}:{v:.2%}' for k,v in sorted(probs.items(), key=lambda x: -x[1])[:3]])}")
    
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
