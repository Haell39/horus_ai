# Heimdall Audio Model Ultra V1 - Pipeline de Inferência

## Estratégia: Heurística + Modelo CNN

Este documento descreve a estratégia completa de inferência que obteve os melhores resultados na validação.

---

## FASE 1: Detecção de Silêncio por Heurística (ANTES do modelo)

**Objetivo:** Segmentos com silêncio puro são classificados diretamente, sem usar o modelo.

### Constantes
```python
SILENCE_RMS_THRESHOLD = 0.008
SILENCE_PEAK_THRESHOLD = 0.15
```

### Lógica
```python
def is_silence(audio_segment):
    """Se RMS < 0.008 E Peak < 0.15 → É silêncio (sem usar modelo)"""
    rms = np.sqrt(np.mean(audio_segment ** 2))
    peak = np.max(np.abs(audio_segment))
    return (rms < SILENCE_RMS_THRESHOLD and peak < SILENCE_PEAK_THRESHOLD), rms
```

**Resultado:** Se `is_silence() == True`, classifica direto como `ausencia_audio` com 99% de confiança.

---

## FASE 2: Modelo de Classificação CNN

**Objetivo:** Para segmentos não-silenciosos, passa o espectrograma pelo modelo.

### Pré-processamento
```python
SAMPLE_RATE = 22050
DURATION = 3.0  # segundos
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMAX = 8000
IMG_SIZE = 128

def audio_to_mel(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=FMAX
    )
    # IMPORTANTE: Normalização fixa (ref=1.0), NÃO ref=np.max
    mel_db = librosa.power_to_db(mel, ref=1.0)
    
    # Clip e normalização para [0, 1]
    MIN_DB, MAX_DB = -80.0, 80.0
    mel_db = np.clip(mel_db, MIN_DB, MAX_DB)
    mel_norm = (mel_db - MIN_DB) / (MAX_DB - MIN_DB)
    
    mel_resized = cv2.resize(mel_norm, (IMG_SIZE, IMG_SIZE))
    return mel_resized[..., np.newaxis]  # Shape: (128, 128, 1)
```

**Saída do Modelo:** Vetor de 5 probabilidades na ordem:
`['ausencia_audio', 'eco_reverb', 'ruido_hiss', 'sinal_teste', 'normal']`

---

## FASE 3: Pós-Processamento com Thresholds

**Objetivo:** Filtrar predições com baixa confiança e aplicar regras especiais por classe.

### Constantes de Confiança
```python
MIN_CONFIDENCE = 0.68           # Geral (hiss, sinal_teste)
MIN_CONFIDENCE_ECO = 0.85       # Eco precisa de confiança alta
MIN_CONFIDENCE_AUSENCIA = 0.75  # Silêncio detectado pelo modelo
SILENCE_STRICT_RMS = 0.006      # RMS muito baixo = silêncio garantido
```

### Regras Especiais para Eco (classe mais difícil)
```python
ECO_MARGIN_OVER_NORMAL = 0.20   # Eco precisa ter 20% de margem sobre "normal"
ECO_MARGIN_OVER_SECOND = 0.15   # Eco precisa ter 15% de margem sobre a 2ª melhor classe
```

### Boost de Confiança por RMS
```python
ECO_MIN_RMS = 0.02
ECO_RMS_BOOST = 3.0
HISS_MIN_RMS = 0.015
HISS_RMS_BOOST = 2.0

def boosted_confidence(pred, conf, rms):
    """Segmentos com RMS alto ganham boost de confiança"""
    if pred == 'eco_reverb':
        return min(1.0, conf + ECO_RMS_BOOST * max(0, rms - ECO_MIN_RMS))
    if pred == 'ruido_hiss':
        return min(1.0, conf + HISS_RMS_BOOST * max(0, rms - HISS_MIN_RMS))
    return conf
```

### Lógica de Decisão por Classe
```python
def decide_error(pred, conf, probs, rms):
    eff_conf = boosted_confidence(pred, conf, rms)
    
    if pred == 'eco_reverb':
        # Eco só é aceito se tiver margem clara sobre outras classes
        eco_prob = probs['eco_reverb']
        normal_prob = probs['normal']
        second_best = max(p for cls, p in probs.items() if cls != 'eco_reverb')
        
        is_error = (
            eff_conf >= 0.85 and
            (eco_prob - normal_prob) >= 0.20 and
            (eco_prob - second_best) >= 0.15
        )
    
    elif pred == 'ausencia_audio':
        # Silêncio: ou modelo tem confiança alta OU RMS é muito baixo
        is_error = eff_conf >= 0.75 or rms <= 0.006
    
    else:  # ruido_hiss, sinal_teste
        is_error = pred != 'normal' and eff_conf >= 0.68
    
    return is_error
```

---

## FASE 4: Suavização Temporal (Merge de Segmentos)

**Objetivo:** Evitar alertas espúrios e unir segmentos consecutivos do mesmo erro.

### Constantes
```python
CLASS_MIN_SEGMENTS = {
    'eco_reverb': 1,      # Eco pode ser detectado em 1 segmento
    'ruido_hiss': 1,      # Chiado pode ser detectado em 1 segmento
    'ausencia_audio': 2,  # Silêncio precisa de 2 segmentos consecutivos
    'sinal_teste': 2      # Sinal de teste precisa de 2 segmentos
}

GAP_TOLERANCE_SEGMENTS = 1  # Tolera 1 segmento "normal" no meio do erro
```

### Regras para Segmento Único
```python
CLASS_SINGLE_SEGMENT_RULES = {
    'eco_reverb': {
        'min_confidence': 0.85,
        'min_rms': 0.03
    },
    'ruido_hiss': {
        'min_confidence': 0.82,
        'min_rms': 0.025
    }
}
```

**Lógica:** Erros só são reportados se:
1. Tiverem o número mínimo de segmentos consecutivos, OU
2. Passarem nas regras de segmento único (alta confiança + RMS alto)

---

## Pipeline Visual Completo

```
┌─────────────────────────────────────────────────────────────────┐
│                         ÁUDIO DE ENTRADA                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Segmentação: 3s com hop de 1s                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
              ┌─────────────────────────────────┐
              │      Para cada segmento:        │
              └─────────────────────────────────┘
                                │
                                ▼
              ┌─────────────────────────────────┐
              │   FASE 1: Heurística de        │
              │   Silêncio                      │
              │   RMS < 0.008 E Peak < 0.15?   │
              └─────────────────────────────────┘
                      │               │
                     SIM             NÃO
                      │               │
                      ▼               ▼
        ┌──────────────────┐  ┌──────────────────────────┐
        │ ausencia_audio   │  │   FASE 2: Modelo CNN     │
        │ (99% confiança)  │  │   (MobileNetV2)          │
        └──────────────────┘  │   → 5 probabilidades     │
                              └──────────────────────────┘
                                        │
                                        ▼
                              ┌──────────────────────────┐
                              │   FASE 3: Thresholds    │
                              │   + Boost de RMS        │
                              │   + Margens para Eco    │
                              └──────────────────────────┘
                                        │
                                        ▼
                              ┌──────────────────────────┐
                              │   FASE 4: Suavização    │
                              │   Temporal              │
                              │   - Merge segmentos     │
                              │   - Min segs por classe │
                              │   - Tolerância de gap   │
                              └──────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              LISTA DE ERROS COM TIMESTAMPS                      │
│                                                                 │
│  [                                                              │
│    {"type": "ausencia_audio", "start": 3.0, "end": 9.0, ...},  │
│    {"type": "ruido_hiss", "start": 15.0, "end": 21.0, ...}     │
│  ]                                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Resultados da Validação

| Classe | Precisão | Recall |
|--------|----------|--------|
| ausencia_audio | 100% | 100% |
| eco_reverb | - | 0%* |
| ruido_hiss | 100% | 100% |
| sinal_teste | 100% | 100% |
| normal | 100% | 100% |

*\*Eco tem recall baixo em vídeos reais, mas o modelo é extremamente estável (zero falsos positivos em vídeos normais).*

**Métricas Gerais:**
- Precision Geral: 75%
- Recall Geral: 75%
- F1-Score: 75%
- Acurácia em Vídeos Normais: 100% (0 falsos positivos)

---

## Notas de Implementação

1. **Sempre use `ref=1.0`** na conversão para dB, nunca `ref=np.max`. Isso preserva diferenças absolutas de volume.

2. **A heurística de silêncio é crítica.** Ela evita que silêncios sejam confundidos com outras classes pelo modelo.

3. **As margens para Eco são importantes.** Sem elas, o modelo tende a "alucinar" eco em áudio normal.

4. **A suavização temporal evita ruído.** Um único frame errado não deve gerar um alerta.
