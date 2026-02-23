# RESUMO TÉCNICO DO PROJETO HORUS AI

**Atualização:** Dezembro 2025 | **Versão:** 2.0 (Produção)

---

## 1. VISÃO GERAL

O **Horus AI** monitora vídeo em tempo real, ingere streams SRT, processa para HLS e executa inferência ML para detecção de anomalias.

| Componente | Tecnologia                | Função                    |
| ---------- | ------------------------- | ------------------------- |
| Backend    | Python/FastAPI            | API, ingestão, inferência |
| Frontend   | Angular 19                | Dashboard, player HLS     |
| Database   | PostgreSQL                | Persistência              |
| ML         | TensorFlow/Keras + TFLite | Detecção                  |

---

## 2. STACK TECNOLÓGICO

### Backend

- Python 3.11+, FastAPI, Uvicorn
- TensorFlow/Keras (`.keras`), TFLite (`.tflite`)
- OpenCV, Librosa, FFmpeg
- PostgreSQL (SQLAlchemy)

### Frontend

- Angular 19, ApexCharts, HLS.js
- jsPDF, VLibras

---

## 3. MODELOS DE IA

| Modelo                | Formato                | Classes                                                     |
| --------------------- | ---------------------- | ----------------------------------------------------------- |
| **Odin v4.5**         | `.keras`               | freeze, fade, fora_de_foco, normal                          |
| **Heimdall Ultra v1** | `.keras`               | ausencia_audio, eco_reverb, ruido_hiss, sinal_teste, normal |
| **SyncNet v2**        | `.tflite` (quantizado) | sincronizado, dessincronizado                               |

---

## 4. ESTRATÉGIA DE DETECÇÃO

### Híbrida (Heurísticas + ML)

1. **Heurísticas (pré-ML):**

   - Blur: Variância Laplaciana
   - Brightness: Média < 30 → fade
   - Motion: Diferença entre frames → freeze

2. **ML (confirmação):**
   - Votação temporal (K=3 frames)
   - Média móvel (M=5 janelas)

### Sistema de Debounce

| Parâmetro           | Valor | Descrição      |
| ------------------- | ----- | -------------- |
| `DEBOUNCE_DURATION` | 3.0s  | Duração mínima |
| `DEBOUNCE_GAP`      | 25.0s | Gap máximo     |

---

## 5. VALIDAÇÃO (Dezembro 2025)

| Modelo  | Acurácia |
| ------- | -------- |
| Vídeo   | 97.6%    |
| Áudio   | 90.9%    |
| Lipsync | 100%     |

---

## 6. ESTRUTURA

```
horus_ai/
├── backend/app/
│   ├── api/endpoints/
│   ├── ml/
│   │   ├── inference.py
│   │   └── models/
│   │       ├── video/odin_model_v4.5/
│   │       ├── audio/heimdall_audio_model_ultra_v1/
│   │       └── lipsync/lipsync_model_v2/
│   ├── streams/srt_reader.py
│   └── db/
├── frontend/src/app/
│   ├── pages/
│   └── services/
├── scripts/
└── videos_test/
```

---

## 7. VARIÁVEIS CRÍTICAS

```dotenv
DATABASE_URL=postgresql://user:pass@localhost:5432/horus_db
VIDEO_VOTE_K=3
VIDEO_MOVING_AVG_M=5
STREAM_DEBOUNCE_DURATION_S=3.0
STREAM_DEBOUNCE_GAP_S=25.0
CLIP_OUTPUT_FPS=15
VIDEO_THRESH_FREEZE=0.80
AUDIO_THRESH_AUSENCIA_AUDIO=0.80
```

---

## 8. MANUTENÇÃO

1. **Modelos:** Vídeo/Áudio `.keras`, Lipsync `.tflite` quantizado
2. **FFmpeg:** Deve estar no PATH, logs em `static/hls/hls_ffmpeg.log`
3. **Debounce:** Ajustar gap se buffer do stream mudar
4. **Clips:** Gerados a 15 FPS por padrão
