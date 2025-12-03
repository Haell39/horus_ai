# Ambiente (.env) — Horus AI

## 1. Criar o arquivo

```powershell
Copy-Item backend\.env.example backend\.env
notepad backend\.env
```

---

## 2. Variáveis Obrigatórias

| Variável       | Descrição         | Exemplo                                          |
| -------------- | ----------------- | ------------------------------------------------ |
| `DATABASE_URL` | URL do PostgreSQL | `postgresql://user:pass@localhost:5432/horus_db` |

---

## 3. Tuning de Detecção - Vídeo

| Variável                    | Descrição                          | Default |
| --------------------------- | ---------------------------------- | ------- |
| `VIDEO_VOTE_K`              | Frames consecutivos para confirmar | `3`     |
| `VIDEO_MOVING_AVG_M`        | Janela para média móvel            | `5`     |
| `VIDEO_THRESH_FREEZE`       | Threshold freeze                   | `0.80`  |
| `VIDEO_THRESH_FADE`         | Threshold fade                     | `0.80`  |
| `VIDEO_THRESH_FORA_DE_FOCO` | Threshold blur                     | `0.75`  |

---

## 4. Tuning de Detecção - Áudio

| Variável                      | Default |
| ----------------------------- | ------- |
| `AUDIO_THRESH_AUSENCIA_AUDIO` | `0.80`  |
| `AUDIO_THRESH_ECO_REVERB`     | `0.85`  |
| `AUDIO_THRESH_RUIDO_HISS`     | `0.80`  |

---

## 5. Sistema de Debounce (Stream)

| Variável                     | Descrição      | Default |
| ---------------------------- | -------------- | ------- |
| `STREAM_DEBOUNCE_DURATION_S` | Duração mínima | `3.0`   |
| `STREAM_DEBOUNCE_GAP_S`      | Gap máximo     | `25.0`  |

---

## 6. Clips

| Variável          | Default |
| ----------------- | ------- |
| `CLIP_OUTPUT_FPS` | `15`    |

---

## Exemplo Mínimo

```dotenv
DATABASE_URL=postgresql://horus_user:senha@localhost:5432/horus_db
VIDEO_VOTE_K=3
STREAM_DEBOUNCE_DURATION_S=3.0
CLIP_OUTPUT_FPS=15
```

⚠️ **NÃO comite** `backend/.env`
