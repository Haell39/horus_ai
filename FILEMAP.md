# FILE MAP — Horus AI

Mapa do repositório. **Atualizado:** Dezembro 2025

---

## ROOT

| Arquivo/Pasta        | Descrição                |
| -------------------- | ------------------------ |
| `README.md`          | Documentação principal   |
| `FILEMAP.md`         | Este arquivo             |
| `docker-compose.yml` | Orquestração Docker      |
| `backend/`           | FastAPI + ML + streaming |
| `frontend/`          | Angular 19               |
| `docs/`              | Documentação técnica     |
| `scripts/`           | Scripts de validação     |
| `videos_test/`       | Vídeos de teste          |

---

## backend/app/

| Arquivo                | Descrição                            |
| ---------------------- | ------------------------------------ |
| `main.py`              | Inicializa FastAPI, routers, modelos |
| `websocket_manager.py` | Gerenciador WebSocket                |

### api/endpoints/

| Arquivo          | Descrição            |
| ---------------- | -------------------- |
| `streams.py`     | Controle de ingestão |
| `ocorrencias.py` | CRUD ocorrências     |
| `analysis.py`    | Upload e análise     |
| `ws.py`          | WebSocket alertas    |
| `ml_info.py`     | Info dos modelos     |
| `admin.py`       | Endpoints admin      |

### ml/models/

```
models/
├── video/odin_model_v4.5/
│   ├── video_model_finetune.keras
│   └── labels.csv
├── audio/heimdall_audio_model_ultra_v1/
│   ├── audio_model.keras
│   └── labels.csv
└── lipsync/lipsync_model_v2/
    └── syncnet_v2_q.tflite
```

### streams/

| Arquivo         | Descrição                         |
| --------------- | --------------------------------- |
| `srt_reader.py` | Ingestão SRT→HLS, detecção, clips |

---

## frontend/src/app/pages/

| Pasta             | Descrição           |
| ----------------- | ------------------- |
| `monitoramento/`  | Player HLS, alertas |
| `dados/`          | Dashboard, gráficos |
| `cortes/`         | Revisão de clips    |
| `configuracoes/`  | Configurações       |
| `acessibilidade/` | VLibras             |

---

## scripts/

| Arquivo                   | Descrição          |
| ------------------------- | ------------------ |
| `test_validate_errors.py` | Validação de erros |
| `test_validate_normal.py` | Validação normal   |
| `validation_results.json` | Resultados         |

---

## videos_test/

| Pasta                   | Descrição      |
| ----------------------- | -------------- |
| `validate_model_video/` | Erros de vídeo |
| `validate_model_audio/` | Erros de áudio |
| `validate_normal/`      | Vídeos normais |
| `test_lipsync/`         | Teste lipsync  |

---

## Classes

**Vídeo:** `['normal', 'freeze', 'fade', 'fora_de_foco']`  
**Áudio:** `['ausencia_audio', 'eco_reverb', 'ruido_hiss', 'sinal_teste', 'normal']`
