# 👁️ Horus AI

**Sistema inteligente de monitoramento e detecção de anomalias em transmissões de vídeo ao vivo.**

Três modelos de Machine Learning analisam o stream simultaneamente vídeo, áudio e lipsync identificando falhas e gerando alertas em tempo real.

| Tipo        | Anomalias Detectadas                                      |
| ----------- | --------------------------------------------------------- |
| **Vídeo**   | Freeze, Fade (tela preta), Fora de foco                   |
| **Áudio**   | Ausência de som, Eco/Reverb, Ruído/Chiado, Sinal de teste |
| **Lipsync** | Dessincronização entre áudio e vídeo                      |

---

## 🛠 Stack

**Backend:** Python 3.11 · FastAPI · PostgreSQL · TensorFlow/Keras · TFLite · OpenCV · Librosa · FFmpeg

**Frontend:** Angular 19 · TypeScript · ApexCharts · HLS.js · WebSocket · jsPDF

---

## 📦 Pré-requisitos

- [Python 3.11+](https://www.python.org/downloads/)
- [Node.js 18+](https://nodejs.org/)
- [PostgreSQL 14+](https://www.postgresql.org/download/)
- [FFmpeg 5.0+](https://ffmpeg.org/download.html)

---

## 🚀 Instalação

**1. Clonar e configurar banco**

```bash
git clone https://github.com/Haell39/horus_ai.git
cd horus_ai
```

```sql
CREATE USER horus_user WITH PASSWORD 'sua_senha';
CREATE DATABASE horus_db OWNER horus_user;
GRANT ALL PRIVILEGES ON DATABASE horus_db TO horus_user;
```

**2. Backend**

```bash
cd backend
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1 | Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # editar DATABASE_URL
```

**3. Frontend**

```bash
cd frontend
npm install
```

---

## ▶️ Executando

```bash
# Terminal 1 — Backend
cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
cd frontend && npm start
```

| Serviço       | URL                                   |
| ------------- | ------------------------------------- |
| Interface Web | http://localhost:4200                 |
| API + Swagger | http://localhost:8000/docs            |
| Stream HLS    | http://localhost:8000/hls/stream.m3u8 |

---

## 📁 Estrutura

```
horus_ai/
├── backend/
│   ├── app/
│   │   ├── api/endpoints/     # REST + WebSocket
│   │   ├── ml/
│   │   │   ├── inference.py   # Pipeline de inferência
│   │   │   └── models/        # video/ · audio/ · lipsync/
│   │   ├── streams/
│   │   │   └── srt_reader.py  # Ingestão SRT → HLS + detecção
│   │   └── db/                # Models + schemas
│   └── static/                # HLS segments + clips gerados
├── frontend/
│   └── src/app/
│       ├── pages/             # monitoramento · dados · cortes · config
│       └── services/          # HTTP + WebSocket
├── scripts/                   # Scripts de validação dos modelos
├── docs/                      # Documentação técnica
└── docker-compose.yml
```

---

## 🤖 Modelos de IA

| Modelo                    | Formato              | Acurácia |
| ------------------------- | -------------------- | -------- |
| Odin v4.5 (vídeo)         | `.keras`             | 97.6%    |
| Heimdall Ultra v1 (áudio) | `.keras`             | 90.9%    |
| SyncNet v2 (lipsync)      | `.tflite` quantizado | 100%     |

Estratégia híbrida: heurísticas OpenCV (detecção rápida) + ML (confirmação com votação temporal).

---

## 📄 Licença

MIT — veja [LICENSE](LICENSE).

O **Horus AI** é um sistema de monitoramento automatizado que detecta falhas técnicas em transmissões de vídeo ao vivo. Três modelos de machine learning analisam o stream simultaneamente — **vídeo**, **áudio** e **lipsync** — identificando anomalias e gerando alertas em tempo real.

### Anomalias Detectadas

| Tipo        | Anomalias                                                 |
| ----------- | --------------------------------------------------------- |
| **Vídeo**   | Freeze (congelamento), Fade (tela preta), Blur (desfoque) |
| **Áudio**   | Ausência de som, Volume baixo, Ruído/Chiado, Eco/Reverb   |
| **Lipsync** | Dessincronização entre áudio e vídeo                      |

### Principais Funcionalidades

- ✅ Monitoramento de streams SRT em tempo real
- ✅ Detecção automática de anomalias com IA (estratégia híbrida: heurísticas + ML)
- ✅ Geração automática de clipes das falhas como evidência
- ✅ Dashboard com estatísticas, gráficos e KPIs
- ✅ Alertas em tempo real via WebSocket
- ✅ Upload e análise de vídeos offline
- ✅ Página de cortes para revisão e download de clipes
- ✅ Exportação de relatórios em PDF
- ✅ Acessibilidade com VLibras integrado

---
