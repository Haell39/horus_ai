👁️ Horus AI

---

## 📋 Índice

- [Sobre o Projeto](#-sobre-o-projeto)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Pré-requisitos](#-pré-requisitos)
- [Guia de Instalação Completo](#-guia-de-instalação-completo)
- [Configuração](#-configuração)
- [Executando o Projeto](#-executando-o-projeto)
- [Uso do Sistema](#-uso-do-sistema)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Sobre o Projeto

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

## 🛠 Tecnologias Utilizadas

### Backend

| Tecnologia       | Versão | Uso                                      |
| ---------------- | ------ | ---------------------------------------- |
| Python           | 3.11+  | Linguagem principal                      |
| FastAPI          | Latest | API REST e WebSocket                     |
| SQLAlchemy       | Latest | ORM para banco de dados                  |
| PostgreSQL       | 14+    | Banco de dados relacional                |
| TensorFlow/Keras | 2.x    | Modelos de ML (vídeo e áudio)            |
| TensorFlow Lite  | 2.x    | Modelo de Lipsync (quantizado)           |
| OpenCV           | 4.x    | Processamento de vídeo e heurísticas     |
| Librosa          | Latest | Processamento de áudio                   |
| FFmpeg           | 5.0+   | Conversão SRT → HLS e extração de frames |

### Frontend

| Tecnologia | Versão | Uso                      |
| ---------- | ------ | ------------------------ |
| Angular    | 19     | Framework frontend       |
| TypeScript | 5.x    | Linguagem                |
| RxJS       | 7.8    | Programação reativa      |
| ApexCharts | 3.54   | Gráficos e visualizações |
| HLS.js     | Latest | Player de vídeo HLS      |
| jsPDF      | 3.0    | Exportação de relatórios |

---

## 📦 Pré-requisitos

Antes de começar, certifique-se de ter instalado:

| Software       | Versão Mínima | Download                                               |
| -------------- | ------------- | ------------------------------------------------------ |
| **Python**     | 3.11+         | [python.org](https://www.python.org/downloads/)        |
| **Node.js**    | 18+           | [nodejs.org](https://nodejs.org/)                      |
| **PostgreSQL** | 14+           | [postgresql.org](https://www.postgresql.org/download/) |
| **FFmpeg**     | 5.0+          | [ffmpeg.org](https://ffmpeg.org/download.html)         |
| **Git**        | 2.0+          | [git-scm.com](https://git-scm.com/downloads)           |

### Verificando as instalações

**Windows (PowerShell):**

```powershell
python --version      # Python 3.11.x
node --version        # v18.x.x ou superior
npm --version         # 9.x.x ou superior
psql --version        # psql (PostgreSQL) 14.x
ffmpeg -version       # ffmpeg version 5.x
git --version         # git version 2.x.x
```

**Linux/Mac (Bash):**

```bash
python3 --version
node --version
npm --version
psql --version
ffmpeg -version
git --version
```

---

## 🚀 Guia de Instalação Completo

### Passo 1: Clonar o Repositório

**Windows:**

```powershell
cd C:\Projetos
git clone https://github.com/Haell39/horus_ai.git
cd horus_ai
```

**Linux/Mac:**

```bash
cd ~/projetos
git clone https://github.com/Haell39/horus_ai.git
cd horus_ai
```

---

### Passo 2: Configurar o Banco de Dados PostgreSQL

#### Windows (pgAdmin ou psql)

1. Abra o **pgAdmin** ou **SQL Shell (psql)**
2. Execute os comandos:

```sql
CREATE USER horus_user WITH PASSWORD 'sua_senha_segura';
CREATE DATABASE horus_db OWNER horus_user;
GRANT ALL PRIVILEGES ON DATABASE horus_db TO horus_user;
```

#### Linux/Mac

```bash
sudo -u postgres psql
```

```sql
CREATE USER horus_user WITH PASSWORD 'sua_senha_segura';
CREATE DATABASE horus_db OWNER horus_user;
GRANT ALL PRIVILEGES ON DATABASE horus_db TO horus_user;
\q
```

---

### Passo 3: Configurar o Backend

#### 3.1 Criar ambiente virtual Python

**Windows PowerShell:**

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/Mac:**

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
```

#### 3.2 Instalar dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3.3 Criar arquivo de configuração

**Windows:**

```powershell
Copy-Item .env.example .env
```

**Linux/Mac:**

```bash
cp .env.example .env
```

#### 3.4 Editar o arquivo `.env`

Abra `backend/.env` no seu editor e configure:

```dotenv
# === OBRIGATÓRIO: Conexão com PostgreSQL ===
DATABASE_URL=postgresql://horus_user:sua_senha_segura@localhost:5432/horus_db

# === OPCIONAL: URL do stream SRT (pode configurar depois na UI) ===
SRT_STREAM_URL_GLOBO=srt://seu.servidor.srt:porta?mode=caller

# === Configurações de Detecção (valores recomendados) ===
VIDEO_VOTE_K=3
VIDEO_MOVING_AVG_M=5
VIDEO_DISABLE_AUDIO_PROCESSING=false
VIDEO_ALLOW_AUDIO_OVERRIDE=false

# === Thresholds de Áudio ===
AUDIO_THRESH_DEFAULT=0.60
AUDIO_THRESH_AUSENCIA_AUDIO=0.80
AUDIO_THRESH_ECO_REVERB=0.85
AUDIO_THRESH_RUIDO_HISS=0.80

# === Thresholds de Vídeo ===
VIDEO_THRESH_FREEZE=0.80
VIDEO_THRESH_FADE=0.80
VIDEO_THRESH_FORA_DE_FOCO=0.75

# === Debounce para Stream (evita falsos positivos) ===
STREAM_DEBOUNCE_DURATION_S=3.0
STREAM_DEBOUNCE_GAP_S=25.0

# === FPS dos Clipes Gerados ===
CLIP_OUTPUT_FPS=15
```

---

### Passo 4: Configurar o Frontend

```bash
cd ../frontend
npm install
```

---

### Passo 5: Verificar FFmpeg no PATH

O FFmpeg deve estar acessível globalmente:

```bash
ffmpeg -version
ffprobe -version
```

**Se não estiver no PATH:**

- **Windows**: Adicione a pasta `bin` do FFmpeg em:
  - Configurações → Sistema → Sobre → Configurações avançadas → Variáveis de Ambiente → Path
- **Linux/Mac**: Adicione ao `~/.bashrc` ou `~/.zshrc`:
  ```bash
  export PATH=$PATH:/caminho/para/ffmpeg/bin
  ```

---

## ▶️ Executando o Projeto

### Execução para Desenvolvimento

Abra **dois terminais**:

#### Terminal 1 — Backend

**Windows PowerShell:**

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Linux/Mac:**

```bash
cd backend
source .venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> ⚠️ **IMPORTANTE**: NÃO use `--reload` ao testar streams! O reload reinicia o processo e fecha conexões FFmpeg/WebSocket.

#### Terminal 2 — Frontend

```bash
cd frontend
npm start
```

---

### Acessando o Sistema

| Serviço                           | URL                                   |
| --------------------------------- | ------------------------------------- |
| **🖥️ Interface Web**              | http://localhost:4200                 |
| **📡 API Backend**                | http://localhost:8000                 |
| **📚 Documentação API (Swagger)** | http://localhost:8000/docs            |
| **📺 Stream HLS**                 | http://localhost:8000/hls/stream.m3u8 |
| **🔌 WebSocket**                  | ws://localhost:8000/ws/ocorrencias    |

---

## 📖 Uso do Sistema

### Páginas Disponíveis

| Página            | Descrição                                           |
| ----------------- | --------------------------------------------------- |
| **Monitoramento** | Player ao vivo + lista de ocorrências em tempo real |
| **Dados**         | Dashboards com gráficos e estatísticas              |
| **Cortes**        | Gerenciamento de clipes gerados                     |
| **Configurações** | Ajustes do sistema                                  |

### Iniciar/Parar Stream via API

```powershell
# Iniciar stream
$body = @{ url = 'srt://servidor:porta?mode=caller'; fps = 1.0 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/streams/start -Body $body -ContentType 'application/json'

# Parar stream
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/v1/streams/stop

# Verificar status
Invoke-RestMethod -Method Get -Uri http://localhost:8000/api/v1/streams/status
```

### Upload de Vídeo para Análise Offline

Na interface web: **Monitoramento** → Botão de Upload

Ou via API:

```bash
curl -X POST "http://localhost:8000/api/v1/analysis/upload" -F "file=@video.mp4"
```

---

## 📁 Estrutura do Projeto

```
horus_ai/
├── backend/                      # API FastAPI + ML
│   ├── app/
│   │   ├── api/endpoints/        # Endpoints REST e WebSocket
│   │   │   ├── analysis.py       # Upload e análise de vídeos
│   │   │   ├── ocorrencias.py    # CRUD de ocorrências
│   │   │   ├── streams.py        # Controle de ingestão SRT
│   │   │   └── ws.py             # WebSocket para alertas
│   │   ├── core/                 # Configurações
│   │   ├── db/                   # Modelos e schemas do banco
│   │   ├── ml/                   # Inferência e modelos de IA
│   │   │   └── models/           # Arquivos .keras e .tflite
│   │   │       ├── video/        # Modelo de vídeo (Keras)
│   │   │       ├── audio/        # Modelo de áudio (Keras)
│   │   │       └── lipsync/      # Modelo de lipsync (TFLite quantizado)
│   │   └── streams/              # Ingestão SRT e processamento
│   │       └── srt_reader.py     # Controlador FFmpeg + análise
│   ├── static/
│   │   ├── hls/                  # Playlist e segmentos HLS
│   │   └── clips/                # Clipes de evidência gerados
│   ├── .env                      # Configurações locais (NÃO committar)
│   ├── .env.example              # Exemplo de configuração
│   └── requirements.txt          # Dependências Python
│
├── frontend/                     # App Angular 19
│   ├── src/app/
│   │   ├── pages/
│   │   │   ├── monitoramento/    # Player + ocorrências ao vivo
│   │   │   ├── dados/            # Dashboards e gráficos
│   │   │   ├── cortes/           # Gerenciamento de clipes
│   │   │   └── configuracoes/    # Configurações
│   │   ├── components/           # Componentes reutilizáveis
│   │   ├── services/             # Serviços (API, WebSocket)
│   │   └── models/               # Interfaces TypeScript
│   └── package.json
│
├── docs/                         # Documentação adicional
├── docker-compose.yml            # Orquestração Docker (opcional)
└── README.md                     # Este arquivo
```

---

## 🔧 Troubleshooting

### ❌ Backend não inicia

```powershell
# Verificar se PostgreSQL está rodando
Get-Service -Name postgresql*  # Windows
sudo systemctl status postgresql  # Linux

# Testar conexão com o banco
psql -U horus_user -d horus_db -h localhost
```

### ❌ Stream não aparece no player

```powershell
# Verificar se ffmpeg está rodando
Get-Process -Name ffmpeg

# Ver logs do ffmpeg
Get-Content backend\static\hls\hls_ffmpeg.log -Tail 50

# Verificar se playlist existe
Test-Path backend\static\hls\stream.m3u8
```

### ❌ Matar processos FFmpeg pendentes

```powershell
# Windows
Get-Process -Name ffmpeg | Stop-Process -Force

# Linux/Mac
pkill -9 ffmpeg
```

### ❌ Erro de CORS no frontend

Verifique se o backend está rodando na porta 8000.

### ❌ Modelos não carregam

```powershell
# Verificar se os arquivos existem
Get-ChildItem backend\app\ml\models -Recurse -Filter "*.keras"
Get-ChildItem backend\app\ml\models -Recurse -Filter "*.tflite"
```

### ❌ Dependências Python com erro

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --force-reinstall
```

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

<p align="center">
  <b>👁️ Horus AI</b> — Monitoramento Inteligente de Broadcast<br>
  <i>Projeto Acadêmico</i>
</p>
