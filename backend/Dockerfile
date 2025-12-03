# Backend Dockerfile - Horus AI
# Python FastAPI + TensorFlow + FFmpeg

FROM python:3.11-slim

# Instalar dependências do sistema (FFmpeg, libpq para PostgreSQL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Diretório de trabalho
WORKDIR /app

# Copiar requirements primeiro (cache de layers)
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fonte
COPY ./app ./app

# Criar diretórios para arquivos estáticos
RUN mkdir -p /app/static/hls /app/static/clips

# Expor porta
EXPOSE 8000

# Comando de inicialização
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
