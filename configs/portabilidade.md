# Portabilidade e Deploy — Hosted Demo / Testes no Local (Guia Completo)

Este documento descreve tudo o que um supervisor técnico precisa saber para colocar o sistema "horus_ai" no ar como um _hosted demo_ ou em um ambiente local (rede interna), de forma que um testador não-técnico possa acessar pelo browser e usar as funcionalidades (uploads, visualização do HLS, ver ocorrências, baixar cortes). O guia inclui: pré-requisitos, opções de deploy (rápido e recomendado), exemplos de configuração (nginx, systemd), Docker Compose esqueleto, comandos PowerShell para testes SRT/capture, checklist de verificação e troubleshooting.

O conteúdo deste documento foi preparado para a estrutura do repositório:

- Backend: `backend/` (FastAPI)
- Frontend: `frontend/` (Angular)
- Modelos TFLite: `backend/app/ml/models/`
- Static: `backend/static/clips/`, `backend/static/hls/`

Versão: 2025-11-08

---

## Sumário rápido

- Para um testador leigo: disponibilizar um URL (HTTPS) do app backend+frontend é suficiente — o testador abre o navegador e não instala nada.
- Para suportar SRT ao vivo e placas de captura, o host que roda o backend deve ter ffmpeg/ffprobe com suporte a SRT, acesso à placa ou receber SRT push do local da captura.
- Recomendações:
  - Demo rápido (temporário): ngrok para expor backend local via HTTPS.
  - Deploy robusto: Docker Compose em VM com Nginx + Certbot (Let's Encrypt) + volumes persistentes.

---

## 1. Requisitos mínimos do host

1. Sistema operacional: Linux (recomendado para produção) ou Windows Server (suportado). A documentação abaixo usa comandos PowerShell para diagnóstico e exemplos multiplataforma para scripts.
2. Python 3.10+ (backend) e Node.js 16+ (apenas para build do frontend). Na produção servimos o build estático do frontend com Nginx.
3. ffmpeg e ffprobe instalados e no PATH. Importante: a build do ffmpeg deve incluir suporte a libsrt (para SRT) se você for consumir/receber SRT.
4. Modelos TFLite no repositório: `backend/app/ml/models/*.tflite`.
5. Permissões de escrita em `backend/static/clips` e `backend/static/hls`.
6. Open ports / regras de firewall conforme a opção escolhida (ex.: 80/443 para web, porta SRT se receber direto).
7. (Opcional) Docker Engine / Docker Desktop para deploy via containers.

---

## 2. Preparação rápida para um demo (ngrok)

Objetivo: expor o backend local para que qualquer pessoa abra uma URL HTTPS e use a aplicação sem instalar nada.

Passos (rápidos):

1. No servidor/PC onde o projeto está clonado, ative o venv e inicie o backend (exemplo PowerShell):

```powershell
cd 'D:\GitHub Desktop\horus_ai\backend'
# ative o virtualenv (ajuste o caminho conforme seu ambiente)
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

2. (Opcional) Inicie o frontend dev (se quiser hot-reload). Para demo simples, é melhor buildar e servir estático via nginx.

3. Instale e rode ngrok (local machine) para expor a porta 8000:

```powershell
# baixe/instale ngrok e em seguida:
ngrok http 8000
# ngrok retorna uma URL pública https://xxxxx.ngrok.io
```

4. Atualize a configuração do frontend `environment.backendBase` para apontar para a URL pública (ou use a URL diretamente no navegador). ngrok suporta WebSocket, então `/ws/ocorrencias` funciona.

Observações:

- ngrok é ideal para demos rápidas. Não é recomendado para produção (limitações, sessão temporária e segurança).
- Se quiser que o testador tenha um URL fixo, considere usar ngrok pago (domínio custom) ou provisionar um VM e configurar DNS/TLS.

---

## 3. Fazer o sistema rodar como serviço em VM (recomendado para teste interno)

Objetivo: disponibilizar app em URL estável (interna ou pública) usando Nginx (proxy + static), uvicorn (backend) e certbot para TLS.

### 3.1 Build frontend (uma vez)

No diretório `frontend/`:

```powershell
cd frontend
npm ci
npm run build -- --configuration production
# output tipicamente em frontend/dist/
```

Copie os artefatos para o servidor (ou para dentro do container que servirá o frontend). Alternativamente, sirva o `dist/` via Nginx ou copie para `backend/static/frontend/` e adapte `main.py` para servir.

### 3.2 Instalar dependências do backend

No servidor (Linux recomendado):

```bash
cd /opt/horus/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# garanta models em backend/app/ml/models
```

### 3.3 Testar ffmpeg

```powershell
ffmpeg -version
ffprobe -version
# verifique que a saída reporta libsrt (se for usar SRT)
```

### 3.4 systemd unit (exemplo)

Crie `/etc/systemd/system/horus_backend.service` com o conteúdo abaixo (ajuste paths/usuário):

```ini
[Unit]
Description=Horus AI Backend
After=network.target

[Service]
User=horus
WorkingDirectory=/opt/horus/backend
Environment="PATH=/opt/horus/backend/.venv/bin"
ExecStart=/opt/horus/backend/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

Depois:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now horus_backend
sudo journalctl -u horus_backend -f
```

### 3.5 Nginx (proxy + WebSocket passthrough)

Exemplo de bloco de servidor (ajuste paths/hostname):

```nginx
server {
    listen 80;
    server_name your.example.domain;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 90;
    }

    location /clips/ {
        alias /opt/horus/backend/static/clips/;
        add_header Cache-Control "public, max-age=3600";
    }

    location /.well-known/acme-challenge/ { }
}
```

Habilite TLS com certbot:

```bash
sudo certbot --nginx -d your.example.domain
```

### 3.6 CORS / Configuração do frontend

Ajuste `environment.backendBase` no build do frontend para apontar para `https://your.example.domain` antes de buildar. Rebuild e sirva os arquivos estáticos por Nginx ou pelo backend.

### 3.7 Testes básicos após deploy

- Acesse `https://your.example.domain/docs` (OpenAPI) — confirma backend ativo.
- Abra a UI no browser e verifique upload e player.
- Teste WebSocket com `wscat` ou pelo browser console.

---

## 4. Docker Compose (deploy recomendado para portabilidade)

Uma forma robusta e reprodutível é empacotar tudo em contêineres e entregar um `docker-compose.yml`. A vantagem: a equipe infra só precisa do Docker Engine e o resto sobe com um único comando.

### 4.1 Esqueleto `docker-compose.yml` (exemplo)

> Atenção: este é um esqueleto. Em produção ajuste imagens, volumes e políticas de restart.

```yaml
version: "3.8"
services:
  backend:
    build: ./backend
    image: horus/backend:latest
    restart: unless-stopped
    volumes:
      - ./backend/static/clips:/app/static/clips
      - ./backend/app/ml/models:/app/app/ml/models:ro
    environment:
      - PYTHONUNBUFFERED=1
      - ENVIRONMENT=production
    ports:
      - "8000:8000"

  nginx:
    image: nginx:stable
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
      - ./backend/static/clips:/usr/share/nginx/html/clips:ro
      - ./frontend/dist:/usr/share/nginx/html:ro
    depends_on:
      - backend

  # opcional: redis + celery
  redis:
    image: redis:7
    restart: unless-stopped

  celery:
    build: ./backend
    command: celery -A app.workers worker --loglevel=info
    depends_on:
      - redis
    volumes:
      - ./backend:/app
```

### 4.2 Dockerfile mínimo para backend (com ffmpeg) — exemplo Debian-based

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg build-essential curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Observações:

- Algumas distribuições slim podem não incluir codecs. Se ffmpeg em apt não tiver SRT, utilize uma imagem Debian/Ubuntu que permita instalar `libsrt` ou construa ffmpeg com libsrt estático.
- Monte os volumes para `static/clips` para persistência.

### 4.3 Subir com docker-compose

```bash
docker-compose up -d --build
# ver logs
docker-compose logs -f backend
```

---

## 5. Suporte a SRT e Placas de Captura — detalhes operacionais

### 5.1 SRT: caller vs listener

- Caller: seu servidor (consumer) abre conexão para um origin (outbound). Exemplo: `ffmpeg -i "srt://origin:9000" ...`.
- Listener (server): o servidor aceita conexões SRT em uma porta: `ffmpeg -listen 1 -i "srt://0.0.0.0:9000" ...`.

Escolha baseada em rede:

- Se o remetente (onde a captura está) pode iniciar conexão para o servidor, use _listener_ no servidor (recebe push). Fácil quando remetente está em campo.
- Se o servidor pode abrir uma conexão para a fonte, use _caller_.

### 5.2 Exemplos ffmpeg (PowerShell)

- Push SRT (do laptop com a placa):

```powershell
ffmpeg -f dshow -i video="Nome do Dispositivo" -c:v libx264 -preset veryfast -f mpegts "srt://seu-servidor.example:9000?pkt_size=1316"
```

- Servidor ouvindo (listener) e salvando HLS local:

```bash
ffmpeg -listen 1 -i "srt://0.0.0.0:9000" -c copy -f hls /app/static/hls/stream.m3u8
```

- Teste simples de leitura (caller):

```powershell
ffmpeg -timeout 5000000 -i "srt://srt-host.example:9000?pkt_size=1316" -t 5 -c copy test_output.ts
```

### 5.3 Placas de captura

- Para placa conectada ao servidor: configure ffmpeg (dshow no Windows, v4l2 no Linux) para gerar HLS ou enviar para a pipeline local.
- Para placa conectada a laptop local: execute ffmpeg/OBS no laptop e faça SRT push para o servidor (exemplo acima). Após o push, o backend deverá ser capaz de processar o HLS/fluxo (se estiver configurado para pegar o SRT/hls).

---

## 6. Testes & Smoke tests (passo a passo)

1. Testes de pré-check (no servidor):

```powershell
python --version
node --version
ffmpeg -version
ffprobe -version
# verificar existem modelos TFLite
ls backend/app/ml/models/
# verificar pastas static
ls backend/static/clips
ls backend/static/hls
```

2. Testar rota OpenAPI:

```powershell
curl.exe http://localhost:8000/docs
```

3. Testar upload (arquivo local):

```powershell
curl.exe -v -F "file=@C:\caminho\teste.mp4" -F "fps=1.0" http://localhost:8000/api/v1/analysis/upload
```

- Resposta esperada: `status` `ok` ou `queued` ou um objeto de ocorrência (se detectado inline).

4. Testar WebSocket (ouvir ocorrências):

```powershell
npm i -g wscat
wscat -c ws://localhost:8000/ws/ocorrencias
# aguarde mensagens quando ocorrências forem criadas
```

5. Testar HLS (se ingest ativo):

- Abra `http://localhost:8000/hls/stream.m3u8` no VLC ou no player do frontend.

6. Testar download de clip:

- Depois de criar uma ocorrência com evidência, pegue `evidence.clip_path` e baixe:

```powershell
curl.exe -o clip.mp4 http://localhost:8000/clips/<nome-do-clipe>.mp4
```

---

## 7. Checklist para um testador leigo (o que pedir a pessoa)

- Abra o link fornecido (ex.: https://demo.horus.example).
- Vá para a página Monitoramento.
- Para teste simples: clique em "Escolher arquivo", selecione um MP4 curto e clique em "Analisar Vídeo".
- Espere notificação/toast — se houver falha detectada, vá para a página Cortes e baixe o clip.
- Não é necessário instalar nada no PC do testador.

Se o teste envolver uma captura ao vivo via SRT ou placa:

- Combine com um técnico para que o fluxo SRT esteja sendo enviado ao servidor (push) ou que a placa esteja fisicamente no servidor.
- Testador leigo só precisa recarregar a página e observar o player/alertas.

---

## 8. Segurança, retenção e operacional

- Limite uploads e implemente autenticação para evitar abuso.
- Defina uma política de retenção: rotina cron para limpar `backend/static/clips` mais antigos que X dias.
- Logs e monitoring: envie logs do Uvicorn/Backend para um agregador (ELK/CloudWatch) para depurar problemas em campo.
- Rate limiting: especialmente para o endpoint de upload.

---

## 9. Troubleshooting comum

- ffmpeg sem SRT: `ffmpeg -version` não mostra libsrt → instale uma build do ffmpeg que inclua SRT ou construa ffmpeg com libsrt.
- Firewall: SRT falha com `I/O error` → teste com `Test-NetConnection` e verifique regras de firewall / NAT.
- Clips não servidos → confirme que arquivos estão em `backend/static/clips` e que o Nginx ou o backend serve `/clips/` a partir deste diretório.
- WebSocket falha por proxy → verifique `proxy_set_header Upgrade` e `Connection "upgrade"` no Nginx.

Comandos de diagnóstico rápidos (PowerShell):

```powershell
# testa conectividade SRT (porta)
Test-NetConnection -ComputerName srt-host.example -Port 9000

# testa se ffmpeg pode consumir a fonte SRT
ffmpeg -timeout 5000000 -i "srt://srt-host.example:9000?pkt_size=1316" -t 5 -c copy test_output.ts

# listar clips
Get-ChildItem -Path .\backend\static\clips
```

---

## 10. Próximos passos que posso entregar (posso gerar os artefatos agora se quiser):

- (A) `docker-compose.yml` + `Dockerfile` para backend pronto para testes (eu gero e adiciono ao repositório).
- (B) Script PowerShell `on_site_smoke_tests.ps1` que a equipe local roda para validar SRT / upload / HLS / WS.
- (C) `nginx` conf pronto com templates para TLS + WebSocket.
- (D) Script `cleanup_clips.sh`/`cleanup_clips.ps1` para retenção de clips.

Diz qual desses você prefere que eu gere primeiro e eu adiciono ao repositório com instruções passo-a-passo.

---

_Fim do documento._
