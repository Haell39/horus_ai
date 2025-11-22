RESUMO TÉCNICO DO PROJETO HORUS AI
Data de Atualização: 22/11/2025
Versão: Atual (Pós-refatoração de Ingestão e Inferência)

1. VISÃO GERAL DO SISTEMA
-------------------------
O Horus AI é uma solução de monitoramento de vídeo em tempo real projetada para ingerir streams SRT (Secure Reliable Transport), processá-los para visualização web (HLS) e executar inferência de Machine Learning (áudio e vídeo) para detecção de anomalias/ocorrências.

O sistema é composto por:
- Backend: API REST em Python (FastAPI) que gerencia processos de ingestão (ffmpeg), inferência (TensorFlow/Keras) e persistência.
- Frontend: Aplicação Angular (v19) para dashboard de monitoramento, visualização de vídeo e gestão de ocorrências.

2. STACK TECNOLÓGICO
--------------------
Backend:
- Linguagem: Python 3.11+
- Framework Web: FastAPI (Uvicorn como servidor ASGI).
- ML/AI: TensorFlow/Keras (.h5/.keras) e TFLite (.tflite).
- Processamento de Imagem: OpenCV (cv2).
- Processamento de Áudio: Librosa, SoundFile, Audioread.
- Manipulação de Mídia: FFmpeg (via subprocesso).
- Banco de Dados: PostgreSQL (via SQLAlchemy/Alembic) - Opcional/Configurável.

Frontend:
- Framework: Angular 19.
- Visualização de Dados: ApexCharts.
- Comunicação: HTTP REST e WebSocket (para alertas em tempo real).
- Player de Vídeo: HLS.js (nativo ou via biblioteca wrapper).

3. ARQUITETURA DETALHADA DO BACKEND
-----------------------------------

3.1. Módulo de Ingestão (`backend/app/streams/srt_reader.py`)
- Responsabilidade: Conectar ao stream SRT remoto, gerar playlist HLS para o frontend e extrair frames para a IA.
- Mecanismo:
  - Inicia um processo `ffmpeg` persistente que recebe SRT e outputa:
    1. Segmentos HLS (`.ts` + `.m3u8`) em `backend/static/hls/`.
    2. Frames JPEG (amostragem definida, ex: 1fps) em diretório temporário.
- Tratamento de Erros de Stream:
  - O código utiliza flags tolerantes do ffmpeg (`-err_detect ignore_err`, `+discardcorrupt`, `+genpts+igndts`) para lidar com streams SRT instáveis que podem apresentar perda de pacotes, falta de PPS (Picture Parameter Sets) ou referências H.264 quebradas.

3.2. Pipeline de Inferência (`backend/app/ml/inference.py`)
- Carregamento de Modelos:
  - Suporta modelos Keras (.h5/.keras) e TFLite.
  - Detecta automaticamente a dimensionalidade de entrada do modelo de vídeo.
    - 4D: `(batch, height, width, channels)` - Inferência frame a frame.
    - 5D: `(batch, frames, height, width, channels)` - Inferência baseada em sequência temporal.
- Lógica de Sequência (Vídeo):
  - Mantém um buffer deslizante (`deque`) dos últimos N frames.
  - Estratégia de Padding: Se o modelo exige N frames e o buffer tem menos (início do stream), o sistema repete o último frame para preencher o tensor (Padding) e permitir inferência de baixa latência, em vez de bloquear aguardando acumulação.
- Heurísticas (Pré-ML):
  - Calcula métricas clássicas por frame antes da inferência pesada:
    - Blur (Variância Laplaciana).
    - Brilho (Média de intensidade).
    - Movimento (Diferença absoluta entre frames consecutivos convertidos para grayscale).
    - Densidade de Bordas (Canny).

3.3. Geração de Clipes e Áudio
- Criação de Clipes (.mp4):
  - Acionado quando uma ocorrência é detectada.
  - Snapshotting: Para evitar "Race Conditions" (onde o ffmpeg tenta ler um frame que ainda está sendo escrito pelo ingestor), o sistema copia os frames relevantes para um diretório de snapshot temporário antes de invocar o ffmpeg para muxing.
- Extração de Áudio:
  - Tenta carregar via `librosa`.
  - Fallback Robusto: Se `librosa` falhar (comum em ambientes Windows sem backend de áudio configurado), invoca `ffmpeg` via subprocesso para extrair WAV.
  - Tratamento de "Sem Áudio": Se o stream original não possuir trilha de áudio, o sistema captura o erro do ffmpeg e retorna um buffer de áudio vazio (tratado como operação normal, sem crash).

3.4. Comunicação em Tempo Real
- WebSocket (`/ws/ocorrencias`):
  - O backend mantém um `ConnectionManager`.
  - Novas ocorrências detectadas são serializadas e enviadas via broadcast para todos os clientes conectados (Frontend).

4. ARQUITETURA DO FRONTEND
--------------------------
- Página de Monitoramento:
  - Exibe o player HLS apontando para o backend.
  - Conecta ao WebSocket para receber alertas instantâneos.
  - Gráficos de série temporal (ApexCharts) atualizados em tempo real.
  - Lógica de Deduplicação: O frontend possui lógica para evitar duplicar ocorrências visuais caso o backend reenvie estados ou haja reconexão.

5. PONTOS CRÍTICOS E MANUTENÇÃO (CONTEXTO PARA O PRÓXIMO ENGENHEIRO)
--------------------------------------------------------------------
1. Compatibilidade Keras/TensorFlow:
   - O código foi ajustado para lidar explicitamente com inputs 4D vs 5D. Se o modelo for trocado, verifique os logs de inicialização onde `model.inputs` é impresso. O código atual faz coerção automática (reshape) se um frame único for passado para um modelo de sequência.

2. FFmpeg e Windows:
   - O ambiente de desenvolvimento é Windows. O caminho do `ffmpeg` deve estar no PATH do sistema.
   - Erros de "Permission Denied" ou "File not found" em arquivos temporários foram mitigados com o uso de diretórios de snapshot e limpeza (cleanup) robusta no `srt_reader.py`.

3. Logs e Debugging:
   - Logs do FFmpeg de ingestão são salvos em `backend/static/hls/hls_ffmpeg.log`.
   - Logs da aplicação (Uvicorn) mostram a saída padrão da inferência.
   - Warnings do Keras (barras de progresso) e Librosa foram suprimidos no código para limpar o console, mas podem ser reativados removendo os contextos `warnings.catch_warnings` e `verbose=0` em `inference.py`.

4. Variáveis de Ambiente (.env):
   - `VIDEO_THRESH_*`: Define limiares de confiança por classe.
   - `VIDEO_VOTE_K`: Janela de votação para suavizar detecções instáveis.

6. PRÓXIMOS PASSOS SUGERIDOS (ROADMAP TÉCNICO)
----------------------------------------------
- Configuração Dinâmica: Mover a política de "Padding de Sequência" (repetir frames vs esperar buffer encher) e "Comportamento Sem Áudio" para variáveis de ambiente.
- Retry/Backoff no Ingest: Implementar reconexão automática robusta caso o servidor SRT remoto caia.
- Testes Automatizados: Adicionar testes unitários para a lógica de bufferização em `inference.py` para garantir que a coerção 4D/5D permaneça estável em futuras atualizações de lib.

Este documento serve como base de conhecimento para portabilidade e manutenção evolutiva do Horus AI.
