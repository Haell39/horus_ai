from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, status, BackgroundTasks
from typing import Optional
import os
import uuid
import shutil
import subprocess
from datetime import datetime, timedelta, timezone

from app.db.base import get_db, SessionLocal
from sqlalchemy.orm import Session
from app.db import models, schemas
from app.ml import inference
from app.core import storage as storage_core

router = APIRouter()

print("INFO: analysis router module imported and ready")

# Configuráveis (podem ser movidos para core.config mais tarde)
# NOTE: main.py monta /clips a partir de backend/static/clips (um nível acima de 'app').
# Use a mesma pasta que o main por padrão, mas permita override via storage config.
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'uploads'))
os.makedirs(UPLOAD_DIR, exist_ok=True)


def _get_duration_seconds(path: str) -> float:
    """Tenta obter a duração do arquivo usando ffprobe; retorna 0 em falha."""
    try:
        proc = subprocess.run(
            [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', path
            ], capture_output=True, text=True, timeout=10
        )
        out = proc.stdout.strip()
        return float(out) if out else 0.0
    except Exception:
        return 0.0


@router.post('/analysis/upload', summary='Envia um arquivo de vídeo para análise')
async def upload_analysis(
    file: UploadFile = File(...),
    fps: Optional[float] = Form(None),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    # Valida tipo
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Arquivo não é um vídeo')

    # Salva temporário
    ext = os.path.splitext(file.filename)[1] or '.mp4'
    uid = uuid.uuid4().hex
    tmp_name = f"upload_{uid}{ext}"
    tmp_path = os.path.join(UPLOAD_DIR, tmp_name)
    try:
        with open(tmp_path, 'wb') as out_f:
            shutil.copyfileobj(file.file, out_f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Erro ao salvar arquivo: {e}')

    # Tamanho e duração
    try:
        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    except Exception:
        size_mb = 0.0
    duration_s = _get_duration_seconds(tmp_path)

    # Limites (simples - podem ser configuráveis)
    HARD_LIMIT_MB = 1024
    SYNC_LIMIT_MB = 50
    SYNC_LIMIT_SECONDS = 30

    if size_mb > HARD_LIMIT_MB:
        os.remove(tmp_path)
        raise HTTPException(status_code=413, detail='Arquivo maior que o limite permitido (1 GB)')

    # Move para clips para poder servir via /clips (clips_dir pode ser configurado)
    clip_name = f"upload_clip_{uid}{ext}"
    clips_dir = storage_core.get_clips_dir()
    clip_path = os.path.join(clips_dir, clip_name)
    try:
        shutil.copy2(tmp_path, clip_path)
        print(f"DEBUG: upload_analysis -> clip copiado para: {clip_path}")
    except Exception:
        # fallback: try move
        try:
            shutil.move(tmp_path, clip_path)
            print(f"DEBUG: upload_analysis -> clip movido para: {clip_path}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Erro ao preparar clip: {e}')

    # Decide sync vs async (MVP: inline processing if pequeno)
    process_inline = (size_mb <= SYNC_LIMIT_MB and duration_s <= SYNC_LIMIT_SECONDS)

    if process_inline:
        # Processa video e, somente se detectar falha de interesse, grava ocorrência.
        try:
            pred_class, confidence, event_time = inference.analyze_video_frames(clip_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Erro na inferência: {e}')

        # Limiar síncrono (alinha com lógica do endpoint /analyze)
        CONFIDENCE_THRESHOLD = 0.60
        # Timestamp UTC para registrar início/fim da ocorrência quando necessário
        now = datetime.now(timezone.utc)

        # Se for 'normal' ou confiança abaixo do limiar, não criamos ocorrência
        if pred_class == 'normal' or (confidence or 0.0) < CONFIDENCE_THRESHOLD:
            # Opcional: remove o clip público salvo, pois não gerou ocorrência
            try:
                if os.path.exists(clip_path):
                    os.remove(clip_path)
            except Exception:
                pass
            return {
                'status': 'ok',
                'message': f'Arquivo analisado — sem falhas detectadas (classe={pred_class}, conf={confidence:.3f}).'
            }

            # Caso haja falha com confiança suficiente, grava ocorrência.
            # Preferimos recortar um trecho curto ao redor do evento (2s antes/2s depois)
            # se tivermos timestamp, para não salvar o arquivo inteiro por padrão.
        clip_to_save = clip_path
        before_s = 2.0
        after_s = 2.0
        try:
            if event_time is not None:
                start = max(0.0, event_time - before_s)
                duration_cut = before_s + after_s
                dest_name = f"clip_{uid}_cut{ext}"
                dest_path = os.path.join(storage_core.get_clips_dir(), dest_name)
                try:
                    subprocess.run([
                        'ffmpeg', '-y', '-ss', f"{start}", '-t', f"{duration_cut}",
                        '-i', clip_path, '-c', 'copy', dest_path
                    ], check=True, capture_output=True, timeout=60)
                    clip_to_save = dest_path
                    print(f"DEBUG: upload_analysis -> recorte inline salvo em: {clip_to_save}")
                except Exception as e:
                    print(f"DEBUG: upload_analysis -> falha ao recortar inline via ffmpeg: {e} — usando clip inteiro")
                    clip_to_save = clip_path

            # Garante que o clipe também esteja disponível na pasta pública montada em /clips
            try:
                # public clips are served from backend/static/clips (one level above app/)
                public_clips_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'static', 'clips'))
                os.makedirs(public_clips_dir, exist_ok=True)
                public_path = os.path.join(public_clips_dir, os.path.basename(clip_to_save))
                if os.path.abspath(clip_to_save) != os.path.abspath(public_path):
                    try:
                        shutil.copy2(clip_to_save, public_path)
                        clip_to_serve = public_path
                    except Exception as e:
                        # Log full error and attempt a second copy attempt to help diagnose transient issues
                        print(f"DEBUG: upload_analysis -> falha ao copiar para pasta pública: {e}")
                        try:
                            # second attempt
                            shutil.copy(clip_to_save, public_path)
                            clip_to_serve = public_path
                        except Exception as e2:
                            print(f"DEBUG: upload_analysis -> segunda tentativa de cópia falhou: {e2}")
                            # fallback: serve the original saved clip (may be outside static mount)
                            clip_to_serve = clip_to_save
                else:
                    clip_to_serve = clip_to_save
            except Exception as e:
                print(f"DEBUG: upload_analysis -> erro preparando public_path: {e}")
                clip_to_serve = clip_to_save

            clip_basename = os.path.basename(clip_to_serve)
            clip_dur_saved = _get_duration_seconds(clip_to_serve)
            # calcula severidade e duração (mesma lógica usada pelo background worker)
            try:
                dur_calc, severity = calcular_severidade_e_duracao(clip_to_save)
            except Exception:
                dur_calc, severity = (clip_dur_saved or duration_s or 0.0), 'Leve (C)'

            oc = schemas.OcorrenciaCreate(
                start_ts=now - timedelta(seconds=dur_calc or duration_s or 0),
                end_ts=now,
                duration_s=dur_calc or clip_dur_saved or duration_s,
                category='video-file',
                type=pred_class,
                severity=severity,
                confidence=float(confidence or 0.0),
                evidence={'clip_path': f'/clips/{clip_basename}', 'clip_duration_s': float(clip_dur_saved or duration_s), 'event_window': {'before_margin_s': (before_s if event_time is not None else 0.0), 'after_margin_s': (after_s if event_time is not None else 0.0)}}
            )

            try:
                db_oc = models.Ocorrencia(**oc.dict())
                db.add(db_oc)
                db.commit()
                db.refresh(db_oc)
                print(f"DEBUG: upload_analysis -> ocorrência criada id={db_oc.id} com evidência {oc.evidence}")
                return db_oc
            except Exception as e:
                db.rollback()
                raise HTTPException(status_code=500, detail=f'Erro ao salvar ocorrência: {e}')
        except Exception:
            # Re-raise to let outer handler return 500
            raise

    else:
        # Agenda processamento em background (usando BackgroundTasks do FastAPI).
        job_id = uuid.uuid4().hex

        def _process_queued(clip_path_local: str, clip_name_local: str, job_id_local: str):
            """Função síncrona executada em background pelo FastAPI; usa SessionLocal
            para abrir uma sessão e processar o arquivo, recortar evento e salvar ocorrência.
            """
            print(f"Background worker: iniciando job {job_id_local} para {clip_path_local}")
            db_session = SessionLocal()
            try:
                try:
                    pred_class, confidence, event_time = inference.analyze_video_frames(clip_path_local)
                except Exception as e:
                    print(f"Background worker: falha na inferência job {job_id_local}: {e}")
                    return

                CONFIDENCE_THRESHOLD = 0.60
                if pred_class == 'normal' or (confidence or 0.0) < CONFIDENCE_THRESHOLD:
                    print(f"Background worker: job {job_id_local} - sem falhas detectadas (class={pred_class} conf={confidence})")
                    return

                # Recorta segmento ao redor do evento se tivermos timestamp
                before_s = 2.0
                after_s = 2.0
                if event_time is None:
                    # fallback: usa o clip inteiro
                    clip_to_save = clip_path_local
                else:
                    start = max(0.0, event_time - before_s)
                    duration = before_s + after_s
                    dest_name = f"clip_{job_id_local}{os.path.splitext(clip_name_local)[1]}"
                    dest_path = os.path.join(storage_core.get_clips_dir(), dest_name)
                    # ffmpeg -ss START -t DURATION -i INPUT -c copy OUTPUT
                    try:
                        subprocess.run([
                            'ffmpeg', '-y', '-ss', f"{start}", '-t', f"{duration}",
                            '-i', clip_path_local, '-c', 'copy', dest_path
                        ], check=True, capture_output=True, timeout=60)
                        clip_to_save = dest_path
                        print(f"DEBUG: Background worker -> recorte salvo em: {clip_to_save}")
                    except Exception as e:
                        print(f"Background worker: falha ao recortar com ffmpeg: {e}. Usando clip inteiro.")
                        clip_to_save = clip_path_local

                # --- Tolerance check: evitar falsos positivos curtos (ex: fade) ---
                # Re-run inference on the small cut (if available) to confirm event persists
                try:
                    if event_time is not None and clip_to_save and clip_to_save != clip_path_local:
                        try:
                            pred2, conf2, _ = inference.analyze_video_frames(clip_to_save)
                        except Exception as e:
                            print(f"Background worker: falha re-analisar trecho cortado: {e}")
                            pred2, conf2 = pred_class, confidence
                        # Se a reanálise não confirmar (mesma classe com confiança suficiente), descarta como transitório
                        if pred2 != pred_class or (conf2 or 0.0) < CONFIDENCE_THRESHOLD:
                            print(f"Background worker: job {job_id_local} - evento transitório detectado (pred2={pred2}, conf2={conf2}) - ignorando.")
                            return
                except Exception as _:
                    pass

                # calcula duração do clip salvo
                clip_dur = _get_duration_seconds(clip_to_save)
                duration_sec = clip_dur or duration_s

                print(f"DEBUG: Background worker -> clip final a salvar: {clip_to_save} (dur={duration_sec}s)")

                # calcula severidade com a função existente
                dur_calc, severity = calcular_severidade_e_duracao(clip_to_save)

                # monta e salva ocorrência
                try:
                    now = datetime.now(timezone.utc)
                    start_ts_calc = now - timedelta(seconds=duration_sec)

                    # garante que o clipe esteja disponível na pasta pública /clips
                    try:
                        # same public path as main.py mounts (/clips -> backend/static/clips)
                        public_clips_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'static', 'clips'))
                        os.makedirs(public_clips_dir, exist_ok=True)
                        public_path = os.path.join(public_clips_dir, os.path.basename(clip_to_save))
                        if os.path.abspath(clip_to_save) != os.path.abspath(public_path):
                            try:
                                shutil.copy2(clip_to_save, public_path)
                                clip_to_serve = public_path
                            except Exception as e:
                                print(f"DEBUG: Background worker -> falha ao copiar para pasta pública: {e}")
                                try:
                                    shutil.copy(clip_to_save, public_path)
                                    clip_to_serve = public_path
                                except Exception as e2:
                                    print(f"DEBUG: Background worker -> segunda tentativa de cópia falhou: {e2}")
                                    clip_to_serve = clip_to_save
                        else:
                            clip_to_serve = clip_to_save
                    except Exception as e:
                        print(f"DEBUG: Background worker -> erro preparando public_path: {e}")
                        clip_to_serve = clip_to_save

                    evidence_dict = {
                        'clip_path': f'/clips/{os.path.basename(clip_to_serve)}',
                        'model': inference.VIDEO_MODEL_FILENAME if hasattr(inference, 'VIDEO_MODEL_FILENAME') else 'video',
                        'original_filename': clip_name_local,
                        'confidence_raw': float(confidence),
                        'clip_duration_s': float(duration_sec),
                        'event_window': {'before_margin_s': before_s, 'after_margin_s': after_s}
                    }

                    oc_data = db_schemas.OcorrenciaCreate(
                        start_ts=start_ts_calc, end_ts=now, duration_s=duration_sec,
                        category='Video Arquivo', type=pred_class, severity=severity,
                        confidence=confidence, evidence=evidence_dict
                    )
                    db_oc = models.Ocorrencia(**oc_data.dict())
                    db_session.add(db_oc)
                    db_session.commit()
                    db_session.refresh(db_oc)

                    # envia via websocket
                    try:
                        ocorrencia_read = db_schemas.OcorrenciaRead.from_orm(db_oc)
                        payload = {'type': 'nova_ocorrencia', 'data': ocorrencia_read.dict()}
                        # manager.broadcast_json is async; use asyncio.run to ensure it runs
                        try:
                            import asyncio
                            asyncio.run(manager.broadcast_json(payload))
                        except Exception as e:
                            print(f"Background worker: falha ao enviar WS (async): {e}")
                    except Exception as e:
                        print(f"Background worker: falha ao preparar/enviar WS: {e}")

                except Exception as e:
                    db_session.rollback()
                    print(f"Background worker: falha ao salvar ocorrência job {job_id_local}: {e}")
            finally:
                try:
                    db_session.close()
                except Exception:
                    pass

        # agenda execução em background
        try:
            background_tasks.add_task(_process_queued, clip_path, clip_name, job_id)
        except Exception:
            # se BackgroundTasks falhar por algum motivo, avisamos o usuário
            return {
                'status': 'queued',
                'message': 'Arquivo aceito, mas falha ao agendar processamento em background.',
                'clip_url': f'/clips/{clip_name}',
                'size_mb': round(size_mb, 2),
                'duration_s': round(duration_s, 2)
            }

        return {
            'status': 'queued',
            'job_id': job_id,
            'message': 'Arquivo aceito para processamento em segundo plano.',
            'clip_url': f'/clips/{clip_name}',
            'size_mb': round(size_mb, 2),
            'duration_s': round(duration_s, 2)
        }
# backend/app/api/endpoints/analysis.py
# (Versão TFLite com Upload e Análise Multi-Segmento - CORRIGIDO)
from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks, UploadFile, File, Form
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import os
import tempfile
import time
# === CORREÇÃO IMPORT MOVIEPY ===
from moviepy.video.io.VideoFileClip import VideoFileClip # Import específico
# === CORREÇÃO IMPORT TYPING ===
from typing import Tuple, Optional, List # Adicionado Tuple
import traceback

# Importa nossa lógica de ML e schemas
from app.ml import inference, schemas as ml_schemas
# Importa coisas do banco
from app.db import schemas as db_schemas
from app.db import models
from app.db.base import get_db
from app.websocket_manager import manager
import shutil

# Reuse the module-level `router` defined above so all routes register on the
# same APIRouter instance. Avoid reassigning `router` which would orphan
# previously-declared routes (this caused `/analysis/upload` to disappear).
# router = APIRouter()

# Função para obter duração e calcular severidade
# A assinatura agora usa o 'Tuple' importado
def calcular_severidade_e_duracao(clip_path: str) -> Tuple[float, str]:
    """ Calcula a duração do clipe e determina a severidade baseada na Cartilha Globo. """
    duration_s = 0.0
    severity = "Leve (C)" # Padrão
    try:
        print(f"DEBUG: Calculando duração para: {clip_path}")
        if not os.path.exists(clip_path) or os.path.getsize(clip_path) == 0:
             print(f"AVISO: Arquivo não encontrado ou vazio: {clip_path}. Duração = 0s.")
             duration_s = 0.0
        else:
             try:
                 # Usa o VideoFileClip importado diretamente
                 with VideoFileClip(clip_path) as clip:
                     duration_s = clip.duration if clip.duration is not None else 0.0
                 print(f"DEBUG: Duração MoviePy: {duration_s:.2f}s")
             except Exception as moviepy_err:
                  print(f"AVISO: Erro MoviePy ({clip_path}): {moviepy_err}. Duração = 0s.")
                  duration_s = 0.0

    except Exception as e:
        print(f"AVISO: Falha geral duração ({clip_path}): {e}. Duração = 0s.")
        duration_s = 0.0

    # Garante float não negativo
    try:
      duration_s = float(duration_s)
      if duration_s < 0: duration_s = 0.0
    except (ValueError, TypeError):
       duration_s = 0.0

    # <<< LÓGICA DA CARTILHA GLOBO >>>
    if duration_s >= 60: severity = "Gravíssima (X)"
    elif duration_s >= 10: severity = "Grave (A)"
    elif duration_s >= 5: severity = "Média (B)"
    else: severity = "Leve (C)"

    print(f"DEBUG: Duração final: {duration_s:.2f}s -> Severidade: {severity}")
    return duration_s, severity

# Função de background para salvar ocorrência (Mantida)
# Função de background para salvar ocorrência (MODIFICADA)
async def salvar_ocorrencia_task(db: Session, ocorrencia_data: db_schemas.OcorrenciaCreate):
    """ Salva a ocorrência no banco E envia via WebSocket. """
    db_ocorrencia = None # Inicializa para saber se salvou
    try:
        print(f"Background Task: Iniciando salvamento: {ocorrencia_data.type}")
        # Cria instância SQLAlchemy
        db_ocorrencia = models.Ocorrencia(**ocorrencia_data.dict())
        db.add(db_ocorrencia)
        db.commit()
        db.refresh(db_ocorrencia) # Pega ID e created_at
        print(f"Background Task: Ocorrência ID {db_ocorrencia.id} ({db_ocorrencia.type}) salva.")

        # <<< NOVA PARTE: ENVIA VIA WEBSOCKET >>>
        if db_ocorrencia:
            # Converte o objeto SQLAlchemy para um schema Pydantic Read (que é serializável)
            ocorrencia_read = db_schemas.OcorrenciaRead.from_orm(db_ocorrencia)
            # Cria um objeto JSON para enviar (podemos simplificar se necessário)
            payload = {
                "type": "nova_ocorrencia", # Tipo da mensagem
                "data": ocorrencia_read.dict() # Converte o Pydantic para dict
            }
            print(f"DEBUG: Enviando broadcast WebSocket: {payload['type']} ID {ocorrencia_read.id}")
            await manager.broadcast_json(payload) # Envia para todos conectados
        # <<< FIM DA NOVA PARTE >>>

    except Exception as e:
        db.rollback()
        print(f"Background Task ERRO: Falha ao salvar/transmitir '{ocorrencia_data.type}': {e}")
        traceback.print_exc()

@router.post(
    "/analyze",
    response_model=ml_schemas.AnalysisOutput,
    summary="Analisa um clipe de mídia (áudio ou vídeo)"
)
async def analyze_media(
    # (Parâmetros mantidos: media_type, file, background_tasks, db)
    media_type: str = Form(..., pattern="^(audio|video)$"),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    """
    Recebe upload, salva temporariamente, executa ANÁLISE MULTI-SEGMENTO/FRAME,
    e agenda salvamento se detectar falha.
    """
    # (Verificação inicial de modelos carregados mantida)
    model_is_available = False
    if media_type == 'audio' and inference.audio_interpreter: model_is_available = True
    elif media_type == 'video' and inference.video_interpreter: model_is_available = True
    if not inference.models_loaded or not model_is_available:
         raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Modelo TFLite para '{media_type}' não carregado.")

    print(f"INFO: Recebido '{file.filename}' ({media_type}). Tamanho: {file.size} bytes.")
    temp_file_path = None
    results: List[ml_schemas.PredictionResult] = []
    error_msg: Optional[str] = None
    ocorrencia_salva = False
    message = "Análise iniciada."

    try:
        # --- Salva Arquivo Temporário (Mantido) ---
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".tmp"
        temp_dir = tempfile.gettempdir(); os.makedirs(temp_dir, exist_ok=True)
        fd, temp_file_path = tempfile.mkstemp(suffix=suffix, dir=temp_dir)
        with os.fdopen(fd, 'wb') as temp_file:
             content = await file.read();
             if not content: raise ValueError("Arquivo vazio.")
             temp_file.write(content)
        print(f"DEBUG: Arquivo salvo em: {temp_file_path}")

        # --- Pré-processamento e Inferência Multi-Segmento ---
        start_time = time.time()
        pred_class: str = "error"
        confidence: float = 0.0
        model_name: str = "unknown"

        if media_type == 'audio':
            pred_class, confidence = inference.analyze_audio_segments(temp_file_path)
            model_name = inference.AUDIO_MODEL_FILENAME
            event_time = None
        elif media_type == 'video':
            pred_class, confidence, event_time = inference.analyze_video_frames(temp_file_path)
            model_name = inference.VIDEO_MODEL_FILENAME

        results.append(ml_schemas.PredictionResult(
            predicted_class=pred_class, confidence=confidence, model_name=model_name
        ))
        end_time = time.time()
        message = f"Análise {media_type} concluída em {(end_time - start_time):.2f}s. Resultado: {pred_class} ({confidence*100:.1f}%)"
        print(f"INFO: {message}")

        # --- Lógica de Decisão e Salvamento (Mantida) ---
        CONFIDENCE_THRESHOLD = 0.60 # Limiar (AJUSTE)

        if pred_class not in ['normal', "Erro_Indice", "Erro_Inferência", "Erro_Análise_Áudio", "Erro_Análise_Vídeo", "Erro_Abertura_Vídeo"] and confidence >= CONFIDENCE_THRESHOLD:
            print(f"ALERTA: Falha '{pred_class}' detectada ({confidence:.2f})!")
            duration_s, severity = calcular_severidade_e_duracao(temp_file_path)

            if duration_s <= 0:
                 print(f"AVISO: Duração inválida ({duration_s:.2f}s). Ocorrência '{pred_class}' NÃO salva.")
                 message += f". Duração inválida, ocorrência não salva."
            else:
                # (Código para criar ocorrencia_data e agendar background_tasks mantido)
                end_ts_now = datetime.now(timezone.utc)
                start_ts_calc = end_ts_now - timedelta(seconds=duration_s)

                # Move o arquivo temporário para a pasta pública de clipes
                try:
                    # Ensure we use the configured clips folder (falling back to static/clips)
                    clips_dir = storage_core.get_clips_dir()
                    os.makedirs(clips_dir, exist_ok=True)

                    # Se tiver event_time, recorta trecho curto ao redor do evento
                    before_s = 2.0
                    after_s = 2.0
                    if media_type == 'video' and event_time:
                        dest_name = f"clip_{int(time.time())}_{os.path.basename(temp_file_path)}"
                        dest_path = os.path.join(clips_dir, dest_name)
                        start = max(0.0, event_time - before_s)
                        duration_cut = before_s + after_s
                        try:
                            subprocess.run([
                                'ffmpeg', '-y', '-ss', f"{start}", '-t', f"{duration_cut}",
                                '-i', temp_file_path, '-c', 'copy', dest_path
                            ], check=True, capture_output=True, timeout=60)
                            clip_saved_path = dest_path
                        except Exception as ff_err:
                            print(f"AVISO: Falha recortar evento via ffmpeg: {ff_err}. Salvando arquivo inteiro.")
                            dest_name = f"clip_{int(time.time())}_{os.path.basename(temp_file_path)}"
                            clip_saved_path = os.path.join(clips_dir, dest_name)
                            shutil.move(temp_file_path, clip_saved_path)
                            temp_file_path = None

                        # Tolerance: re-analisar recorte para evitar falsos positivos curtos
                        try:
                            pred2, conf2, _ = inference.analyze_video_frames(clip_saved_path)
                            if pred2 != pred_class or (conf2 or 0.0) < 0.60:
                                print(f"ANALYZE: recorte não confirmou evento (pred2={pred2} conf2={conf2}) -> não salva ocorrência.")
                                # remove recorte para não poluir clips
                                try:
                                    if os.path.exists(clip_saved_path):
                                        os.remove(clip_saved_path)
                                except Exception:
                                    pass
                                raise ValueError('Evento transitório - não confirmado')
                        except Exception as e:
                            # Se a verificação não passar, consideramos que não houve evento
                            print(f"ANALYZE: verificação de tolerância falhou/descartou evento: {e}")
                            raise
                    else:
                        dest_name = f"clip_{int(time.time())}_{os.path.basename(temp_file_path)}"
                        clip_saved_path = os.path.join(clips_dir, dest_name)
                        shutil.move(temp_file_path, clip_saved_path)
                        temp_file_path = None

                    # monta evidence dict
                    clip_dur_calc = _get_duration_seconds(clip_saved_path)
                    evidence_dict = {
                        "path": f"/clips/{os.path.basename(clip_saved_path)}",
                        "model": model_name,
                        "original_filename": file.filename,
                        "confidence_raw": float(confidence),
                        "clip_duration_s": float(clip_dur_calc or duration_s),
                        "event_window": {"before_margin_s": (before_s if media_type == 'video' else 0.0), "after_margin_s": (after_s if media_type == 'video' else 0.0)},
                    }
                except Exception as mv_err:
                    print(f"AVISO: Falha mover/recortar arquivo temp para clips: {mv_err}")
                    # Fallback: mantém o caminho temporário (pode não existir após finally)
                    evidence_dict = {
                        "clip_path_temp": temp_file_path,
                        "model": model_name,
                        "original_filename": file.filename,
                        "confidence_raw": float(confidence),
                    }
                ocorrencia_data = db_schemas.OcorrenciaCreate(
                    start_ts=start_ts_calc, end_ts=end_ts_now, duration_s=duration_s,
                    category=f"{media_type.capitalize()} Técnico", type=pred_class,
                    severity=severity, confidence=confidence, evidence=evidence_dict
                )
                background_tasks.add_task(salvar_ocorrencia_task, db, ocorrencia_data.copy(deep=True))
                message += f". Falha '{pred_class}' ({severity}). Agendando salvamento."
                ocorrencia_salva = True
                print(f"INFO: Task salvar '{pred_class}' agendada.")
        else:
             message += f". Classe '{pred_class}' não acionou salvamento (Conf: {confidence:.2f}, Limiar: {CONFIDENCE_THRESHOLD})."
             print(f"INFO: {message}")

    except Exception as e:
        error_msg = f"Erro fatal análise '{file.filename}': {e}"
        message = error_msg
        print(f"ERRO: {error_msg}")
        traceback.print_exc()
        results = []

    finally:
        # (Limpeza do arquivo temporário mantida)
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print(f"DEBUG: Temp removido: {temp_file_path}")
            except Exception as unlink_err:
                print(f"AVISO: Falha remover temp {temp_file_path}: {unlink_err}")

    # Retorna o resultado
    return ml_schemas.AnalysisOutput(
        filename=file.filename, media_type=media_type, predictions=results,
        ocorrencia_salva=ocorrencia_salva, message=message
    )

print("INFO: Endpoint de Análise (/analyze) TFLite (Multi-Segmento) configurado.")