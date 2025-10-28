# backend/app/api/endpoints/analysis.py
# (Versão TFLite com Upload - Passo 5)

from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks, UploadFile, File, Form
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import os
import tempfile
import time
# moviepy 2.2+ doesn't provide `moviepy.editor`; import VideoFileClip directly
from moviepy.video.io.VideoFileClip import VideoFileClip  # Para pegar a duração
import traceback # Para log de erro
from typing import Tuple, List, Optional
import numpy as np

# Importa nossa lógica de ML e schemas
from app.ml import inference, schemas as ml_schemas
# Importa coisas do banco
from app.db import schemas as db_schemas
from app.db import models
from app.db.base import get_db

router = APIRouter()

# Função para obter duração e calcular severidade
def calcular_severidade_e_duracao(clip_path: str) -> Tuple[float, str]:
    """ Calcula a duração do clipe e determina a severidade baseada na Cartilha Globo. """
    duration_s = 0.0
    severity = "Leve (C)" # Padrão
    try:
        # Tenta obter a duração usando MoviePy
        print(f"DEBUG: Calculando duração para: {clip_path}")
        # Verifica se o arquivo existe antes de tentar abrir
        if not os.path.exists(clip_path):
             print(f"AVISO: Arquivo não encontrado para cálculo de duração: {clip_path}. Usando fallback (5s).")
             duration_s = 5.0
        else:
             with VideoFileClip(clip_path) as clip:
                 duration_s = clip.duration if clip.duration else 0.0
             print(f"DEBUG: Duração obtida com MoviePy: {duration_s:.2f}s")

    except Exception as e:
        print(f"AVISO: Falha ao obter duração com MoviePy ({clip_path}): {e}. Usando fallback (5s).")
        # Fallback em caso de erro do MoviePy
        duration_s = 5.0

    # Garante que duration_s seja um float válido
    try:
      duration_s = float(duration_s)
      if duration_s < 0: # Duração não pode ser negativa
          print(f"AVISO: Duração negativa '{duration_s}', usando fallback 5.0s")
          duration_s = 5.0
    except (ValueError, TypeError):
       print(f"AVISO: Duração inválida '{duration_s}', usando fallback 5.0s")
       duration_s = 5.0


    # <<< LÓGICA DA CARTILHA GLOBO >>>
    if duration_s >= 60:
        severity = "Gravíssima (X)"
    elif duration_s >= 10: # 10s a 59.99...s
        severity = "Grave (A)"
    elif duration_s >= 5: # 5s a 9.99...s
        severity = "Média (B)"
    else: # 0s a 4.99...s
        severity = "Leve (C)"

    print(f"DEBUG: Duração final: {duration_s:.2f}s -> Severidade: {severity}")
    return duration_s, severity

# Função de background para salvar ocorrência
async def salvar_ocorrencia_task(db: Session, ocorrencia_data: db_schemas.OcorrenciaCreate):
    """ Salva a ocorrência no banco de dados. """
    try:
        print(f"Background Task: Iniciando salvamento da ocorrência: {ocorrencia_data.type}")
        # Cria a instância do modelo SQLAlchemy
        db_ocorrencia = models.Ocorrencia(
            start_ts=ocorrencia_data.start_ts,
            end_ts=ocorrencia_data.end_ts,
            duration_s=ocorrencia_data.duration_s,
            category=ocorrencia_data.category,
            type=ocorrencia_data.type,
            severity=ocorrencia_data.severity,
            confidence=ocorrencia_data.confidence,
            evidence=ocorrencia_data.evidence
            # created_at é DEFAULT now() no banco
        )
        db.add(db_ocorrencia)
        db.commit() # Salva as mudanças
        db.refresh(db_ocorrencia) # Pega o ID gerado pelo banco
        print(f"Background Task: Ocorrência ID {db_ocorrencia.id} ({db_ocorrencia.type}) salva com sucesso.")
    except Exception as e:
        db.rollback() # Desfaz em caso de erro
        print(f"Background Task ERRO: Falha ao salvar ocorrência tipo '{ocorrencia_data.type}': {e}")
        traceback.print_exc()

@router.post(
    "/analyze",
    response_model=ml_schemas.AnalysisOutput,
    summary="Analisa um clipe de mídia (áudio ou vídeo)"
)
async def analyze_media(
    # Recebe os dados como 'form-data'
    media_type: str = Form(..., pattern="^(audio|video)$"),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db) # Injeta a sessão do DB
):
    """
    Recebe um upload de arquivo, salva temporariamente, executa o modelo TFLite,
    e se detectar uma falha, agenda o salvamento da ocorrência no banco.
    """
    # Verifica se os modelos foram carregados corretamente no início
    if not inference.models_loaded or (media_type == 'audio' and not inference.audio_interpreter) or (media_type == 'video' and not inference.video_interpreter):
        message = f"Erro: Modelo TFLite para '{media_type}' não está carregado."
        print(f"ERRO: {message}")
        # Retorna um erro 503 Service Unavailable, mas no formato AnalysisOutput
        return ml_schemas.AnalysisOutput(
            filename=file.filename,
            media_type=media_type,
            predictions=[],
            message=message,
            ocorrencia_salva=False
        )

    print(f"INFO: Recebido upload '{file.filename}' para análise ({media_type}). Tamanho: {file.size} bytes.")
    temp_file_path = None # Inicializa variável
    results: List[ml_schemas.PredictionResult] = []
    error_msg: Optional[str] = None
    ocorrencia_salva = False
    message = "Análise iniciada."

    try:
        # Salva o arquivo temporariamente de forma segura
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".tmp"
        # Usamos 'wb' para escrita binária
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode='wb') as temp_file:
            content = await file.read()
            if not content:
                 raise ValueError("Arquivo recebido está vazio.")
            temp_file.write(content)
            temp_file_path = temp_file.name
        print(f"DEBUG: Arquivo salvo temporariamente em: {temp_file_path}")

        start_time = time.time()
        input_data: Optional[np.ndarray] = None
        pred_class: str = "error"
        confidence: float = 0.0
        model_name: str = "unknown"

        # --- Pré-processamento e Inferência ---
        if media_type == 'audio':
            input_data = inference.preprocess_audio(temp_file_path)
            if input_data is None: raise ValueError("Falha no pré-processamento do áudio.")
            pred_class, confidence = inference.run_audio_inference(input_data)
            model_name = inference.AUDIO_MODEL_FILENAME

        elif media_type == 'video':
            input_data = inference.preprocess_video_frame(temp_file_path)
            if input_data is None: raise ValueError("Falha no pré-processamento do vídeo.")
            pred_class, confidence = inference.run_video_inference(input_data)
            model_name = inference.VIDEO_MODEL_FILENAME

        results.append(ml_schemas.PredictionResult(
            predicted_class=pred_class,
            confidence=confidence,
            model_name=model_name
        ))

        end_time = time.time()
        message = f"Inferência {media_type} concluída em {(end_time - start_time):.2f}s. Resultado: {pred_class} ({confidence*100:.1f}%)"
        print(f"INFO: {message}")

        # --- Lógica de Decisão e Salvamento ---
        CONFIDENCE_THRESHOLD = 0.60 # Limiar de confiança (AJUSTE SE NECESSÁRIO)

        # Só salva se NÃO for normal E confiança for alta
        if pred_class != 'normal' and confidence >= CONFIDENCE_THRESHOLD:
            print(f"ALERTA: Falha '{pred_class}' detectada com confiança suficiente ({confidence:.2f})!")

            # Calcula duração e severidade usando o arquivo temporário
            duration_s, severity = calcular_severidade_e_duracao(temp_file_path)

            # Prepara os dados para salvar
            # Simula start_ts baseado na duração calculada
            end_ts_now = datetime.now()
            start_ts_calc = end_ts_now - timedelta(seconds=duration_s)

            ocorrencia_data = db_schemas.OcorrenciaCreate(
                start_ts=start_ts_calc,
                end_ts=end_ts_now,
                duration_s=duration_s,
                category=f"{media_type.capitalize()} Técnico", # Ex: "Áudio Técnico"
                type=pred_class,                              # Ex: "ruido"
                severity=severity,                            # Ex: "Média (B)"
                confidence=confidence,
                # IMPORTANTE: No futuro, salve o PATH REAL do clipe permanente aqui
                evidence={"clip_path_temp": temp_file_path, "model": model_name, "original_filename": file.filename}
            )

            # Agenda a tarefa de salvar no banco (passa cópia dos dados)
            background_tasks.add_task(salvar_ocorrencia_task, db, ocorrencia_data.copy(deep=True))
            message += f". Falha '{pred_class}' ({severity}) detectada. Agendando salvamento."
            ocorrencia_salva = True
            print(f"INFO: Tarefa de salvar ocorrência '{pred_class}' agendada.")
        else:
             message += ". Nenhuma falha detectada acima do limiar ou classe 'normal'."
             print(f"INFO: {message}")


    except Exception as e:
        error_msg = f"Erro fatal durante a análise: {e}"
        message = error_msg # Atualiza a mensagem de retorno
        print(f"ERRO: {error_msg}")
        traceback.print_exc()
        # Neste caso, podemos levantar uma exceção para retornar um erro 500 mais claro
        # Mas vamos manter o retorno no formato AnalysisOutput por consistência
        # raise HTTPException(status_code=500, detail=error_msg)


    finally:
        # Garante a exclusão do arquivo temporário, mesmo se der erro
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print(f"DEBUG: Arquivo temporário removido: {temp_file_path}")
            except Exception as unlink_err:
                print(f"AVISO: Falha ao remover arquivo temporário {temp_file_path}: {unlink_err}")

    # Retorna o resultado da análise (sucesso ou falha)
    return ml_schemas.AnalysisOutput(
        filename=file.filename,
        media_type=media_type,
        predictions=results,
        ocorrencia_salva=ocorrencia_salva,
        message=message
    )

print("INFO: Endpoint de Análise (/analyze) TFLite configurado.")