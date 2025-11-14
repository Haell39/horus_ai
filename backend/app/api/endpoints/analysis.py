from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, status, BackgroundTasks
from typing import Optional, Tuple
import os
import uuid
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
import json
import tempfile
import time
import traceback

from app.db.base import get_db, SessionLocal
from sqlalchemy.orm import Session
from app.db import models, schemas
from app.ml import inference
from app.core.config import settings as core_settings
from app.core import storage as storage_core

# Optional MoviePy for duration extraction
try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
except Exception:
    VideoFileClip = None

router = APIRouter()

# Diretório absoluto onde os diagnostics serão salvos (sempre o mesmo lugar)
DIAGNOSTICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'diagnostics'))
os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)

# Uploads dir
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'uploads'))
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Small helper: try ffprobe for duration, fallback to MoviePy if available
def _get_duration_seconds(path: str) -> float:
    try:
        proc = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', path
        ], capture_output=True, text=True, timeout=10)
        out = proc.stdout.strip()
        return float(out) if out else 0.0
    except Exception:
        # fallback to MoviePy if installed
        try:
            if VideoFileClip:
                with VideoFileClip(path) as clip:
                    return float(clip.duration or 0.0)
        except Exception:
            pass
    return 0.0


def calcular_severidade_e_duracao(clip_path: str) -> Tuple[float, str]:
    """Calcula duração do clipe e determina severidade (heurística simples)."""
    duration_s = _get_duration_seconds(clip_path)
    severity = "Leve (C)"
    try:
        duration_s = float(duration_s)
    except Exception:
        duration_s = 0.0
    if duration_s >= 60:
        severity = "Gravíssima (X)"
    elif duration_s >= 10:
        severity = "Grave (A)"
    elif duration_s >= 5:
        severity = "Média (B)"
    else:
        severity = "Leve (C)"
    return duration_s, severity


@router.post('/analysis/upload', summary='Envia um arquivo de vídeo para análise')
async def upload_analysis(
    file: UploadFile = File(...),
    fps: Optional[float] = Form(None),
    debug: Optional[bool] = Form(False),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    # Basic validation
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Arquivo não é um vídeo')

    ext = os.path.splitext(file.filename)[1] or '.mp4'
    uid = uuid.uuid4().hex
    tmp_name = f"upload_{uid}{ext}"
    tmp_path = os.path.join(UPLOAD_DIR, tmp_name)

    try:
        with open(tmp_path, 'wb') as out_f:
            shutil.copyfileobj(file.file, out_f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Erro ao salvar arquivo: {e}')

    try:
        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    except Exception:
        size_mb = 0.0
    duration_s = _get_duration_seconds(tmp_path)

    HARD_LIMIT_MB = 1024
    SYNC_LIMIT_MB = 50
    SYNC_LIMIT_SECONDS = 30

    if size_mb > HARD_LIMIT_MB:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise HTTPException(status_code=413, detail='Arquivo maior que o limite permitido (1 GB)')

    # copy to clips dir so it's served
    clip_name = f"upload_clip_{uid}{ext}"
    clips_dir = storage_core.get_clips_dir()
    os.makedirs(clips_dir, exist_ok=True)
    clip_path = os.path.join(clips_dir, clip_name)
    try:
        shutil.copy2(tmp_path, clip_path)
    except Exception:
        try:
            shutil.move(tmp_path, clip_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Erro ao preparar clip: {e}')

    process_inline = (size_mb <= SYNC_LIMIT_MB and duration_s <= SYNC_LIMIT_SECONDS)

    if process_inline:
        # Inline processing
        try:
            pred_class, confidence, event_time = inference.analyze_video_frames(clip_path, sample_rate_hz=2.0)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Erro na inferência de vídeo: {e}')

        # By default the project runs in visual-only mode. Skip audio
        # processing entirely when VIDEO_DISABLE_AUDIO_PROCESSING is True
        # to avoid unnecessary work and to ensure audio cannot influence
        # detection decisions.
        audio_class, audio_conf = 'normal', 0.0
        try:
            if not bool(core_settings.VIDEO_DISABLE_AUDIO_PROCESSING):
                audio_class, audio_conf = inference.analyze_audio_segments(clip_path)
        except Exception:
            audio_class, audio_conf = 'normal', 0.0

        try:
            diag = inference.analyze_video_frames_diagnostic(clip_path, k=3, sample_rate_hz=2.0, max_samples=200)
        except Exception:
            diag = []

        # aggregate top1 votes with per-class thresholds and vote K
        aggregated = None
        try:
            if diag:
                score_sum = {}
                count_above = {}
                total_samples = 0
                # get per-class thresholds from settings
                per_class_thresh = core_settings.video_thresholds()
                vote_k = int(core_settings.VIDEO_VOTE_K)
                for item in diag:
                    total_samples += 1
                    topk = item.get('topk') or []
                    if not topk:
                        continue
                    top1 = topk[0]
                    cls = top1.get('class')
                    sc = float(top1.get('score') or 0.0)
                    score_sum[cls] = score_sum.get(cls, 0.0) + sc
                    # threshold for this class
                    thr = per_class_thresh.get(cls.upper(), per_class_thresh.get('DEFAULT', float(core_settings.VIDEO_THRESH_DEFAULT)))
                    # count sample as supporting this class if score >= threshold
                    if sc >= float(thr):
                        count_above[cls] = count_above.get(cls, 0) + 1
                if score_sum:
                    # choose best by average score
                    best_cls = max(score_sum.items(), key=lambda x: x[1])[0]
                    summed = score_sum[best_cls]
                    avg_conf = summed / (total_samples or 1)
                    samples = total_samples
                    # if there are at least vote_k supporting samples for best_cls, prefer it
                    supporting = count_above.get(best_cls, 0)
                    if supporting >= vote_k or avg_conf >= float(per_class_thresh.get(best_cls.upper(), per_class_thresh.get('DEFAULT', float(core_settings.VIDEO_THRESH_DEFAULT)))):
                        aggregated = {'class': best_cls, 'confidence': float(avg_conf), 'samples': samples, 'supporting': supporting}
                    else:
                        # not enough supporting evidence to prefer aggregated class
                        aggregated = None
        except Exception:
            aggregated = None

        # save debug diagnostic file if requested
        if debug:
            try:
                payload = {'clip': os.path.basename(clip_path), 'diagnostic': diag, 'audio': {'class': audio_class, 'confidence': float(audio_conf or 0.0)}, 'aggregated': aggregated}
                fname = f"upload_{uid}.diagnostic.json"
                fpath = os.path.join(DIAGNOSTICS_DIR, fname)
                with open(fpath, 'w', encoding='utf-8') as jf:
                    json.dump(payload, jf, ensure_ascii=False, indent=2)
                print(f"DEBUG: diagnostic salvo em: {fpath}")
            except Exception as _e:
                print(f"DEBUG: falha ao salvar diagnostic: {_e}")

        # Determine final prediction using thresholds and aggregated evidence
        # Base confidence threshold (can be tuned via env)
        CONFIDENCE_THRESHOLD = float(core_settings.VIDEO_THRESH_DEFAULT or 0.60)

        # start with video raw
        video_effective_conf = float(confidence or 0.0)
        video_pred = pred_class
        # If aggregated supports a class strongly, prefer aggregated
        if aggregated:
            # prefer aggregated class if supporting votes or avg_conf above per-class threshold
            video_effective_conf = float(aggregated.get('confidence') or video_effective_conf)
            video_pred = aggregated.get('class')

        final_pred_class = video_pred
        final_confidence = float(video_effective_conf or 0.0)

        # If the video pipeline strongly indicates 'normal' (no visual faults),
        # prefer that and suppress audio influence entirely. This avoids creating
        # audio-only occurrences when the visual analysis shows no problem.
        try:
            per_class_thresh = core_settings.video_thresholds()
            video_thr = float(per_class_thresh.get(video_pred.upper(), per_class_thresh.get('DEFAULT', float(core_settings.VIDEO_THRESH_DEFAULT))))
            if (video_pred == 'normal' or (str(video_pred).lower() == 'normal')) and (final_confidence >= video_thr):
                # Force final to normal and ignore audio results regardless of audio model output
                final_pred_class = 'normal'
                final_confidence = float(final_confidence)
                try:
                    audio_class, audio_conf = 'normal', 0.0
                except Exception:
                    audio_class, audio_conf = 'normal', 0.0
        except Exception:
            # If threshold lookup fails, fall back to existing behavior
            pass

        # Compare audio signal: decide final winner between video/audio
        # use separate audio thresholds (AUDIO_THRESH_<CLASS> or AUDIO_THRESH_DEFAULT)
        try:
            audio_thresh_map = core_settings.audio_thresholds()
            audio_thresh = audio_thresh_map.get(audio_class.upper(), audio_thresh_map.get("DEFAULT", float(core_settings.AUDIO_THRESH_DEFAULT))) if audio_class else float(core_settings.AUDIO_THRESH_DEFAULT)
        except Exception:
            audio_thresh = float(core_settings.AUDIO_THRESH_DEFAULT)

        # prefer audio if stronger. Two conditions to promote audio:
        # 1) audio_conf >= audio_thresh (strong enough on its own)
        # 2) or audio_conf is significantly higher than video (delta) and reasonably confident
        # This helps cases where audio identifies 'freeze' but both are below global env thresholds.
        try:
            # Optionally allow audio to override video decisions. Controlled by
            # VIDEO_ALLOW_AUDIO_OVERRIDE (default: False) to avoid audio-driven
            # false positives for visual-only errors.
            if bool(core_settings.VIDEO_ALLOW_AUDIO_OVERRIDE):
                if audio_class and (audio_conf or 0.0) > final_confidence:
                    delta = (audio_conf or 0.0) - final_confidence
                    if (audio_conf or 0.0) >= float(audio_thresh) or (delta >= 0.04 and (audio_conf or 0.0) >= max(0.55, float(audio_thresh) - 0.05)):
                        final_pred_class = audio_class
                        final_confidence = float(audio_conf or 0.0)
            else:
                # Audio override disabled: keep video decision but log for debug
                if audio_class and (audio_conf or 0.0) > final_confidence:
                    try:
                        print(f"DEBUG: Audio override suppressed (audio={audio_class} conf={audio_conf:.3f}) by VIDEO_ALLOW_AUDIO_OVERRIDE=False")
                    except Exception:
                        pass
        except Exception:
            pass

        now = datetime.now(timezone.utc)

        # Decide which threshold to use for final occurrence creation:
        # - If the final prediction was chosen based on audio, use the audio threshold for that class
        # - Otherwise, use the per-class video threshold or VIDEO_THRESH_DEFAULT
        try:
            if final_pred_class and audio_class and final_pred_class == audio_class:
                final_threshold = float(audio_thresh)
            else:
                per_class_thresh = core_settings.video_thresholds()
                final_threshold = float(per_class_thresh.get(final_pred_class.upper(), per_class_thresh.get('DEFAULT', float(core_settings.VIDEO_THRESH_DEFAULT))))
        except Exception:
            final_threshold = float(core_settings.VIDEO_THRESH_DEFAULT)

        if final_pred_class == 'normal' or (final_confidence or 0.0) < final_threshold:
            # If video says normal but audio detected a strong audio-only fault,
            # create a separate audio-file occurrence (do not override video decision).
            try:
                audio_thresh_map = core_settings.audio_thresholds()
                audio_thresh = audio_thresh_map.get(audio_class.upper(), audio_thresh_map.get('DEFAULT', float(core_settings.AUDIO_THRESH_DEFAULT))) if audio_class else float(core_settings.AUDIO_THRESH_DEFAULT)
            except Exception:
                audio_thresh = float(core_settings.AUDIO_THRESH_DEFAULT)

            try:
                if audio_class and str(audio_class).lower() != 'normal' and float(audio_conf or 0.0) >= float(audio_thresh):
                    # create an audio occurrence and return it
                    dur_calc, severity = calcular_severidade_e_duracao(clip_path)
                    now = datetime.now(timezone.utc)
                    start_ts_calc = now - timedelta(seconds=dur_calc or 0)

                    # ensure public copy
                    public_clips_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'static', 'clips'))
                    os.makedirs(public_clips_dir, exist_ok=True)
                    public_path = os.path.join(public_clips_dir, os.path.basename(clip_path))
                    try:
                        if os.path.abspath(clip_path) != os.path.abspath(public_path):
                            shutil.copy2(clip_path, public_path)
                            clip_to_serve = public_path
                        else:
                            clip_to_serve = clip_path
                    except Exception:
                        clip_to_serve = clip_path

                    clip_dur_saved = _get_duration_seconds(clip_to_serve)
                    evidence = {
                        'clip_path': f'/clips/{os.path.basename(clip_to_serve)}',
                        'clip_duration_s': float(clip_dur_saved or duration_s),
                        'audio_pred': audio_class,
                        'audio_confidence': float(audio_conf or 0.0),
                        'fusion': {'video_pred': pred_class, 'video_conf': float(confidence or 0.0), 'audio_pred': audio_class, 'audio_conf': float(audio_conf or 0.0), 'final_pred': final_pred_class, 'final_conf': float(final_confidence or 0.0)}
                    }

                    try:
                        db_oc = models.Ocorrencia(
                            start_ts=start_ts_calc, end_ts=now, duration_s=dur_calc or clip_dur_saved or duration_s,
                            category='audio-file', type=audio_class, severity=severity,
                            confidence=float(audio_conf or 0.0), evidence=evidence
                        )
                        db.add(db_oc)
                        db.commit()
                        db.refresh(db_oc)
                        if debug:
                            # attach diagnostic info in the inline response for convenience
                            out = db_oc.__dict__
                            out['diagnostic'] = diag
                            out['audio_debug'] = {'class': audio_class, 'confidence': float(audio_conf or 0.0)}
                            return out
                        return db_oc
                    except Exception as e:
                        db.rollback()
                        print(f"Erro ao salvar ocorrência de áudio inline: {e}")

            except Exception:
                pass

            # cleanup clip if no occurrence
            try:
                if os.path.exists(clip_path):
                    os.remove(clip_path)
            except Exception:
                pass
            msg = f"Arquivo analisado — sem falhas detectadas (classe={final_pred_class}, conf={final_confidence:.3f})." if final_pred_class == 'normal' else (
                f"Arquivo analisado — detecção fraca abaixo do limiar ({CONFIDENCE_THRESHOLD:.2f}). Classe detectada: {final_pred_class}, confiança={final_confidence:.3f}. Sem ocorrência criada.")
            out = {'status': 'ok', 'message': msg, 'prediction': {'class': final_pred_class, 'confidence': float(final_confidence or 0.0)}}
            if debug:
                out['diagnostic'] = diag
                out['audio_debug'] = {'class': audio_class, 'confidence': float(audio_conf or 0.0)}
            return out

        # Otherwise, create occurrence and save evidence
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
                        'ffmpeg', '-y', '-ss', f"{start}", '-t', f"{duration_cut}", '-i', clip_path, '-c', 'copy', dest_path
                    ], check=True, capture_output=True, timeout=60)
                    clip_to_save = dest_path
                except Exception:
                    clip_to_save = clip_path

            # ensure public copy
            public_clips_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'static', 'clips'))
            os.makedirs(public_clips_dir, exist_ok=True)
            public_path = os.path.join(public_clips_dir, os.path.basename(clip_to_save))
            try:
                if os.path.abspath(clip_to_save) != os.path.abspath(public_path):
                    shutil.copy2(clip_to_save, public_path)
                    clip_to_serve = public_path
                else:
                    clip_to_serve = clip_to_save
            except Exception:
                clip_to_serve = clip_to_save

            clip_basename = os.path.basename(clip_to_serve)
            clip_dur_saved = _get_duration_seconds(clip_to_serve)
            dur_calc, severity = calcular_severidade_e_duracao(clip_to_save)

            # Use the final chosen class/confidence (after video/audio fusion) when creating the occurrence
            # Decide category based on whether the final prediction is an audio fault
            try:
                audio_label_set = [c.lower() for c in getattr(inference, 'AUDIO_CLASSES', [])]
            except Exception:
                audio_label_set = []
            is_audio_occ = str(final_pred_class).lower() in audio_label_set
            occ_category = 'audio-file' if is_audio_occ else 'video-file'

            evidence = {
                'clip_path': f'/clips/{clip_basename}',
                'clip_duration_s': float(clip_dur_saved or duration_s),
                'event_window': {'before_margin_s': (before_s if event_time is not None else 0.0), 'after_margin_s': (after_s if event_time is not None else 0.0)},
                'fusion': {'video_pred': pred_class, 'video_conf': float(confidence or 0.0), 'audio_pred': audio_class, 'audio_conf': float(audio_conf or 0.0), 'final_pred': final_pred_class, 'final_conf': float(final_confidence or 0.0)}
            }

            oc = schemas.OcorrenciaCreate(
                start_ts=now - timedelta(seconds=dur_calc or duration_s or 0),
                end_ts=now,
                duration_s=dur_calc or clip_dur_saved or duration_s,
                category=occ_category,
                type=final_pred_class,
                severity=severity,
                confidence=float(final_confidence or 0.0),
                evidence=evidence
            )

            try:
                db_oc = models.Ocorrencia(**oc.dict())
                db.add(db_oc)
                db.commit()
                db.refresh(db_oc)
                return db_oc
            except Exception as e:
                db.rollback()
                raise HTTPException(status_code=500, detail=f'Erro ao salvar ocorrência: {e}')
        except Exception:
            raise

    else:
        # Schedule background processing
        job_id = uuid.uuid4().hex

        def _process_queued(clip_path_local: str, clip_name_local: str, job_id_local: str):
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

                # cut around event if available
                before_s = 2.0
                after_s = 2.0
                clip_to_save = clip_path_local
                if event_time is not None:
                    start = max(0.0, event_time - before_s)
                    duration = before_s + after_s
                    dest_name = f"clip_{job_id_local}{os.path.splitext(clip_name_local)[1]}"
                    dest_path = os.path.join(storage_core.get_clips_dir(), dest_name)
                    try:
                        subprocess.run([
                            'ffmpeg', '-y', '-ss', f"{start}", '-t', f"{duration}", '-i', clip_path_local, '-c', 'copy', dest_path
                        ], check=True, capture_output=True, timeout=60)
                        clip_to_save = dest_path
                    except Exception as e:
                        clip_to_save = clip_path_local

                # re-run inference on the small cut to avoid transient events
                try:
                    if event_time is not None and clip_to_save and clip_to_save != clip_path_local:
                        pred2, conf2, _ = inference.analyze_video_frames(clip_to_save)
                        if pred2 != pred_class or (conf2 or 0.0) < CONFIDENCE_THRESHOLD:
                            print(f"Background worker: evento transitório detectado (pred2={pred2}, conf2={conf2}) - ignorando.")
                            return
                except Exception:
                    pass
                # Optionally run audio analysis on the cut and create audio occurrence
                audio_class = 'normal'
                audio_conf = 0.0
                try:
                    if not bool(core_settings.VIDEO_DISABLE_AUDIO_PROCESSING):
                        audio_class, audio_conf = inference.analyze_audio_segments(clip_to_save)
                except Exception:
                    audio_class, audio_conf = 'normal', 0.0

                # Determine audio threshold
                try:
                    audio_thresh_map = core_settings.audio_thresholds()
                    audio_thresh = audio_thresh_map.get(audio_class.upper(), audio_thresh_map.get('DEFAULT', float(core_settings.AUDIO_THRESH_DEFAULT)))
                except Exception:
                    audio_thresh = float(core_settings.AUDIO_THRESH_DEFAULT)

                # If audio indicates a strong audio-only fault, create an audio occurrence and return
                try:
                    if audio_class and str(audio_class).lower() != 'normal' and float(audio_conf or 0.0) >= float(audio_thresh):
                        dur_calc, severity = calcular_severidade_e_duracao(clip_to_save)
                        now = datetime.now(timezone.utc)
                        start_ts_calc = now - timedelta(seconds=dur_calc or 0)
                        public_clips_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'static', 'clips'))
                        os.makedirs(public_clips_dir, exist_ok=True)
                        public_path = os.path.join(public_clips_dir, os.path.basename(clip_to_save))
                        try:
                            if os.path.abspath(clip_to_save) != os.path.abspath(public_path):
                                shutil.copy2(clip_to_save, public_path)
                                clip_to_serve = public_path
                            else:
                                clip_to_serve = clip_to_save
                        except Exception:
                            clip_to_serve = clip_to_save

                        clip_dur = _get_duration_seconds(clip_to_serve)
                        evidence_dict = {
                            'clip_path': f'/clips/{os.path.basename(clip_to_serve)}',
                            'original_filename': clip_name_local,
                            'audio_pred': audio_class,
                            'audio_confidence': float(audio_conf or 0.0),
                            'clip_duration_s': float(clip_dur or 0.0),
                        }

                        try:
                            db_oc = models.Ocorrencia(
                                start_ts=start_ts_calc, end_ts=now, duration_s=dur_calc or clip_dur or 0.0,
                                category='audio-file', type=audio_class, severity=severity,
                                confidence=float(audio_conf or 0.0), evidence=evidence_dict
                            )
                            db_session.add(db_oc)
                            db_session.commit()
                            db_session.refresh(db_oc)
                            print(f"Background worker: ocorrência de áudio criada id={db_oc.id} tipo={db_oc.type}")
                            return
                        except Exception as e:
                            db_session.rollback()
                            print(f"Background worker: falha ao salvar ocorrência de áudio: {e}")
                except Exception:
                    pass

                # save occurrence (video) if audio didn't produce one
                dur_calc, severity = calcular_severidade_e_duracao(clip_to_save)
                now = datetime.now(timezone.utc)
                start_ts_calc = now - timedelta(seconds=dur_calc or 0)

                # ensure public copy
                public_clips_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'static', 'clips'))
                os.makedirs(public_clips_dir, exist_ok=True)
                public_path = os.path.join(public_clips_dir, os.path.basename(clip_to_save))
                try:
                    if os.path.abspath(clip_to_save) != os.path.abspath(public_path):
                        shutil.copy2(clip_to_save, public_path)
                        clip_to_serve = public_path
                    else:
                        clip_to_serve = clip_to_save
                except Exception:
                    clip_to_serve = clip_to_save

                clip_dur = _get_duration_seconds(clip_to_serve)
                evidence_dict = {
                    'clip_path': f'/clips/{os.path.basename(clip_to_serve)}',
                    'original_filename': clip_name_local,
                    'confidence_raw': float(confidence),
                    'clip_duration_s': float(clip_dur or 0.0),
                }

                try:
                    db_oc = models.Ocorrencia(
                        start_ts=start_ts_calc, end_ts=now, duration_s=dur_calc or clip_dur or 0.0,
                        category='video-file', type=pred_class, severity=severity,
                        confidence=float(confidence or 0.0), evidence=evidence_dict
                    )
                    db_session.add(db_oc)
                    db_session.commit()
                    db_session.refresh(db_oc)
                    print(f"Background worker: ocorrência criada id={db_oc.id} tipo={db_oc.type}")
                except Exception as e:
                    db_session.rollback()
                    print(f"Background worker: falha ao salvar ocorrência: {e}")
            finally:
                try:
                    db_session.close()
                except Exception:
                    pass

        try:
            background_tasks.add_task(_process_queued, clip_path, clip_name, job_id)
        except Exception:
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
        