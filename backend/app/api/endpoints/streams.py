from fastapi import APIRouter, HTTPException
import threading
import os
from pydantic import BaseModel
from typing import Optional
from app.streams.srt_reader import srt_ingestor

router = APIRouter()


class StreamStartRequest(BaseModel):
    url: Optional[str] = None
    streamId: Optional[str] = None
    fps: Optional[float] = 1.0
    # mode: 'srt' (default) or 'capture'
    mode: Optional[str] = 'srt'
    # capture device input spec (optional). On Windows (dshow) this can be like
    # 'video="Device Name":audio="Microphone"' or on Linux '/dev/video0'
    device: Optional[str] = None


def _resolve_srt_url(req: StreamStartRequest) -> str:
    # priority: explicit url if provided
    if req.url and req.url.startswith('srt://'):
        return req.url
    # resolve by streamId via environment, e.g., SRT_STREAM_URL_GLOBO
    if req.streamId:
        env_key = f"SRT_STREAM_URL_{req.streamId.upper()}"
        val = os.getenv(env_key, '')
        if val.startswith('srt://'):
            return val
    # fallback: single default
    default_val = os.getenv('SRT_STREAM_URL_DEFAULT', '')
    if default_val.startswith('srt://'):
        return default_val
    return ''


@router.post('/streams/start')
def start_stream(req: StreamStartRequest):
    if srt_ingestor._running:
        raise HTTPException(status_code=400, detail='Stream already running')

    # start synchronously and return success only if start() succeeded.
    try:
        # set fps on the appropriate ingestor
        fps = req.fps or 1.0
        if req.mode and req.mode.lower() == 'capture':
            # start capture-based ingest; require a device spec or env fallback
            from app.streams.srt_reader import capture_ingestor
            capture_ingestor.fps = fps
            device = req.device or os.getenv('CAPTURE_INPUT', '')
            if not device:
                raise HTTPException(status_code=400, detail='No capture device provided (pass device or set CAPTURE_INPUT env)')
            ok = capture_ingestor.start(device)
            if not ok:
                raise HTTPException(status_code=500, detail='Failed to start capture ingest (ffmpeg error)')
            return {'status': 'started', 'mode': 'capture'}
        else:
            srt_ingestor.fps = fps
            resolved_url = _resolve_srt_url(req)
            if not resolved_url:
                raise HTTPException(status_code=400, detail='No SRT URL available (provide url or valid streamId)')
            ok = srt_ingestor.start(resolved_url)
            if not ok:
                raise HTTPException(status_code=500, detail='Failed to start SRT ingest (ffmpeg error)')
            return {'status': 'started', 'mode': 'srt'}
    except HTTPException:
        raise
    except Exception as e:
        print(f"ERRO: Exception while starting srt_ingestor: {e}")
        raise HTTPException(status_code=500, detail='Internal server error while starting stream')


@router.post('/streams/stop')
def stop_stream():
    if not srt_ingestor._running:
        return {'status': 'not running'}
    srt_ingestor.stop()
    return {'status': 'stopped'}


@router.get('/streams/status')
def status():
    return {'running': bool(srt_ingestor._running), 'fps': srt_ingestor.fps}


@router.post('/streams/cleanup')
def cleanup():
    # Stop if running and force cleanup of HLS artifacts
    try:
        srt_ingestor.stop()
    except Exception:
        pass
    return {'status': 'cleaned'}
