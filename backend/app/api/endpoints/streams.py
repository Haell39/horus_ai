from fastapi import APIRouter, HTTPException
import threading
from pydantic import BaseModel
from typing import Optional
from app.streams.srt_reader import srt_ingestor

router = APIRouter()


class StreamStartRequest(BaseModel):
    url: str
    fps: Optional[float] = 1.0


@router.post('/streams/start')
def start_stream(req: StreamStartRequest):
    if srt_ingestor._running:
        raise HTTPException(status_code=400, detail='Stream already running')

    # start synchronously and return success only if start() succeeded.
    try:
        srt_ingestor.fps = req.fps or 1.0
        ok = srt_ingestor.start(req.url)
        if not ok:
            raise HTTPException(status_code=500, detail='Failed to start SRT ingest (ffmpeg error)')
        return {'status': 'started', 'url': req.url}
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
