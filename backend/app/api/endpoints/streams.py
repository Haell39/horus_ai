from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.streams.srt_reader import srt_ingestor

router = APIRouter()


class StreamStartRequest(BaseModel):
    url: str
    fps: Optional[float] = 1.0


@router.post('/streams/start')
def start_stream(req: StreamStartRequest, background_tasks: BackgroundTasks):
    if srt_ingestor._running:
        raise HTTPException(status_code=400, detail='Stream already running')
    # start in background
    def _start():
        srt_ingestor.fps = req.fps or 1.0
        srt_ingestor.start(req.url)

    background_tasks.add_task(_start)
    return {'status': 'starting', 'url': req.url}


@router.post('/streams/stop')
def stop_stream():
    if not srt_ingestor._running:
        return {'status': 'not running'}
    srt_ingestor.stop()
    return {'status': 'stopped'}


@router.get('/streams/status')
def status():
    return {'running': bool(srt_ingestor._running), 'fps': srt_ingestor.fps}
