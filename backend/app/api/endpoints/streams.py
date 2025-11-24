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


@router.get('/streams/devices')
def list_capture_devices():
    """Auto-detect available capture devices (webcams, capture cards, etc.)"""
    import subprocess
    import platform
    
    devices = []
    system = platform.system()
    
    try:
        if system == 'Windows':
            # Use DirectShow on Windows
            cmd = ['ffmpeg', '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy']
        else:
            # Use v4l2 on Linux (list /dev/video* devices)
            cmd = ['v4l2-ctl', '--list-devices']
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        output = result.stderr + result.stdout  # ffmpeg outputs to stderr
        
        if system == 'Windows':
            # Parse DirectShow output
            lines = output.split('\n')
            in_video_section = False
            for line in lines:
                if 'DirectShow video devices' in line:
                    in_video_section = True
                    continue
                if 'DirectShow audio devices' in line:
                    in_video_section = False
                    break
                if in_video_section and '"' in line:
                    # Extract device name between quotes
                    parts = line.split('"')
                    if len(parts) >= 2:
                        device_name = parts[1]
                        devices.append({
                            'name': device_name,
                            'value': f'video={device_name}',
                            'type': 'video'
                        })
        else:
            # Parse v4l2-ctl output for Linux
            lines = output.split('\n')
            current_device = None
            for line in lines:
                line = line.strip()
                if line and not line.startswith('/dev/'):
                    current_device = line.rstrip(':')
                elif line.startswith('/dev/video'):
                    device_path = line.strip()
                    devices.append({
                        'name': f'{current_device} ({device_path})' if current_device else device_path,
                        'value': device_path,
                        'type': 'video'
                    })
        
        # If ffmpeg/v4l2-ctl not available or no devices found, return friendly message
        if not devices:
            # Try fallback: common default devices
            if system == 'Windows':
                devices.append({
                    'name': 'Dispositivo padrão (manual)',
                    'value': '',
                    'type': 'default'
                })
            else:
                devices.append({
                    'name': '/dev/video0 (padrão)',
                    'value': '/dev/video0',
                    'type': 'default'
                })
        
        return {'devices': devices, 'count': len(devices)}
    
    except Exception as e:
        print(f"Erro ao listar dispositivos: {e}")
        # Return empty list with error info
        return {'devices': [], 'count': 0, 'error': str(e)}

