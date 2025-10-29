import os
import tempfile
import threading
import subprocess
import time
import glob
import shutil
from typing import Optional
from datetime import datetime, timedelta

from app.ml import inference
from app.api.endpoints import ws as ws_router
from app.db.base import SessionLocal
from app.db import models


class SRTIngestor:
    def __init__(self, fps: float = 1.0, confidence_threshold: float = 0.6):
        self.fps = fps
        self.confidence_threshold = confidence_threshold
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._tmpdir: Optional[str] = None

    def start(self, url: str):
        if self._running:
            return False
        self._running = True
        # create tmpdir to hold frames
        self._tmpdir = tempfile.mkdtemp(prefix='srt_frames_')
        # ffmpeg command: write images to tmpdir/frame_%06d.jpg at self.fps
        out_pattern = os.path.join(self._tmpdir, 'frame_%06d.jpg')
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel',
            'error',
            '-i',
            url,
            '-vf',
            f'fps={self.fps}',
            '-q:v',
            '2',
            out_pattern,
        ]
        # Start ffmpeg
        self._proc = subprocess.Popen(cmd)
        # Start watcher thread
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._running = False
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=3)
                except Exception:
                    self._proc.kill()
        finally:
            self._proc = None
        # join thread
        if self._thread:
            self._thread.join(timeout=1)
            self._thread = None
        # cleanup tmpdir
        if self._tmpdir and os.path.exists(self._tmpdir):
            try:
                shutil.rmtree(self._tmpdir)
            except Exception:
                pass
        self._tmpdir = None

    def _watch_loop(self):
        last_seen_index = 0
        while self._running:
            try:
                files = sorted(glob.glob(os.path.join(self._tmpdir, 'frame_*.jpg')))
                for f in files:
                    # determine index
                    basename = os.path.basename(f)
                    idx = int(basename.replace('frame_', '').replace('.jpg', ''))
                    if idx <= last_seen_index:
                        continue
                    last_seen_index = idx
                    # run inference on image
                    try:
                        import cv2
                        import numpy as np

                        img = cv2.imread(f)
                        if img is None:
                            continue
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_resized = cv2.resize(img_rgb, (inference.INPUT_WIDTH, inference.INPUT_HEIGHT))
                        img_normalized = img_resized.astype('float32') / 255.0
                        input_data = np.expand_dims(img_normalized, axis=0)
                        pred_class, confidence = inference.run_video_inference(input_data)
                        # broadcast and save if needed
                        if pred_class != 'normal' and confidence >= self.confidence_threshold:
                            # Save occurrence to DB
                            db = SessionLocal()
                            try:
                                end_ts_now = datetime.now()
                                start_ts_calc = end_ts_now - timedelta(seconds=1)
                                db_oc = models.Ocorrencia(
                                    start_ts=start_ts_calc,
                                    end_ts=end_ts_now,
                                    duration_s=1.0,
                                    category='Vídeo Técnico',
                                    type=pred_class,
                                    severity='Auto',
                                    confidence=float(confidence),
                                    evidence={'frame': f}
                                )
                                db.add(db_oc)
                                db.commit()
                                db.refresh(db_oc)
                                # broadcast
                                message = {
                                    'type': 'nova_ocorrencia',
                                    'data': {
                                        'id': db_oc.id,
                                        'start_ts': db_oc.start_ts.isoformat(),
                                        'end_ts': db_oc.end_ts.isoformat(),
                                        'duration_s': db_oc.duration_s,
                                        'category': db_oc.category,
                                        'type': db_oc.type,
                                        'severity': db_oc.severity,
                                        'confidence': db_oc.confidence,
                                        'evidence': db_oc.evidence,
                                    },
                                }
                                # async broadcast — schedule via event loop
                                import asyncio

                                try:
                                    loop = asyncio.get_event_loop()
                                except RuntimeError:
                                    loop = None
                                if loop and loop.is_running():
                                    asyncio.run_coroutine_threadsafe(ws_router.manager.broadcast(message), loop)
                                else:
                                    # No running loop — try to start a temporary loop
                                    try:
                                        asyncio.run(ws_router.manager.broadcast(message))
                                    except Exception:
                                        pass
                            finally:
                                db.close()
                    except Exception:
                        # ignore per-frame exceptions to keep ingest running
                        pass
                time.sleep(max(0.1, 1.0 / max(1, int(self.fps))))
            except Exception:
                time.sleep(1)


# Singleton instance
srt_ingestor = SRTIngestor()
