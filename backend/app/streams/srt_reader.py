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
        self._proc_hls: Optional[subprocess.Popen] = None
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
        # Also start an ffmpeg process to generate HLS served by backend/static/hls
        try:
            # Use the repository-level static folder (backend/static/hls) so it matches
            # the StaticFiles mount configured in app.main (which points to backend/static/hls).
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            hls_dir = os.path.join(repo_root, 'static', 'hls')
            os.makedirs(hls_dir, exist_ok=True)
            out_hls = os.path.join(hls_dir, 'stream.m3u8')
            # Try to avoid re-encoding video (which may require libx264 in the ffmpeg build).
            # Use copy for video stream and transcode audio to AAC (widely available) so
            # HLS segments can be produced without heavy CPU and encoder dependencies.
            cmd_hls = [
                'ffmpeg', '-hide_banner', '-i', url,
                '-c:v', 'copy', '-c:a', 'aac', '-ar', '44100', '-ac', '2',
                '-f', 'hls', '-hls_time', '2', '-hls_list_size', '5', '-hls_flags', 'delete_segments',
                out_hls
            ]
            # Redirect stderr to a logfile so we can inspect ffmpeg connection errors
            log_path = os.path.join(hls_dir, 'hls_ffmpeg.log')
            log_file = open(log_path, 'ab')
            self._proc_hls = subprocess.Popen(cmd_hls, stdout=log_file, stderr=log_file)
            # wait a moment and verify process is alive; if exited, log warning
            time.sleep(1.0)
            if self._proc_hls.poll() is not None:
                print(f"ERRO: ffmpeg HLS process terminated immediately. See {log_path} for details.")
            else:
                print(f"INFO: Started ffmpeg HLS process for {url} -> {out_hls} (log: {log_path})")
        except Exception as e:
            print(f"AVISO: Falha iniciar ffmpeg HLS: {e}")
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
            if self._proc_hls and self._proc_hls.poll() is None:
                self._proc_hls.terminate()
                try:
                    self._proc_hls.wait(timeout=3)
                except Exception:
                    self._proc_hls.kill()
        finally:
            self._proc = None
            self._proc_hls = None
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
                                # Tenta gerar um pequeno clipe a partir dos frames próximos ao índice
                                evidence_obj = {'frame': f}
                                try:
                                    # calcula janela de frames (ex: 2 frames antes e depois)
                                    idx_base = idx
                                    start_idx = max(1, idx_base - 2)
                                    end_idx = idx_base + 2
                                    num_frames = end_idx - start_idx + 1
                                    # cria arquivo temporário de saída no tmpdir
                                    out_name = f"clip_{int(time.time())}_{idx_base}.mp4"
                                    out_tmp = os.path.join(self._tmpdir, out_name)
                                    # ffmpeg: usa sequência de imagens com start_number
                                    ffmpeg_cmd = [
                                        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                                        '-framerate', str(self.fps),
                                        '-start_number', str(start_idx),
                                        '-i', os.path.join(self._tmpdir, 'frame_%06d.jpg'),
                                        '-frames:v', str(num_frames),
                                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_tmp
                                    ]
                                    try:
                                        subprocess.run(ffmpeg_cmd, check=False)
                                        # move para pasta pública de clipes
                                        clips_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'clips'))
                                        os.makedirs(clips_dir, exist_ok=True)
                                        dest_path = os.path.join(clips_dir, out_name)
                                        shutil.move(out_tmp, dest_path)
                                        evidence_obj['clip_path'] = f"/clips/{out_name}"
                                    except Exception as clip_err:
                                        # fallback: apenas registra o frame path
                                        print(f"AVISO: Falha gerar clipe de frames: {clip_err}")
                                except Exception:
                                    # Se algo falhar, deixamos apenas o frame
                                    evidence_obj = {'frame': f}

                                db_oc = models.Ocorrencia(
                                    start_ts=start_ts_calc,
                                    end_ts=end_ts_now,
                                    duration_s=1.0,
                                    category='Vídeo Técnico',
                                    type=pred_class,
                                    severity='Auto',
                                    confidence=float(confidence),
                                    evidence=evidence_obj
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
