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
        self._hls_log_file: Optional[object] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._tmpdir: Optional[str] = None

    def start(self, url: str):
        if self._running:
            return False
        # create tmpdir to hold frames
        self._tmpdir = tempfile.mkdtemp(prefix='srt_frames_')
        # Prepare HLS output folder
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        hls_dir = os.path.join(repo_root, 'static', 'hls')
        os.makedirs(hls_dir, exist_ok=True)
        out_hls = os.path.join(hls_dir, 'stream.m3u8')

        # HLS ffmpeg command: start HLS producer first, with tolerant flags for SRT/H264
        cmd_hls = [
            'ffmpeg', '-hide_banner', '-i', url,
            '-fflags', '+genpts+igndts', '-avoid_negative_ts', 'make_zero',
            '-use_wallclock_as_timestamps', '1',
            '-c:v', 'copy', '-c:a', 'aac', '-ar', '44100', '-ac', '2',
            '-f', 'hls', '-hls_time', '2', '-hls_list_size', '5', '-hls_flags', 'delete_segments',
            out_hls
        ]
        # start HLS process first
        log_path = os.path.join(hls_dir, 'hls_ffmpeg.log')
        try:
            self._hls_log_file = open(log_path, 'ab')
            self._proc_hls = subprocess.Popen(cmd_hls, stdout=self._hls_log_file, stderr=self._hls_log_file)
        except Exception as e:
            print(f"ERRO: falha iniciar ffmpeg HLS: {e}")
            try:
                if self._tmpdir and os.path.exists(self._tmpdir):
                    shutil.rmtree(self._tmpdir)
            except Exception:
                pass
            try:
                if self._hls_log_file:
                    self._hls_log_file.close()
                self._hls_log_file = None
            except Exception:
                pass
            return False
        # give HLS process a bit longer to produce an initial playlist/segments
        time.sleep(2.0)
        # check if hls process is alive
        if self._proc_hls.poll() is not None:
            print(f"ERRO: ffmpeg HLS process terminated immediately. See {log_path} for details.")
            try:
                if self._hls_log_file:
                    self._hls_log_file.flush()
            except Exception:
                pass
            try:
                if self._tmpdir and os.path.exists(self._tmpdir):
                    shutil.rmtree(self._tmpdir)
            except Exception:
                pass
            try:
                if self._hls_log_file:
                    self._hls_log_file.close()
            except Exception:
                pass
            self._proc_hls = None
            self._hls_log_file = None
            return False
        else:
            print(f"INFO: Started ffmpeg HLS process for {url} -> {out_hls} (log: {log_path})")
            # wait until playlist file appears (or timeout)
            playlist_ready = False
            try:
                for _ in range(10):
                    if os.path.exists(out_hls) and os.path.getsize(out_hls) > 0:
                        playlist_ready = True
                        break
                    time.sleep(0.3)
                if not playlist_ready:
                    print(f"WARN: HLS playlist not ready after wait for {out_hls}")
            except Exception:
                pass

        # ffmpeg frame extractor: write images to tmpdir/frame_%06d.jpg at self.fps
        out_pattern = os.path.join(self._tmpdir, 'frame_%06d.jpg')
        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', url,
            '-vf', f'fps={self.fps}', '-q:v', '2', out_pattern,
        ]
        # Start ffmpeg (frame extractor)
        try:
            self._proc = subprocess.Popen(cmd)
        except Exception as e:
            print(f"ERRO: falha iniciar ffmpeg frames extractor: {e}")
            # cleanup tmpdir and hls
            try:
                if self._proc_hls and self._proc_hls.poll() is None:
                    try:
                        self._proc_hls.terminate()
                        self._proc_hls.wait(timeout=2)
                    except Exception:
                        try:
                            self._proc_hls.kill()
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                if self._tmpdir and os.path.exists(self._tmpdir):
                    shutil.rmtree(self._tmpdir)
            except Exception:
                pass
            try:
                if self._hls_log_file:
                    self._hls_log_file.close()
            except Exception:
                pass
            self._proc_hls = None
            self._hls_log_file = None
            return False
        # HLS process already started above with tolerant flags; avoid starting a duplicate.
        # both processes started correctly -> mark running and start watcher
        self._running = True
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
            # Close HLS log file handle if open
            try:
                if self._hls_log_file:
                    try:
                        self._hls_log_file.close()
                    except Exception:
                        pass
            finally:
                self._hls_log_file = None
        # Remove any transient HLS files from the public hls folder so we don't
        # accumulate segments on disk after stopping. We only remove files in
        # the hls folder; clips remain in static/clips.
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            hls_dir = os.path.join(repo_root, 'static', 'hls')
            if os.path.exists(hls_dir):
                # remove individual files safely
                for fname in glob.glob(os.path.join(hls_dir, '*')):
                    try:
                        # never remove the directory itself here; remove contents
                        if os.path.isdir(fname):
                            shutil.rmtree(fname)
                        else:
                            os.remove(fname)
                    except Exception:
                        # ignore individual file delete errors
                        pass
                # after cleaning segments, write a minimal playlist with ENDLIST
                try:
                    playlist_path = os.path.join(hls_dir, 'stream.m3u8')
                    with open(playlist_path, 'w', encoding='utf-8') as plf:
                        plf.write('#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-PLAYLIST-TYPE:VOD\n#EXT-X-ENDLIST\n')
                except Exception:
                    pass
        except Exception:
            pass
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
                                    # calcula janela com margem em segundos (ex: 2s antes e 2s depois)
                                    idx_base = idx
                                    margin_seconds = 2.0
                                    before_frames = max(0, int(self.fps * margin_seconds))
                                    after_frames = max(0, int(self.fps * margin_seconds))
                                    start_idx = max(1, idx_base - before_frames)
                                    end_idx = idx_base + after_frames
                                    num_frames = max(1, end_idx - start_idx + 1)
                                    clip_duration_s = float(num_frames) / max(1.0, float(self.fps))
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
                                        evidence_obj['clip_duration_s'] = clip_duration_s
                                        evidence_obj['event_window'] = {
                                            'before_margin_s': float(before_frames) / max(1.0, float(self.fps)),
                                            'after_margin_s': float(after_frames) / max(1.0, float(self.fps)),
                                        }
                                    except Exception as clip_err:
                                        # fallback: apenas registra o frame path
                                        print(f"AVISO: Falha gerar clipe de frames: {clip_err}")
                                except Exception:
                                    # Se algo falhar, deixamos apenas o frame
                                    evidence_obj = {'frame': f}

                                # Deriva severidade pela duração (cartilha Globo)
                                def _sev(d: float) -> str:
                                    try:
                                        d = float(d)
                                        if d >= 60:
                                            return 'Gravíssima (X)'
                                        if d >= 10:
                                            return 'Grave (A)'
                                        if d >= 5:
                                            return 'Média (B)'
                                        return 'Leve (C)'
                                    except Exception:
                                        return 'Leve (C)'

                                db_oc = models.Ocorrencia(
                                    start_ts=start_ts_calc,
                                    end_ts=end_ts_now,
                                    duration_s=1.0,
                                    category='Vídeo Técnico',
                                    type=pred_class,
                                    severity=_sev(1.0),
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
