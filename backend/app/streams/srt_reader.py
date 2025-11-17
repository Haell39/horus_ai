import os
import numpy as np
import tempfile
import threading
import subprocess
import time
import glob
import shutil
from typing import Optional
from datetime import datetime, timedelta, timezone

from app.ml import inference
from app.api.endpoints import ws as ws_router
from app.db.base import SessionLocal
from app.db import models
from app.core import storage as storage_core


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
        # Voting and thresholds
        self.vote_k = int(os.getenv('VIDEO_VOTE_K', '3'))
        self.mavg_window = int(os.getenv('VIDEO_MOVING_AVG_M', '5'))
        # class thresholds via env: VIDEO_THRESH_<UPPERCLASS>=float
        self.class_thresholds = {}
        try:
            for cls in getattr(inference, 'VIDEO_CLASSES', []):
                env_key = f"VIDEO_THRESH_{cls.upper()}".replace(' ', '_')
                val = os.getenv(env_key)
                if val is not None:
                    self.class_thresholds[cls] = float(val)
        except Exception:
            pass
        # If no thresholds were provided via ENV, try to load thresholds shipped with the model package
        try:
            if not self.class_thresholds:
                pkg_thresh = getattr(inference, 'CLASS_THRESHOLDS', {}) or {}
                for k, v in pkg_thresh.items():
                    # key names in thresholds may be e.g. 'fade' -> map to class if present
                    if k in getattr(inference, 'MODEL_CLASSES', []):
                        try:
                            self.class_thresholds[k] = float(v)
                        except Exception:
                            pass
        except Exception:
            pass
        # recent predictions
        from collections import deque
        self._recent_classes = deque(maxlen=max(3, self.mavg_window))
        self._recent_conf = deque(maxlen=max(3, self.mavg_window))
        self._streak_class: Optional[str] = None
        self._streak_count: int = 0
        # time-based aggregation and reporting
        # window in seconds used to throttle periodic diagnostics prints
        self._diag_throttle_s = float(os.getenv('VIDEO_DIAG_THROTTLE_S', '5.0'))
        self._last_diag_print = None
        # minimum event duration (seconds) required to report an occurrence
        self.min_event_duration_s = float(os.getenv('VIDEO_MIN_EVENT_DURATION_S', '2.0'))
        # currently active candidate event: {'class', 'start_ts', 'last_ts', 'max_conf', 'reported', 'start_idx'}
        self._active_event = None
        # fixed-window evaluation (collect preds for window_s seconds and decide by majority)
        self.fixed_window_enabled = bool(int(os.getenv('VIDEO_FIXED_WINDOW_ENABLED', '1')))
        self.window_s = float(os.getenv('VIDEO_FIXED_WINDOW_S', '5.0'))
        # sliding window stride (seconds) -> evaluates window every `window_stride_s`
        self.window_stride_s = float(os.getenv('VIDEO_WINDOW_STRIDE_S', '1.0'))
        # timestamp when we last evaluated the sliding window
        self._last_window_eval_ts = None
        # buffer of (timestamp, idx, class, confidence)
        self._window_preds = []
        # Sequence model support: detect sequence length from inference metadata and create frame buffer
        try:
            meta = getattr(inference, 'MODEL_METADATA', {}) or {}
            seq_len = None
            if meta and isinstance(meta.get('input_shape'), (list, tuple)):
                seq_len = int(meta.get('input_shape')[0])
            self.seq_len = seq_len
            from collections import deque
            self._seq_buffer = deque(maxlen=seq_len) if seq_len and seq_len > 1 else None
            self._last_seq_infer_ts = None
        except Exception:
            self.seq_len = None
            self._seq_buffer = None
            self._last_seq_infer_ts = None
        # last time we reported an occurrence (to avoid duplicates)
        self._last_report_ts = None

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
                    try:
                        idx = int(basename.replace('frame_', '').replace('.jpg', ''))
                    except Exception:
                        continue
                    if idx <= last_seen_index:
                        continue
                    last_seen_index = idx

                    # run inference on image
                    try:
                        import cv2

                        img = cv2.imread(f)
                        if img is None:
                            continue
                        # Prepare both resized raw frame and normalized input
                        try:
                            raw_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            raw_resized = cv2.resize(raw_rgb, (inference.INPUT_WIDTH, inference.INPUT_HEIGHT)).astype(np.float32)
                        except Exception:
                            # fallback to previous preprocess
                            input_data = inference.preprocess_video_single_frame(img)
                            if input_data is None:
                                continue
                            raw_resized = None
                        # normalized input (same preprocessing as preprocess_video_single_frame)
                        if raw_resized is not None:
                            input_norm = (raw_resized - 127.5) / 127.5
                        else:
                            input_norm = None

                        pred_class, confidence = 'normal', 0.0
                        top3 = []

                        # If we have a sequence-model (seq_len > 1), buffer frames and run sequence inference
                        if getattr(self, 'seq_len', None) and self.seq_len and self.seq_len > 1 and self._seq_buffer is not None:
                            # choose frame representation according to model rescaling presence
                            try:
                                frame_for_seq = raw_resized if getattr(inference, 'keras_video_model_has_rescaling', False) and raw_resized is not None else (input_norm if input_norm is not None else raw_resized)
                            except Exception:
                                frame_for_seq = input_norm if input_norm is not None else raw_resized

                            if frame_for_seq is None:
                                # nothing useful to append
                                continue

                            # append single-frame (H,W,C) to buffer
                            try:
                                self._seq_buffer.append(frame_for_seq)
                            except Exception:
                                # if buffer broken, recreate
                                try:
                                    from collections import deque
                                    self._seq_buffer = deque(maxlen=self.seq_len)
                                    self._seq_buffer.append(frame_for_seq)
                                except Exception:
                                    pass

                            now_ts = datetime.now(timezone.utc)
                            # evaluate sequence inference only when buffer full and stride elapsed
                            if len(self._seq_buffer) >= self.seq_len and (self._last_seq_infer_ts is None or (now_ts - self._last_seq_infer_ts).total_seconds() >= float(self.window_stride_s)):
                                try:
                                    seq_arr = np.stack(list(self._seq_buffer), axis=0).astype(np.float32)  # (seq_len,H,W,C)
                                    seq_batch = np.expand_dims(seq_arr, axis=0)  # (1,seq_len,H,W,C)
                                    pred_class, confidence = inference.run_keras_sequence_inference(getattr(inference, 'keras_video_model'), seq_batch, getattr(inference, 'MODEL_CLASSES', []))
                                    top3 = []
                                except Exception as e:
                                    print(f"AVISO: Falha na inferência de sequência: {e}")
                                    pred_class, confidence = 'Erro_Inferência', 0.0
                                finally:
                                    self._last_seq_infer_ts = now_ts
                            else:
                                # not time yet to evaluate; skip to next frame
                                continue
                        else:
                            # legacy single-frame inference path
                            input_data = None
                            if input_norm is not None:
                                input_data = np.expand_dims(input_norm, axis=0)
                            else:
                                input_data = inference.preprocess_video_single_frame(img)
                                if input_data is None:
                                    continue

                            top3 = inference.run_video_topk(input_data, k=3) or []
                            if top3:
                                vc = [v.lower() for v in inference.VIDEO_CLASSES]
                                found = None
                                for c, s in top3:
                                    if str(c).lower() in vc:
                                        found = (c, s)
                                        break
                                if found:
                                    pred_class, confidence = found
                                else:
                                    pred_class, confidence = inference.run_video_inference(input_data)
                            else:
                                pred_class, confidence = inference.run_video_inference(input_data)

                        now = datetime.now(timezone.utc)
                        # diagnostic throttle
                        do_print = False
                        if top3:
                            if pred_class != 'normal':
                                do_print = True
                            elif self._last_diag_print is None:
                                do_print = True
                            else:
                                try:
                                    if (now - self._last_diag_print).total_seconds() >= float(self._diag_throttle_s):
                                        do_print = True
                                except Exception:
                                    do_print = False
                        if do_print and top3:
                            try:
                                print(f"DEBUG TOP3 frame#{idx}: " + ", ".join([f"{c}:{s:.3f}" for c, s in top3]))
                                self._last_diag_print = now
                            except Exception:
                                pass

                        # update history
                        self._recent_classes.append(pred_class)
                        self._recent_conf.append(float(confidence))

                        eff_thresh = self.class_thresholds.get(pred_class, self.confidence_threshold)
                        try:
                            mavg_conf = sum(self._recent_conf) / max(1, len(self._recent_conf))
                        except Exception:
                            mavg_conf = float(confidence)

                        if self.fixed_window_enabled:
                            # add to sliding window buffer
                            self._window_preds.append((now, idx, pred_class, float(confidence)))
                            # evict old
                            while self._window_preds and (now - self._window_preds[0][0]).total_seconds() > self.window_s:
                                self._window_preds.pop(0)

                            # evaluate window if it spans at least window_s and stride elapsed
                            if (self._window_preds
                                    and (now - self._window_preds[0][0]).total_seconds() >= self.window_s
                                    and (self._last_window_eval_ts is None or (now - self._last_window_eval_ts).total_seconds() >= self.window_stride_s)):
                                # weighted vote by confidence
                                weights = {}
                                sum_weights = 0.0
                                first_pos = None
                                last_pos = None
                                # collect indices for top-class clip boundaries
                                class_indices = {}
                                for ts_i, idx_i, cls_i, conf_i in self._window_preds:
                                    thresh_i = self.class_thresholds.get(cls_i, self.confidence_threshold)
                                    if cls_i != 'normal' and conf_i >= thresh_i:
                                        weights[cls_i] = weights.get(cls_i, 0.0) + float(conf_i)
                                        sum_weights += float(conf_i)
                                        if cls_i not in class_indices:
                                            class_indices[cls_i] = {'first_ts': ts_i, 'last_ts': ts_i, 'min_idx': idx_i, 'max_idx': idx_i}
                                        else:
                                            class_indices[cls_i]['last_ts'] = ts_i
                                            class_indices[cls_i]['min_idx'] = min(class_indices[cls_i]['min_idx'], idx_i)
                                            class_indices[cls_i]['max_idx'] = max(class_indices[cls_i]['max_idx'], idx_i)

                                if weights and sum_weights > 0.0:
                                    top_class, top_weight = max(weights.items(), key=lambda x: x[1])
                                else:
                                    top_class, top_weight = (None, 0.0)

                                # require weighted majority (>50% of sum weights)
                                if top_class and top_weight > (sum_weights / 2.0):
                                    # positive span duration check for top_class
                                    ci = class_indices.get(top_class)
                                    if ci:
                                        first_pos = ci.get('first_ts')
                                        last_pos = ci.get('last_ts')
                                    if first_pos and last_pos and (last_pos - first_pos).total_seconds() >= self.min_event_duration_s:
                                        if self._last_report_ts is None or (now - self._last_report_ts).total_seconds() > max(1.0, self.window_s / 2.0):
                                            # persist occurrence (weighted confidence fraction)
                                            db = SessionLocal()
                                            try:
                                                start_ts_calc = first_pos
                                                end_ts_now = last_pos
                                                event_duration = (end_ts_now - start_ts_calc).total_seconds()
                                                evidence_obj = {'frame': f}
                                                try:
                                                    # build clip using min/max indices for the top class
                                                    if ci:
                                                        start_idx = max(1, ci.get('min_idx', idx) - int(self.fps * 2.0))
                                                        end_idx = ci.get('max_idx', idx) + int(self.fps * 2.0)
                                                    else:
                                                        start_idx = max(1, idx - int(self.fps * 2.0))
                                                        end_idx = idx + int(self.fps * 2.0)
                                                    num_frames = max(1, end_idx - start_idx + 1)
                                                    clip_duration_s = float(num_frames) / max(1.0, float(self.fps))
                                                    out_name = f"clip_{int(time.time())}_{idx}.mp4"
                                                    out_tmp = os.path.join(self._tmpdir, out_name)
                                                    ffmpeg_cmd = [
                                                        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                                                        '-framerate', str(self.fps),
                                                        '-start_number', str(start_idx),
                                                        '-i', os.path.join(self._tmpdir, 'frame_%06d.jpg'),
                                                        '-frames:v', str(num_frames),
                                                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_tmp
                                                    ]
                                                    expected_files = [os.path.join(self._tmpdir, f'frame_{i:06d}.jpg') for i in range(start_idx, end_idx + 1)]
                                                    missing = [p for p in expected_files if not os.path.exists(p)]
                                                    if missing:
                                                        print(f"AVISO: Frames faltando para gerar clipe ({len(missing)}) — pulando geração de clipe")
                                                    else:
                                                        subprocess.run(ffmpeg_cmd, check=False)
                                                        clips_dir = storage_core.get_clips_dir()
                                                        os.makedirs(clips_dir, exist_ok=True)
                                                        dest_path = os.path.join(clips_dir, out_name)
                                                        shutil.move(out_tmp, dest_path)
                                                        evidence_obj['clip_path'] = f"/clips/{out_name}"
                                                        evidence_obj['clip_duration_s'] = clip_duration_s
                                                        evidence_obj['event_window'] = {
                                                            'before_margin_s': float(int(self.fps * 2.0)) / max(1.0, float(self.fps)),
                                                            'after_margin_s': float(int(self.fps * 2.0)) / max(1.0, float(self.fps)),
                                                        }
                                                except Exception as clip_err:
                                                    print(f"AVISO: Falha gerar clipe de frames: {clip_err}")

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

                                                confidence_fraction = float(top_weight) / max(1.0, float(sum_weights))
                                                db_oc = models.Ocorrencia(
                                                    start_ts=start_ts_calc,
                                                    end_ts=end_ts_now,
                                                    duration_s=event_duration,
                                                    category='Vídeo Técnico',
                                                    type=top_class,
                                                    severity=_sev(event_duration),
                                                    confidence=confidence_fraction,
                                                    evidence=evidence_obj,
                                                )
                                                db.add(db_oc)
                                                db.commit()
                                                db.refresh(db_oc)
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
                                                # broadcast
                                                import asyncio
                                                try:
                                                    loop = asyncio.get_event_loop()
                                                except RuntimeError:
                                                    loop = None
                                                if loop and loop.is_running():
                                                    asyncio.run_coroutine_threadsafe(ws_router.manager.broadcast(message), loop)
                                                else:
                                                    try:
                                                        asyncio.run(ws_router.manager.broadcast(message))
                                                    except Exception:
                                                        pass

                                                self._last_report_ts = now
                                                # clear window preds to avoid immediate repeat
                                                self._window_preds = []
                                            finally:
                                                db.close()
                                # update last eval timestamp regardless (throttle evaluations)
                                self._last_window_eval_ts = now
                        else:
                            # fallback: original streak-based reporting
                            if pred_class == self._streak_class:
                                self._streak_count += 1
                            else:
                                self._streak_class = pred_class
                                self._streak_count = 1

                            eff_thresh = self.class_thresholds.get(pred_class, self.confidence_threshold)
                            try:
                                mavg_conf = sum(self._recent_conf) / max(1, len(self._recent_conf))
                            except Exception:
                                mavg_conf = float(confidence)

                            if pred_class != 'normal' and self._streak_count >= max(1, self.vote_k) and mavg_conf >= eff_thresh:
                                db = SessionLocal()
                                try:
                                    end_ts_now = datetime.now(timezone.utc)
                                    event_duration = max(1.0 / max(1.0, float(self.fps)), float(self._streak_count) / max(1.0, float(self.fps)))
                                    start_ts_calc = end_ts_now - timedelta(seconds=event_duration)
                                    evidence_obj = {'frame': f}
                                    db_oc = models.Ocorrencia(
                                        start_ts=start_ts_calc,
                                        end_ts=end_ts_now,
                                        duration_s=event_duration,
                                        category='Vídeo Técnico',
                                        type=pred_class,
                                        severity='Leve (C)',
                                        confidence=float(confidence),
                                        evidence=evidence_obj,
                                    )
                                    db.add(db_oc)
                                    db.commit()
                                    db.refresh(db_oc)
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
                                    import asyncio
                                    try:
                                        loop = asyncio.get_event_loop()
                                    except RuntimeError:
                                        loop = None
                                    if loop and loop.is_running():
                                        asyncio.run_coroutine_threadsafe(ws_router.manager.broadcast(message), loop)
                                    else:
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


class CaptureIngestor(SRTIngestor):
    """Ingestor that reads from a local capture device (dshow on Windows, v4l2 on Linux).
    It reuses most of SRTIngestor behaviour (writing HLS, extracting frames) but
    constructs ffmpeg commands for device input.
    """
    def start(self, device: str):
        if self._running:
            return False
        # create tmpdir to hold frames
        self._tmpdir = tempfile.mkdtemp(prefix='capture_frames_')
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        hls_dir = os.path.join(repo_root, 'static', 'hls')
        os.makedirs(hls_dir, exist_ok=True)
        out_hls = os.path.join(hls_dir, 'stream.m3u8')

        # Determine platform and build input spec
        import platform
        system = platform.system().lower()
        # device may be provided as a raw ffmpeg -i argument (good), or a simple path (/dev/video0)
        # For Windows (dshow) the caller should pass a dshow spec like: 'video="Device Name":audio="Microphone"'
        # We'll assemble the -f and -i parts accordingly
        if 'windows' in system:
            input_args = ['-f', 'dshow', '-i', device]
        elif 'linux' in system:
            # assume v4l2 for video-only devices; device could be '/dev/video0' or similar
            input_args = ['-f', 'v4l2', '-i', device]
        else:
            # fallback: treat device as generic -i value
            input_args = ['-i', device]

        # HLS ffmpeg command using the device input
        cmd_hls = ['ffmpeg', '-hide_banner'] + input_args + [
            '-fflags', '+genpts+igndts', '-avoid_negative_ts', 'make_zero',
            '-use_wallclock_as_timestamps', '1',
            '-c:v', 'copy', '-c:a', 'aac', '-ar', '44100', '-ac', '2',
            '-f', 'hls', '-hls_time', '2', '-hls_list_size', '5', '-hls_flags', 'delete_segments',
            out_hls
        ]

        log_path = os.path.join(hls_dir, 'hls_ffmpeg_capture.log')
        try:
            self._hls_log_file = open(log_path, 'ab')
            self._proc_hls = subprocess.Popen(cmd_hls, stdout=self._hls_log_file, stderr=self._hls_log_file)
        except Exception as e:
            print(f"ERRO: falha iniciar ffmpeg HLS (capture): {e}")
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

        time.sleep(2.0)
        if self._proc_hls.poll() is not None:
            print(f"ERRO: ffmpeg HLS (capture) process terminated immediately. See {log_path} for details.")
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

        # Frame extractor command: use same input_args and capture frames at fps
        out_pattern = os.path.join(self._tmpdir, 'frame_%06d.jpg')
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error'] + input_args + ['-vf', f'fps={self.fps}', '-q:v', '2', out_pattern]

        try:
            self._proc = subprocess.Popen(cmd)
        except Exception as e:
            print(f"ERRO: falha iniciar ffmpeg frames extractor (capture): {e}")
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

        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        return True


# Singleton instance for capture
capture_ingestor = CaptureIngestor()
