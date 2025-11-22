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
from app.core.config import settings as core_settings
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
        # periodic stream-buffer analysis
        try:
            self.stream_buffer_s = int(float(os.getenv('VIDEO_STREAM_BUFFER_S', '20')))
        except Exception:
            self.stream_buffer_s = 20
        self._last_stream_analysis_ts = None

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
                    try:
                        self._hls_log_file.close()
                    except Exception:
                        pass
            finally:
                self._hls_log_file = None
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
        prev_gray = None
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
                        # compute heuristics used later for window voting
                        try:
                            blur_var = inference._compute_blur_var(img)
                        except Exception:
                            blur_var = 0.0
                        try:
                            brightness = inference._compute_brightness(img)
                        except Exception:
                            brightness = 0.0
                        try:
                            motion = inference._compute_motion(prev_gray, img)
                        except Exception:
                            motion = float('inf')
                        try:
                            edge_density = inference._compute_edge_density(img)
                        except Exception:
                            edge_density = 0.0
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
                            # add to sliding window buffer (include heuristics so we can use them when model scores are low)
                            heur = {'blur_var': blur_var, 'brightness': brightness, 'edge_density': edge_density, 'motion': motion}
                            self._window_preds.append((now, idx, pred_class, float(confidence), heur))
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
                                for entry in self._window_preds:
                                    # unpack entries which now include heuristics
                                    try:
                                        ts_i, idx_i, cls_i, conf_i, heur_i = entry
                                    except Exception:
                                        # backward compatibility: skip malformed entries
                                        continue
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
                                    else:
                                        # consider heuristic-only support when model confidence is low/normal
                                        try:
                                            blur_v = float(heur_i.get('blur_var') or 999999.0)
                                        except Exception:
                                            blur_v = 999999.0
                                        try:
                                            brightness = float(heur_i.get('brightness') or 0.0)
                                        except Exception:
                                            brightness = 0.0
                                        try:
                                            edge_d = float(heur_i.get('edge_density') or 1.0)
                                        except Exception:
                                            edge_d = 1.0
                                        try:
                                            motion_v = heur_i.get('motion')
                                        except Exception:
                                            motion_v = None
                                        # thresholds (from env / config)
                                        try:
                                            blur_thr = float(os.getenv('VIDEO_BLUR_VAR_THRESHOLD', str(getattr(inference, 'DEFAULT_BLUR_THRESHOLD', 518.0))))
                                        except Exception:
                                            blur_thr = 518.0
                                        try:
                                            edge_thr = float(os.getenv('VIDEO_EDGE_DENSITY_THRESHOLD', str(getattr(inference, 'DEFAULT_EDGE_THRESHOLD', 0.015))))
                                        except Exception:
                                            edge_thr = 0.015
                                        try:
                                            motion_thr = float(os.getenv('VIDEO_MOTION_THRESHOLD', '2.0'))
                                        except Exception:
                                            motion_thr = 2.0
                                        heur_cls = None
                                        heur_conf = 0.0
                                        if brightness < float(os.getenv('VIDEO_BRIGHTNESS_LOW', '50.0')):
                                            heur_cls = 'fade'
                                            heur_conf = max(0.85, float(conf_i))
                                        elif motion_v is not None and motion_v != float('inf') and motion_v < motion_thr:
                                            heur_cls = 'freeze'
                                            heur_conf = max(0.80, float(conf_i))
                                        elif blur_v < blur_thr or edge_d < edge_thr:
                                            heur_cls = 'fora_foco'
                                            heur_conf = max(0.75, float(conf_i))
                                        if heur_cls:
                                            weights[heur_cls] = weights.get(heur_cls, 0.0) + float(heur_conf)
                                            sum_weights += float(heur_conf)
                                            if heur_cls not in class_indices:
                                                class_indices[heur_cls] = {'first_ts': ts_i, 'last_ts': ts_i, 'min_idx': idx_i, 'max_idx': idx_i}
                                            else:
                                                class_indices[heur_cls]['last_ts'] = ts_i
                                                class_indices[heur_cls]['min_idx'] = min(class_indices[heur_cls]['min_idx'], idx_i)
                                                class_indices[heur_cls]['max_idx'] = max(class_indices[heur_cls]['max_idx'], idx_i)

                                if weights and sum_weights > 0.0:
                                    top_class, top_weight = max(weights.items(), key=lambda x: x[1])
                                else:
                                    top_class, top_weight = (None, 0.0)

                                # require weighted majority (>50% of sum weights)
                                if top_class and top_weight > (sum_weights / 2.0):
                                    # positive span duration check for top_class
                                    ci = class_indices.get(top_class)
                                    # no muxing at this stage; proceed to build clip from frame indices
                                    muxed = False
                                    # build clip using min/max indices for the top class
                                    PRE_PAD_FRAMES = int(round(self.fps * 1.0))
                                    POST_PAD_FRAMES = int(round(self.fps * 1.0))
                                    if ci:
                                        start_idx = max(1, ci.get('min_idx', idx) - PRE_PAD_FRAMES)
                                        end_idx = ci.get('max_idx', idx) + POST_PAD_FRAMES
                                    else:
                                        start_idx = max(1, idx - PRE_PAD_FRAMES)
                                        end_idx = idx + POST_PAD_FRAMES
                                    num_frames = max(1, end_idx - start_idx + 1)
                                    clip_duration_s = float(num_frames) / max(1.0, float(self.fps))
                                    # ensure minimum clip duration (env VIDEO_MIN_EVENT_DURATION_S or 2s)
                                    try:
                                        min_span_s = float(os.getenv('VIDEO_MIN_EVENT_DURATION_S', '2.0'))
                                    except Exception:
                                        min_span_s = 2.0
                                    if clip_duration_s < min_span_s:
                                        needed_frames = int(max(1, round(min_span_s * float(self.fps))))
                                        extra = max(0, needed_frames - num_frames)
                                        # distribute extra frames before/after (prefer after)
                                        add_pre = extra // 2
                                        add_post = extra - add_pre
                                        start_idx = max(1, start_idx - add_pre)
                                        end_idx = end_idx + add_post
                                        num_frames = max(1, end_idx - start_idx + 1)
                                        clip_duration_s = float(num_frames) / max(1.0, float(self.fps))
                                    # cap clip duration to 65s; if larger, shorten from start keeping end anchored
                                    MAX_CLIP_S = 65.0
                                    if clip_duration_s > MAX_CLIP_S:
                                        desired_end_frame = end_idx
                                        max_frames = int(round(MAX_CLIP_S * float(self.fps)))
                                        start_idx = max(1, desired_end_frame - max_frames + 1)
                                        num_frames = max(1, desired_end_frame - start_idx + 1)
                                        clip_duration_s = float(num_frames) / max(1.0, float(self.fps))
                                    out_name = f"clip_{int(time.time())}_{idx}.mp4"
                                    out_tmp = os.path.join(self._tmpdir, out_name)
                                    ffmpeg_cmd = [
                                        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                                        '-fflags', '+genpts+igndts+discardcorrupt', '-err_detect', 'ignore_err',
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
                                        try:
                                            # To avoid race conditions where ffmpeg tries to read
                                            # frame files while they are still being written by
                                            # the extractor, copy the needed frames to a
                                            # temporary directory and run ffmpeg on that
                                            # stable snapshot.
                                            tmp_clip_dir = tempfile.mkdtemp(prefix='srt_clip_')
                                            try:
                                                for src in expected_files:
                                                    try:
                                                        shutil.copy2(src, os.path.join(tmp_clip_dir, os.path.basename(src)))
                                                    except Exception:
                                                        # if copying fails, break and abort
                                                        raise
                                                # run ffmpeg on the snapshot
                                                snap_pattern = os.path.join(tmp_clip_dir, 'frame_%06d.jpg')
                                                ffmpeg_cmd_snapshot = ffmpeg_cmd.copy()
                                                # replace input pattern (it's located at index of '-i' + 1)
                                                try:
                                                    i_idx = ffmpeg_cmd_snapshot.index('-i') + 1
                                                    ffmpeg_cmd_snapshot[i_idx] = snap_pattern
                                                except Exception:
                                                    pass
                                                proc = subprocess.run(ffmpeg_cmd_snapshot, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                                                if proc.returncode != 0:
                                                    stderr = proc.stderr.decode(errors='ignore') if proc.stderr is not None else ''
                                                    print(f"AVISO: ffmpeg clip generation failed rc={proc.returncode} stderr={stderr}")
                                                    # cleanup and skip
                                                    try:
                                                        if os.path.exists(out_tmp):
                                                            os.remove(out_tmp)
                                                    except Exception:
                                                        pass
                                                else:
                                                    clips_dir = storage_core.get_clips_dir()
                                                    os.makedirs(clips_dir, exist_ok=True)
                                                    dest_path = os.path.join(clips_dir, out_name)
                                                    shutil.move(out_tmp, dest_path)
                                            finally:
                                                try:
                                                    shutil.rmtree(tmp_clip_dir)
                                                except Exception:
                                                    pass
                                            evidence_obj['clip_path'] = f"/clips/{out_name}"
                                            evidence_obj['clip_duration_s'] = clip_duration_s
                                            evidence_obj['event_window'] = {
                                                'before_margin_s': float(PRE_PAD_FRAMES) / max(1.0, float(self.fps)),
                                                'after_margin_s': float(POST_PAD_FRAMES) / max(1.0, float(self.fps)),
                                            }
                                            # Attempt audio analysis on generated clip (conservative, non-blocking)
                                            try:
                                                from app.ml import inference as ml_infer
                                                # analyze_audio_segments expects a path to a file and returns (class, conf, event_time_s)
                                                try:
                                                    audio_cls, audio_conf, audio_time = ml_infer.analyze_audio_segments(dest_path)
                                                except Exception:
                                                    audio_cls, audio_conf, audio_time = 'normal', 0.0, None
                                                evidence_obj['audio_analysis'] = {
                                                    'class': audio_cls,
                                                    'confidence': float(audio_conf),
                                                    'event_time_s': audio_time,
                                                }
                                            except Exception:
                                                # do not fail clip persistence if audio analysis fails
                                                pass
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
                                    db = SessionLocal()
                                    try:
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
                                    finally:
                                        try:
                                            db.close()
                                        except Exception:
                                            pass

                                    # clear window preds to avoid immediate repeat
                                    self._window_preds = []
                                    self._last_report_ts = now
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
                        # update prev_gray for motion calculation in next frame
                        try:
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            prev_gray = cv2.resize(gray, (inference.INPUT_WIDTH, inference.INPUT_HEIGHT))
                        except Exception:
                            prev_gray = None
                    except Exception:
                        # ignore per-frame exceptions to keep ingest running
                        pass
                time.sleep(max(0.1, 1.0 / max(1, int(self.fps))))
                # periodic stream-buffer analysis (non-blocking)
                try:
                    now_loop = datetime.now(timezone.utc)
                    if (self._last_stream_analysis_ts is None or (now_loop - self._last_stream_analysis_ts).total_seconds() >= max(1.0, float(self.stream_buffer_s))):
                        # decide end index from last_seen_index
                        try:
                            end_idx_local = last_seen_index
                        except Exception:
                            end_idx_local = None
                        if end_idx_local and end_idx_local > 0:
                            try:
                                # spawn background thread to avoid blocking main loop
                                t = threading.Thread(target=self._run_stream_buffer_analysis, args=(end_idx_local,), daemon=True)
                                t.start()
                                self._last_stream_analysis_ts = now_loop
                            except Exception:
                                pass
                except Exception:
                    pass
            except Exception:
                time.sleep(1)
    
    def _run_stream_buffer_analysis(self, end_idx: int):
        """Create a temporary clip from the last `stream_buffer_s` seconds of frames,
        run the same video+audio analysis used by the upload endpoint, and
        persist an occurrence if the aggregated decision passes thresholds.
        This runs in a background thread and must not raise.
        """
        try:
            if not self._tmpdir or not os.path.exists(self._tmpdir):
                return
            # compute frames window
            try:
                num_frames = max(1, int(round(float(self.stream_buffer_s) * float(self.fps))))
            except Exception:
                num_frames = int(max(1, round(20 * max(1.0, float(self.fps)))))
            start_idx = max(1, int(end_idx) - num_frames + 1)
            end_idx = int(end_idx)

            expected_files = [os.path.join(self._tmpdir, f'frame_{i:06d}.jpg') for i in range(start_idx, end_idx + 1)]
            existing = [p for p in expected_files if os.path.exists(p)]
            if not existing:
                return

            out_name = f"streambuf_{int(time.time())}_{end_idx}.mp4"
            out_tmp = os.path.join(self._tmpdir, out_name)
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                '-fflags', '+genpts+igndts+discardcorrupt', '-err_detect', 'ignore_err',
                '-framerate', str(self.fps),
                '-start_number', str(start_idx),
                '-i', os.path.join(self._tmpdir, 'frame_%06d.jpg'),
                '-frames:v', str(max(1, end_idx - start_idx + 1)),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_tmp
            ]
            try:
                subprocess.run(ffmpeg_cmd, check=False)
            except Exception:
                return

            # move to public clips dir
            try:
                clips_dir = storage_core.get_clips_dir()
                os.makedirs(clips_dir, exist_ok=True)
                dest_path = os.path.join(clips_dir, out_name)
                shutil.move(out_tmp, dest_path)
                public_clip = f"/clips/{out_name}"
            except Exception:
                try:
                    if os.path.exists(out_tmp):
                        os.remove(out_tmp)
                except Exception:
                    pass
                return

            # Run analyses (video + optional audio)
            try:
                video_pred, video_conf, video_time = inference.analyze_video_frames(dest_path, sample_rate_hz=2.0)
            except Exception:
                video_pred, video_conf, video_time = 'normal', 0.0, None

            audio_pred, audio_conf, audio_time = 'normal', 0.0, None
            try:
                if not bool(getattr(core_settings, 'VIDEO_DISABLE_AUDIO_PROCESSING', False)):
                    try:
                        audio_pred, audio_conf, audio_time = inference.analyze_audio_segments(dest_path)
                    except Exception:
                        audio_pred, audio_conf, audio_time = 'normal', 0.0, None
                    else:
                        # Log that audio analysis was executed (helpful to debug stream vs upload parity)
                        try:
                            print(f"INFO: Stream buffer audio analysis result -> class={audio_pred}, conf={audio_conf:.3f}, time_s={audio_time}")
                        except Exception:
                            pass
            except Exception:
                audio_pred, audio_conf, audio_time = 'normal', 0.0, None

            try:
                diag = inference.analyze_video_frames_diagnostic(dest_path, k=3, sample_rate_hz=2.0, max_samples=200)
            except Exception:
                diag = []

            # Aggregation logic (mirrors upload endpoint)
            aggregated = None
            try:
                if diag:
                    score_sum = {}
                    count_above = {}
                    total_samples = 0
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
                        thr = per_class_thresh.get(cls.upper(), per_class_thresh.get('DEFAULT', float(core_settings.VIDEO_THRESH_DEFAULT)))
                        if sc >= float(thr):
                            count_above[cls] = count_above.get(cls, 0) + 1
                    if score_sum:
                        best_cls = max(score_sum.items(), key=lambda x: x[1])[0]
                        summed = score_sum[best_cls]
                        avg_conf = summed / (total_samples or 1)
                        supporting = count_above.get(best_cls, 0)
                        if supporting >= vote_k or avg_conf >= float(per_class_thresh.get(best_cls.upper(), per_class_thresh.get('DEFAULT', float(core_settings.VIDEO_THRESH_DEFAULT)))):
                            aggregated = {'class': best_cls, 'confidence': float(avg_conf), 'samples': total_samples, 'supporting': supporting}
                        else:
                            aggregated = None
            except Exception:
                aggregated = None

            # Decide final prediction
            final_pred = video_pred
            final_conf = float(video_conf or 0.0)
            if aggregated:
                final_pred = aggregated.get('class') or final_pred
                final_conf = float(aggregated.get('confidence') or final_conf)

            try:
                print(f"DEBUG: Stream buffer initial video decision -> {video_pred} ({video_conf:.3f}), aggregated -> {aggregated}, audio -> {audio_pred} ({audio_conf:.3f})")
            except Exception:
                pass

            # strong video-normal suppression of audio
            try:
                per_class_thresh = core_settings.video_thresholds()
                video_thr = float(per_class_thresh.get(str(video_pred).upper(), per_class_thresh.get('DEFAULT', float(core_settings.VIDEO_THRESH_DEFAULT))))
                if (str(video_pred).lower() == 'normal') and (final_conf >= video_thr):
                    final_pred = 'normal'
                    final_conf = float(final_conf)
                    audio_pred, audio_conf = 'normal', 0.0
            except Exception:
                pass

            # Audio override rules (conservative)
            try:
                audio_thresh_map = core_settings.audio_thresholds()
                audio_thresh = audio_thresh_map.get(str(audio_pred).upper(), audio_thresh_map.get('DEFAULT', float(core_settings.AUDIO_THRESH_DEFAULT))) if audio_pred else float(core_settings.AUDIO_THRESH_DEFAULT)
            except Exception:
                audio_thresh = float(core_settings.AUDIO_THRESH_DEFAULT)

            try:
                audio_conf_val = float(audio_conf or 0.0)
            except Exception:
                audio_conf_val = 0.0
            try:
                video_conf_val = float(final_conf or 0.0)
            except Exception:
                video_conf_val = 0.0

            try:
                if audio_pred and audio_conf_val > video_conf_val and audio_conf_val >= 0.90:
                    if str(audio_pred).lower() != 'normal':
                        final_pred = audio_pred
                        final_conf = float(audio_conf_val)
                    else:
                        if str(video_pred).lower() != 'normal' and video_conf_val <= 0.75:
                            final_pred = 'normal'
                            final_conf = float(audio_conf_val)
                else:
                    if bool(core_settings.VIDEO_ALLOW_AUDIO_OVERRIDE):
                        if audio_pred and (audio_conf or 0.0) > final_conf:
                            delta = (audio_conf or 0.0) - final_conf
                            if (audio_conf or 0.0) >= float(audio_thresh) or (delta >= 0.04 and (audio_conf or 0.0) >= max(0.55, float(audio_thresh) - 0.05)):
                                final_pred = audio_pred
                                final_conf = float(audio_conf or 0.0)
            except Exception:
                pass

            try:
                if final_pred and audio_pred and final_pred == audio_pred:
                    final_threshold = float(audio_thresh)
                else:
                    per_class_thresh = core_settings.video_thresholds()
                    final_threshold = float(per_class_thresh.get(str(final_pred).upper(), per_class_thresh.get('DEFAULT', float(core_settings.VIDEO_THRESH_DEFAULT))))
            except Exception:
                final_threshold = float(core_settings.VIDEO_THRESH_DEFAULT)

            # If decision passes threshold and it's not 'normal', persist occurrence
            if final_pred and str(final_pred).lower() != 'normal' and float(final_conf or 0.0) >= float(final_threshold):
                try:
                    # avoid duplicates
                    now = datetime.now(timezone.utc)
                    if self._last_report_ts and (now - self._last_report_ts).total_seconds() < max(1.0, float(self.stream_buffer_s) / 2.0):
                        return

                    # compute clip duration (ffprobe)
                    clip_dur = 0.0
                    try:
                        proc = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', dest_path], capture_output=True, text=True, timeout=10)
                        out = proc.stdout.strip()
                        clip_dur = float(out) if out else 0.0
                    except Exception:
                        clip_dur = float(max(1.0, float(self.stream_buffer_s)))

                    def _sev_local(d: float) -> str:
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

                    start_ts_calc = now - timedelta(seconds=float(clip_dur or self.stream_buffer_s))
                    end_ts_now = now
                    evidence_obj = {'clip_path': public_clip, 'clip_duration_s': float(clip_dur or self.stream_buffer_s)}
                    try:
                        evidence_obj['audio_analysis'] = {'class': audio_pred, 'confidence': float(audio_conf or 0.0), 'event_time_s': audio_time}
                    except Exception:
                        pass

                    db = SessionLocal()
                    try:
                        db_oc = models.Ocorrencia(
                            start_ts=start_ts_calc,
                            end_ts=end_ts_now,
                            duration_s=float(clip_dur or self.stream_buffer_s),
                            category='Stream Buffered',
                            type=final_pred,
                            severity=_sev_local(float(clip_dur or self.stream_buffer_s)),
                            confidence=float(final_conf or 0.0),
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

                        self._last_report_ts = now
                    finally:
                        db.close()
                except Exception:
                    pass
        except Exception:
            # never let this background analysis crash
            try:
                return
            except Exception:
                return
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
