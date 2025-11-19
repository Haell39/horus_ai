#!/usr/bin/env python3
"""
Run audio and video inference over every MP4 in test_clips/ and save results.

Outputs JSON to `backend/tmp/test_clips_results.json` with one entry per file:
{ filename, audio: {class, confidence, segments}, video: {class, confidence, event_time} }

Run from repository root: python backend/scripts/run_models_on_test_clips.py
"""
import os
import json
import math
import traceback
import sys

# Ensure repository root is on sys.path so we can import backend.* packages when running
ROOT = os.path.abspath(os.path.join(os.getcwd()))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from backend.app.ml import inference
except Exception as e:
    print("ERROR importing inference module:", e)
    raise

import librosa


TEST_DIR = os.path.abspath(os.path.join(os.getcwd(), 'test_clips'))
OUT_DIR = os.path.abspath(os.path.join(os.getcwd(), 'backend', 'tmp'))
OUT_FILE = os.path.join(OUT_DIR, 'test_clips_results.json')

os.makedirs(OUT_DIR, exist_ok=True)


def count_audio_segments(file_path, sr, seg_dur, hop_dur):
    try:
        y, _ = librosa.load(file_path, sr=sr)
        seg_samples = int(seg_dur * sr)
        hop_samples = max(1, int(hop_dur * sr))
        if len(y) < seg_samples:
            return 0
        return 1 + (len(y) - seg_samples) // hop_samples
    except Exception:
        return 0


def process_file(path):
    out = {'filename': os.path.basename(path), 'path': path}
    try:
        # Audio analysis
        md = inference.MODEL_METADATA or {}
        sr = int(md.get('sample_rate', 16000))
        seg_dur = float(md.get('segment_duration', 3.0))
        overlap = float(md.get('overlap', 0.5))
        hop = seg_dur * (1.0 - overlap) if 0 < overlap < 1 else float(md.get('hop_duration', 1.0))

        audio_class, audio_conf, audio_event_time = inference.analyze_audio_segments(path)
        segments = count_audio_segments(path, sr, seg_dur, hop)
        out['audio'] = {'class': audio_class, 'confidence': float(audio_conf), 'segments': int(segments), 'event_time_s': float(audio_event_time) if audio_event_time is not None else None}

        # Video analysis
        video_class, video_conf, event_time = inference.analyze_video_frames(path, sample_rate_hz=2.0)
        out['video'] = {'class': video_class, 'confidence': float(video_conf), 'event_time_s': float(event_time) if event_time is not None else None}

    except Exception as e:
        out['error'] = str(e)
        out['traceback'] = traceback.format_exc()
    return out


def main():
    results = []
    if not os.path.exists(TEST_DIR):
        print("Test clips directory not found:", TEST_DIR)
        return
    files = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if f.lower().endswith('.mp4')]
    if not files:
        print("No mp4 files found in:", TEST_DIR)
        return

    for f in files:
        print("Processing:", f)
        r = process_file(f)
        results.append(r)
        # save incrementally to avoid losing progress
        with open(OUT_FILE, 'w', encoding='utf-8') as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)

    print("Wrote results to:", OUT_FILE)


if __name__ == '__main__':
    main()
