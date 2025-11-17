"""
Simple diagnostic runner for the video model.
Usage:
    python run_video_diagnostic.py /path/to/video.mp4

It imports `inference` and prints the top-k per sampled frame (diagnostic) and the overall analyze_video_frames result.
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.ml import inference


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_video_diagnostic.py <video_file>")
        return
    video = sys.argv[1]
    if not Path(video).exists():
        print(f"Video file not found: {video}")
        return

    # If metadata indicates sequence model (frames > 1) use sequence diagnostic
    meta = getattr(inference, 'MODEL_METADATA', {}) or {}
    seq_len = None
    sample_hz = meta.get('frame_rate_sampled', 2)
    seg_dur = meta.get('segment_duration_s')
    if seg_dur and meta.get('input_shape') and isinstance(meta.get('input_shape'), list):
        seq_len = int(meta.get('input_shape')[0])

    if seq_len and seq_len > 1:
        print(f"Running sequence diagnostic using seq_len={seq_len}, sample_hz={sample_hz}...")
        # open video and sample frames, build sliding windows of seq_len
        import cv2
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print("Failed to open video for diagnostic")
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        try:
            fps = float(fps)
        except Exception:
            fps = 30.0
        frame_step = max(1, int(round(fps / float(max(0.001, sample_hz)))))
        frames = []
        read = 0
        while True:
            ret, fr = cap.read()
            if not ret:
                break
            read += 1
            if read % frame_step != 0:
                continue
            # resize to model expected size if available
            try:
                h = inference.INPUT_HEIGHT
                w = inference.INPUT_WIDTH
                fr_rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                fr_rs = cv2.resize(fr_rgb, (w, h)).astype('float32')
            except Exception:
                continue
            frames.append(fr_rs)
        cap.release()

        # build sliding windows
        import numpy as np
        windows = []
        for i in range(0, max(0, len(frames) - seq_len + 1)):
            seq = np.stack(frames[i:i+seq_len], axis=0) # shape (seq_len, H, W, C)
            seq = np.expand_dims(seq, axis=0) # (1, seq_len, H, W, C)
            windows.append((i, seq))

        print(f"Diagnostic windows to evaluate: {len(windows)}")
        for idx, seq in windows[:50]:
            top_class, conf = inference.run_keras_sequence_inference(getattr(inference, 'keras_video_model'), seq, getattr(inference, 'MODEL_CLASSES', []))
            print(f"win#{idx} -> {top_class}:{conf:.3f}")

        # also run the heuristic analyzer for comparison
        print('\nRunning high-level analyze_video_frames (heuristic + model combined)...')
        cls, conf, t = inference.analyze_video_frames(video, sample_rate_hz=sample_hz)
        print(f"Result: class={cls} conf={conf:.4f} time_s={t}")
    else:
        print("Running diagnostic top-k for sampled frames...")
        diag = inference.analyze_video_frames_diagnostic(video, k=3, sample_rate_hz=2.0, max_samples=50)
        for item in diag[:50]:
            t = item.get('time_s')
            topk = item.get('topk', [])
            print(f"t={t:.2f}s -> " + ", ".join([f"{d['class']}:{d['score']:.3f}" for d in topk]))

        print('\nRunning high-level analyze_video_frames (heuristic + model combined)...')
        cls, conf, t = inference.analyze_video_frames(video, sample_rate_hz=2.0)
        print(f"Result: class={cls} conf={conf:.4f} time_s={t}")


if __name__ == '__main__':
    main()
