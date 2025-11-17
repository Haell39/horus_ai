"""Minimal inference test runner.

Given a media file path, it walks the media with 3s windows (hop 1s), runs the Keras model and prints top-1 + confidence per segment.

Usage:
  python run_inference_test.py --model models/audio_model_finetune.keras --input /path/to/file.mp4
"""
from __future__ import annotations

import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from preprocess import load_audio_segment, audio_segment_to_input, SEGMENT_DURATION, DEFAULT_SR
import csv


def load_labels(labels_path: Path):
    with open(labels_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if r]
    if not rows:
        return []
    if rows[0][0].lower() == "class":
        return [r[0] for r in rows[1:]]
    return [r[0] for r in rows]


def run(model_path: Path, input_path: Path, labels_path: Path):
    model = tf.keras.models.load_model(str(model_path))
    labels = load_labels(labels_path)
    if not labels:
        raise SystemExit("No labels found")

    # estimate duration via librosa.get_duration
    import librosa

    duration = librosa.get_duration(filename=str(input_path))
    t = 0.0
    print(f"File duration: {duration:.2f}s, running segments of {SEGMENT_DURATION}s with hop 1s")
    while t + SEGMENT_DURATION <= duration:
        y = load_audio_segment(str(input_path), t, duration_s=SEGMENT_DURATION, sr=DEFAULT_SR)
        x = audio_segment_to_input(y, sr=DEFAULT_SR)  # (1,160,160,3)
        preds = model(x, training=False)
        # if logits, apply softmax
        probs = tf.nn.softmax(preds, axis=-1).numpy()[0]
        top_idx = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        print(f"segment {t:.1f}s - {t+SEGMENT_DURATION:.1f}s => {labels[top_idx]} ({top_conf:.3f})")
        t += 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--labels", default=Path(__file__).parents[1].parent / "labels.csv")
    args = parser.parse_args()
    run(Path(args.model), Path(args.input), Path(args.labels))


if __name__ == "__main__":
    main()
