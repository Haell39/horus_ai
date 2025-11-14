"""Preprocessing utilities used by training and inference test scripts.

The pipeline matches the repo backend expectations:
- compute mel-spectrogram (n_mels=128, n_fft=2048, hop_length=512, fmax=8000)
- convert to dB
- min-max normalize per-segment to [0,1]
- stack to 3 channels, convert to uint8 0..255
- resize to 160x160
- convert to float32 and apply MobileNetV2 preprocessing: (x - 127.5) / 127.5

Returns a float32 numpy array shaped (1,160,160,3).
"""
from __future__ import annotations

import numpy as np
from PIL import Image
import librosa


DEFAULT_SR = 16000
SEGMENT_DURATION = 3.0  # seconds
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMAX = 8000
TARGET_SIZE = (160, 160)


def audio_segment_to_input(y: np.ndarray, sr: int = DEFAULT_SR) -> np.ndarray:
    """Convert a 1D audio segment (numpy) to model input tensor (1,160,160,3) float32.

    Expects y to contain exactly SEGMENT_DURATION*sr samples (or will be padded/truncated).
    """
    # mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, fmax=FMAX
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # min-max normalize per-segment to [0,1]
    minv = float(np.min(mel_db))
    maxv = float(np.max(mel_db))
    if maxv - minv < 1e-6:
        norm = np.zeros_like(mel_db, dtype=np.float32)
    else:
        norm = (mel_db - minv) / (maxv - minv)

    # scale to 0..255 uint8 and stack to 3 channels
    img = (norm * 255.0).astype(np.uint8)
    img_3 = np.stack([img, img, img], axis=-1)  # (H,W,3)

    # resize to TARGET_SIZE with PIL
    pil = Image.fromarray(img_3)
    pil = pil.resize(TARGET_SIZE, resample=Image.BILINEAR)
    arr = np.array(pil).astype(np.float32)

    # MobileNetV2 preprocessing: (x - 127.5) / 127.5
    arr = (arr - 127.5) / 127.5

    # expand batch dim
    return np.expand_dims(arr, axis=0)


def load_audio_segment(path: str, start_s: float, duration_s: float = SEGMENT_DURATION, sr: int = DEFAULT_SR) -> np.ndarray:
    """Load a segment from an audio file (or video container supported by librosa).

    Returns a mono numpy array of length ~ duration_s*sr (may be shorter at file end).
    """
    # librosa.load supports many audio/video containers if ffmpeg is installed
    y, file_sr = librosa.load(path, sr=sr, mono=True, offset=start_s, duration=duration_s)
    # pad if needed
    expected_len = int(round(duration_s * sr))
    if y.shape[0] < expected_len:
        y = np.pad(y, (0, expected_len - y.shape[0]))
    elif y.shape[0] > expected_len:
        y = y[:expected_len]
    return y
