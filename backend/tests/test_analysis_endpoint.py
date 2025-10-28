import io
from fastapi.testclient import TestClient
from app.main import app
from app import ml


def make_dummy_file_bytes(size=1024):
    return b"\0" * size


def test_analyze_audio_monkeypatched(monkeypatch, tmp_path):
    # monkeypatch heavy ML preprocess and inference to make test deterministic
    def fake_preprocess_audio(path):
        import numpy as np
        return np.zeros((1, 224, 224, 3), dtype=np.float32)

    def fake_run_audio(input_data):
        return ("normal", 0.95)

    monkeypatch.setattr(ml.inference, "preprocess_audio", fake_preprocess_audio)
    monkeypatch.setattr(ml.inference, "run_audio_inference", fake_run_audio)

    client = TestClient(app)
    sample = tmp_path / "sample.wav"
    sample.write_bytes(make_dummy_file_bytes(2048))

    with open(sample, "rb") as fh:
        files = {"file": ("sample.wav", fh, "audio/wav")}
        data = {"media_type": "audio"}
        resp = client.post("/api/v1/analyze", files=files, data=data)

    assert resp.status_code == 200
    body = resp.json()
    assert body["media_type"] == "audio"
    assert isinstance(body.get("predictions"), list)


def test_analyze_video_monkeypatched(monkeypatch, tmp_path):
    # monkeypatch heavy ML preprocess and inference to make test deterministic
    def fake_preprocess_video(path):
        import numpy as np
        return np.zeros((1, 224, 224, 3), dtype=np.float32)

    def fake_run_video(input_data):
        return ("normal", 0.88)

    monkeypatch.setattr(ml.inference, "preprocess_video_frame", fake_preprocess_video)
    monkeypatch.setattr(ml.inference, "run_video_inference", fake_run_video)

    client = TestClient(app)
    sample = tmp_path / "sample.mp4"
    sample.write_bytes(make_dummy_file_bytes(4096))

    with open(sample, "rb") as fh:
        files = {"file": ("sample.mp4", fh, "video/mp4")}
        data = {"media_type": "video"}
        resp = client.post("/api/v1/analyze", files=files, data=data)

    assert resp.status_code == 200
    body = resp.json()
    assert body["media_type"] == "video"
    assert isinstance(body.get("predictions"), list)
