import io
from fastapi.testclient import TestClient
from app.main import app
from app import ml


def make_dummy_file_bytes(size=1024):
    return b"\0" * size


def test_analyze_audio_monkeypatched(monkeypatch, tmp_path):
    # monkeypatch high-level analyzer to make test deterministic
    def fake_analyze_audio_segments(path: str):
        return ("normal", 0.95, None)

    monkeypatch.setattr(ml.inference, "analyze_audio_segments", fake_analyze_audio_segments)

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
    # monkeypatch high-level analyzer to make test deterministic
    def fake_analyze_video_frames(path: str):
        return ("normal", 0.88, None)

    monkeypatch.setattr(ml.inference, "analyze_video_frames", fake_analyze_video_frames)

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
