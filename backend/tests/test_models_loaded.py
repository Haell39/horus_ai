from app.ml import inference


def test_models_loaded_and_interpreters():
    """Ensure TFLite models were loaded at import time and interpreters exist."""
    assert hasattr(inference, "models_loaded")
    assert inference.models_loaded is True
    # interpreters must be present (audio/video)
    assert getattr(inference, "audio_interpreter", None) is not None
    assert getattr(inference, "video_interpreter", None) is not None
