import sys
sys.path.append(r"d:\GitHub Desktop\horus_ai\backend")
from app.ml import inference
print('models_loaded=', inference.models_loaded)
print('keras_audio_model=', type(inference.keras_audio_model))
print('keras_video_model=', type(inference.keras_video_model))
try:
    print('video output shape', inference.keras_video_model.output_shape)
except Exception as e:
    print('video output shape: error', e)
