import os
import tensorflow as tf
import sys

# Paths from the logs
video_path = r"D:\GitHub Desktop\horus_ai\backend\app\ml\models\video\odin_model_v4.5\video_model_finetune.keras"
audio_path = r"D:\GitHub Desktop\horus_ai\backend\app\ml\models\audio\heimdall_audio_model_ultra_v1\audio_model.keras"

print(f"Python executable: {sys.executable}")
print(f"TensorFlow version: {tf.__version__}")

def test_load(path, name):
    print(f"\nTesting {name} model at: {path}")
    if os.path.exists(path):
        print(f"File exists. Size: {os.path.getsize(path)} bytes")
        try:
            model = tf.keras.models.load_model(path)
            print(f"Successfully loaded {name} model.")
            model.summary()
        except Exception as e:
            print(f"Failed to load {name} model: {e}")
    else:
        print("File does not exist.")

test_load(video_path, "Video")
test_load(audio_path, "Audio")
