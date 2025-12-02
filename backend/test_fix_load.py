import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import tensorflow as tf
import shutil

# Path to the model (renamed to .h5 for testing)
original_path = r"D:\GitHub Desktop\horus_ai\backend\app\ml\models\video\odin_model_v4.5\video_model_finetune.keras"
temp_h5 = "temp_fix_test.h5"

print(f"Keras version: {keras.__version__}")

def test_load():
    try:
        shutil.copy2(original_path, temp_h5)
        print(f"Copied to {temp_h5}")
        
        # Try loading with keras directly
        print("Attempting load with keras.models.load_model...")
        try:
            model = keras.models.load_model(temp_h5)
            print("SUCCESS: Loaded with keras.models.load_model")
            return
        except Exception as e:
            print(f"FAIL keras load: {e}")

        # Try registering Functional (hack)
        print("Attempting to register Functional...")
        try:
            @keras.saving.register_keras_serializable(package='keras.models')
            class Functional(keras.models.Model):
                pass
            
            model = keras.models.load_model(temp_h5)
            print("SUCCESS: Loaded after registering Functional")
        except Exception as e:
            print(f"FAIL register hack: {e}")

    finally:
        if os.path.exists(temp_h5):
            os.remove(temp_h5)

test_load()
