import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import h5py
import shutil

AUDIO_MODEL_PATH = r"d:\GitHub Desktop\horus_ai\backend\app\ml\models\audio\heimdall_audio_model_ultra_v1\audio_model.keras"
TEMP_H5_PATH = r"d:\GitHub Desktop\horus_ai\backend\app\ml\models\audio\heimdall_audio_model_ultra_v1\audio_model_temp.h5"

def try_load_as_h5():
    print(f"\nCopying to {TEMP_H5_PATH} to test loading as .h5...")
    shutil.copy2(AUDIO_MODEL_PATH, TEMP_H5_PATH)
    
    print("Attempting to load with keras.models.load_model (as .h5)...")
    try:
        model = keras.models.load_model(TEMP_H5_PATH)
        print("Success! Model loaded as .h5.")
        model.summary()
        
        # If successful, we can save it as a proper .keras file
        FIXED_PATH = r"d:\GitHub Desktop\horus_ai\backend\app\ml\models\audio\heimdall_audio_model_ultra_v1\audio_model_fixed.keras"
        print(f"Saving as proper Keras 3 format to {FIXED_PATH}...")
        model.save(FIXED_PATH)
        print("Saved.")
        
    except Exception as e:
        print(f"Failed to load as .h5: {e}")
    finally:
        if os.path.exists(TEMP_H5_PATH):
            os.remove(TEMP_H5_PATH)

if __name__ == "__main__":
    try_load_as_h5()
