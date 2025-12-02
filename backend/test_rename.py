import os
import tensorflow as tf
import shutil

# Paths
video_path = r"D:\GitHub Desktop\horus_ai\backend\app\ml\models\video\odin_model_v4.5\video_model_finetune.keras"

def test_rename_load(path):
    print(f"Testing rename load for: {path}")
    temp_h5 = "temp_test_model.h5"
    
    try:
        shutil.copy2(path, temp_h5)
        print(f"Copied to {temp_h5}")
        
        try:
            model = tf.keras.models.load_model(temp_h5)
            print("SUCCESS: Loaded model after renaming to .h5")
            model.summary()
        except Exception as e:
            print(f"FAIL: Could not load even as .h5: {e}")
            
    finally:
        if os.path.exists(temp_h5):
            os.remove(temp_h5)

test_rename_load(video_path)
