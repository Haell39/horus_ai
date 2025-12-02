import os
import tensorflow as tf
import sys
import zipfile
import shutil

# Paths
video_path = r"D:\GitHub Desktop\horus_ai\backend\app\ml\models\video\odin_model_v4.5\video_model_finetune.keras"
audio_path = r"D:\GitHub Desktop\horus_ai\backend\app\ml\models\audio\heimdall_audio_model_ultra_v1\audio_model.keras"

def check_and_load(path, name):
    print(f"\n--- Testing {name} ---")
    print(f"Original Path: {path}")
    
    if not os.path.exists(path):
        print("File does not exist.")
        return

    print(f"Size: {os.path.getsize(path)} bytes")
    
    # 1. Check if valid zip
    if zipfile.is_zipfile(path):
        print("VALID: File is a valid zip archive.")
        try:
            with zipfile.ZipFile(path, 'r') as z:
                print(f"Zip contents: {z.namelist()[:5]} ...")
        except Exception as e:
            print(f"Error reading zip: {e}")
    else:
        print("INVALID: File is NOT a valid zip archive (corrupted?).")
        # If it's not a zip, maybe it's an H5 file?
        try:
            import h5py
            if h5py.is_hdf5(path):
                print("VALID: File is a valid HDF5 file (legacy format?).")
            else:
                print("INVALID: File is neither zip nor HDF5.")
        except ImportError:
            print("h5py not installed, cannot check HDF5.")

    # 2. Try loading from original path
    try:
        print("Attempting load from original path...")
        tf.keras.models.load_model(path)
        print("SUCCESS: Loaded from original path.")
    except Exception as e:
        print(f"FAIL: Could not load from original path: {e}")

    # 3. Try copying to a temp file in the current directory (to test path issues)
    temp_name = f"temp_{name.lower()}.keras"
    try:
        shutil.copy2(path, temp_name)
        abs_temp = os.path.abspath(temp_name)
        print(f"Copied to {abs_temp}")
        
        try:
            print(f"Attempting load from temp copy: {temp_name}")
            tf.keras.models.load_model(temp_name)
            print("SUCCESS: Loaded from temp copy.")
        except Exception as e:
            print(f"FAIL: Could not load from temp copy: {e}")
            
    except Exception as e:
        print(f"Error copying file: {e}")
    finally:
        if os.path.exists(temp_name):
            os.remove(temp_name)

check_and_load(video_path, "Video")
check_and_load(audio_path, "Audio")
