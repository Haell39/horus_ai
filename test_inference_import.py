import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

try:
    from backend.app.ml import inference
    print("Successfully imported backend.app.ml.inference")
    
    if hasattr(inference, 'analyze_video_frames'):
        print("analyze_video_frames is defined")
    else:
        print("ERROR: analyze_video_frames is NOT defined")
        
    if hasattr(inference, 'analyze_audio_segments'):
        print("analyze_audio_segments is defined")
    else:
        print("ERROR: analyze_audio_segments is NOT defined")

except Exception as e:
    print(f"Failed to import inference: {e}")
