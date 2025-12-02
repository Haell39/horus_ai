import os
os.environ["TF_USE_LEGACY_KERAS"] = "0"
import keras
import json

print(f"Keras version: {keras.__version__}")
print("Available classes in keras.models:")
print(dir(keras.models))

try:
    from keras.models import Functional
    print("Functional class found in keras.models")
except ImportError:
    print("Functional class NOT found in keras.models")

try:
    from keras.src.models.functional import Functional
    print("Functional class found in keras.src.models.functional")
except ImportError:
    print("Functional class NOT found in keras.src.models.functional")
