import os
import sys
import numpy as np

# Ensure backend package is importable when pytest is run from repo root
THIS_DIR = os.path.dirname(__file__)
BACKEND_ROOT = os.path.abspath(os.path.join(THIS_DIR, '..'))
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from types import ModuleType
from PIL import Image

# Create a minimal fake `cv2` module so tests can run in environments
# where OpenCV isn't installed. We implement only the small subset used
# by `analyze_video_frames` (cvtColor, resize, Laplacian, Canny, absdiff
# and a few constants). The implementations are simple and rely on numpy/PIL.
fake_cv2 = ModuleType('cv2')

# Constants
fake_cv2.COLOR_BGR2GRAY = 0
fake_cv2.CV_64F = 6
fake_cv2.CAP_PROP_FPS = 5
fake_cv2.CAP_PROP_POS_MSEC = 0
fake_cv2.COLOR_BGR2RGB = 1

def _cvtColor_bgr2gray(img, code):
    # img: HxWx3 BGR uint8 -> return HxW uint8 grayscale via luminance
    if img is None:
        return None
    # If requested conversion is to RGB, simply swap channels
    if code == fake_cv2.COLOR_BGR2RGB:
        return img[..., ::-1]
    arr = img.astype(np.float32)
    # Use simple luminosity formula (RGB order assumed BGR but uniform frames are symmetric)
    r = arr[..., 2]
    g = arr[..., 1]
    b = arr[..., 0]
    gray = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    return gray

def _resize(img, size):
    # size: (width, height) like cv2.resize
    if img is None:
        return None
    w, h = size
    pil = Image.fromarray(img)
    pil2 = pil.resize((w, h))
    return np.array(pil2)

def _laplacian(img, dtype):
    # simple discrete Laplacian kernel
    arr = img.astype(np.float64)
    lap = (np.roll(arr, 1, axis=0) + np.roll(arr, -1, axis=0) + np.roll(arr, 1, axis=1) + np.roll(arr, -1, axis=1) - 4 * arr)
    return lap

def _absdiff(a, b):
    return np.abs(a.astype(np.int32) - b.astype(np.int32)).astype(np.uint8)

def _canny(img, t1, t2):
    # very naive edge detector: gradient magnitude > small epsilon
    gx = np.abs(np.diff(img, axis=1, prepend=img[:, :1]))
    gy = np.abs(np.diff(img, axis=0, prepend=img[:1, :]))
    mag = (gx.astype(np.float32) + gy.astype(np.float32))
    edges = (mag > 10).astype(np.uint8) * 255
    return edges

fake_cv2.cvtColor = _cvtColor_bgr2gray
fake_cv2.resize = _resize
fake_cv2.Laplacian = _laplacian
fake_cv2.absdiff = _absdiff
fake_cv2.Canny = _canny

# Insert into sys.modules so importers see it
sys.modules['cv2'] = fake_cv2

# Provide a very small fake `tensorflow` so the module import succeeds in environments
# without full TF installed. We only implement the parts used during import and
# softmax computation in tests.
fake_tf = ModuleType('tensorflow')

class _SoftmaxWrapper:
    def __init__(self, arr):
        self._arr = np.array(arr, dtype=np.float32)
    def numpy(self):
        a = np.exp(self._arr - np.max(self._arr))
        s = np.sum(a)
        return a / s if s > 0 else a

class _NN:
    @staticmethod
    def softmax(x):
        return _SoftmaxWrapper(x)

class _KerasModels:
    @staticmethod
    def load_model(p):
        raise FileNotFoundError("no keras model in test environment")

class _Image:
    @staticmethod
    def resize(x, size, method=None):
        # x is a numpy array; use PIL for resizing
        pil = Image.fromarray(x.astype(np.uint8))
        w, h = size
        return np.array(pil.resize((w, h))).astype(np.float32)

fake_tf.nn = _NN()
fake_tf.keras = ModuleType('keras')
fake_tf.keras.models = _KerasModels()
fake_tf.keras.Model = type('Model', (), {})
fake_tf.convert_to_tensor = lambda x, dtype=None: x
fake_tf.image = _Image()
fake_tf.lite = ModuleType('lite')
class _Interpreter:
    pass
fake_tf.lite.Interpreter = _Interpreter

sys.modules['tensorflow'] = fake_tf

# Minimal placeholder for librosa (used by inference.py but not invoked in this unit test)
sys.modules['librosa'] = ModuleType('librosa')

from app.ml import inference


class FakeModel:
    def __init__(self, num_classes=8, targ_idx=None):
        self.num_classes = num_classes
        # target index to be highest in softmax; if None produce low-confidence other
        self.targ_idx = targ_idx if targ_idx is not None else (num_classes - 1)

    def predict(self, input_data):
        # return a single batch of logits with target index highest
        logits = np.zeros((1, self.num_classes), dtype=np.float32)
        logits[0, :] = 0.1
        logits[0, self.targ_idx] = 0.2
        return logits


class FakeCapture:
    """A minimal fake VideoCapture that serves a list of frames."""
    def __init__(self, frames, fps=30):
        self.frames = frames
        self.idx = 0
        self._fps = fps

    def isOpened(self):
        return True

    def read(self):
        if self.idx < len(self.frames):
            f = self.frames[self.idx]
            self.idx += 1
            return True, f
        return False, None

    def get(self, prop):
        # support CAP_PROP_FPS and CAP_PROP_POS_MSEC
        if prop == fake_cv2.CAP_PROP_FPS:
            return float(self._fps)
        if prop == fake_cv2.CAP_PROP_POS_MSEC:
            # return time of last delivered frame in ms
            return float(max(0, self.idx - 1)) * (1000.0 / float(self._fps))
        return 0

    def release(self):
        pass


def make_dark_blurred_frame(width=320, height=240, value=30):
    """Return a dark, low-detail (nearly uniform) BGR frame as uint8 array."""
    arr = np.full((height, width, 3), fill_value=value, dtype=np.uint8)
    return arr


def test_analyze_video_frames_blur_override(monkeypatch):
    """Simulate several dark, blurred, static frames and assert heuristics force 'fora_foco'."""
    # Prepare 4 identical dark blurred frames -> low motion, low brightness, near-zero edges
    frames = [make_dark_blurred_frame() for _ in range(4)]

    # Monkeypatch VideoCapture to return our frames
    def fake_videocap_ctor(path):
        return FakeCapture(frames, fps=30)

    # ensure inference uses our fake cv2 module and replace VideoCapture
    monkeypatch.setattr(inference, 'cv2', fake_cv2, raising=False)
    monkeypatch.setattr(fake_cv2, 'VideoCapture', fake_videocap_ctor, raising=False)

    # Replace the keras_video_model with a fake model that does NOT strongly predict fora_foco
    # The MODEL_CLASSES ordering in the module is AUDIO + VIDEO; fora_foco index is near the end.
    num_classes = max(8, len(inference.MODEL_CLASSES))
    # pick an index that would correspond to the 'fora_foco' position if present
    targ_idx = len(inference.MODEL_CLASSES) - 1 if len(inference.MODEL_CLASSES) > 0 else num_classes - 1
    fake_model = FakeModel(num_classes=num_classes, targ_idx=targ_idx)
    monkeypatch.setitem(inference.__dict__, 'keras_video_model', fake_model)

    # Lower thresholds slightly for the unit test to make heuristics trigger easily
    monkeypatch.setattr(inference.core_settings, 'VIDEO_BLUR_VAR_THRESHOLD', 100.0)
    monkeypatch.setattr(inference.core_settings, 'VIDEO_EDGE_DENSITY_THRESHOLD', 0.5)
    monkeypatch.setattr(inference.core_settings, 'VIDEO_MOTION_THRESHOLD', 5.0)
    monkeypatch.setattr(inference.core_settings, 'VIDEO_BRIGHTNESS_LOW', 100.0)
    monkeypatch.setattr(inference.core_settings, 'VIDEO_VOTE_K', 2)

    # Use a high sample rate so our small number of frames are sampled
    pred_class, conf, t = inference.analyze_video_frames("dummy_path.mp4", sample_rate_hz=30.0)

    # The heuristics (blur + low edges + low motion) should force 'fora_foco'
    assert isinstance(pred_class, str)
    assert pred_class.lower().replace(' ', '_') == 'fora_foco'
    assert conf >= 0.75
