# backend/app/ml/inference.py
# (Versão Keras-only com Análise Multi-Segmento/Frame)

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import librosa
import cv2 # OpenCV
from typing import Tuple, Optional, List, Dict
import traceback
import math
from collections import deque
from ..core.config import settings as core_settings
import json
import tempfile
import subprocess
import shutil
import warnings

# === Definições (Mantidas) ===
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
# Prefer models under mobilenetv2/ to keep versions organized; fallback to root filenames
MOBILENET_DIR = os.path.join(MODEL_DIR, 'mobilenetv2')
# Per-model subtype directories (allow audio models under models/audio/)
AUDIO_MODEL_DIR = os.path.join(MODEL_DIR, 'audio')

# New model filenames (user supplied INT8 quantized models)
VIDEO_MODEL_FILENAME = os.path.join('video', 'odin_model_v4.5', 'video_model_finetune_fixed.keras')
AUDIO_MODEL_FILENAME = os.path.join('audio', 'heimdall_audio_model_ultra_v1', 'audio_model_fixed.keras')

VIDEO_MODEL_PATH = os.path.join(MODEL_DIR, VIDEO_MODEL_FILENAME)
AUDIO_MODEL_PATH = os.path.join(MODEL_DIR, AUDIO_MODEL_FILENAME)

# Classes for the new models (as provided)
# Default fallbacks (kept for backward compat) — real class list is loaded from training_files/labels.csv when available
AUDIO_CLASSES = ['ausencia_audio', 'eco_reverb', 'ruido_hiss', 'sinal_teste', 'normal']
VIDEO_CLASSES = ['normal', 'freeze', 'fade', 'fora_de_foco']

# Additional package folder (user-provided model bundle). If you placed a model package
# at repository root `horus_package_video_model_v2` we attempt to use its metadata/labels
PACKAGE_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'horus_package_video_model_v2'))

# Load metadata if present in the package model dir or in MODEL_DIR
MODEL_METADATA = {}
THRESHOLDS = {}
try:
    # prefer metadata in MODEL_DIR/audio (for audio-specific bundles), then MODEL_DIR, else PACKAGE_MODEL_DIR
    metadata_candidates = [
        os.path.join(MODEL_DIR, 'audio', 'heimdall_audio_model_ultra_v1', 'metadata.json'),
        os.path.join(MODEL_DIR, 'video', 'odin_model_v4.5', 'metadata.json'),
        os.path.join(AUDIO_MODEL_DIR, 'metadata.json'),
        os.path.join(MODEL_DIR, 'video_model_finetune.metadata.json'),
        os.path.join(MODEL_DIR, 'metadata.json'),
        os.path.join(PACKAGE_MODEL_DIR, 'video_model_finetune.metadata.json')
    ]
    for mpath in metadata_candidates:
        if os.path.exists(mpath):
            with open(mpath, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
                MODEL_METADATA.update(data) # Merge metadata
except Exception:
    MODEL_METADATA = {}

try:
    thresh_candidates = [
        os.path.join(MODEL_DIR, 'video', 'odin_model_v4.5', 'thresholds.yaml'),
        os.path.join(MODEL_DIR, 'thresholds.yaml'), 
        os.path.join(PACKAGE_MODEL_DIR, 'thresholds.yaml')
    ]
    for tpath in thresh_candidates:
        if os.path.exists(tpath):
            # simple YAML parser for key: value lines (no dependency on PyYAML)
            with open(tpath, 'r', encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    if ':' in line:
                        k, v = line.split(':', 1)
                        try:
                            THRESHOLDS[k.strip()] = float(v.strip())
                        except Exception:
                            try:
                                THRESHOLDS[k.strip()] = float(v.strip())
                            except Exception:
                                THRESHOLDS[k.strip()] = v.strip()
            break
except Exception:
    THRESHOLDS = {}

# Expose class thresholds for other modules
CLASS_THRESHOLDS = THRESHOLDS.copy() if isinstance(THRESHOLDS, dict) else {}

# Canonical model class list (loaded from training metadata if present). This should match the
# output ordering of your trained Keras models. If training_files/labels.csv exists we will
# extract the unique classes in file order and use that as MODEL_CLASSES. Otherwise we fall back
# to AUDIO_CLASSES + VIDEO_CLASSES as a reasonable default.
MODEL_CLASSES = []

# One-time diagnostic flag to log model input specs
keras_video_inputs_logged = False

def _load_model_classes_from_training_files():
    try:
        import csv
        # Priority: 
        # 1. labels.csv in specific audio/video model folders (if we can guess which one is active)
        # 2. training_files/labels.csv
        # 3. labels.csv in model root
        
        candidates = [
            os.path.join(MODEL_DIR, 'audio', 'heimdall_audio_model_ultra_v1', 'labels.csv'),
            os.path.join(MODEL_DIR, 'video', 'odin_model_v4.5', 'labels.csv'),
            os.path.join(MODEL_DIR, 'training_files', 'labels.csv'),
            os.path.join(MODEL_DIR, 'labels.csv')
        ]
        
        seen = []
        for training_labels in candidates:
            if os.path.exists(training_labels):
                with open(training_labels, 'r', encoding='utf-8') as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        # Support both 'class' and 'class_name' columns
                        cls = row.get('class') or row.get('class_name')
                        if not cls:
                            continue
                        # If multi-label (pipe-separated), take each
                        parts = [p.strip() for p in cls.split('|') if p.strip()]
                        for p in parts:
                            if p not in seen:
                                seen.append(p)
                # If we found labels in a specific file, we might want to stop or merge. 
                # For now, let's merge all unique labels found across files to be safe.
        return seen
    except Exception:
        return []

# Try to populate MODEL_CLASSES at import time
_loaded_classes = _load_model_classes_from_training_files()
if _loaded_classes:
    MODEL_CLASSES = _loaded_classes
else:
    # fallback: concatenate audio and video defaults (deduplicated)
    temp = []
    for c in (AUDIO_CLASSES + VIDEO_CLASSES):
        if c not in temp:
            temp.append(c)
    MODEL_CLASSES = temp

# If an audio-specific classes file exists under MODEL_DIR/audio, prefer it for AUDIO_CLASSES and MODEL_CLASSES ordering
try:
    audio_classes_path = os.path.join(AUDIO_MODEL_DIR, 'classes.txt')
    audio_labels_csv = os.path.join(AUDIO_MODEL_DIR, 'labels.csv')
    prefer_classes = None
    if os.path.exists(audio_classes_path):
        with open(audio_classes_path, 'r', encoding='utf-8') as fh:
            prefer_classes = [l.strip() for l in fh.readlines() if l.strip()]
    elif os.path.exists(audio_labels_csv):
        import csv
        with open(audio_labels_csv, 'r', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            prefer_classes = []
            for row in reader:
                cls = row.get('class')
                if cls and cls not in prefer_classes:
                    prefer_classes.append(cls)
    if prefer_classes:
        # Update AUDIO_CLASSES and MODEL_CLASSES to match the audio-trained ordering
        AUDIO_CLASSES = prefer_classes
        # if MODEL_CLASSES currently contains these classes, reorder MODEL_CLASSES to put audio classes first
        MODEL_CLASSES = prefer_classes + [c for c in MODEL_CLASSES if c not in prefer_classes]
except Exception:
    pass

# If metadata provided with explicit labels, prefer that ordering
try:
    if MODEL_METADATA and 'labels' in MODEL_METADATA and isinstance(MODEL_METADATA['labels'], list):
        MODEL_CLASSES = MODEL_METADATA['labels']
        # Derive sensible video classes list (exclude 'normal' if present)
        VIDEO_CLASSES = [c for c in MODEL_CLASSES if str(c).lower() != 'normal']
        # If metadata contains input shape, set input size accordingly
        ishape = MODEL_METADATA.get('input_shape') or MODEL_METADATA.get('input_shape')
        try:
            if ishape and isinstance(ishape, (list, tuple)) and len(ishape) >= 3:
                # assume format [batch, frames, height, width, channels] or [frames, height, width, channels] or [height, width, channels]
                if len(ishape) == 5:
                    _, _, INPUT_HEIGHT, INPUT_WIDTH, _ = ishape
                elif len(ishape) == 4:
                    _, INPUT_HEIGHT, INPUT_WIDTH, _ = ishape
                elif len(ishape) == 3:
                    INPUT_HEIGHT, INPUT_WIDTH, _ = ishape
                INPUT_HEIGHT = int(INPUT_HEIGHT)
                INPUT_WIDTH = int(INPUT_WIDTH)
        except Exception:
            pass
except Exception:
    pass

# Default input size; may be overridden by MODEL_METADATA or detected from model inputs
try:
    INPUT_HEIGHT
    INPUT_WIDTH
except NameError:
    INPUT_HEIGHT = 160
    INPUT_WIDTH = 160

# === Carregamento dos Intérpretes (Mantido) ===
# Keras models (preferred if present)
keras_audio_model = None
keras_video_model = None

audio_interpreter = None
video_interpreter = None
# ... (restante das variáveis globais e função load_all_models idêntica à anterior) ...
audio_input_details = None
audio_output_details = None
video_input_details = None
video_output_details = None
models_loaded = False

# Flags set after loading the video model to indicate extra expected inputs
keras_video_model_requires_motion_brightness = False
keras_video_model_has_rescaling = False

import keras

def flatten_time(x):
    s = keras.ops.shape(x)
    return keras.ops.reshape(x, (s[0] * s[1], s[2], s[3], s[4]))

def unflatten_time(args):
    flat, original_input = args
    s = keras.ops.shape(original_input)
    f_shape = keras.ops.shape(flat)
    return keras.ops.reshape(flat, (s[0], s[1], f_shape[1]))

def load_all_models():
    """Carrega modelos Keras a partir de arquivos HDF5 (.h5).
    Estratégia: tenta primeiro os artefatos finetune (audio_model_finetune.h5 / video_model_finetune.h5)
    e, se não encontrados, tenta os modelos base (audio_model.h5 / video_model.h5).
    Não tenta carregar TFLite ou .keras (o usuário pediu explicitamente .h5).
    """
    global models_loaded, keras_audio_model, keras_video_model
    print("INFO: Carregando modelos Keras (.h5) via TensorFlow...")
    loaded_any = False
    try:
        # Prefer native Keras format (.keras) if present, fall back to legacy HDF5 (.h5)
        # audio model candidates: prefer finetune in MODEL_DIR, then accept models in MODEL_DIR/audio
        audio_candidates = [
            AUDIO_MODEL_PATH,
            os.path.join(MODEL_DIR, 'audio', 'heimdall_audio_model_ultra_v1', 'audio_model.keras'),
            os.path.join(MODEL_DIR, 'audio_model_finetune.keras'),
            os.path.join(MODEL_DIR, 'audio_model_finetune.h5'),
            os.path.join(MODEL_DIR, 'audio_model.h5'),
            os.path.join(MODEL_DIR, 'audio_model.keras'),
            os.path.join(AUDIO_MODEL_DIR, 'audio_model.keras'),
            os.path.join(AUDIO_MODEL_DIR, 'audio_model_head.keras'),
            os.path.join(AUDIO_MODEL_DIR, 'audio_model.h5'),
        ]
        video_candidates = [
            VIDEO_MODEL_PATH,
            os.path.join(MODEL_DIR, 'video', 'odin_model_v4.5', 'video_model_finetune.keras'),
            os.path.join(MODEL_DIR, 'video_model_finetune.keras'),
            os.path.join(MODEL_DIR, 'video_model_finetune.h5'),
            os.path.join(MODEL_DIR, 'video_model.h5')
        ]
        # also accept models placed in the package folder (user-provided bundle)
        video_candidates += [
            os.path.join(PACKAGE_MODEL_DIR, 'video_model_finetune.keras'),
            os.path.join(PACKAGE_MODEL_DIR, 'video_model_finetune.h5')
        ]

        # Load audio model (finetune preferred)
        for p in audio_candidates:
            if os.path.exists(p):
                try:
                    keras_audio_model = keras.models.load_model(p)
                    print(f"INFO: Modelo de Áudio Keras carregado. Path={p}")
                    try:
                        print(f"INFO: Audio model output shape: {keras_audio_model.output_shape}")
                        # Update AUDIO_CLASSES if specific labels.csv exists next to the model
                        model_dir = os.path.dirname(p)
                        labels_path = os.path.join(model_dir, 'labels.csv')
                        if os.path.exists(labels_path):
                            import csv
                            with open(labels_path, 'r', encoding='utf-8') as fh:
                                reader = csv.DictReader(fh)
                                new_classes = []
                                for row in reader:
                                    cls = row.get('class') or row.get('class_name')
                                    if cls: new_classes.append(cls)
                                if new_classes:
                                    global AUDIO_CLASSES
                                    AUDIO_CLASSES = new_classes
                                    print(f"INFO: Updated AUDIO_CLASSES from {labels_path}: {AUDIO_CLASSES}")
                    except Exception:
                        pass
                    loaded_any = True
                    break
                except Exception as e:
                    print(f"AVISO: falha ao carregar modelo Keras de áudio {p}: {e}")

        # Load video model (finetune preferred)
        for p in video_candidates:
            if os.path.exists(p):
                try:
                    custom_objects = {'flatten_time': flatten_time, 'unflatten_time': unflatten_time}
                    keras_video_model = keras.models.load_model(p, custom_objects=custom_objects)
                    print(f"INFO: Modelo de Vídeo Keras carregado. Path={p}")
                    try:
                        print(f"INFO: Video model output shape: {keras_video_model.output_shape}")
                        # Update VIDEO_CLASSES if specific labels.csv exists next to the model
                        model_dir = os.path.dirname(p)
                        labels_path = os.path.join(model_dir, 'labels.csv')
                        if os.path.exists(labels_path):
                            import csv
                            with open(labels_path, 'r', encoding='utf-8') as fh:
                                reader = csv.DictReader(fh)
                                new_classes = []
                                for row in reader:
                                    cls = row.get('class') or row.get('class_name')
                                    if cls: new_classes.append(cls)
                                if new_classes:
                                    global VIDEO_CLASSES
                                    VIDEO_CLASSES = new_classes
                                    print(f"INFO: Updated VIDEO_CLASSES from {labels_path}: {VIDEO_CLASSES}")
                    except Exception:
                        pass
                    loaded_any = True
                    break
                except Exception as e:
                    print(f"AVISO: falha ao carregar modelo Keras de vídeo {p}: {e}")

        models_loaded = loaded_any
        if models_loaded:
            print("INFO: Carregamento de modelos concluído.")
        else:
            print("AVISO: Nenhum modelo .h5 foi carregado. Verifique os arquivos em backend/app/ml/models.")

    except Exception as e:
        print(f"ERRO CRÍTICO ao carregar modelos Keras: {e}")
        traceback.print_exc()
        models_loaded = False


# Carrega os modelos na importação
load_all_models()

# After loading, detect whether the loaded video model expects auxiliary inputs
try:
    if globals().get('keras_video_model') is not None:
        vm = globals().get('keras_video_model')
        try:
            inputs = vm.inputs if hasattr(vm, 'inputs') else ([vm.input] if hasattr(vm, 'input') else [])
            input_names = [getattr(i, 'name', '') for i in inputs]
            lname = [n.lower() for n in input_names if isinstance(n, str)]
            if any('motion' in n or 'brightness' in n for n in lname) or len(inputs) > 1:
                keras_video_model_requires_motion_brightness = True
                print('INFO: video model expects auxiliary inputs: motion/brightness')
        except Exception:
            pass
        try:
            for layer in vm.layers:
                if layer.__class__.__name__.lower() == 'rescaling' or 'rescaling' in getattr(layer, 'name', '').lower():
                    keras_video_model_has_rescaling = True
                    print('INFO: video model contains Rescaling layer (do not rescale externally)')
                    break
        except Exception:
            pass
except Exception:
    pass

# === Função de Inferência TFLite Genérica (Mantida) ===
def run_tflite_inference(interpreter: tf.lite.Interpreter, input_details: dict, output_details: dict, input_data: np.ndarray, classes: List[str]) -> Tuple[str, float]:
    """ Executa a inferência TFLite genérica. """
    # (Código idêntico ao da resposta anterior)
    if not interpreter: raise RuntimeError("Intérprete TFLite não está disponível.")
    try:
        # Prepare input according to interpreter quantization parameters.
        input_dtype = input_details['dtype']
        # Ensure input_data is float32 preprocessed according to model (e.g. MobileNetV2: (x-127.5)/127.5)
        input_float = input_data.astype(np.float32)
        if input_dtype in (np.uint8, np.int8):
            scale, zero_point = input_details.get('quantization', (1.0, 0))
            # Quantize: q = round(input_float / scale + zero_point)
            input_data_quant = np.round(input_float / scale + zero_point).astype(input_dtype)
            interpreter.set_tensor(input_details['index'], input_data_quant)
        else:
            interpreter.set_tensor(input_details['index'], input_float)

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details['index'])[0]

        # Dequantize outputs if needed
        output_dtype = output_details['dtype']
        if output_dtype in (np.uint8, np.int8):
            scale, zero_point = output_details.get('quantization', (1.0, 0))
            output_data = scale * (output_data.astype(np.float32) - zero_point)

        # --- Softmax Opcional ---
        # output_data = tf.nn.softmax(output_data).numpy()

        predicted_index = np.argmax(output_data)
        confidence = float(output_data[predicted_index])

        if predicted_index >= len(classes):
            predicted_class = "Erro_Indice"
            confidence = 0.0
        else:
            predicted_class = classes[predicted_index]

        # print(f"DEBUG: Inferência Segmento: Classe={predicted_class}, Confiança={confidence:.4f}") # Log por segmento
        return predicted_class, confidence
    except Exception as e:
        print(f"ERRO durante a execução da inferência TFLite: {e}")
        traceback.print_exc()
        return "Erro_Inferência", 0.0


def _apply_softmax(scores: np.ndarray) -> np.ndarray:
    try:
        probs = tf.nn.softmax(scores).numpy()
        return probs
    except Exception:
        # fallback: normalize to sum 1 if possible
        arr = np.array(scores, dtype=np.float32)
        s = np.sum(arr)
        if s > 0:
            return arr / s
        return arr


def run_keras_inference(model: tf.keras.Model, input_data: np.ndarray, classes: List[str]) -> Tuple[str, float]:
    """Roda inferência usando modelo Keras (assume saída logits ou probabilidades)."""
    if model is None:
        raise RuntimeError("Modelo Keras não está carregado.")
    try:
        # Detectar se o modelo espera input 5D (batch, frames, H, W, C) vs 4D (batch, H, W, C)
        # Se input_data é 4D mas modelo espera 5D, adicionar dimensão de frames
        inputs_spec = getattr(model, 'inputs', None)
        if inputs_spec and len(inputs_spec) >= 1:
            first_input = inputs_spec[0]
            expected_shape = getattr(first_input, 'shape', None)
            if expected_shape is not None and len(expected_shape) == 5 and len(input_data.shape) == 4:
                # Modelo espera 5D mas recebeu 4D - adicionar dimensão de frames
                # Shape: (batch, H, W, C) -> (batch, 1, H, W, C)
                input_data = np.expand_dims(input_data, axis=1)
                # print(f"DEBUG: Reshape 4D->5D para modelo de sequência: {input_data.shape}")
        
        # If the model expects multiple inputs but we received a single ndarray,
        # attempt to build a list of ordered inputs (image, motion, brightness)
        preds = None
        try:
            if inputs_spec and len(inputs_spec) > 1 and isinstance(input_data, np.ndarray):
                # build fallback auxiliary inputs (zeros) matching common small shapes
                ordered = []
                for inp in inputs_spec:
                    in_name = getattr(inp, 'name', '') or ''
                    lname = in_name.lower()
                    shape = getattr(inp, 'shape', None)
                    if ('motion' in lname) or (shape is not None and len(shape) == 2):
                        ordered.append(np.zeros((1, 1), dtype=np.float32))
                    elif ('bright' in lname) or ('brightness' in lname):
                        ordered.append(np.zeros((1, 1), dtype=np.float32))
                    elif ('image' in lname) or ('input' in lname) or (shape is not None and len(shape) in (4, 5)):
                        ordered.append(input_data)
                    else:
                        # default fallback: if expects 4D/5D assume image, else scalar zero
                        if shape is not None and len(shape) in (4, 5):
                            ordered.append(input_data)
                        else:
                            ordered.append(np.zeros((1, 1), dtype=np.float32))
                preds = model.predict(ordered, verbose=0)
            else:
                preds = model.predict(input_data, verbose=0)
        except Exception:
            # fallback to direct predict
            preds = model.predict(input_data, verbose=0)
        if preds is None:
            return "Erro_Inferência", 0.0
        scores = np.array(preds[0], dtype=np.float32)
        # Aplica softmax para obter probabilidades
        probs = _apply_softmax(scores)
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        predicted_class = classes[idx] if idx < len(classes) else f"idx_{idx}"
        return predicted_class, conf
    except Exception as e:
        print(f"ERRO durante inferência Keras: {e}")
        traceback.print_exc()
        return "Erro_Inferência", 0.0


def run_keras_inference_topk(model: tf.keras.Model, input_data: np.ndarray, classes: List[str], k: int = 3) -> List[Tuple[str, float]]:
    if model is None:
        return []
    try:
        # Detectar se o modelo espera input 5D (batch, frames, H, W, C) vs 4D (batch, H, W, C)
        inputs_spec = getattr(model, 'inputs', None)
        if inputs_spec and len(inputs_spec) >= 1:
            first_input = inputs_spec[0]
            expected_shape = getattr(first_input, 'shape', None)
            if expected_shape is not None and len(expected_shape) == 5 and len(input_data.shape) == 4:
                # Modelo espera 5D mas recebeu 4D - adicionar dimensão de frames
                input_data = np.expand_dims(input_data, axis=1)
        
        # same multi-input guard as run_keras_inference
        if inputs_spec and len(inputs_spec) > 1 and isinstance(input_data, np.ndarray):
            ordered = []
            for inp in inputs_spec:
                in_name = getattr(inp, 'name', '') or ''
                lname = in_name.lower()
                shape = getattr(inp, 'shape', None)
                if ('motion' in lname) or (shape is not None and len(shape) == 2):
                    ordered.append(np.zeros((1, 1), dtype=np.float32))
                elif ('bright' in lname) or ('brightness' in lname):
                    ordered.append(np.zeros((1, 1), dtype=np.float32))
                elif ('image' in lname) or ('input' in lname) or (shape is not None and len(shape) in (4, 5)):
                    ordered.append(input_data)
                else:
                    if shape is not None and len(shape) in (4, 5):
                        ordered.append(input_data)
                    else:
                        ordered.append(np.zeros((1, 1), dtype=np.float32))
            preds = model.predict(ordered, verbose=0)
        else:
            preds = model.predict(input_data, verbose=0)
        scores = np.array(preds[0], dtype=np.float32)
        probs = _apply_softmax(scores)
        indices = np.argsort(probs)[::-1]
        top = []
        for i in indices[:max(1, k)]:
            name = classes[i] if i < len(classes) else f"idx_{i}"
            top.append((name, float(probs[i])))
        return top
    except Exception:
        return []


def run_keras_sequence_inference(model: tf.keras.Model, seq_data: np.ndarray, classes: List[str]) -> Tuple[str, float]:
    """Run inference on a sequence tensor (batch, frames, H, W, C) when model expects 5D input.
    Falls back to frame-wise inference if model is single-input.
    Returns (predicted_class, confidence).
    """
    if model is None:
        raise RuntimeError("Modelo Keras não está carregado.")
    try:
        # If model expects multiple inputs, try to map similarly to run_keras_inference
        inputs_spec = getattr(model, 'inputs', None)
        preds = None
        try:
            if inputs_spec and len(inputs_spec) > 1:
                # Attempt to order inputs: find a 5D input (frames) and pass seq_data there
                ordered = []
                for inp in inputs_spec:
                    in_name = getattr(inp, 'name', '') or ''
                    lname = in_name.lower()
                    shape = getattr(inp, 'shape', None)
                    if shape is not None and len(shape) == 5:
                        ordered.append(seq_data)
                    elif 'motion' in lname:
                        ordered.append(np.zeros((seq_data.shape[0], 1), dtype=np.float32))
                    elif 'bright' in lname or 'brightness' in lname:
                        ordered.append(np.zeros((seq_data.shape[0], 1), dtype=np.float32))
                    else:
                        # best-effort: if expects 4D, pass middle frame
                        if shape is not None and len(shape) == 4:
                            mid = seq_data.shape[1] // 2
                            ordered.append(np.expand_dims(seq_data[0, mid].astype(np.float32), axis=0))
                        else:
                            ordered.append(np.zeros((seq_data.shape[0], 1), dtype=np.float32))
                preds = model.predict(ordered, verbose=0)
            else:
                # model may accept 5D directly
                preds = model.predict(seq_data, verbose=0)
        except Exception:
            # fallback: try predicting with sequence, else with middle frame
            try:
                preds = model.predict(seq_data, verbose=0)
            except Exception:
                try:
                    mid = seq_data.shape[1] // 2
                    preds = model.predict(np.expand_dims(seq_data[0, mid].astype(np.float32), axis=0), verbose=0)
                except Exception:
                    return "Erro_Inferência", 0.0

        if preds is None:
            return "Erro_Inferência", 0.0
        scores = np.array(preds[0], dtype=np.float32)
        probs = _apply_softmax(scores)
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        predicted_class = classes[idx] if idx < len(classes) else f"idx_{idx}"
        return predicted_class, conf
    except Exception as e:
        print(f"ERRO durante inferência de sequência Keras: {e}")
        traceback.print_exc()
        return "Erro_Inferência", 0.0

def run_tflite_inference_topk(
    interpreter: tf.lite.Interpreter,
    input_details: dict,
    output_details: dict,
    input_data: np.ndarray,
    classes: List[str],
    k: int = 3
) -> List[Tuple[str, float]]:
    """ Executa a inferência e retorna top-k (classe, score). """
    if not interpreter:
        return []
    try:
        input_dtype = input_details['dtype']
        input_float = input_data.astype(np.float32)
        if input_dtype in (np.uint8, np.int8):
            scale, zero_point = input_details.get('quantization', (1.0, 0))
            input_data_quant = np.round(input_float / scale + zero_point).astype(input_dtype)
            interpreter.set_tensor(input_details['index'], input_data_quant)
        else:
            interpreter.set_tensor(input_details['index'], input_float)

        interpreter.invoke()
        scores = interpreter.get_tensor(output_details['index'])[0]
        out_dtype = output_details['dtype']
        if out_dtype in (np.uint8, np.int8):
            scale, zero_point = output_details.get('quantization', (1.0, 0))
            scores = scale * (scores.astype(np.float32) - zero_point)

        indices = np.argsort(scores)[::-1]
        top = []
        for i in indices[:max(1, k)]:
            name = classes[i] if i < len(classes) else f"idx_{i}"
            top.append((name, float(scores[i])))
        return top
    except Exception:
        return []

# =======================================================
# === NOVAS FUNÇÕES: ANÁLISE DE ÁUDIO MULTI-SEGMENTO ===
# =======================================================

def preprocess_audio_segment(y_segment: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """
    Preprocess audio segment for Heimdall Audio Model Ultra V1.
    Converts audio waveform to Mel Spectrogram with fixed normalization.
    """
    try:
        # Constants for Ultra V1
        TARGET_SR = 22050
        N_MELS = 128
        N_FFT = 2048
        HOP_LENGTH = 512
        FMAX = 8000
        IMG_SIZE = 128
        REF_DB = 1.0
        MIN_DB = -80.0
        MAX_DB = 80.0

        # Resample if necessary
        if sr != TARGET_SR:
            y_segment = librosa.resample(y_segment, orig_sr=sr, target_sr=TARGET_SR)
        
        # Ensure correct length (3.0s)
        target_len = int(3.0 * TARGET_SR)
        if len(y_segment) < target_len:
            y_segment = np.pad(y_segment, (0, target_len - len(y_segment)))
        elif len(y_segment) > target_len:
            y_segment = y_segment[:target_len]

        # 1. Mel Spectrogram
        mel = librosa.feature.melspectrogram(
            y=y_segment, sr=TARGET_SR, n_mels=N_MELS,
            n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=FMAX
        )
        
        # 2. Power to DB (Fixed Reference)
        mel_db = librosa.power_to_db(mel, ref=REF_DB)
        
        # 3. Fixed Normalization [-80, 80] -> [0, 1]
        mel_db = np.clip(mel_db, MIN_DB, MAX_DB)
        mel_norm = (mel_db - MIN_DB) / (MAX_DB - MIN_DB)
        
        # 4. Resize to 128x128
        mel_resized = cv2.resize(mel_norm, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        
        # 5. Add batch and channel dims -> (1, 128, 128, 1)
        return mel_resized.astype(np.float32)[np.newaxis, ..., np.newaxis]

    except Exception as e:
        print(f"ERRO no pré-processamento de áudio: {e}")
        traceback.print_exc()
        return None
        # print(f"ERRO pré-processando segmento de áudio: {e}") # Log menos verboso
        return None

def analyze_audio_segments(file_path: str) -> Tuple[str, float, Optional[float], Optional[float]]:
    """
    Analisa múltiplos segmentos de áudio usando estratégia Heurística + Modelo CNN.
    
    Retorna: (classe, confiança, tempo_início_erro, tempo_fim_erro)
    
    Pipeline conforme INFERENCE_PIPELINE.md do Heimdall Audio Model Ultra V1:
    - FASE 1: Detecção de silêncio por heurística (RMS + Peak)
    - FASE 2: Modelo CNN para classificação
    - FASE 3: Pós-processamento com thresholds e boost de RMS
    - FASE 4: Suavização temporal (merge de segmentos)
    """
    if not globals().get('keras_audio_model'):
        raise RuntimeError("Modelo de áudio Keras não está carregado.")

    try:
        # =====================================================
        # === CONSTANTES DO HEIMDALL ULTRA V1 (INFERENCE_PIPELINE.md) ===
        # =====================================================
        TARGET_SR = 22050
        SEGMENT_DURATION_S = 3.0
        HOP_DURATION_S = 1.0
        
        # --- FASE 1: Detecção de Silêncio por Heurística ---
        SILENCE_RMS_THRESHOLD = 0.008
        SILENCE_PEAK_THRESHOLD = 0.15
        SILENCE_STRICT_RMS = 0.006  # RMS muito baixo = silêncio garantido
        
        # --- FASE 3: Thresholds de Confiança por Classe ---
        MIN_CONFIDENCE = 0.80           # Geral (hiss, sinal_teste)
        MIN_CONFIDENCE_ECO = 0.85       # Eco precisa de confiança alta
        MIN_CONFIDENCE_AUSENCIA = 0.75  # Silêncio detectado pelo modelo
        
        # --- Regras Especiais para Eco ---
        ECO_MARGIN_OVER_NORMAL = 0.20   # Eco precisa ter 20% de margem sobre "normal"
        ECO_MARGIN_OVER_SECOND = 0.15   # Eco precisa ter 15% de margem sobre a 2ª melhor classe
        
        # --- Boost de Confiança por RMS ---
        ECO_MIN_RMS = 0.02
        ECO_RMS_BOOST = 3.0
        HISS_MIN_RMS = 0.015
        HISS_RMS_BOOST = 2.0
        
        # --- FASE 4: Suavização Temporal ---
        CLASS_MIN_SEGMENTS = {
            'eco_reverb': 1,      # Eco pode ser detectado em 1 segmento
            'ruido_hiss': 1,      # Chiado pode ser detectado em 1 segmento
            'ausencia_audio': 2,  # Silêncio precisa de 2 segmentos consecutivos
            'sinal_teste': 2      # Sinal de teste precisa de 2 segmentos
        }
        GAP_TOLERANCE_SEGMENTS = 1  # Tolera 1 segmento "normal" no meio do erro
        
        # --- Regras para Segmento Único ---
        CLASS_SINGLE_SEGMENT_RULES = {
            'eco_reverb': {'min_confidence': 0.85, 'min_rms': 0.03},
            'ruido_hiss': {'min_confidence': 0.82, 'min_rms': 0.025}
        }
        # =====================================================
        
        # Helper: Boost de confiança por RMS (conforme documento)
        def boosted_confidence(pred_class: str, conf: float, rms: float) -> float:
            """Segmentos com RMS alto ganham boost de confiança"""
            if pred_class == 'eco_reverb':
                return min(1.0, conf + ECO_RMS_BOOST * max(0, rms - ECO_MIN_RMS))
            if pred_class == 'ruido_hiss':
                return min(1.0, conf + HISS_RMS_BOOST * max(0, rms - HISS_MIN_RMS))
            return conf

        print(f"DEBUG: Analisando segmentos de áudio: {file_path}")
        
        # Helper para carregar áudio com fallback ffmpeg
        def _librosa_load_with_ffmpeg_fallback(path, sr=None):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    warnings.simplefilter("ignore", FutureWarning)
                    y, sr_loaded = librosa.load(path, sr=sr)
                    if len(y) == 0:
                        raise RuntimeError("Librosa returned empty audio array")
                    return y, sr_loaded
            except Exception:
                ffmpeg_path = shutil.which('ffmpeg')
                if not ffmpeg_path:
                    raise
                tmpf = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                tmp_path = tmpf.name
                tmpf.close()
                cmd = [ffmpeg_path, '-y', '-i', path, '-vn', '-acodec', 'pcm_s16le', 
                       '-ar', str(sr or 16000), '-ac', '1', tmp_path]
                proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stderr = proc.stderr.decode(errors='ignore') if proc.stderr else ''
                if proc.returncode != 0 or not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                    no_audio_indicators = ['does not contain any stream', 'could not find audio', 
                                           'Invalid data found when processing input']
                    try:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                    except Exception:
                        pass
                    if any(ind in stderr for ind in no_audio_indicators):
                        return np.array([], dtype=np.float32), sr or 0
                    raise RuntimeError(f"ffmpeg failed: {stderr}")
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        y, sr_native = librosa.load(tmp_path, sr=sr)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                return y, sr_native

        y_full, sr = _librosa_load_with_ffmpeg_fallback(file_path, sr=TARGET_SR)
        if len(y_full) == 0:
            print(f"DEBUG: Áudio vazio detectado: {file_path}")
            return 'ausencia_audio', 0.95, 0.0, 0.0

        segment_samples = int(SEGMENT_DURATION_S * sr)
        hop_samples = int(HOP_DURATION_S * sr)
        
        # Armazena todas as detecções por classe para suavização temporal
        class_detections = {cls: [] for cls in AUDIO_CLASSES}
        all_segments = []  # Lista de todas as análises
        processed_segments = 0

        for i in range(0, len(y_full) - segment_samples + 1, hop_samples):
            y_segment = y_full[i : i + segment_samples]
            seg_start_s = float(i) / float(sr)
            
            # === REGRA 1: Detecção de silêncio por RMS (sem modelo) ===
            rms = float(np.sqrt(np.mean(y_segment ** 2)))
            peak = float(np.max(np.abs(y_segment)))
            
            if rms < SILENCE_RMS_THRESHOLD and peak < SILENCE_PEAK_THRESHOLD:
                # Silêncio detectado diretamente - não precisa do modelo
                class_detections['ausencia_audio'].append({
                    'time': seg_start_s,
                    'confidence': 0.95,
                    'method': 'rms_silence'
                })
                all_segments.append({
                    'time': seg_start_s,
                    'class': 'ausencia_audio',
                    'confidence': 0.99,  # 99% confiança para silêncio por heurística
                    'rms': rms,
                    'method': 'heuristic_silence'
                })
                processed_segments += 1
                continue
            
            # === FASE 2: Modelo CNN ===
            input_data = preprocess_audio_segment(y_segment, sr)
            if input_data is None:
                continue
            
            model = globals().get('keras_audio_model')
            probs = model.predict(input_data, verbose=0)[0]
            processed_segments += 1
            
            # Mapeia probabilidades para classes
            class_probs = {AUDIO_CLASSES[j]: float(probs[j]) for j in range(len(AUDIO_CLASSES))}
            
            # Ordena por probabilidade
            sorted_classes = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
            top_class, top_prob = sorted_classes[0]
            second_class, second_prob = sorted_classes[1] if len(sorted_classes) > 1 else (None, 0)
            normal_prob = class_probs.get('normal', 0)
            
            # === FASE 3: Pós-Processamento com Thresholds ===
            detected_class = 'normal'
            detected_confidence = normal_prob
            
            # Aplica boost de RMS para obter confiança efetiva
            eff_conf = boosted_confidence(top_class, top_prob, rms)
            
            # --- Regras para ECO (classe mais difícil) ---
            if top_class == 'eco_reverb':
                # Eco só é aceito se tiver margem clara sobre outras classes
                eco_prob = class_probs['eco_reverb']
                passes_threshold = eff_conf >= MIN_CONFIDENCE_ECO
                passes_normal_margin = (eco_prob - normal_prob) >= ECO_MARGIN_OVER_NORMAL
                passes_second_margin = (eco_prob - second_prob) >= ECO_MARGIN_OVER_SECOND if second_class != 'normal' else True
                
                if passes_threshold and passes_normal_margin and passes_second_margin:
                    detected_class = 'eco_reverb'
                    detected_confidence = eff_conf
            
            # --- Regras para AUSENCIA_AUDIO (modelo detectou) ---
            elif top_class == 'ausencia_audio':
                # Silêncio: ou modelo tem confiança alta OU RMS é muito baixo
                if eff_conf >= MIN_CONFIDENCE_AUSENCIA or rms <= SILENCE_STRICT_RMS:
                    detected_class = 'ausencia_audio'
                    detected_confidence = max(eff_conf, 0.95 if rms <= SILENCE_STRICT_RMS else eff_conf)
            
            # --- Regras para HISS e SINAL_TESTE ---
            elif top_class in ['ruido_hiss', 'sinal_teste']:
                if eff_conf >= MIN_CONFIDENCE:
                    detected_class = top_class
                    detected_confidence = eff_conf
            
            # Registra detecção
            all_segments.append({
                'time': seg_start_s,
                'class': detected_class,
                'confidence': detected_confidence,
                'rms': rms,
                'probs': class_probs,
                'method': 'model'
            })
            
            if detected_class != 'normal':
                class_detections[detected_class].append({
                    'time': seg_start_s,
                    'confidence': detected_confidence,
                    'rms': rms
                })

        # === SUAVIZAÇÃO TEMPORAL: Agregar detecções ===
        # Conta votos por classe
        class_votes = {cls: len(dets) for cls, dets in class_detections.items()}
        total_faults = sum(v for cls, v in class_votes.items() if cls != 'normal')
        
        # DEBUG: mostra contagem
        debug_counts = {cls: v for cls, v in class_votes.items() if v > 0}
        print(f"DEBUG: Votos por classe: {debug_counts}")
        
        # === FASE 4: Suavização Temporal ===
        if total_faults == 0:
            # Nenhuma falha detectada
            avg_confidence = np.mean([s['confidence'] for s in all_segments if s['class'] == 'normal']) if all_segments else 0.5
            result = ('normal', float(avg_confidence), None, None)
        else:
            # Encontra a classe de falha dominante com regras de suavização
            best_fault_class = None
            best_fault_score = 0
            best_fault_time = None
            best_fault_end_time = None
            best_fault_confidence = 0
            
            for cls, detections in class_detections.items():
                if cls == 'normal' or len(detections) == 0:
                    continue
                
                avg_conf = np.mean([d['confidence'] for d in detections])
                avg_rms = np.mean([d.get('rms', 0) for d in detections])
                n_segments = len(detections)
                
                # Verifica se passa nos critérios de suavização
                min_segs = CLASS_MIN_SEGMENTS.get(cls, 2)
                passes_min_segments = n_segments >= min_segs
                
                # Regras para segmento único (alta confiança + RMS alto)
                single_seg_rules = CLASS_SINGLE_SEGMENT_RULES.get(cls)
                passes_single_segment = False
                if single_seg_rules and n_segments == 1:
                    passes_single_segment = (
                        avg_conf >= single_seg_rules['min_confidence'] and 
                        avg_rms >= single_seg_rules['min_rms']
                    )
                
                # Aceita se passa em qualquer critério
                if passes_min_segments or passes_single_segment or avg_conf > 0.92:
                    score = n_segments * avg_conf
                    if score > best_fault_score:
                        best_fault_score = score
                        best_fault_class = cls
                        best_fault_confidence = avg_conf
                        # Pega PRIMEIRO tempo (início do erro)
                        best_fault_time = detections[0]['time']
                        # Pega ÚLTIMO tempo (fim do erro) + duração do segmento
                        # Ordena por tempo para garantir que pegamos o último
                        sorted_detections = sorted(detections, key=lambda x: x['time'])
                        last_detection_time = sorted_detections[-1]['time']
                        # Adiciona a duração do segmento (SEGMENT_DURATION_S = 3s) para pegar o fim real
                        best_fault_end_time = last_detection_time + SEGMENT_DURATION_S
            
            if best_fault_class:
                result = (best_fault_class, float(best_fault_confidence), float(best_fault_time), float(best_fault_end_time))
            else:
                avg_normal = np.mean([s['confidence'] for s in all_segments if s['class'] == 'normal']) if any(s['class'] == 'normal' for s in all_segments) else 0.6
                result = ('normal', float(avg_normal), None, None)

        # Debug output - handle None values for normal results
        start_str = f"{result[2]}s" if result[2] is not None else "N/A"
        end_str = f"{result[3]}s" if result[3] is not None else "N/A"
        print(f"DEBUG: Áudio - {processed_segments} segmentos processados. "
              f"Resultado: {result[0]} ({result[1]:.4f}) start={start_str} end={end_str}")
        
        return result

    except Exception as e:
        print(f"ERRO durante análise de segmentos de áudio ({file_path}): {e}")
        traceback.print_exc()
        return "Erro_Análise_Áudio", 0.0, None, None

# =====================================================
# === ANÁLISE DE VÍDEO: ESTRATÉGIA HÍBRIDA (HEURÍSTICA + MODELO) ===
# =====================================================

def preprocess_video_single_frame(frame: np.ndarray) -> Optional[np.ndarray]:
    """ Pré-processa UM frame de vídeo (já lido). """
    try:
        if frame is None: return None
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (INPUT_WIDTH, INPUT_HEIGHT))
        img_resized_f = img_resized.astype(np.float32)
        # Apply MobileNetV2 preprocessing per user: (x - 127.5) / 127.5
        img_preproc = (img_resized_f - 127.5) / 127.5
        input_data = np.expand_dims(img_preproc, axis=0)

        return input_data if input_data.shape == (1, INPUT_HEIGHT, INPUT_WIDTH, 3) else None
    except Exception as e:
        # print(f"ERRO pré-processando frame de vídeo: {e}") # Log menos verboso
        return None


def _compute_blur_var(frame: np.ndarray) -> float:
    """Compute variance of Laplacian as a blur score. Lower -> more blur."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Use smaller size to make metric stable
        small = cv2.resize(gray, (INPUT_WIDTH, INPUT_HEIGHT))
        lap = cv2.Laplacian(small, cv2.CV_64F)
        var = float(lap.var())
        return var
    except Exception:
        return 0.0


def _compute_brightness(frame: np.ndarray) -> float:
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (INPUT_WIDTH, INPUT_HEIGHT))
        return float(small.mean())
    except Exception:
        return 0.0


def _compute_motion(prev_gray: Optional[np.ndarray], frame: np.ndarray) -> float:
    """Compute mean absolute difference between prev_gray and current frame. If prev_gray is None return large value."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (INPUT_WIDTH, INPUT_HEIGHT))
        if prev_gray is None:
            return float('inf')
        diff = cv2.absdiff(small, prev_gray)
        return float(diff.mean())
    except Exception:
        return float('inf')


def _compute_edge_density(frame: np.ndarray) -> float:
    """Compute edge density using Canny; returns fraction of edge pixels in the small image."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (INPUT_WIDTH, INPUT_HEIGHT))
        # Canny thresholds can be tuned; use defaults that work reasonably across clips
        edges = cv2.Canny(small, 100, 200)
        if edges is None:
            return 0.0
        nonzero = float(np.count_nonzero(edges))
        total = float(edges.size)
        return nonzero / total if total > 0 else 0.0
    except Exception:
        return 0.0

def analyze_video_frames(file_path: str, sample_rate_hz: float = 2.0) -> Tuple[str, float, Optional[float]]:
    """
    Analisa múltiplos frames de vídeo usando estratégia Híbrida: Heurística + Modelo.
    
    Pipeline conforme ESTRATEGIA_HEURISTICA.md do Odin Model V4.5:
    - Heurísticas: Freeze (diff < 2.0), Fade (brightness < 15), Blur (sharpness < 130)
    - Override: Se modelo tem < 95% certeza em "Normal", heurística pode sobrescrever
    - Regra de negócio: Só reporta erros que persistam por mais de 2 segundos
    
    Retorna (pred_class, confidence, event_time_s)
    """
    if not globals().get('keras_video_model'):
        raise RuntimeError("Modelo de vídeo Keras não está carregado.")

    # =====================================================
    # === CONSTANTES DO ODIN V4.5 (ESTRATEGIA_HEURISTICA.md) ===
    # =====================================================
    
    # === THRESHOLDS CALIBRADOS COM BASE EM DADOS REAIS ===
    # Analisando diagnostics de clips reais:
    # - Freeze real: motion < 1.5 por vários frames consecutivos (pixels quase idênticos)
    # - Cena parada normal: motion 1.5-3.0 (há pequenas variações)
    # - Movimento normal: motion = 5 a 40+
    # - Zona cinza (slow motion): motion = 3 a 5
    FREEZE_DIFF_THRESHOLD = 1.5       # Motion < 1.5 = suspeita de freeze (mais conservador)
    FREEZE_MAX_DIFF_THRESHOLD = 2.5   # Max permitido (menos tolerante a variações)
    FADE_BRIGHTNESS_THRESHOLD = 15    # Brilho médio < 15 = fade (tela preta)
    BLUR_SHARPNESS_THRESHOLD = 130.0  # Variância do Laplaciano < 130 = blur
    
    # Override: modelo precisa ter >= 95% em "normal" para ignorar heurísticas
    MODEL_OVERRIDE_THRESHOLD = 0.95
    
    # Regra de negócio: só reporta erros > 2 segundos
    MIN_ERROR_DURATION_S = 2.0
    # =====================================================

    # Helpers para heurísticas (conforme documento)
    def _check_freeze(frames: List[np.ndarray]) -> Tuple[bool, float]:
        """
        Detecta congelamento comparando o FRAME ATUAL com o ANTERIOR.
        Freeze real = pixels quase idênticos entre frames (motion < 1.5).
        
        Diferença de freeze vs cena parada:
        - Freeze real: motion ~0.2-1.0 (só ruído de compressão H.264)
        - Cena parada com câmera fixa: motion ~1.5-3.0 (pequenas variações de iluminação)
        """
        if len(frames) < 2:
            return False, 0.0
        
        # Usa os 2 ÚLTIMOS frames (frame atual vs anterior)
        current_frame = frames[-1]
        previous_frame = frames[-2]
        
        # Se a tela está muito escura, não é freeze - é fade
        current_brightness = np.mean(current_frame)
        if current_brightness < 20:
            return False, 0.0
        
        # Converte para grayscale para alinhar com a métrica 'motion' do diagnostic
        if len(current_frame.shape) == 3:
            gray_curr = cv2.cvtColor(current_frame.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gray_prev = cv2.cvtColor(previous_frame.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray_curr = current_frame
            gray_prev = previous_frame
        
        diff = cv2.absdiff(gray_curr, gray_prev)
        motion_val = float(diff.mean())
        
        # Freeze verdadeiro: motion muito baixo (< 1.5)
        is_freeze = motion_val < FREEZE_DIFF_THRESHOLD
        
        # Confiança proporcional: quanto menor o motion, maior a confiança
        if is_freeze:
            # motion=0 → conf=1.0, motion=THRESHOLD → conf=0.85
            conf = 0.85 + 0.15 * (1.0 - motion_val / FREEZE_DIFF_THRESHOLD)
            conf = min(1.0, max(0.85, conf))
        else:
            conf = 0.0
        
        return is_freeze, conf

    def _check_fade(frames: List[np.ndarray]) -> Tuple[bool, float]:
        """
        Detecta fade/tela preta pelo brilho do ÚLTIMO frame (frame atual).
        Não usa média do buffer pois dilui detecções em transições.
        Confiança proporcional à escuridão:
        - brightness = 0 → conf = 1.0 (tela completamente preta)
        - brightness = THRESHOLD → conf = 0.85
        """
        if len(frames) < 1:
            return False, 0.0
        
        # Usa o ÚLTIMO frame (frame atual), não a média do buffer
        # Isso permite detectar fade em transições sem diluir com frames normais
        current_frame = frames[-1]
        current_brightness = np.mean(current_frame)
        
        is_fade = current_brightness < FADE_BRIGHTNESS_THRESHOLD
        
        if is_fade:
            # Confiança proporcional: quanto mais escuro, maior a confiança
            # brightness=0 → conf=1.0, brightness=THRESHOLD → conf=0.85
            darkness_ratio = 1.0 - (current_brightness / FADE_BRIGHTNESS_THRESHOLD)
            conf = 0.85 + (0.15 * darkness_ratio)  # Range: 0.85 a 1.0
            conf = min(1.0, max(0.85, conf))
        else:
            conf = 0.0
        return is_fade, conf

    def _check_blur(frames: List[np.ndarray]) -> Tuple[bool, float]:
        """
        Detecta desfoque usando variância do Laplaciano do FRAME ATUAL.
        Não usa média do buffer pois dilui detecções em transições.
        NÃO detecta blur em tela preta (sharpness muito baixo = tela preta, não blur).
        """
        if len(frames) < 1:
            return False, 0.0
        
        # Usa o ÚLTIMO frame (frame atual), não a média do buffer
        current_frame = frames[-1]
        
        if len(current_frame.shape) == 3:
            gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY) if current_frame.shape[2] == 3 else current_frame[:,:,0]
        else:
            gray = current_frame
        
        current_sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        current_brightness = np.mean(current_frame)
        
        # Se tela está muito escura, NÃO é blur - é fade/tela preta
        if current_brightness < 20:
            return False, 0.0
        
        # Blur real: sharpness baixo MAS não zero (zero = tela preta)
        is_blur = current_sharpness < BLUR_SHARPNESS_THRESHOLD and current_sharpness > 10
        
        if is_blur:
            # Confiança proporcional: quanto menor o sharpness, maior a confiança
            conf = min(0.99, 1.0 - (current_sharpness / BLUR_SHARPNESS_THRESHOLD))
            conf = max(0.80, conf)  # Mínimo 0.80 para blur detectado
        else:
            conf = 0.0
        
        return is_blur, conf

    # Estado da análise
    processed_frames = 0
    frames_read = 0
    recent_frames: List[np.ndarray] = []  # Últimos N frames para heurísticas
    seq_queue = deque(maxlen=6)  # Para modelo de sequência
    
    # Contadores para suavização temporal
    heuristic_counts = {'freeze': 0, 'fade': 0, 'fora_de_foco': 0}
    heuristic_first_time = {'freeze': None, 'fade': None, 'fora_de_foco': None}
    model_predictions = []  # Lista de (class, conf, time)
    
    try:
        print(f"DEBUG: Analisando frames de vídeo: {file_path} (sample_rate_hz={sample_rate_hz})")
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"ERRO: Não foi possível abrir vídeo: {file_path}")
            return "Erro_Abertura_Vídeo", 0.0, None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps = float(fps) if fps > 0 else 30.0
        FRAME_SKIP = max(1, int(round(fps / float(max(0.001, sample_rate_hz)))))

        # Configuração do modelo
        model = globals().get('keras_video_model')
        expected_frames = 6
        expected_h = INPUT_HEIGHT
        expected_w = INPUT_WIDTH
        
        try:
            if model is not None and hasattr(model, 'inputs') and len(model.inputs) > 0:
                shape = getattr(model.inputs[0], 'shape', None)
                if shape is not None and len(shape) == 5:
                    if shape[1] is not None:
                        expected_frames = int(shape[1])
                    expected_h = int(shape[2]) if shape[2] is not None else INPUT_HEIGHT
                    expected_w = int(shape[3]) if shape[3] is not None else INPUT_WIDTH
        except Exception:
            pass

        seq_queue = deque(maxlen=expected_frames)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames_read += 1
            if frames_read % FRAME_SKIP != 0:
                continue

            # Tempo atual do frame
            try:
                ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                current_time_s = float(ms) / 1000.0 if ms is not None else None
            except Exception:
                current_time_s = None

            # Prepara frame para heurísticas e modelo
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (expected_w, expected_h))
            except Exception:
                continue

            # Mantém lista de frames recentes para heurísticas
            recent_frames.append(frame_resized)
            if len(recent_frames) > expected_frames:
                recent_frames.pop(0)
            
            # Adiciona ao seq_queue para modelo
            seq_queue.append(frame_resized.astype(np.float32))
            
            processed_frames += 1

            # === HEURÍSTICAS (rodam em paralelo ao modelo) ===
            is_freeze, freeze_conf = _check_freeze(recent_frames)
            is_fade, fade_conf = _check_fade(recent_frames)
            is_blur, blur_conf = _check_blur(recent_frames)

            # Registra detecções de heurísticas
            if is_freeze:
                heuristic_counts['freeze'] += 1
                if heuristic_first_time['freeze'] is None:
                    heuristic_first_time['freeze'] = current_time_s
            if is_fade:
                heuristic_counts['fade'] += 1
                if heuristic_first_time['fade'] is None:
                    heuristic_first_time['fade'] = current_time_s
            if is_blur:
                heuristic_counts['fora_de_foco'] += 1
                if heuristic_first_time['fora_de_foco'] is None:
                    heuristic_first_time['fora_de_foco'] = current_time_s

            # === MODELO DE DEEP LEARNING ===
            model_class = 'normal'
            model_conf = 0.0
            
            try:
                # Prepara sequência para o modelo
                seq_list = list(seq_queue)
                if len(seq_list) < expected_frames:
                    pad_frame = seq_list[-1] if seq_list else np.zeros((expected_h, expected_w, 3), dtype=np.float32)
                    while len(seq_list) < expected_frames:
                        seq_list.insert(0, pad_frame)
                
                seq_arr = np.stack(seq_list, axis=0)
                seq_batch = np.expand_dims(seq_arr, axis=0).astype(np.float32)
                
                # Normalização MobileNetV2
                if not globals().get('keras_video_model_has_rescaling'):
                    seq_batch = (seq_batch - 127.5) / 127.5
                
                # Inferência
                preds = model.predict(seq_batch, verbose=0)
                scores = np.array(preds[0], dtype=np.float32)
                probs = _apply_softmax(scores)
                idx = int(np.argmax(probs))
                model_conf = float(probs[idx])
                model_class = VIDEO_CLASSES[idx] if idx < len(VIDEO_CLASSES) else 'normal'
                
            except Exception as e:
                print(f"ERRO na inferência do modelo: {e}")
                model_class = 'normal'
                model_conf = 0.0

            # === LÓGICA DE DECISÃO (ENSEMBLE) ===
            final_class = model_class
            final_conf = model_conf
            method = "model"

            # OVERRIDE HEURÍSTICO: Heurísticas têm prioridade em casos óbvios
            # Se heurística detectou fade (tela preta), SEMPRE usar heurística
            # porque é um caso óbvio que não precisa de modelo
            if is_fade:
                final_class = 'fade'
                final_conf = max(fade_conf, 0.90)
                method = "heuristic_override"
            # Para freeze/blur, só override se modelo não tem certeza
            elif model_class == 'normal' and model_conf < MODEL_OVERRIDE_THRESHOLD:
                if is_freeze:
                    final_class = 'freeze'
                    final_conf = max(freeze_conf, 0.85)
                    method = "heuristic_override"
                elif is_blur:
                    final_class = 'fora_de_foco'
                    final_conf = max(blur_conf, 0.80)
                    method = "heuristic_override"

            # Registra predição
            model_predictions.append({
                'class': final_class,
                'confidence': final_conf,
                'time': current_time_s,
                'method': method,
                'heuristics': {'freeze': is_freeze, 'fade': is_fade, 'blur': is_blur}
            })

        cap.release()

        # === AGREGAÇÃO FINAL ===
        # Conta votos por classe
        class_votes = {}
        for pred in model_predictions:
            cls = pred['class']
            class_votes[cls] = class_votes.get(cls, 0) + 1

        # Encontra a classe dominante (excluindo 'normal')
        best_fault_class = 'normal'
        best_fault_confidence = 0.0
        best_fault_time = None

        fault_classes = {k: v for k, v in class_votes.items() if k != 'normal'}
        
        if fault_classes:
            # Classe com mais votos
            dominant_class = max(fault_classes, key=fault_classes.get)
            dominant_count = fault_classes[dominant_class]
            
            # Separa predições por método
            class_preds = [p for p in model_predictions if p['class'] == dominant_class]
            heuristic_preds = [p for p in class_preds if p.get('method') == 'heuristic_override']
            model_preds = [p for p in class_preds if p.get('method') != 'heuristic_override']
            
            # Se há muitas detecções de heurística (>= 2 segundos), usar confiança da heurística
            if len(heuristic_preds) >= 4:
                # Usa a máxima confiança das heurísticas - tela preta deve ser ~1.0
                max_heuristic_conf = np.max([p['confidence'] for p in heuristic_preds])
                avg_heuristic_conf = np.mean([p['confidence'] for p in heuristic_preds])
                # 80% máx + 20% média para manter alta confiança
                combined_conf = 0.8 * max_heuristic_conf + 0.2 * avg_heuristic_conf
            elif len(heuristic_preds) > 0:
                # Algumas detecções de heurística - média ponderada
                max_conf = np.max([p['confidence'] for p in class_preds])
                avg_conf = np.mean([p['confidence'] for p in class_preds])
                combined_conf = 0.6 * max_conf + 0.4 * avg_conf
            else:
                # Só modelo - usa média simples
                combined_conf = np.mean([p['confidence'] for p in class_preds])
            
            # Verifica se passa na regra de duração mínima (2 segundos)
            # Cada frame é ~0.5s com sample_rate_hz=2.0, então 4 frames = 2s
            min_frames_for_error = max(1, int(MIN_ERROR_DURATION_S * sample_rate_hz))
            
            if dominant_count >= min_frames_for_error or combined_conf > 0.90:
                best_fault_class = dominant_class
                best_fault_confidence = combined_conf
                # Pega o primeiro tempo de detecção
                first_detection = next((p for p in model_predictions if p['class'] == dominant_class), None)
                best_fault_time = first_detection['time'] if first_detection else None
        
        # Se não encontrou falha, retorna normal
        if best_fault_class == 'normal':
            normal_preds = [p for p in model_predictions if p['class'] == 'normal']
            best_fault_confidence = np.mean([p['confidence'] for p in normal_preds]) if normal_preds else 0.5

        # Debug
        debug_heuristics = {k: v for k, v in heuristic_counts.items() if v > 0}
        print(f"DEBUG: Vídeo - {processed_frames} frames processados ({frames_read} lidos). "
              f"Resultado: {best_fault_class} ({best_fault_confidence:.4f}) time={best_fault_time} "
              f"(heuristics: {debug_heuristics})")

        return best_fault_class, best_fault_confidence, best_fault_time

    except Exception as e:
        print(f"ERRO durante análise de frames de vídeo ({file_path}): {e}")
        traceback.print_exc()
        if 'cap' in locals() and cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
        return "Erro_Análise_Vídeo", 0.0, None

# === Wrappers Antigos (NÃO USADOS DIRETAMENTE PELO ENDPOINT AGORA) ===
# Mantidos para referência ou testes unitários futuros, se necessário

def run_audio_inference_single_segment(input_data: np.ndarray) -> Tuple[str, float]:
    """ Roda inferência em um único segmento de áudio pré-processado. """
    if globals().get('keras_audio_model') is not None:
        return run_keras_inference(globals().get('keras_audio_model'), input_data, AUDIO_CLASSES)
    raise RuntimeError("Modelo de áudio Keras não carregado. Por favor coloque 'audio_model_finetune.keras' ou 'audio_model_finetune.h5' em backend/app/ml/models/.")

def run_video_inference_single_frame(input_data: np.ndarray) -> Tuple[str, float]:
    """ Roda inferência em um único frame de vídeo pré-processado. """
    if globals().get('keras_video_model') is not None:
        return run_keras_inference(globals().get('keras_video_model'), input_data, MODEL_CLASSES)
    raise RuntimeError("Modelo de vídeo Keras não carregado. Por favor coloque 'video_model_finetune.keras' ou 'video_model_finetune.h5' em backend/app/ml/models/.")

print("INFO: Módulo de inferência carregado (Keras-only mode).")

# --- Compat wrappers (legacy names expected elsewhere/tests) ---
def run_audio_inference(input_data: np.ndarray) -> Tuple[str, float]:
    return run_audio_inference_single_segment(input_data)

def run_video_inference(input_data: np.ndarray) -> Tuple[str, float]:
    return run_video_inference_single_frame(input_data)

def run_video_topk(input_data: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
    # Keras-only top-k
    if globals().get('keras_video_model') is not None:
        return run_keras_inference_topk(globals().get('keras_video_model'), input_data, MODEL_CLASSES, k)
    return []


def analyze_video_frames_diagnostic(file_path: str, k: int = 3, sample_rate_hz: float = 2.0, max_samples: int = 200) -> List[dict]:
    """Analisa frames e retorna uma lista de dicionários com timestamp e top-k scores.
    Útil para diagnóstico (não altera comportamento principal).
    Cada item: {'time_s': float, 'topk': [{'class': str, 'score': float}, ...]}.
    sample_rate_hz: approximate number of samples per second to capture (default 2 Hz).
    """
    results = []
    if not globals().get('keras_video_model'):
        return results
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return results

        frames_read = 0
        samples = 0
        prev_gray = None

        # Determine FPS and compute frame skip
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        try:
            fps = float(fps) if fps > 0 else 30.0
        except Exception:
            fps = 30.0
        frame_skip = max(1, int(round(fps / float(max(0.001, sample_rate_hz)))))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_read += 1
            if frames_read % frame_skip != 0:
                continue

            input_data = preprocess_video_single_frame(frame)
            if input_data is None:
                continue

            # Run model predict directly to capture raw score vector length for debugging
            try:
                model = globals().get('keras_video_model')
                preds = model.predict(input_data)
                scores = np.array(preds[0], dtype=np.float32)
                probs = _apply_softmax(scores)
                indices = np.argsort(probs)[::-1]
                top = []
                for i in indices[:max(1, k)]:
                    name = VIDEO_CLASSES[i] if i < len(VIDEO_CLASSES) else f"idx_{i}"
                    top.append((name, float(probs[i])))
                output_len = int(scores.shape[0])
            except Exception:
                top = []
                output_len = 0

            # Get current time of the frame
            try:
                ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                time_s = float(ms) / 1000.0 if ms is not None else None
            except Exception:
                time_s = None

            # also compute heuristic scores for diagnostics: blur_var, brightness, motion, edge_density
            try:
                blur_var = _compute_blur_var(frame)
                brightness = _compute_brightness(frame)
                edge_density = _compute_edge_density(frame)
            except Exception:
                blur_var = 0.0
                brightness = 0.0
                edge_density = 0.0
            # compute motion using previous sampled frame if available
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small_gray = cv2.resize(gray, (INPUT_WIDTH, INPUT_HEIGHT))
                if prev_gray is None:
                    motion_val = float('inf')
                else:
                    diff = cv2.absdiff(small_gray, prev_gray)
                    motion_val = float(diff.mean())
                prev_gray = small_gray
            except Exception:
                motion_val = None
            results.append({
                'time_s': time_s,
                'topk': [{'class': t[0], 'score': float(t[1])} for t in top],
                'output_len': output_len,
                'heuristics': {'blur_var': float(blur_var), 'brightness': float(brightness), 'edge_density': float(edge_density), 'motion': motion_val}
            })
            samples += 1
            if samples >= max_samples:
                break

        cap.release()
        return results
    except Exception:
        try:
            if 'cap' in locals() and cap.isOpened():
                cap.release()
        except Exception:
            pass
        return results