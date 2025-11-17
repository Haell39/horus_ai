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

# === Definições (Mantidas) ===
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
# Prefer models under mobilenetv2/ to keep versions organized; fallback to root filenames
MOBILENET_DIR = os.path.join(MODEL_DIR, 'mobilenetv2')

# New model filenames (user supplied INT8 quantized models)
VIDEO_MODEL_FILENAME = os.path.join('mobilenetv2', 'video_model_int8.tflite')
AUDIO_MODEL_FILENAME = os.path.join('mobilenetv2', 'audio_model_int8.tflite')

VIDEO_MODEL_PATH = os.path.join(MODEL_DIR, VIDEO_MODEL_FILENAME)
AUDIO_MODEL_PATH = os.path.join(MODEL_DIR, AUDIO_MODEL_FILENAME)

# Classes for the new models (as provided)
# Default fallbacks (kept for backward compat) — real class list is loaded from training_files/labels.csv when available
AUDIO_CLASSES = ['ausencia_audio', 'volume_baixo', 'eco', 'ruido', 'sinal_1khz']
VIDEO_CLASSES = ['freeze', 'fade', 'fora_foco']

# Additional package folder (user-provided model bundle). If you placed a model package
# at repository root `horus_package_video_model_v2` we attempt to use its metadata/labels
PACKAGE_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'horus_package_video_model_v2'))

# Load metadata if present in the package model dir or in MODEL_DIR
MODEL_METADATA = {}
THRESHOLDS = {}
try:
    # prefer metadata in MODEL_DIR, else PACKAGE_MODEL_DIR
    metadata_candidates = [os.path.join(MODEL_DIR, 'video_model_finetune.metadata.json'), os.path.join(PACKAGE_MODEL_DIR, 'video_model_finetune.metadata.json')]
    for mpath in metadata_candidates:
        if os.path.exists(mpath):
            with open(mpath, 'r', encoding='utf-8') as fh:
                MODEL_METADATA = json.load(fh)
            break
except Exception:
    MODEL_METADATA = {}

try:
    thresh_candidates = [os.path.join(MODEL_DIR, 'thresholds.yaml'), os.path.join(PACKAGE_MODEL_DIR, 'thresholds.yaml')]
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

def _load_model_classes_from_training_files():
    try:
        import csv
        training_labels = os.path.join(MODEL_DIR, 'training_files', 'labels.csv')
        if not os.path.exists(training_labels):
            return []
        seen = []
        with open(training_labels, 'r', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                cls = row.get('class')
                if not cls:
                    continue
                # If multi-label (pipe-separated), take each
                parts = [p.strip() for p in cls.split('|') if p.strip()]
                for p in parts:
                    if p not in seen:
                        seen.append(p)
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
                # assume format [frames, height, width, channels] or [height, width, channels]
                if len(ishape) == 4:
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
        audio_candidates = [
            os.path.join(MODEL_DIR, 'audio_model_finetune.keras'),
            os.path.join(MODEL_DIR, 'audio_model_finetune.h5'),
            os.path.join(MODEL_DIR, 'audio_model.h5')
        ]
        video_candidates = [
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
                    keras_audio_model = tf.keras.models.load_model(p)
                    print(f"INFO: Modelo de Áudio Keras carregado. Path={p}")
                    try:
                        print(f"INFO: Audio model output shape: {keras_audio_model.output_shape}")
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
                    keras_video_model = tf.keras.models.load_model(p)
                    print(f"INFO: Modelo de Vídeo Keras carregado. Path={p}")
                    try:
                        print(f"INFO: Video model output shape: {keras_video_model.output_shape}")
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
        # If the model expects multiple inputs but we received a single ndarray,
        # attempt to build a list of ordered inputs (image, motion, brightness)
        preds = None
        try:
            inputs_spec = getattr(model, 'inputs', None)
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
                    elif ('image' in lname) or ('input' in lname) or (shape is not None and len(shape) == 4):
                        ordered.append(input_data)
                    else:
                        # default fallback: if expects 4D assume image, else scalar zero
                        if shape is not None and len(shape) == 4:
                            ordered.append(input_data)
                        else:
                            ordered.append(np.zeros((1, 1), dtype=np.float32))
                preds = model.predict(ordered)
            else:
                preds = model.predict(input_data)
        except Exception:
            # fallback to direct predict
            preds = model.predict(input_data)
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
        # same multi-input guard as run_keras_inference
        inputs_spec = getattr(model, 'inputs', None)
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
                elif ('image' in lname) or ('input' in lname) or (shape is not None and len(shape) == 4):
                    ordered.append(input_data)
                else:
                    if shape is not None and len(shape) == 4:
                        ordered.append(input_data)
                    else:
                        ordered.append(np.zeros((1, 1), dtype=np.float32))
            preds = model.predict(ordered)
        else:
            preds = model.predict(input_data)
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
                preds = model.predict(ordered)
            else:
                # model may accept 5D directly
                preds = model.predict(seq_data)
        except Exception:
            # fallback: try predicting with sequence, else with middle frame
            try:
                preds = model.predict(seq_data)
            except Exception:
                try:
                    mid = seq_data.shape[1] // 2
                    preds = model.predict(np.expand_dims(seq_data[0, mid].astype(np.float32), axis=0))
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
    """ Pré-processa UM segmento de áudio (y). """
    try:
        # --- PARÂMETROS (DEVEM SER OS MESMOS DO TREINO) ---
        N_MELS = 128
        FMAX = 8000
        HOP_LENGTH = 512
        N_FFT = 2048
        # --- FIM PARÂMETROS ---

        if len(y_segment) == 0: return None

        mel_spec = librosa.feature.melspectrogram(y=y_segment, sr=sr, n_mels=N_MELS, fmax=FMAX, hop_length=HOP_LENGTH, n_fft=N_FFT)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        min_db, max_db = np.min(mel_spec_db), np.max(mel_spec_db)
        img_gray = np.zeros_like(mel_spec_db, dtype=np.float32)
        if max_db > min_db: # Evita divisão por zero
             img_gray = (mel_spec_db - min_db) / (max_db - min_db)

        img_rgb = np.stack([img_gray]*3, axis=-1)
        # Convert to 0-255 uint8 image then resize
        img_uint8 = (np.clip(img_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        img_tf = tf.convert_to_tensor(img_uint8, dtype=tf.float32)
        img_resized_tf = tf.image.resize(img_tf, [INPUT_HEIGHT, INPUT_WIDTH], method=tf.image.ResizeMethod.BILINEAR)
        img_resized = img_resized_tf.numpy().astype(np.float32)
        # Apply MobileNetV2 preprocessing per user: (x - 127.5) / 127.5
        img_preproc = (img_resized - 127.5) / 127.5
        input_data = np.expand_dims(img_preproc, axis=0)

        return input_data if input_data.shape == (1, INPUT_HEIGHT, INPUT_WIDTH, 3) else None
    except Exception as e:
        # print(f"ERRO pré-processando segmento de áudio: {e}") # Log menos verboso
        return None

def analyze_audio_segments(file_path: str) -> Tuple[str, float]:
    """ Analisa múltiplos segmentos de áudio e retorna a falha mais confiante. """
    if not globals().get('keras_audio_model'):
        raise RuntimeError("Modelo de áudio Keras não está carregado. Remova dependências TFLite ou instale o modelo Keras em backend/app/ml/models/.")

    best_fault_class = 'normal'
    max_confidence = 0.0
    processed_segments = 0

    try:
        # --- PARÂMETROS DE SEGMENTAÇÃO ---
        TARGET_SR = None # Use None ou sr específico (ex: 16000)
        SEGMENT_DURATION_S = 3.0 # Duração de cada segmento para análise
        HOP_DURATION_S = 1.0     # Sobreposição (avança 1s por vez)
        # --- FIM PARÂMETROS ---

        print(f"DEBUG: Analisando segmentos de áudio: {file_path}")
        # Carrega o áudio completo uma vez
        y_full, sr = librosa.load(file_path, sr=TARGET_SR)
        if len(y_full) == 0:
            print(f"AVISO: Áudio completo vazio: {file_path}")
            return 'normal', 0.0 # Se vazio, é normal

        segment_samples = int(SEGMENT_DURATION_S * sr)
        hop_samples = int(HOP_DURATION_S * sr)

        for i in range(0, len(y_full) - segment_samples + 1, hop_samples):
            y_segment = y_full[i : i + segment_samples]

            # Pré-processa o segmento
            input_data = preprocess_audio_segment(y_segment, sr)
            if input_data is None:
                continue # Pula segmento se pré-processamento falhar

            # Roda a inferência no segmento (Keras preferido)
            # Keras-only inference
            pred_class, confidence = run_keras_inference(globals().get('keras_audio_model'), input_data, MODEL_CLASSES)
            processed_segments += 1

            # Estratégia: Guarda a falha (não-normal) com maior confiança encontrada até agora
            if pred_class != 'normal' and confidence > max_confidence:
                max_confidence = confidence
                best_fault_class = pred_class
            # Se ainda não achamos falha, mas achamos 'normal' com alta confiança, guardamos isso
            elif best_fault_class == 'normal' and pred_class == 'normal' and confidence > max_confidence:
                 max_confidence = confidence # Atualiza a confiança do 'normal'

        print(f"DEBUG: Áudio - {processed_segments} segmentos processados. Resultado: {best_fault_class} ({max_confidence:.4f})")
        # Se nenhuma falha foi encontrada (best_fault_class ainda é 'normal'), retorna 'normal'
        # com a maior confiança de 'normal' encontrada (ou 0 se nenhum segmento foi processado).
        # Se uma falha foi encontrada, retorna a falha e sua confiança.
        return best_fault_class, max_confidence

    except Exception as e:
        print(f"ERRO durante análise de segmentos de áudio ({file_path}): {e}")
        traceback.print_exc()
        return "Erro_Análise_Áudio", 0.0

# =====================================================
# === NOVAS FUNÇÕES: ANÁLISE DE VÍDEO MULTI-FRAME ===
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
    """ Analisa múltiplos frames de vídeo e retorna a falha mais confiante.
    Retorna (pred_class, confidence, event_time_s) onde event_time_s é o tempo em
    segundos do frame com maior confiança para uma classe não-'normal', ou None.
    """
    if not globals().get('keras_video_model'):
        raise RuntimeError("Modelo de vídeo Keras não está carregado. Remova dependências TFLite ou instale o modelo Keras em backend/app/ml/models/.")

    best_fault_class = 'normal'
    max_confidence = 0.0
    event_time_s: Optional[float] = None
    processed_frames = 0
    frames_read = 0
    prev_gray = None
    # window to track recent brightness values for fade detection
    try:
        bw_len = max(1, int(core_settings.VIDEO_MOVING_AVG_M))
    except Exception:
        bw_len = 5
    brightness_window = deque(maxlen=bw_len)
    # counters for heuristic support and first supporting time
    blur_support = 0
    fade_support = 0
    freeze_support = 0
    first_blur_time = None
    first_fade_time = None
    first_freeze_time = None
    # flag: if any frame showed a sudden brightness drop (strong fade)
    fade_drop_detected = False
    try:
        print(f"DEBUG: Analisando frames de vídeo: {file_path} (sample_rate_hz={sample_rate_hz})")
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"ERRO: Não foi possível abrir vídeo para análise de frames: {file_path}")
            return "Erro_Abertura_Vídeo", 0.0, None

        # Compute frame skip using video FPS and desired sample rate
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        try:
            fps = float(fps) if fps > 0 else 30.0
        except Exception:
            fps = 30.0
        FRAME_SKIP = max(1, int(round(fps / float(max(0.001, sample_rate_hz)))))

        while True:
            # Lê o frame
            ret, frame = cap.read()
            if not ret:
                break # Fim do vídeo

            frames_read += 1
            # Pula frames de acordo com taxa
            if frames_read % FRAME_SKIP != 0:
                continue

            # Pré-processa o frame selecionado
            input_data = preprocess_video_single_frame(frame)
            if input_data is None:
                continue # Pula frame se pré-processamento falhar

            # Heurísticas por frame
            blur_var = _compute_blur_var(frame)
            brightness = _compute_brightness(frame)
            motion = _compute_motion(prev_gray, frame)
            edge_density = _compute_edge_density(frame)

            # Keras-only inference (handle models that expect auxiliary inputs motion/brightness)
            model = globals().get('keras_video_model')
            pred_class, confidence = 'normal', 0.0
            if model is None:
                raise RuntimeError("Modelo de vídeo Keras não está carregado.")
            try:
                # prepare raw resized RGB frame (no normalization) for models that include Rescaling
                try:
                    raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    raw_resized = cv2.resize(raw_rgb, (INPUT_WIDTH, INPUT_HEIGHT)).astype(np.float32)
                except Exception:
                    raw_resized = None

                # If the model expects auxiliary inputs, build inputs according to model.inputs
                if globals().get('keras_video_model_requires_motion_brightness'):
                    # motion normalized to [0,1]
                    motion_norm = 1.0 if motion == float('inf') else float(motion) / 255.0
                    motion_arr = np.array([[motion_norm]], dtype=np.float32)
                    brightness_norm = float(brightness) / 255.0
                    bright_arr = np.array([[brightness_norm]], dtype=np.float32)

                    # image input: if the model already rescales internally, pass raw uint8/resized image
                    if globals().get('keras_video_model_has_rescaling') and raw_resized is not None:
                        img_for = np.expand_dims(raw_resized, axis=0)
                    else:
                        img_for = input_data

                    # Build inputs in the order the model expects by inspecting model.inputs
                    try:
                        ordered_inputs = []
                        for inp in model.inputs:
                            in_name = getattr(inp, 'name', '') or ''
                            lname = in_name.lower()
                            shape = getattr(inp, 'shape', None)
                            # Heuristics to match input by name or shape
                            if 'motion' in lname:
                                ordered_inputs.append(motion_arr)
                            elif 'brightness' in lname or 'bright' in lname:
                                ordered_inputs.append(bright_arr)
                            elif ('image' in lname) or ('input' in lname) or (shape is not None and len(shape) == 4):
                                ordered_inputs.append(img_for)
                            else:
                                # Fallback: if input expects a scalar / 1D vector, try brightness/motion
                                if shape is not None and len(shape) == 2:
                                    ordered_inputs.append(motion_arr)
                                else:
                                    ordered_inputs.append(img_for)

                        preds = model.predict(ordered_inputs)
                    except Exception:
                        # As a last resort try dict mapping by common keys, then single-input
                        try:
                            preds = model.predict({'image': img_for, 'motion': motion_arr, 'brightness': bright_arr})
                        except Exception:
                            try:
                                preds = model.predict([img_for, motion_arr, bright_arr])
                            except Exception:
                                preds = model.predict(img_for)
                else:
                    # simple single-input model
                    if globals().get('keras_video_model_has_rescaling') and raw_resized is not None:
                        img_for = np.expand_dims(raw_resized, axis=0)
                    else:
                        img_for = input_data
                    preds = model.predict(img_for)

                scores = np.array(preds[0], dtype=np.float32)
                probs = _apply_softmax(scores)
                idx = int(np.argmax(probs))
                confidence = float(probs[idx])
                pred_class = MODEL_CLASSES[idx] if idx < len(MODEL_CLASSES) else f"idx_{idx}"
                # Guard against cases where the video model outputs labels
                # that belong to the audio label set (e.g. 'ausencia_audio').
                # For the video pipeline we only care about visual faults
                # (freeze/fade/fora_foco). If the predicted class is not one
                # of the expected video classes, treat it as 'normal' so that
                # audio-only labels do not cause visual occurrences.
                try:
                    expected_video = [c.lower() for c in VIDEO_CLASSES]
                    if str(pred_class).lower() not in expected_video and str(pred_class).lower() != 'normal':
                        pred_class = 'normal'
                        confidence = 0.0
                except Exception:
                    pass
            except Exception as e:
                print(f"ERRO durante inferência Keras no loop de frames: {e}")
                traceback.print_exc()
                pred_class, confidence = 'Erro_Inferência', 0.0
            processed_frames += 1

            # Combine heuristics with model prediction
            effective_conf = confidence
            # If heuristic strongly indicates a condition, boost confidence or override
            # Blur detection: use either low Laplacian variance OR low edge density
            blur_thresh = float(core_settings.VIDEO_BLUR_VAR_THRESHOLD)
            edge_thresh = float(getattr(core_settings, 'VIDEO_EDGE_DENSITY_THRESHOLD', 0.015))
            is_blur_by_var = (blur_var < blur_thresh)
            is_blur_by_edges = (edge_density < edge_thresh)
            if is_blur_by_var or is_blur_by_edges:
                # If model also predicted blur-like class increase confidence
                lc = pred_class.lower()
                if 'fora_foco' in lc or 'fora' in lc or 'borr' in lc:
                    effective_conf = max(effective_conf, 0.85)
                else:
                    # If model didn't predict blur but heuristic strong, propose 'fora_foco' with mid confidence
                    effective_conf = max(effective_conf, 0.75)
                    pred_class = 'fora_foco'
                # debug print for heuristic triggers (helpful during tuning)
                try:
                    print(f"DEBUG: Blur heuristic triggered (var={blur_var:.1f} edge_density={edge_density:.4f}) -> pred={pred_class} eff_conf={effective_conf:.3f}")
                except Exception:
                    pass
                blur_support += 1
                try:
                    # capture first supporting time
                    if first_blur_time is None:
                        ms_tmp = cap.get(cv2.CAP_PROP_POS_MSEC)
                        first_blur_time = float(ms_tmp) / 1000.0 if ms_tmp is not None else None
                except Exception:
                    pass

            # Freeze (low motion) -> prefer 'freeze'
            motion_thresh = float(core_settings.VIDEO_MOTION_THRESHOLD)
            if motion != float('inf') and motion < motion_thresh:
                if 'freeze' in pred_class.lower():
                    effective_conf = max(effective_conf, 0.85)
                else:
                    effective_conf = max(effective_conf, 0.8)
                    pred_class = 'freeze'
                freeze_support += 1
                try:
                    if first_freeze_time is None:
                        ms_tmp = cap.get(cv2.CAP_PROP_POS_MSEC)
                        first_freeze_time = float(ms_tmp) / 1000.0 if ms_tmp is not None else None
                except Exception:
                    pass

            # Fade detection: sustained low brightness or a sudden drop relative
            # to a short moving-average window is considered a 'fade' (tela
            # preta/escurecimento). We use both an absolute low threshold and a
            # relative drop ratio to catch quick fades.
            try:
                brightness_window.append(brightness)
                avg_brightness = float(sum(brightness_window) / len(brightness_window)) if len(brightness_window) > 0 else float(brightness)
            except Exception:
                avg_brightness = float(brightness)

            bright_low = float(core_settings.VIDEO_BRIGHTNESS_LOW)
            drop_ratio_thr = float(getattr(core_settings, 'VIDEO_BRIGHTNESS_DROP_RATIO', 0.5))
            # relative drop (how much current brightness dropped vs recent average)
            try:
                drop_ratio = (avg_brightness - brightness) / (avg_brightness + 1e-9)
            except Exception:
                drop_ratio = 0.0

            is_fade_by_low = (brightness < bright_low)
            is_fade_by_drop = (avg_brightness > 1.0 and drop_ratio >= drop_ratio_thr)

            if is_fade_by_low or is_fade_by_drop:
                # If model already predicted 'fade', boost strongly; otherwise set
                # pred_class to 'fade' and give it a high effective confidence so
                # it will be preferred by aggregation/voting.
                if 'fade' in pred_class.lower():
                    effective_conf = max(effective_conf, 0.95)
                else:
                    # stronger boost for relative drop (sudden fade)
                    if is_fade_by_drop:
                        effective_conf = max(effective_conf, 0.92)
                    else:
                        effective_conf = max(effective_conf, 0.85)
                    pred_class = 'fade'

                fade_support += 1
                if is_fade_by_drop:
                    fade_drop_detected = True
                try:
                    if first_fade_time is None:
                        ms_tmp = cap.get(cv2.CAP_PROP_POS_MSEC)
                        first_fade_time = float(ms_tmp) / 1000.0 if ms_tmp is not None else None
                except Exception:
                    pass
                try:
                    print(f"DEBUG: Fade heuristic triggered (bright={brightness:.1f} avg={avg_brightness:.1f} drop_ratio={drop_ratio:.3f}) -> pred={pred_class} eff_conf={effective_conf:.3f}")
                except Exception:
                    pass

            # Estratégia: Guarda a falha (não-normal) com maior confiança efetiva
            if pred_class != 'normal' and effective_conf > max_confidence:
                max_confidence = effective_conf
                best_fault_class = pred_class
                # pega o tempo atual do vídeo em ms
                try:
                    ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    event_time_s = float(ms) / 1000.0 if ms is not None else None
                except Exception:
                    event_time_s = None
            elif best_fault_class == 'normal' and pred_class == 'normal' and confidence > max_confidence:
                max_confidence = confidence # Guarda a maior confiança do 'normal'

            # update prev_gray
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.resize(gray, (INPUT_WIDTH, INPUT_HEIGHT))
            except Exception:
                prev_gray = None

        cap.release()
        # If heuristics show consistent evidence (N-of-M) prefer that class even
        # if aggregated model confidences were lower. This helps deterministic
        # detection for blur/fade/freeze when the frame-level model is noisy.
        try:
            vote_k = int(core_settings.VIDEO_VOTE_K)
        except Exception:
            vote_k = 2

        # prefer heuristics if they triggered on at least vote_k samples.
        # When tied, prefer a detected FADE if any frame had a sudden drop; otherwise
        # prefer freeze over blur because frozen frames often have near-zero motion
        # while blur can be noisy when frames are black or low-contrast.
        max_support = max(blur_support, fade_support, freeze_support)
        if max_support >= vote_k:
            # decide winner by support counts with tie-breaker that prefers fade drops
            if fade_support == max_support or (fade_drop_detected and fade_support > 0):
                chosen = 'fade'
                ts = first_fade_time
            elif freeze_support == max_support:
                chosen = 'freeze'
                ts = first_freeze_time
            elif blur_support == max_support:
                chosen = 'fora_foco'
                ts = first_blur_time
            else:
                # fallback safety
                chosen = 'fora_foco'
                ts = first_blur_time
            # override previous best if heuristics are convincing
            best_fault_class = chosen
            # boost confidence to ensure heuristics dominate when convincing
            max_confidence = max(max_confidence, 0.85 if chosen == 'fade' else 0.80)
            if ts is not None:
                event_time_s = event_time_s or ts

        print(f"DEBUG: Vídeo - {processed_frames} frames processados ({frames_read} lidos). Resultado: {best_fault_class} ({max_confidence:.4f}) time={event_time_s} (supports blur={blur_support} fade={fade_support} freeze={freeze_support})")
        return best_fault_class, max_confidence, event_time_s

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
        return run_keras_inference(globals().get('keras_audio_model'), input_data, MODEL_CLASSES)
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
                    name = MODEL_CLASSES[i] if i < len(MODEL_CLASSES) else f"idx_{i}"
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
            # motion cannot be computed reliably here without tracking prev frame; set to None in this diagnostic
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