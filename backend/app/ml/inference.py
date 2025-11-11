# backend/app/ml/inference.py
# (Versão TFLite com Análise Multi-Segmento/Frame)

import os
import numpy as np
import tensorflow as tf
from tensorflow.lite.python.interpreter import Interpreter
from PIL import Image
import librosa
import cv2 # OpenCV
from typing import Tuple, Optional, List, Dict
import traceback
import math

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
AUDIO_CLASSES = ['ausencia_audio', 'volume_baixo', 'eco', 'ruido', 'sinal_1khz']
VIDEO_CLASSES = ['freeze', 'fade', 'fora_foco']

# Input size as specified by the user (160x160x3)
INPUT_HEIGHT = 160
INPUT_WIDTH = 160

# === Carregamento dos Intérpretes (Mantido) ===
audio_interpreter = None
video_interpreter = None
# ... (restante das variáveis globais e função load_all_models idêntica à anterior) ...
audio_input_details = None
audio_output_details = None
video_input_details = None
video_output_details = None
models_loaded = False

def load_all_models():
    """ Carrega os modelos TFLite usando tf.lite.Interpreter. """
    global audio_interpreter, video_interpreter, models_loaded
    global audio_input_details, audio_output_details, video_input_details, video_output_details
    print("INFO: Carregando modelos TFLite via TensorFlow...")
    loaded_any = False
    try:
        # Prefer new mobilenetv2 models; if missing, keep previous filenames (backwards compat)
        audio_path_to_try = AUDIO_MODEL_PATH
        video_path_to_try = VIDEO_MODEL_PATH

        # Backwards-compatible fallbacks (older filenames present in repo)
        if not os.path.exists(audio_path_to_try):
            alt = os.path.join(MODEL_DIR, 'audio_model_quant.tflite')
            if os.path.exists(alt):
                audio_path_to_try = alt

        if not os.path.exists(video_path_to_try):
            alt = os.path.join(MODEL_DIR, 'video_model_quant.tflite')
            if os.path.exists(alt):
                video_path_to_try = alt

        if os.path.exists(audio_path_to_try):
            audio_interpreter = tf.lite.Interpreter(model_path=audio_path_to_try)
            audio_interpreter.allocate_tensors()
            audio_input_details = audio_interpreter.get_input_details()[0]
            audio_output_details = audio_interpreter.get_output_details()[0]
            print(f"INFO: Modelo de Áudio TFLite carregado. Path={audio_path_to_try} Input: {audio_input_details['shape']}, dtype: {audio_input_details['dtype']}")
            loaded_any = True
        else:
            print(f"AVISO: Modelo de Áudio TFLite não encontrado (tentadas: {AUDIO_MODEL_PATH}, audio_model_quant.tflite)")

        if os.path.exists(video_path_to_try):
            video_interpreter = tf.lite.Interpreter(model_path=video_path_to_try)
            video_interpreter.allocate_tensors()
            video_input_details = video_interpreter.get_input_details()[0]
            video_output_details = video_interpreter.get_output_details()[0]
            print(f"INFO: Modelo de Vídeo TFLite carregado. Path={video_path_to_try} Input: {video_input_details['shape']}, dtype: {video_input_details['dtype']}")
            loaded_any = True
        else:
            print(f"AVISO: Modelo de Vídeo TFLite não encontrado (tentadas: {VIDEO_MODEL_PATH}, video_model_quant.tflite)")

        models_loaded = loaded_any
        if models_loaded:
            print("INFO: Carregamento de modelos TFLite concluído.")
        else:
            print("AVISO: Nenhum modelo TFLite foi carregado.")

    except Exception as e:
        print(f"ERRO CRÍTICO ao carregar modelos TFLite: {e}")
        traceback.print_exc()
        models_loaded = False
load_all_models() # Carrega na importação

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
    if not audio_interpreter: raise RuntimeError("Intérprete de áudio TFLite não carregado.")

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

            # Roda a inferência no segmento
            pred_class, confidence = run_tflite_inference(
                audio_interpreter, audio_input_details, audio_output_details, input_data, AUDIO_CLASSES
            )
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

def analyze_video_frames(file_path: str) -> Tuple[str, float, Optional[float]]:
    """ Analisa múltiplos frames de vídeo e retorna a falha mais confiante.
    Retorna (pred_class, confidence, event_time_s) onde event_time_s é o tempo em
    segundos do frame com maior confiança para uma classe não-'normal', ou None.
    """
    if not video_interpreter: raise RuntimeError("Intérprete de vídeo TFLite não carregado.")

    best_fault_class = 'normal'
    max_confidence = 0.0
    event_time_s: Optional[float] = None
    processed_frames = 0
    frames_read = 0

    try:
        # --- PARÂMETROS DE ANÁLISE ---
        FRAME_SKIP = 30 # Analisa 1 frame a cada 30 (aprox. 1 por segundo se 30fps)
        # --- FIM PARÂMETROS ---

        print(f"DEBUG: Analisando frames de vídeo: {file_path}")
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"ERRO: Não foi possível abrir vídeo para análise de frames: {file_path}")
            return "Erro_Abertura_Vídeo", 0.0

        while True:
            # Lê o frame
            ret, frame = cap.read()
            if not ret:
                break # Fim do vídeo

            frames_read += 1
            # Pula frames
            if frames_read % FRAME_SKIP != 0:
                continue

            # Pré-processa o frame selecionado
            input_data = preprocess_video_single_frame(frame)
            if input_data is None:
                continue # Pula frame se pré-processamento falhar

            # Roda a inferência no frame
            pred_class, confidence = run_tflite_inference(
                video_interpreter, video_input_details, video_output_details, input_data, VIDEO_CLASSES
            )
            processed_frames += 1

            # Estratégia: Guarda a falha (não-normal) com maior confiança
            # captura também o timestamp do frame quando detectar a melhor falha
            if pred_class != 'normal' and confidence > max_confidence:
                max_confidence = confidence
                best_fault_class = pred_class
                # pega o tempo atual do vídeo em ms
                try:
                    ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    event_time_s = float(ms) / 1000.0 if ms is not None else None
                except Exception:
                    event_time_s = None
            elif best_fault_class == 'normal' and pred_class == 'normal' and confidence > max_confidence:
                max_confidence = confidence # Guarda a maior confiança do 'normal'

        cap.release()
        print(f"DEBUG: Vídeo - {processed_frames} frames processados ({frames_read} lidos). Resultado: {best_fault_class} ({max_confidence:.4f}) time={event_time_s}")
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
    if not audio_interpreter: raise RuntimeError("Intérprete de áudio TFLite não carregado.")
    return run_tflite_inference(audio_interpreter, audio_input_details, audio_output_details, input_data, AUDIO_CLASSES)

def run_video_inference_single_frame(input_data: np.ndarray) -> Tuple[str, float]:
    """ Roda inferência em um único frame de vídeo pré-processado. """
    if not video_interpreter: raise RuntimeError("Intérprete de vídeo TFLite não carregado.")
    return run_tflite_inference(video_interpreter, video_input_details, video_output_details, input_data, VIDEO_CLASSES)

print("INFO: Módulo de inferência TFLite (Multi-Segmento) inicializado.")

# --- Compat wrappers (legacy names expected elsewhere/tests) ---
def run_audio_inference(input_data: np.ndarray) -> Tuple[str, float]:
    return run_audio_inference_single_segment(input_data)

def run_video_inference(input_data: np.ndarray) -> Tuple[str, float]:
    return run_video_inference_single_frame(input_data)

def run_video_topk(input_data: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
    if not video_interpreter:
        return []
    return run_tflite_inference_topk(video_interpreter, video_input_details, video_output_details, input_data, VIDEO_CLASSES, k)