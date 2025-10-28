# backend/app/ml/inference.py
# (Versão TFLite usando TensorFlow Interpreter - Completo)

import os
import numpy as np
import tensorflow as tf # Usaremos tf.lite.Interpreter e tf.image
# Removido import de tflite_runtime
from PIL import Image
import librosa
import cv2 # OpenCV
from typing import Tuple, Optional, List
import traceback # Para logar erros detalhados

# === Definições ===
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
AUDIO_MODEL_FILENAME = 'audio_model_quant.tflite'
VIDEO_MODEL_FILENAME = 'video_model_quant.tflite'
AUDIO_MODEL_PATH = os.path.join(MODEL_DIR, AUDIO_MODEL_FILENAME)
VIDEO_MODEL_PATH = os.path.join(MODEL_DIR, VIDEO_MODEL_FILENAME)

AUDIO_CLASSES = ['baixo', 'eco', 'normal', 'ruido']
VIDEO_CLASSES = ['bloco', 'borrado', 'normal']

INPUT_HEIGHT = 224
INPUT_WIDTH = 224

# === Carregamento dos Intérpretes TFLite (via TensorFlow) ===
audio_interpreter = None
video_interpreter = None
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
        if os.path.exists(AUDIO_MODEL_PATH):
            audio_interpreter = tf.lite.Interpreter(model_path=AUDIO_MODEL_PATH) # <-- Mudança aqui
            audio_interpreter.allocate_tensors()
            audio_input_details = audio_interpreter.get_input_details()[0]
            audio_output_details = audio_interpreter.get_output_details()[0]
            print(f"INFO: Modelo de Áudio TFLite carregado de {AUDIO_MODEL_PATH}")
            loaded_any = True
        else:
            print(f"AVISO: Modelo de Áudio TFLite não encontrado em {AUDIO_MODEL_PATH}")

        if os.path.exists(VIDEO_MODEL_PATH):
            video_interpreter = tf.lite.Interpreter(model_path=VIDEO_MODEL_PATH) # <-- Mudança aqui
            video_interpreter.allocate_tensors()
            video_input_details = video_interpreter.get_input_details()[0]
            video_output_details = video_interpreter.get_output_details()[0]
            print(f"INFO: Modelo de Vídeo TFLite carregado de {VIDEO_MODEL_PATH}")
            loaded_any = True
        else:
            print(f"AVISO: Modelo de Vídeo TFLite não encontrado em {VIDEO_MODEL_PATH}")

        models_loaded = loaded_any
        print("INFO: Carregamento de modelos TFLite concluído.")

    except Exception as e:
        print(f"ERRO CRÍTICO ao carregar modelos TFLite: {e}")
        traceback.print_exc()
        models_loaded = False

# Carrega na importação
load_all_models()

# === Funções de Pré-processamento ===

def preprocess_audio(file_path: str) -> Optional[np.ndarray]:
    """ Carrega áudio, gera espectrograma MEL, redimensiona e normaliza. """
    try:
        # --- AJUSTE OS PARÂMETROS ABAIXO CONFORME SEU TREINAMENTO ---
        TARGET_SR = None # Use None para sr original ou defina (ex: 16000)
        DURATION = 5.0 # Segundos a carregar
        N_MELS = 128   # Número de bandas Mel
        FMAX = 8000    # Frequência máxima
        HOP_LENGTH = 512 # Salto (afeta largura do espectrograma)
        N_FFT = 2048   # Tamanho da FFT
        # --- FIM AJUSTES ---

        print(f"DEBUG: Processando áudio: {file_path}")
        y, sr = librosa.load(file_path, sr=TARGET_SR, duration=DURATION)
        if len(y) == 0:
             print(f"AVISO: Arquivo de áudio vazio ou muito curto: {file_path}")
             return None

        print(f"DEBUG: Áudio carregado. Duração: {len(y)/sr:.2f}s, SR: {sr}")
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX, hop_length=HOP_LENGTH, n_fft=N_FFT)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        print(f"DEBUG: Espectrograma Mel gerado. Shape: {mel_spec_db.shape}")

        # Normaliza para [0, 1] antes de converter para imagem
        img_gray = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db) + 1e-6)

        # Converte para 3 canais replicando
        img_rgb = np.stack([img_gray]*3, axis=-1)

        # Redimensiona usando TensorFlow Image
        img_tf = tf.convert_to_tensor(img_rgb, dtype=tf.float32)
        img_resized_tf = tf.image.resize(img_tf, [INPUT_HEIGHT, INPUT_WIDTH], method=tf.image.ResizeMethod.BILINEAR)
        img_resized = img_resized_tf.numpy()
        print(f"DEBUG: Imagem redimensionada para {img_resized.shape}")

        # Normalização final [0, 1] float32 (comum para MobileNetV2 TFLite quantizado ou float)
        img_normalized = img_resized.astype(np.float32) # Já deve estar em [0, 1]

        input_data = np.expand_dims(img_normalized, axis=0)

        if input_data.shape != (1, INPUT_HEIGHT, INPUT_WIDTH, 3):
            print(f"ERRO: Shape final do tensor de áudio inesperado: {input_data.shape}")
            return None

        print(f"DEBUG: Pré-processamento de áudio concluído. Shape final: {input_data.shape}")
        return input_data

    except Exception as e:
        print(f"ERRO no pré-processamento de áudio ({file_path}): {e}")
        traceback.print_exc()
        return None

def preprocess_video_frame(file_path: str) -> Optional[np.ndarray]:
    """ Abre vídeo, pega um frame, redimensiona, normaliza. """
    try:
        print(f"DEBUG: Processando vídeo: {file_path}")
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"ERRO: Não foi possível abrir o vídeo: {file_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            print(f"ERRO: Vídeo sem frames: {file_path}")
            cap.release()
            return None
        print(f"DEBUG: Vídeo com {total_frames} frames.")

        middle_frame_idx = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            print(f"ERRO: Não foi possível ler o frame do meio de {file_path}")
            return None
        print(f"DEBUG: Frame lido. Shape original: {frame.shape}")

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (INPUT_WIDTH, INPUT_HEIGHT))
        print(f"DEBUG: Frame redimensionado para {img_resized.shape}")

        # Normalização [0, 1] float32
        img_normalized = img_resized.astype(np.float32) / 255.0

        input_data = np.expand_dims(img_normalized, axis=0)

        if input_data.shape != (1, INPUT_HEIGHT, INPUT_WIDTH, 3):
             print(f"ERRO: Shape do tensor de vídeo inesperado: {input_data.shape}")
             return None

        print(f"DEBUG: Pré-processamento de vídeo concluído. Shape final: {input_data.shape}")
        return input_data

    except Exception as e:
        print(f"ERRO no pré-processamento de vídeo ({file_path}): {e}")
        traceback.print_exc()
        return None

# === Função de Inferência TFLite Genérica ===
def run_tflite_inference(interpreter: tf.lite.Interpreter, input_details: dict, output_details: dict, input_data: np.ndarray, classes: List[str]) -> Tuple[str, float]:
    """ Executa a inferência TFLite genérica. """
    if not interpreter or not models_loaded:
        raise RuntimeError("Intérprete TFLite não carregado.")

    try:
        input_dtype = input_details['dtype']
        # Verifica se o tipo de entrada é quantizado e ajusta
        if input_dtype == np.uint8 or input_dtype == np.int8:
            scale, zero_point = input_details['quantization']
            input_data_quant = (input_data / scale + zero_point).astype(input_dtype)
            interpreter.set_tensor(input_details['index'], input_data_quant)
            # print("DEBUG: Input TFLite quantizado.")
        else:
            input_data = input_data.astype(np.float32) # Garante float32
            interpreter.set_tensor(input_details['index'], input_data)
            # print("DEBUG: Input TFLite float32.")

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details['index'])[0]

        # Desquantiza a saída se necessário
        output_dtype = output_details['dtype']
        if output_dtype == np.uint8 or output_dtype == np.int8:
            scale, zero_point = output_details['quantization']
            output_data = scale * (output_data.astype(np.float32) - zero_point)
            # print("DEBUG: Output TFLite desquantizado.")

        predicted_index = np.argmax(output_data)
        confidence = float(output_data[predicted_index])

        if predicted_index >= len(classes):
            print(f"ERRO: Índice previsto ({predicted_index}) fora dos limites para as classes {classes}")
            predicted_class = "Erro_Indice"
            confidence = 0.0
        else:
            predicted_class = classes[predicted_index]

        print(f"DEBUG: Inferência: Classe={predicted_class}, Confiança={confidence:.4f}")
        return predicted_class, confidence

    except Exception as e:
        print(f"ERRO durante a execução da inferência TFLite: {e}")
        traceback.print_exc()
        return "Erro_Inferência", 0.0


# === Wrappers Específicos ===
def run_audio_inference(input_data: np.ndarray) -> Tuple[str, float]:
    """ Wrapper para inferência de áudio TFLite. """
    if not audio_interpreter: raise RuntimeError("Intérprete de áudio TFLite não carregado.")
    return run_tflite_inference(audio_interpreter, audio_input_details, audio_output_details, input_data, AUDIO_CLASSES)

def run_video_inference(input_data: np.ndarray) -> Tuple[str, float]:
    """ Wrapper para inferência de vídeo TFLite. """
    if not video_interpreter: raise RuntimeError("Intérprete de vídeo TFLite não carregado.")
    return run_tflite_inference(video_interpreter, video_input_details, video_output_details, input_data, VIDEO_CLASSES)

print("INFO: Módulo de inferência TFLite (via TF) inicializado.")