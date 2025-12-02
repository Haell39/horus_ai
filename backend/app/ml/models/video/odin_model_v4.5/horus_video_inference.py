#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HORUS V4.5 - Módulo de Inferência (Precisão Temporal)
=====================================================
Versão otimizada para detectar início/fim de falhas com maior precisão.
Usa janelas de 1.5s (6 frames) e step de 0.25s.

Uso:
    from horus_inference import HorusDetector
    
    detector = HorusDetector("models/horus_v4_5")
    result = detector.detect(frames)
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union

# Suprimir logs do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class HorusDetector:
    """
    Detector de anomalias em vídeo para o sistema HORUS V4.5.
    """
    
    LABELS = ['normal', 'freeze', 'fade', 'fora_de_foco'] # Ordem do treino V4.5
    
    # Configurações padrão (Ajustadas para 1.5s / 6 frames)
    DEFAULT_CONFIG = {
        # Heurísticas
        "freeze_diff_threshold": 2.0,
        "freeze_min_frames": 4,            # Ajustado para janela de 6 frames
        "fade_brightness_threshold": 15,
        "fade_min_frames": 3,              # Ajustado para janela de 6 frames
        "blur_sharpness_threshold": 130.0,
        "blur_min_frames": 4,              # Ajustado para janela de 6 frames
        
        # Modelo
        "model_confidence_threshold": 0.5,
        "sequence_length": 6,              # 1.5s @ 4 FPS
        "img_size": (192, 192),
        
        # Detecção de anomalias
        "min_anomaly_duration_sec": 1.0,   # Menor duração reportável
        "fps": 4,
    }
    
    def __init__(self, model_dir: str, use_tflite: bool = False):
        self.model_dir = Path(model_dir)
        self.use_tflite = use_tflite
        self.model = None
        self.tflite_interpreter = None
        self.config = self.DEFAULT_CONFIG.copy()
        
        self._load_metadata()
        self._load_model()
        
        print(f"[HorusDetector V4.5] Inicializado: {self.model_dir}")
    
    def _load_metadata(self):
        metadata_path = self.model_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                if 'inference' in metadata:
                    self.config.update(metadata['inference'])
        
        thresholds_path = self.model_dir / "thresholds.yaml"
        if thresholds_path.exists():
            try:
                import yaml
                with open(thresholds_path, 'r', encoding='utf-8') as f:
                    thresholds = yaml.safe_load(f)
                    self.class_thresholds = thresholds
            except ImportError:
                self.class_thresholds = {c: 0.5 for c in self.LABELS}
        else:
            self.class_thresholds = {c: 0.5 for c in self.LABELS}
    
    def _load_model(self):
        if self.use_tflite:
            tflite_path = self.model_dir / "model.tflite"
            if not tflite_path.exists():
                raise FileNotFoundError(f"TFLite não encontrado: {tflite_path}")
            
            import tensorflow as tf
            self.tflite_interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            self.tflite_interpreter.allocate_tensors()
            self.input_details = self.tflite_interpreter.get_input_details()
            self.output_details = self.tflite_interpreter.get_output_details()
        else:
            model_path = self.model_dir / "model.h5"
            if not model_path.exists():
                raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
            
            from tensorflow import keras
            self.model = keras.models.load_model(str(model_path))
    
    # =========================================================================
    # HEURÍSTICAS
    # =========================================================================
    
    def _check_freeze(self, frames: np.ndarray) -> Tuple[bool, float]:
        if len(frames) < 2: return False, 0.0
        diffs = []
        for i in range(1, len(frames)):
            diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
            diffs.append(diff)
        avg_diff = np.mean(diffs)
        is_freeze = avg_diff < self.config["freeze_diff_threshold"]
        conf = 1.0 if is_freeze else 0.0
        return is_freeze, conf
    
    def _check_fade(self, frames: np.ndarray) -> Tuple[bool, float]:
        if len(frames) < 1: return False, 0.0
        brightnesses = [np.mean(f) for f in frames]
        avg_brightness = np.mean(brightnesses)
        threshold = self.config["fade_brightness_threshold"]
        is_fade = avg_brightness < threshold
        conf = 1.0 if is_fade else 0.0
        return is_fade, conf
    
    def _check_blur(self, frames: np.ndarray) -> Tuple[bool, float]:
        import cv2
        if len(frames) < 1: return False, 0.0
        sharpness_values = []
        for frame in frames:
            if len(frame.shape) == 3: gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else: gray = frame
            sharpness_values.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        avg_sharpness = np.mean(sharpness_values)
        threshold = self.config["blur_sharpness_threshold"]
        is_blur = avg_sharpness < threshold
        conf = min(0.99, 1.0 - (avg_sharpness / threshold)) if is_blur else 0.0
        return is_blur, conf
    
    # =========================================================================
    # MODELO ML
    # =========================================================================
    
    def _preprocess(self, frames: np.ndarray) -> np.ndarray:
        import cv2
        img_size = tuple(self.config["img_size"]) # (192, 192)
        seq_len = self.config["sequence_length"]
        
        processed = []
        for frame in frames:
            if frame.shape[:2] != img_size[::-1]: # cv2 resize expects (W, H)
                frame = cv2.resize(frame, img_size)
            processed.append(frame)
        
        # Pad or truncate
        while len(processed) < seq_len:
            processed.append(processed[-1] if processed else np.zeros((*img_size[::-1], 3), dtype=np.uint8))
        processed = processed[:seq_len]
        
        # Normalize [0, 1] - MATCHING TRAINING SCRIPT
        arr = np.array(processed, dtype=np.float32)
        arr = arr / 255.0
        
        return arr
    
    def _predict_model(self, frames: np.ndarray) -> Dict[str, float]:
        batch = np.expand_dims(frames, 0)
        
        if self.use_tflite:
            self.tflite_interpreter.set_tensor(self.input_details[0]['index'], batch)
            self.tflite_interpreter.invoke()
            probs = self.tflite_interpreter.get_tensor(self.output_details[0]['index'])[0]
        else:
            probs = self.model.predict(batch, verbose=0)[0]
        
        return {label: float(probs[i]) for i, label in enumerate(self.LABELS)}
    
    # =========================================================================
    # DETECÇÃO
    # =========================================================================
    
    def detect(self, frames: Union[np.ndarray, List[np.ndarray]]) -> Dict:
        if isinstance(frames, list): frames = np.array(frames)
        
        # Ensure uint8 for heuristics
        frames_uint8 = frames.astype(np.uint8) if frames.dtype != np.uint8 else frames
        
        result = {
            "class": "normal",
            "confidence": 0.0,
            "method": "unknown",
            "probabilities": {c: 0.0 for c in self.LABELS},
            "details": {}
        }
        
        # 1. Run Model (Primary for V4.5)
        # V4.5 is trained to be precise, so we trust it more, but heuristics help with "obvious" cases.
        # Let's run model first to get probabilities.
        
        frames_pre = self._preprocess(frames_uint8)
        probs = self._predict_model(frames_pre)
        result["probabilities"] = probs
        
        # Get top class from model
        model_class = max(probs, key=probs.get)
        model_conf = probs[model_class]
        
        # 2. Run Heuristics (Validation)
        is_freeze, freeze_conf = self._check_freeze(frames_uint8)
        is_fade, fade_conf = self._check_fade(frames_uint8)
        is_blur, blur_conf = self._check_blur(frames_uint8)
        
        result["details"]["heuristics"] = {
            "freeze": float(freeze_conf),
            "fade": float(fade_conf),
            "blur": float(blur_conf)
        }
        
        # 3. Decision Logic (Ensemble)
        # If heuristic is VERY strong, override model if model is weak or disagrees?
        # Or just use model if confidence is high.
        
        final_class = model_class
        final_conf = model_conf
        method = "model"
        
        # Override logic:
        # If model says Normal but Heuristic says Error with high confidence -> Trust Heuristic?
        # If model says Error but Heuristic says Normal -> Trust Model (heuristics might miss subtle things).
        
        if model_class == "normal" and model_conf < 0.95:
            if is_freeze:
                final_class = "freeze"
                final_conf = freeze_conf
                method = "heuristic_override"
            elif is_fade:
                final_class = "fade"
                final_conf = fade_conf
                method = "heuristic_override"
            elif is_blur:
                final_class = "fora_de_foco"
                final_conf = blur_conf
                method = "heuristic_override"
                
        result["class"] = final_class
        result["confidence"] = final_conf
        result["method"] = method
        
        return result
