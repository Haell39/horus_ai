#!/usr/bin/env python3
"""
SyncNet v2 - Inferência para integração com Horus
Detecta dessincronização de áudio/vídeo (lipsync)
"""

import numpy as np
import cv2
import librosa
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union
import os

class SyncStatus(Enum):
    SINCRONIZADO = "sincronizado"
    DESSINCRONIZADO = "dessincronizado"
    SEM_FALA = "sem_fala"

@dataclass
class SyncResult:
    status: SyncStatus
    confidence: float
    offset_ms: float
    probabilities: dict

class SyncNetInference:
    """
    Classe de inferência SyncNet v2
    
    Suporta modelos .keras e .tflite
    
    Uso:
        model = SyncNetInference("syncnet_v2.keras")  # ou .tflite
        result = model.predict("video.mp4")
        print(result.status)  # SyncStatus.SINCRONIZADO ou DESSINCRONIZADO
    """
    
    def __init__(self, model_path: str):
        """
        Inicializa o modelo
        
        Args:
            model_path: Caminho para arquivo .keras ou .tflite
        """
        self.model_path = model_path
        
        if model_path.endswith('.tflite'):
            self._load_tflite(model_path)
        else:
            self._load_keras(model_path)
    
    def _load_keras(self, path: str):
        """Carrega modelo Keras"""
        import tensorflow as tf
        self.model = tf.keras.models.load_model(path)
        self.use_tflite = False
        print(f"✅ Modelo Keras carregado: {path}")
    
    def _load_tflite(self, path: str):
        """Carrega modelo TFLite"""
        try:
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=path)
        except:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=path)
        
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.use_tflite = True
        print(f"✅ Modelo TFLite carregado: {path}")
    
    def _extract_video_features(self, video_path: str) -> Optional[np.ndarray]:
        """Extrai 5 frames do vídeo"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total < 5:
            cap.release()
            return None
        
        # 5 frames uniformemente distribuídos
        for idx in np.linspace(0, total-1, 5, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame / 255.0)
        
        cap.release()
        
        if len(frames) < 5:
            return None
        
        return np.array(frames[:5], dtype=np.float32)
    
    def _extract_audio_features(self, video_path: str) -> Optional[np.ndarray]:
        """Extrai MFCC do áudio"""
        try:
            y, sr = librosa.load(video_path, sr=16000, mono=True, duration=2.0)
            
            if len(y) < 8000:  # Menos de 0.5s
                return None
            
            # MFCC: 13 coeficientes, 20 time steps
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=256)
            
            # Padding/truncate para 20 frames
            if mfcc.shape[1] < 20:
                mfcc = np.pad(mfcc, ((0,0), (0, 20-mfcc.shape[1])), mode='edge')
            else:
                mfcc = mfcc[:, :20]
            
            # Normalização
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
            
            return mfcc[:,:,np.newaxis].astype(np.float32)
        except Exception as e:
            print(f"Erro extraindo áudio: {e}")
            return None
    
    def predict(self, video_path: str) -> SyncResult:
        """
        Analisa um vídeo e retorna resultado de sincronização
        
        Args:
            video_path: Caminho do arquivo de vídeo
            
        Returns:
            SyncResult com status, confiança e offset estimado
        """
        # Extrai features
        video_feat = self._extract_video_features(video_path)
        audio_feat = self._extract_audio_features(video_path)
        
        if video_feat is None or audio_feat is None:
            return SyncResult(
                status=SyncStatus.SEM_FALA,
                confidence=0.0,
                offset_ms=0.0,
                probabilities={"sincronizado": 0, "dessincronizado": 0, "sem_fala": 1}
            )
        
        # Batch de 1
        audio_batch = audio_feat[np.newaxis, ...]
        video_batch = video_feat[np.newaxis, ...]
        
        # Inferência
        if self.use_tflite:
            probs, offset = self._predict_tflite(audio_batch, video_batch)
        else:
            probs, offset = self._predict_keras(audio_batch, video_batch)
        
        # Decisão
        class_idx = np.argmax(probs)
        confidence = float(probs[class_idx])
        
        status_map = {0: SyncStatus.SINCRONIZADO, 1: SyncStatus.DESSINCRONIZADO, 2: SyncStatus.SEM_FALA}
        status = status_map.get(class_idx, SyncStatus.SEM_FALA)
        
        return SyncResult(
            status=status,
            confidence=confidence,
            offset_ms=float(offset * 1000),
            probabilities={
                "sincronizado": float(probs[0]),
                "dessincronizado": float(probs[1]),
                "sem_fala": float(probs[2])
            }
        )
    
    def _predict_keras(self, audio: np.ndarray, video: np.ndarray):
        """Inferência Keras"""
        outputs = self.model.predict([audio, video], verbose=0)
        probs = outputs['classification'][0]
        offset = outputs['offset_prediction'][0][0]
        return probs, offset
    
    def _predict_tflite(self, audio: np.ndarray, video: np.ndarray):
        """Inferência TFLite"""
        # Encontra índices corretos
        audio_idx = None
        video_idx = None
        
        for detail in self.input_details:
            shape = detail['shape']
            if len(shape) == 4 and shape[1] == 13:  # Audio: (1, 13, 20, 1)
                audio_idx = detail['index']
            elif len(shape) == 5:  # Video: (1, 5, 224, 224, 3)
                video_idx = detail['index']
        
        if audio_idx is None or video_idx is None:
            raise ValueError("Não foi possível identificar inputs do modelo")
        
        self.interpreter.set_tensor(audio_idx, audio)
        self.interpreter.set_tensor(video_idx, video)
        self.interpreter.invoke()
        
        # Encontra outputs
        probs = None
        offset = 0.0
        
        for detail in self.output_details:
            tensor = self.interpreter.get_tensor(detail['index'])
            if 'classification' in detail['name']:
                probs = tensor[0]
            elif 'offset' in detail['name']:
                offset = tensor[0][0]
        
        if probs is None:
            # Fallback: primeiro output com 3 valores
            for detail in self.output_details:
                tensor = self.interpreter.get_tensor(detail['index'])
                if tensor.shape[-1] == 3:
                    probs = tensor[0]
                    break
        
        return probs if probs is not None else np.array([0.33, 0.33, 0.34]), offset


# Função utilitária para uso direto
def analyze_video(video_path: str, model_path: str = None) -> dict:
    """
    Função simples para analisar um vídeo
    
    Args:
        video_path: Caminho do vídeo
        model_path: Caminho do modelo (opcional, usa padrão se não fornecido)
        
    Returns:
        Dict com resultado da análise
    """
    if model_path is None:
        # Procura modelo no mesmo diretório
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for ext in ['.tflite', '.keras']:
            for name in ['syncnet_v2', 'syncnet_v2_q']:
                path = os.path.join(script_dir, f"{name}{ext}")
                if os.path.exists(path):
                    model_path = path
                    break
            if model_path:
                break
    
    if model_path is None:
        raise FileNotFoundError("Modelo não encontrado. Especifique o caminho.")
    
    model = SyncNetInference(model_path)
    result = model.predict(video_path)
    
    return {
        "status": result.status.value,
        "confidence": result.confidence,
        "offset_ms": result.offset_ms,
        "probabilities": result.probabilities,
        "is_synced": result.status == SyncStatus.SINCRONIZADO,
        "is_desynced": result.status == SyncStatus.DESSINCRONIZADO
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python syncnet_inference.py <video.mp4> [modelo.keras|.tflite]")
        sys.exit(1)
    
    video = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = analyze_video(video, model)
    
    print(f"\n{'='*50}")
    print(f"  Resultado: {result['status'].upper()}")
    print(f"  Confiança: {result['confidence']:.1%}")
    print(f"  Offset: {result['offset_ms']:.1f}ms")
    print(f"{'='*50}")
