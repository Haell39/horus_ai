# SyncNet v2 - Lipsync Detection Model

Modelo de detecção de dessincronização áudio/vídeo (lipsync) para integração com o sistema Horus.

## 📁 Arquivos

| Arquivo | Tamanho | Descrição |
|---------|---------|-----------|
| `syncnet_v2.keras` | ~56 MB | Modelo Keras completo |
| `syncnet_v2.tflite` | ~56 MB | Modelo TFLite |
| `syncnet_v2_q.tflite` | ~14 MB | Modelo TFLite quantizado (menor) |
| `syncnet_inference.py` | - | Script de inferência |
| `model_config.json` | - | Configurações do modelo |

## 🚀 Uso Rápido

### Python
```python
from syncnet_inference import SyncNetInference, analyze_video

# Opção 1: Classe completa
model = SyncNetInference("syncnet_v2.keras")  # ou .tflite
result = model.predict("video.mp4")

print(result.status)       # SyncStatus.SINCRONIZADO ou DESSINCRONIZADO
print(result.confidence)   # 0.0 a 1.0
print(result.offset_ms)    # Offset em milissegundos

# Opção 2: Função simples
result = analyze_video("video.mp4")
print(result["status"])    # "sincronizado" ou "dessincronizado"
```

### CLI
```bash
python syncnet_inference.py video.mp4
python syncnet_inference.py video.mp4 syncnet_v2.tflite
```

## 📊 Classes de Saída

| Classe | Descrição |
|--------|-----------|
| `sincronizado` | Áudio e vídeo estão sincronizados (offset < 80ms) |
| `dessincronizado` | Áudio e vídeo estão dessincronizados (offset > 80ms) |
| `sem_fala` | Não foi possível detectar fala no vídeo |

## 📐 Especificações

### Input
- **Vídeo**: 5 frames RGB, 224x224, normalizados [0,1]
- **Áudio**: MFCC 13 coeficientes × 20 frames, sample rate 16kHz

### Output
- **classification**: Probabilidades [sync, desync, sem_fala]
- **offset_prediction**: Offset estimado em segundos

## ⚙️ Requisitos

```
tensorflow>=2.10.0
opencv-python>=4.5.0
librosa>=0.9.0
numpy>=1.20.0
```

## 📈 Performance

- **Acurácia Treino**: 100%
- **Acurácia Validação**: 100%
- **Acurácia Teste**: 100% (7/7 vídeos)

## 🔧 Integração com Horus

```python
# Exemplo de integração
from syncnet_inference import SyncNetInference

class HorusLipsyncAnalyzer:
    def __init__(self):
        self.model = SyncNetInference("syncnet_v2_q.tflite")
    
    def analyze(self, video_path):
        result = self.model.predict(video_path)
        return {
            "lipsync_ok": result.status.value == "sincronizado",
            "confidence": result.confidence,
            "error_type": "lipsync" if result.status.value == "dessincronizado" else None
        }
```

---
**Versão**: 2.0.0  
**Data**: 2025-12-02  
**Framework**: TensorFlow/Keras 2.10
