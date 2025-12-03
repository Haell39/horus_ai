# Horus AI — Documentação

Sistema Inteligente de Monitoramento e Detecção de Anomalias em Transmissões de Vídeo.

---

## 📚 Índice

| Documento                 | Descrição                   |
| ------------------------- | --------------------------- |
| [Ambiente (.env)](env.md) | Configuração de variáveis   |
| [API (endpoints)](api.md) | Referência REST e WebSocket |

---

## 🔧 Arquitetura

```
┌─────────────────┐     ┌─────────────────┐
│    Frontend     │────▶│     Backend     │
│   Angular 19    │◀────│    FastAPI      │
└─────────────────┘     └────────┬────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
┌───────────────┐    ┌───────────────────┐    ┌───────────────┐
│  PostgreSQL   │    │   ML Inference    │    │    FFmpeg     │
│   Database    │    │ Video/Audio/Sync  │    │   SRT→HLS     │
└───────────────┘    └───────────────────┘    └───────────────┘
```

---

## 📊 Modelos de IA

| Modelo                | Formato                | Uso                |
| --------------------- | ---------------------- | ------------------ |
| **Odin v4.5**         | `.keras`               | Anomalias de vídeo |
| **Heimdall Ultra v1** | `.keras`               | Anomalias de áudio |
| **SyncNet v2**        | `.tflite` (quantizado) | Lipsync            |

---

## 🎯 Anomalias Detectadas

**Vídeo:** freeze, fade, fora_de_foco  
**Áudio:** ausencia_audio, eco_reverb, ruido_hiss, sinal_teste  
**Lipsync:** dessincronização
