Pacote Horus para modelo de vídeo (video_model_v2)

Conteúdo incluído:
- `video_model_finetune.keras` : checkpoint Keras do modelo (classe por segmento).
- `video_model_finetune.metadata.json` : metadados (input shape, fps, segment_duration_s, labels).
- `thresholds.yaml` : limiares por classe (usado no pós-processamento de eventos).

Instruções rápidas de deploy:
1. Colocar toda a pasta `video_model_v2/` no diretório de modelos do Horus.
2. Carregar o modelo Keras com TensorFlow 2.x (CPU compatível).
3. Usar `metadata.json` para configurar pré-processamento:
   - normalização: (x - 127.5) / 127.5
   - resize: conforme `input_shape` no metadata
   - frames por segmento = segment_duration_s * frame_rate_sampled
4. Aplicar thresholds do `thresholds.yaml` para decidir rótulos por segmento e consolidar eventos.

Notas:
- Este pacote foi treinado/ajustado para priorizar recall de `fora_de_foco`.
- Recomenda-se validar no ambiente Horus com um pequeno conjunto de vídeos de teste antes de ativar produção.

Gerado em: 2025-11-16
