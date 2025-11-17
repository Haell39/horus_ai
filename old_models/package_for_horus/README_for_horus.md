Pacote para Horus_AI — instruções e arquivos

Arquivos incluídos (caminhos relativos dentro deste pacote):

- audio/

  - model_retrain_dryrun.h5 # Keras HDF5 (legacy)
  - model_retrain_dryrun.keras # Keras native format (preferível para Keras 3)
  - checkpoints/ckpt_1.weights.h5 # Pesos de checkpoint
  - metadata.json # Metadados do modelo (input_size, class_names, preprocess)
  - thresholds.yaml # Thresholds sugeridos (se houver)
  - training_summary_phase2_audio.txt
  - training_history_audio.json

- vision/
  - model_retrain_dryrun.h5
  - model_retrain_dryrun.keras
  - checkpoints/ckpt_1.weights.h5
  - metadata.json
  - thresholds.yaml
  - training_summary_phase2_vision.txt
  - training_history_vision.json

Observações importantes para Horus_AI (copiar e colar para o outro agente)

1. Formatos de modelo:

   - Preferido: arquivo .keras (formato nativo Keras 3). Use: tf.keras.models.load_model('model.retrain.keras')
   - Alternativa compatível: arquivo .h5 (legacy). Use: tf.keras.models.load_model('model.h5')
   - Nota: o formato SavedModel (pasta) pode ter problemas de recarga via load_model() em Keras 3; se necessário, use keras.layers.TFSMLayer para carregar um SavedModel.

2. Preprocessamento:

   - Os modelos usam preprocessamento MobileNetV2-style: valores escalados para [-1, 1]. No código usamos:
     layers.Rescaling(1/127.5, offset=-1.0)
   - Entrada: imagens RGB 160x160 (channels=3). As espectrogramas devem ser 224x224 originalmente para geração, mas o modelo foi treinado com image_size=160 para o dry-run. Verifique `input_size` no metadata.json.

3. Ordem de classes:

   - A ordem canônica das classes (classe -> índice) foi lida de `data/labels.csv` durante o treino. Está refletida em `metadata.json` incluído.
   - Garanta usar a mesma ordem de classes no pipeline de inferência para interpretar as previsões corretamente.

4. Verificação rápida (exemplo de checks que o Horus agent deve executar):

   - Carregar o modelo (.keras ou .h5) e confirmar que possui as camadas `global_average_pooling2d` e `logits`.
   - Fazer um `model.predict()` em uma batch aleatória de entrada com shape (1, input_size, input_size, 3) para garantir que a saída tem shape (1, num_classes).

5. Arquivos que eu recomendo sempre enviar junto com o modelo:
   - `metadata.json` (obrigatório): descreve input_size, preprocess e class_names.
   - `thresholds.yaml` (opcional): limiares operacionais para classes.
   - `training_history_*.json` e `training_summary_*.txt`: ajudam a auditoria e compreensão do treinamento.

