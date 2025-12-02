Treine um classificador de áudio para o sistema Horus com foco em 6 classes:

ausencia_audio, volume_baixo, eco_reverb, ruido_hiss, sinal_teste (1 kHz), normal.
Regras e restrições:

Use somente os dados que eu fornecer (vídeos originais + versões augmentadas de eco/reverb). NÃO baixar datasets externos.
Trabalhe bem com poucos dados: priorize transfer learning, forte augmentação e validação robusta por cross‑validation.
Preprocessamento e formato:

Taxa de amostragem alvo: 16000 Hz.
Duração de segmento recomendada: 5s com 50% de overlap (evita perda de contexto de eco); se já tiver modelos/artefatos em 3s, documente diferenças.
Gere espectrogramas/mel-spectrogramas seguindo o preprocess existente (veja preprocess.py), e mantenha o mesmo input_shape.
Exportar modelo final em Keras (arquivo .keras ou .h5) como audio_model_finetune.keras.
Gerar um arquivo de metadados audio_model_finetune.metadata.json contendo: sample_rate, segment_duration_s, input_shape, labels (ordem), e qualquer pré‑processo aplicado.
Produzir também training_files/labels.csv.
Augmentação (essencial com poucos dados):

RIR convolution: várias RIRs sintéticas reais (RT60 variados) para simular reverb/eco (varie decay/delay).
Use os seus vídeos de eco/reverb como fontes de augmentação — NÃO baixar IRs de terceiros sem permissão.
Ajustes de nível (volume scaling) para diferenciar volume baixo de mudo (não transformar mudo em volume_baixo).
Ruído aditivo: hiss/static com SNR variados.
Pequeno clipping/distortion e compressão (para robustez), mas não exagerar para não confundir classes.
Time-shift e pequenos cortes (não quebrar o sinal de 1 kHz para sinal_teste).
Treino (configurações recomendadas para poucos dados):

Estratégia: transfer learning sobre backbone leve (ex.: MobileNet-like) + head denso; congele o backbone inicialmente, treine head, então fine-tune com lr reduzido.
LR inicial: 1e-4 para head; fine-tune: 1e-5.
Epochs: 20–40 com EarlyStopping (patience 5) e checkpoint para melhor val_loss.
Batch size: 8–32 (ajuste por memória).
Balanceamento: usar oversampling ou class weights para classes raras (ex.: ausencia_audio, sinal_teste).
Cross‑validation K=4 (quando possível) e manter um holdout set com pelo menos alguns arquivos originais não augmentados.
Métricas e validação:

Reportar por-classe: precision, recall, F1, e matriz de confusão — priorizar reduzir falsos-positivos em volume_baixo e eco_reverb.
Calibrar thresholds para produção (retornar softmax + thresholds configuráveis AUDIO_THRESH_<CLASSE>).
Salvar checkpoints e o melhor modelo final (.keras) + metadata.
Integração com Horus (outputs esperados):

Modelo salvo em audio_model_finetune.keras.
Metadata em audio_model_finetune.metadata.json.
labels.csv em labels.csv.
Para cada segmento de inferência, a API deve poder receber: timestamp_start, timestamp_end, predicted_class, confidence.
Produzir diagnóstico por segmento compatível com Horus: inserir no JSON diagnóstico a chave "audio": {"class": "<label>", "confidence": 0.XXX, "start_s": X.X, "end_s": Y.Y}.
Testes pós-treino (verificação mínima):

Rodar validate_audio_model_for_horus.py com um tom de 1 kHz — deve prever sinal_teste com alta confiança.
Rodar run_inference_test.py sobre arquivos de validação e inspecionar segmentos (3s/5s conforme escolhido).
Gerar relatório com métricas, confusion matrix e exemplos de segmentos mal classificadas (salvar WAVs para inspeção manual).
Entrega final:

Código de treino (script), checkpoints, audio_model_finetune.keras, audio_model_finetune.metadata.json, labels.csv, e um README curto com comandos para validar e exportar para Horus.
Instruções para ajustar thresholds (AUDIO_THRESH_*) no .env do Horus.