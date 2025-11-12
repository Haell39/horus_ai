#!/usr/bin/env python3
"""Treino MobileNetV2 - Visão

Uso: aceita argumentos via argparse. Faz dry-run quando --dry_run True.
Gera: model.h5, history JSON, history PNG, summary log em outputs/results/
"""
import argparse
import json
import os
import time
import traceback

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(image_size, num_classes, freeze_backbone, augmentation=None):
    base = keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3), include_top=False, weights='imagenet')
    base.trainable = not freeze_backbone
    inputs = keras.Input(shape=(image_size, image_size, 3))
    x = inputs
    if augmentation is not None:
        x = augmentation(x)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def get_augmentation():
    return keras.Sequential([
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.06),
        layers.RandomZoom(0.05),
    ], name='augmentation')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--image_size', type=int, default=160)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--freeze_backbone', type=lambda s: s.lower() in ['true','1','yes'], default=True)
    parser.add_argument('--dry_run', type=lambda s: s.lower() in ['true','1','yes'], default=False)
    parser.add_argument('--output', required=True, help='caminho para salvar model.h5')
    args = parser.parse_args()

    os.makedirs('outputs/results', exist_ok=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    summary_path = 'outputs/results/training_summary_phase2_vision.txt'
    history_json = 'outputs/results/training_history_vision.json'
    history_png = 'outputs/results/training_history_vision.png'

    start = time.time()
    try:
        tf_version = tf.__version__
        gpus = tf.config.list_physical_devices('GPU')

        # Datasets
        img_size = (args.image_size, args.image_size)
        train_ds = tf.keras.utils.image_dataset_from_directory(
            args.data_dir, image_size=img_size, batch_size=args.batch_size, shuffle=True)

        class_names = train_ds.class_names
        num_classes = len(class_names)

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.prefetch(AUTOTUNE)

        # create augmentation once and integrate into the model to avoid variable creation inside tf.data.map
        augmentation_layer = get_augmentation()

        if args.dry_run:
            epochs = 1
        else:
            epochs = args.epochs

        model = build_model(args.image_size, num_classes, args.freeze_backbone, augmentation=augmentation_layer)

        callbacks = []
        ckpt_dir = os.path.join(os.path.dirname(args.output), 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, 'ckpt_{epoch}.weights.h5'), save_weights_only=True))
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True))

        history = model.fit(train_ds, epochs=epochs, callbacks=callbacks)

        # Save outputs
        model.save(args.output)
        with open(history_json, 'w', encoding='utf-8') as f:
            json.dump(history.history, f, indent=2)

        # plot history
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(history.history.get('loss', []), label='loss')
            plt.plot(history.history.get('accuracy', []), label='accuracy')
            plt.legend()
            plt.xlabel('epoch')
            plt.savefig(history_png)
            plt.close()
        except Exception:
            pass

        duration = time.time() - start
        summary = {
            'tf_version': tf_version,
            'gpus': [str(x) for x in gpus],
            'duration_seconds': duration,
            'model_path': args.output,
            'history_json': history_json,
            'history_png': history_png
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(summary, indent=2))

        print('Treino (vision) finalizado com sucesso. Resumo salvo em', summary_path)

    except Exception as e:
        duration = time.time() - start
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('ERROR:\n')
            f.write(traceback.format_exc())
        print('Erro durante o dry-run vision. Ver', summary_path)


if __name__ == '__main__':
    main()
