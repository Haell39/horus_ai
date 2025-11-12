#!/usr/bin/env python3
"""Inferência simples para `models/newmodel/video_model_finetune.keras`.
Uso:
  python scripts/infer_video_model.py --model models/newmodel/video_model_finetune.keras --image path/to/image.jpg
Retorna logits (3 valores na ordem freeze,fade,fora_foco) e probabilidades softmax.
"""
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='models/newmodel/video_model_finetune.keras')
parser.add_argument('--image', required=True)
parser.add_argument('--size', type=int, default=160)
args = parser.parse_args()

m = tf.keras.models.load_model(args.model)
img = Image.open(args.image).convert('RGB').resize((args.size, args.size))
arr = np.asarray(img).astype('float32')
# preprocess (x - 127.5) / 127.5
arr = (arr - 127.5) / 127.5
arr = np.expand_dims(arr, axis=0)
logits = m.predict(arr)
probs = tf.nn.softmax(logits, axis=-1).numpy()
print('logits:', logits)
print('probs:', probs)
