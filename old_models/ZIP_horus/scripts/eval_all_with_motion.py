#!/usr/bin/env python3
"""Avalia o modelo que recebe image + motion + brightness para as 3 classes: freeze, fade, fora_foco.
Percorre `data/processed/vision/{freeze,fade,fora_foco}` recursivamente e salva relatório JSON em outputs/results/val_report_all_with_motion.json
"""
from pathlib import Path
import numpy as np
import json
from PIL import Image
import tensorflow as tf
from tensorflow import keras

MODEL_PATH = Path('models/newmodel/video_model_finetune_with_motion.keras')
ROOT = Path('data/processed/vision')
CLASSES = ['freeze','fade','fora_foco']
IMG_SIZE = (160,160)
OUT = Path('outputs/results/val_report_all_with_motion.json')

print('Loading model', MODEL_PATH)
model = keras.models.load_model(str(MODEL_PATH))
print('Model loaded')

report = {'counts':{}, 'correct':{}, 'per_sample': []}
for cls_idx, cls in enumerate(CLASSES):
    folder = ROOT / cls
    report['counts'][cls] = 0
    report['correct'][cls] = 0
    if not folder.exists():
        print('Folder not found, skipping', folder)
        continue
    files = [p for p in folder.rglob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png')]
    if not files:
        print('No images for class', cls)
        continue
    # treat each file in sorted order; compute motion relative to previous file in same parent
    groups = {}
    for p in files:
        rel = p.relative_to(folder)
        parts = rel.parts
        group = parts[0] if len(parts) > 1 else '.'
        groups.setdefault(group, []).append(p)
    for gname, plist in groups.items():
        plist_sorted = sorted(plist)
        prev_arr = None
        for p in plist_sorted:
            try:
                img = Image.open(p).convert('RGB').resize(IMG_SIZE)
                arr = np.asarray(img).astype('float32')
                bright = float(arr.mean()/255.0)
                if prev_arr is None:
                    motion = 1.0
                else:
                    motion = float(np.mean(np.abs(arr/255.0 - prev_arr/255.0)))
                prev_arr = arr
                img_batch = np.expand_dims(arr, axis=0)
                motion_batch = np.array([[motion]], dtype='float32')
                bright_batch = np.array([[bright]], dtype='float32')
                preds = model.predict({'image': img_batch, 'motion': motion_batch, 'brightness': bright_batch}, verbose=0)
                pred_label = int(np.argmax(preds, axis=-1)[0])
                report['per_sample'].append({'path': str(p), 'true': cls, 'pred_idx': pred_label, 'pred': CLASSES[pred_label], 'motion': motion, 'brightness': bright})
                report['counts'][cls] += 1
                if pred_label == cls_idx:
                    report['correct'][cls] += 1
            except Exception as e:
                print('Failed processing', p, e)

for c in CLASSES:
    n = report['counts'].get(c, 0)
    correct = report['correct'].get(c, 0)
    acc = None
    if n>0:
        acc = correct / n
    report[c] = {'n': n, 'accuracy': acc}
report['overall'] = None
all_n = sum(report['counts'].values())
if all_n>0:
    total_correct = sum(report['correct'].values())
    report['overall'] = total_correct / all_n

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as fh:
    json.dump(report, fh, indent=2)

print('Saved report to', OUT)
print('Summary:')
for c in CLASSES:
    print(c, report[c])
print('Overall:', report['overall'])
