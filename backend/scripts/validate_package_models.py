import sys
import os
sys.path.append(r"d:\GitHub Desktop\horus_ai\backend")
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

BASE = r"d:\GitHub Desktop\horus_ai\package_for_horus"
MODELS = {
    'audio': os.path.join(BASE, 'audio', 'model_retrain_dryrun.keras'),
    'vision': os.path.join(BASE, 'vision', 'model_retrain_dryrun.keras')
}
METAS = {
    'audio': os.path.join(BASE, 'audio', 'metadata.json'),
    'vision': os.path.join(BASE, 'vision', 'metadata.json')
}

print('Validate package_for_horus models')
for key in ['audio','vision']:
    print('\n----', key, '----')
    model_path = MODELS[key]
    meta_path = METAS[key]
    print('model_path=', model_path)
    print('meta_path=', meta_path)
    classes = None
    if os.path.exists(meta_path):
        try:
            with open(meta_path,'r',encoding='utf-8') as fh:
                meta = json.load(fh)
                classes = meta.get('class_names') or meta.get('classes') or meta.get('labels')
                print('metadata class names count=', 0 if classes is None else len(classes))
        except Exception as e:
            print('failed reading metadata.json:', e)
    else:
        print('metadata file not found')

    if not os.path.exists(model_path):
        print('model file not found, skipping')
        continue

    try:
        m = load_model(model_path)
        print('loaded model OK')
    except Exception as e:
        print('ERROR loading model:', e)
        continue

    # basic info
    try:
        print('model input_shape=', getattr(m, 'input_shape', None))
        print('model output_shape=', getattr(m, 'output_shape', None))
    except Exception:
        pass

    # list layers of interest
    layer_names = [l.name for l in m.layers]
    print('num layers=', len(layer_names))
    # check for Rescaling
    has_rescaling = any('rescal' in l.lower() for l in layer_names)
    print('has_rescaling_layer_by_name?', has_rescaling)
    # check for GlobalAveragePooling
    gap_layer = None
    for l in m.layers:
        if l.__class__.__name__.lower().find('globalaverage') != -1:
            gap_layer = l
            break
    print('gap_layer=', getattr(gap_layer, 'name', None))

    # find final Dense layer
    dense_layers = [l for l in m.layers if l.__class__.__name__.lower()=='dense']
    print('dense_layers count=', len(dense_layers))
    for dl in dense_layers[-3:]:
        try:
            w = dl.get_weights()
            print(' dense', dl.name, 'units(?)', getattr(dl, 'units', None), 'weights shapes', [x.shape for x in w])
        except Exception as e:
            print(' error inspecting dense', dl.name, e)

    # quick predict on random input
    shape = getattr(m, 'input_shape', None)
    if shape and len(shape) >= 4:
        bs = (1, int(shape[1]) or 160, int(shape[2]) or 160, int(shape[3]) or 3)
    else:
        bs = (1,160,160,3)
    x = np.random.uniform(0,255,size=bs).astype('float32')
    # apply recommended preprocessing before passing? The model may include Rescaling.
    try:
        preds = m.predict(x)
        preds_arr = np.array(preds[0], dtype=float)
        print('preds shape', preds.shape, 'first8', preds_arr[:min(8,len(preds_arr))])
        # softmax
        probs = tf.nn.softmax(preds_arr).numpy()
        print('probs first8', probs[:min(8,len(probs))])
    except Exception as e:
        print('error running predict on raw 0-255 input:', e)
        # try preprocessing (mobile net style)
        try:
            x2 = (x - 127.5) / 127.5
            preds = m.predict(x2)
            preds_arr = np.array(preds[0], dtype=float)
            print('preds (with mobile preprocess) first8', preds_arr[:min(8,len(preds_arr))])
            print('probs', tf.nn.softmax(preds_arr).numpy()[:min(8,len(preds_arr))])
        except Exception as e2:
            print('failed with preprocess too:', e2)

    # inspect GAP stats if present
    if gap_layer is not None:
        try:
            from tensorflow.keras.models import Model
            inter_model = Model(inputs=m.input, outputs=gap_layer.output)
            inter = inter_model.predict(x)
            arr = np.array(inter[0], dtype=float)
            print('GAP stats shape', arr.shape, 'min', arr.min(), 'max', arr.max(), 'mean', arr.mean(), 'std', arr.std())
        except Exception as e:
            print('failed to compute GAP on raw input:', e)
            try:
                x2 = (x - 127.5) / 127.5
                inter = inter_model.predict(x2)
                arr = np.array(inter[0], dtype=float)
                print('GAP stats (with preprocess) min', arr.min(), 'std', arr.std())
            except Exception as e2:
                print('failed GAP with preprocess too:', e2)

print('\nDone')
