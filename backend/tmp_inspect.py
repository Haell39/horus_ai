import sys
sys.path.append(r"d:\GitHub Desktop\horus_ai\backend")
from app.ml import inference
import numpy as np
m = inference.keras_video_model
if not m:
    print('NO_MODEL')
    sys.exit(0)
shape = getattr(m, 'input_shape', (None,160,160,3))
batch_shape = (1, int(shape[1]) or 160, int(shape[2]) or 160, int(shape[3]) or 3)
x = np.random.uniform(-1.0,1.0,size=batch_shape).astype('float32')
from tensorflow.keras.models import Model
lay = m.get_layer('global_average_pooling2d')
interp = Model(inputs=m.input, outputs=lay.output)
inter = interp.predict(x)
arr = inter[0]
print('rand_input_stats:', float(x.min()), float(x.max()), float(x.mean()), float(x.std()))
print('inter_stats:', arr.shape, float(arr.min()), float(arr.max()), float(arr.mean()), float(arr.std()))
# Also run full model
preds = m.predict(x)
print('preds_stats:', preds.shape, preds[0][:8])

# Try several preprocessing variants on the same random uint8 image to see effect
img = ((np.random.uniform(0, 255, size=(INPUT_HEIGHT:=160, INPUT_WIDTH:=160,3))).astype(np.float32))
variants = {
    'mobile_net_preproc': (img - 127.5) / 127.5,
    'zero_one': img / 255.0,
    'raw_255': img,
}
from tensorflow.keras.models import Model
lay = m.get_layer('global_average_pooling2d')
interp = Model(inputs=m.input, outputs=lay.output)
for name, v in variants.items():
    xin = np.expand_dims(v.astype('float32'), axis=0)
    inter = interp.predict(xin)
    arr = inter[0]
    preds = m.predict(xin)
    print('VARIANT', name, 'inter_std', float(np.std(arr)), 'preds', np.round(preds[0],4))
