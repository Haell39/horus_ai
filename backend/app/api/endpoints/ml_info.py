from fastapi import APIRouter
from typing import Dict, Any
from app.ml import inference

router = APIRouter()

@router.get('/models/info', summary='Return loaded model shapes and class mapping')
async def models_info() -> Dict[str, Any]:
    info = {
        'models_loaded': bool(inference.models_loaded),
        'model_classes': inference.MODEL_CLASSES,
    }
    try:
        audio = inference.keras_audio_model
        video = inference.keras_video_model
        if audio is not None:
            info['audio'] = {
                'input_shape': getattr(audio, 'input_shape', None),
                'output_shape': getattr(audio, 'output_shape', None),
            }
        else:
            info['audio'] = None
        if video is not None:
            info['video'] = {
                'input_shape': getattr(video, 'input_shape', None),
                'output_shape': getattr(video, 'output_shape', None),
            }
        else:
            info['video'] = None
    except Exception as e:
        info['error'] = str(e)
    return info


@router.get('/models/sanity_random', summary='Sanity check: run model.predict on random input')
async def models_sanity_random() -> Dict[str, Any]:
    """Run a quick random-input predict on loaded models and return raw scores + softmax.
    This helps detect whether models produce non-informative uniform outputs.
    """
    out: Dict[str, Any] = {'models_loaded': bool(inference.models_loaded)}
    try:
        import numpy as np
        # audio
        audio_model = inference.keras_audio_model
        video_model = inference.keras_video_model
        if audio_model is not None:
            try:
                in_shape = getattr(audio_model, 'input_shape', None)
                # build a random tensor matching the model input (batch dim = 1)
                if in_shape and len(in_shape) >= 4:
                    shape = (1, int(in_shape[1]) or 160, int(in_shape[2]) or 160, int(in_shape[3]) or 3)
                else:
                    shape = (1, 160, 160, 3)
                x = np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)
                preds = audio_model.predict(x)
                scores = np.array(preds[0], dtype=float).tolist()
                probs = inference._apply_softmax(np.array(scores)).tolist()
                out['audio_sanity'] = {'input_shape': shape, 'raw': scores, 'probs': probs}
            except Exception as e:
                out['audio_sanity_error'] = str(e)
        else:
            out['audio'] = None

        if video_model is not None:
            try:
                in_shape = getattr(video_model, 'input_shape', None)
                if in_shape and len(in_shape) >= 4:
                    shape = (1, int(in_shape[1]) or 160, int(in_shape[2]) or 160, int(in_shape[3]) or 3)
                else:
                    shape = (1, 160, 160, 3)
                x = np.random.uniform(-1.0, 1.0, size=shape).astype(np.float32)
                preds = video_model.predict(x)
                scores = np.array(preds[0], dtype=float).tolist()
                probs = inference._apply_softmax(np.array(scores)).tolist()
                out['video_sanity'] = {'input_shape': shape, 'raw': scores, 'probs': probs}
            except Exception as e:
                out['video_sanity_error'] = str(e)
        else:
            out['video'] = None

    except Exception as e:
        out['error'] = str(e)
    return out


@router.get('/models/last_layers', summary='Inspect last N layers of models')
async def models_last_layers(count: int = 3) -> Dict[str, Any]:
    """Return weight stats for the last `count` layers of each loaded model."""
    import numpy as np
    out: Dict[str, Any] = {'models_loaded': bool(inference.models_loaded)}
    try:
        for name, model in [('audio', inference.keras_audio_model), ('video', inference.keras_video_model)]:
            if model is None:
                out[name] = None
                continue
            layers = model.layers[-count:]
            info = []
            for layer in layers:
                try:
                    w = layer.get_weights()
                except Exception:
                    w = []
                weights_summary = []
                for arr in w:
                    a = np.array(arr, dtype=np.float32)
                    stats = {
                        'shape': list(a.shape),
                        'min': float(np.min(a)) if a.size else None,
                        'max': float(np.max(a)) if a.size else None,
                        'mean': float(np.mean(a)) if a.size else None,
                        'std': float(np.std(a)) if a.size else None,
                        'all_zero': bool(np.all(a == 0)) if a.size else False,
                        'any_nan': bool(np.isnan(a).any()) if a.size else False,
                    }
                    weights_summary.append(stats)
                info.append({'layer_name': layer.name, 'class_name': layer.__class__.__name__, 'weights': weights_summary})
            out[name] = info
    except Exception as e:
        out['error'] = str(e)
    return out


@router.get('/models/intermediate', summary='Return intermediate layer output for a clip')
async def models_intermediate(path: str, layer_name: str = None) -> Dict[str, Any]:
    """Extract a frame from a clip and return the output of an intermediate layer.
    If layer_name is omitted, returns the global average pooling layer output.
    """
    import os
    import numpy as np
    import cv2
    out: Dict[str, Any] = {'requested_path': path, 'layer': layer_name}
    try:
        # Resolve path
        if os.path.isabs(path):
            clip_path = path
        else:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            clip_path = os.path.abspath(os.path.join(repo_root, path))

        if not os.path.exists(clip_path):
            return {'error': f'Arquivo não encontrado: {clip_path}'}

        model = inference.keras_video_model
        if model is None:
            return {'error': 'Modelo de vídeo não carregado.'}

        # determine default layer (global average pooling)
        if not layer_name:
            # try common names
            for candidate in ['global_average_pooling2d_1', 'global_average_pooling2d', 'global_average_pooling2d']:
                try:
                    model.get_layer(candidate)
                    layer_name = candidate
                    break
                except Exception:
                    layer_name = None

        if not layer_name:
            return {'error': 'layer_name não informado e não foi possível determinar camada padrão.'}

        try:
            target_layer = model.get_layer(layer_name)
        except Exception:
            return {'error': f'Camada não encontrada no modelo: {layer_name}'}

        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            return {'error': f'Falha ao abrir vídeo: {clip_path}'}
        cap.set(cv2.CAP_PROP_POS_MSEC, 500)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return {'error': 'Não foi possível extrair frame do vídeo.'}

        input_data = inference.preprocess_video_single_frame(frame)
        if input_data is None:
            return {'error': 'Falha no pré-processamento do frame.'}

        # build a model to fetch intermediate output
        try:
            from tensorflow.keras.models import Model
            intermediate_model = Model(inputs=model.input, outputs=target_layer.output)
            inter = intermediate_model.predict(input_data)
            arr = np.array(inter[0], dtype=float)
            stats = {'min': float(np.min(arr)), 'max': float(np.max(arr)), 'mean': float(np.mean(arr)), 'std': float(np.std(arr)), 'shape': list(arr.shape)}
            # include a small slice of values for inspection
            vals = arr.flatten().tolist()[:64]
            out.update({'clip_path': clip_path, 'layer': layer_name, 'stats': stats, 'values_preview': vals})
            return out
        except Exception as e:
            return {'error': f'Falha ao extrair saída intermediária: {e}'}
    except Exception as e:
        return {'error': str(e)}


@router.get('/models/predict_clip', summary='Run model.predict on a single frame from a local clip')
async def models_predict_clip(path: str) -> Dict[str, Any]:
    """Extract a single frame from a server-local clip and run the video model on it.
    Query param `path` may be an absolute path or relative to the repository root (e.g. 'backend/test_out.mp4').
    Returns raw scores, softmax probabilities, top-k and basic input tensor stats.
    """
    import os
    import numpy as np
    import cv2
    out: Dict[str, Any] = {'requested_path': path}
    try:
        # Resolve path: accept absolute or path relative to repo root
        if os.path.isabs(path):
            clip_path = path
        else:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            clip_path = os.path.abspath(os.path.join(repo_root, path))

        if not os.path.exists(clip_path):
            return {'error': f'Arquivo não encontrado: {clip_path}'}

        model = inference.keras_video_model
        if model is None:
            return {'error': 'Modelo de vídeo não carregado no servidor.'}

        cap = cv2.VideoCapture(clip_path)
        if not cap.isOpened():
            return {'error': f'Falha ao abrir vídeo: {clip_path}'}

        # seek to 500ms (near start) to capture a frame with early freeze
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, 500)
        except Exception:
            pass
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return {'error': 'Não foi possível extrair frame do vídeo.'}

        input_data = inference.preprocess_video_single_frame(frame)
        if input_data is None:
            return {'error': 'Falha no pré-processamento do frame.'}

        # Input stats
        stats = {
            'min': float(np.min(input_data)),
            'max': float(np.max(input_data)),
            'mean': float(np.mean(input_data)),
            'std': float(np.std(input_data)),
            'shape': list(input_data.shape)
        }

        # Run predict
        preds = model.predict(input_data)
        scores = np.array(preds[0], dtype=float)
        probs = inference._apply_softmax(scores).tolist()
        # top-k
        indices = np.argsort(probs)[::-1]
        topk = []
        for i in indices[:max(1, min(5, len(probs)))]:
            name = inference.MODEL_CLASSES[i] if i < len(inference.MODEL_CLASSES) else f'idx_{i}'
            topk.append({'class': name, 'score': float(probs[i])})

        out.update({'clip_path': clip_path, 'input_stats': stats, 'raw': scores.tolist(), 'probs': probs, 'topk': topk})
        return out

    except Exception as e:
        return {'error': str(e)}


@router.get('/models/weights', summary='Return basic stats for model weights')
async def models_weights_info(layers: int = 8) -> Dict[str, Any]:
    """Return summary statistics for the first N layers' weights of loaded models.
    Useful to detect if weights are all zeros or contain NaNs.
    Query param `layers` controls how many layers to inspect (default 8).
    """
    import numpy as np
    out: Dict[str, Any] = {'models_loaded': bool(inference.models_loaded)}
    try:
        for name, model in [('audio', inference.keras_audio_model), ('video', inference.keras_video_model)]:
            if model is None:
                out[name] = None
                continue
            layer_infos = []
            cnt = 0
            for layer in model.layers:
                if cnt >= layers:
                    break
                try:
                    w = layer.get_weights()
                except Exception:
                    w = []
                if not w:
                    # skip layers without weights (e.g., pooling)
                    continue
                weights_summary = []
                for arr in w:
                    a = np.array(arr, dtype=np.float32)
                    stats = {
                        'shape': list(a.shape),
                        'min': float(np.min(a)) if a.size else None,
                        'max': float(np.max(a)) if a.size else None,
                        'mean': float(np.mean(a)) if a.size else None,
                        'std': float(np.std(a)) if a.size else None,
                        'all_zero': bool(np.all(a == 0)) if a.size else False,
                        'any_nan': bool(np.isnan(a).any()) if a.size else False,
                    }
                    weights_summary.append(stats)
                layer_infos.append({'layer_name': layer.name, 'class_name': layer.__class__.__name__, 'weights': weights_summary})
                cnt += 1
            out[name] = layer_infos
    except Exception as e:
        out['error'] = str(e)
    return out
