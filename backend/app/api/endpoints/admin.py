from fastapi import APIRouter, HTTPException, Query, Body
import os
import shutil
from typing import Dict, Any

from app.core import storage as storage_core

router = APIRouter()


@router.post('/admin/sync-clips')
def sync_clips():
    """Copia todos os arquivos do diretório de storage configurado para a pasta pública `backend/static/clips`.

    Retorna lista de arquivos copiados e erros. Útil para sincronizar clipes salvos em um diretório local externo.
    NOTA: endpoint administrativo — proteger em produção.
    """
    try:
        src = storage_core.get_clips_dir()
        if not os.path.exists(src):
            raise HTTPException(status_code=404, detail=f"Storage clips dir not found: {src}")

        public_clips_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'clips'))
        os.makedirs(public_clips_dir, exist_ok=True)

        results = {'copied': [], 'skipped': [], 'errors': []}
        for name in os.listdir(src):
            src_path = os.path.join(src, name)
            if not os.path.isfile(src_path):
                continue
            dest_path = os.path.join(public_clips_dir, name)
            try:
                if os.path.exists(dest_path):
                    results['skipped'].append(name)
                    continue
                shutil.copy2(src_path, dest_path)
                results['copied'].append(name)
            except Exception as e:
                results['errors'].append({'file': name, 'error': str(e)})

        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/admin/disk-usage')
def disk_usage(path: str = Query(..., description='Path to check disk usage')):
    """Return disk usage stats (GB) for a given path. Used by frontend to display local disk usage."""
    try:
        if not path:
            raise HTTPException(status_code=400, detail='path is required')
        # Make sure the path exists; if file given, use its mount point
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f'Path not found: {path}')
        # Resolve to the mount point / directory
        target = path
        if os.path.isfile(path):
            target = os.path.dirname(path) or path

        usage = shutil.disk_usage(target)
        total_gb = round(usage.total / (1024**3), 2)
        used_gb = round((usage.total - usage.free) / (1024**3), 2)
        free_gb = round(usage.free / (1024**3), 2)
        percent = round((used_gb / total_gb) * 100 if total_gb else 0.0, 2)
        return {
            'path': os.path.abspath(target),
            'total_gb': total_gb,
            'used_gb': used_gb,
            'free_gb': free_gb,
            'percent': percent,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/admin/storage-config')
def get_storage_config():
    """Retorna a configuração de storage atual (modo, local_path, oneDriveLink)."""
    try:
        cfg = storage_core.read_storage_config()
        return cfg
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post('/admin/storage-config')
def set_storage_config(payload: Dict[str, Any] = Body(...)):
    """Define a configuração de storage. Exemplo payload: {"mode":"local","local_path":"C:\\clips"}

    Nota: este endpoint deve ser protegido em produção (autenticação/autorização).
    """
    try:
        # validação mínima
        mode = payload.get('mode')
        if mode not in ('local', 'onedrive'):
            raise HTTPException(status_code=400, detail='mode must be "local" or "onedrive"')
        cfg = storage_core.read_storage_config()
        cfg.update(payload)
        storage_core.write_storage_config(cfg)
        return cfg
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
