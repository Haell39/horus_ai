from fastapi import APIRouter, HTTPException, Query
import os
from typing import List, Dict
from datetime import datetime

router = APIRouter()

# Define allowed folders relative to repository root
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
ALLOWED = {
    'docs': os.path.join(BASE, 'docs'),
    'tools': os.path.join(BASE, 'tools'),
    'configs': os.path.join(BASE, 'configs'),
}


def _safe_list(folder: str) -> List[Dict]:
    path = ALLOWED.get(folder)
    if not path or not os.path.exists(path):
        return []
    out = []
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if os.path.isfile(full) and name.lower().endswith(('.md', '.markdown')):
            stat = os.stat(full)
            out.append({
                'name': name,
                'size': stat.st_size,
                'mtime': datetime.utcfromtimestamp(stat.st_mtime).isoformat() + 'Z',
            })
    return out


@router.get('/docs/list')
def list_docs():
    """Lista arquivos markdown nas pastas docs, tools e configs."""
    try:
        return {k: _safe_list(k) for k in ALLOWED.keys()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/docs/file')
def get_doc(folder: str = Query(..., description='docs|tools|configs'), name: str = Query(...)):
    """Retorna o conteúdo raw (texto) de um arquivo markdown.

    Uso: /api/v1/docs/file?folder=docs&name=README.md
    """
    try:
        if folder not in ALLOWED:
            raise HTTPException(status_code=400, detail='folder must be one of: docs, tools, configs')
        safe_name = os.path.basename(name)
        target_dir = ALLOWED[folder]
        target = os.path.join(target_dir, safe_name)
        # ensure file is inside allowed dir
        if not os.path.commonpath([os.path.abspath(target_dir)]) == os.path.commonpath([os.path.abspath(target_dir), os.path.abspath(target)]):
            raise HTTPException(status_code=400, detail='invalid path')
        if not os.path.exists(target) or not os.path.isfile(target):
            raise HTTPException(status_code=404, detail='file not found')
        # only markdown
        if not safe_name.lower().endswith(('.md', '.markdown')):
            raise HTTPException(status_code=400, detail='only markdown files are allowed')
        with open(target, 'r', encoding='utf-8') as f:
            content = f.read()
        return {'name': safe_name, 'folder': folder, 'content': content}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
