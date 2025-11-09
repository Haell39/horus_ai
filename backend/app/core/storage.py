import os
import json
from typing import Dict, Any

# Arquivo de configuração simples para armazenamento (podemos usar DB depois)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
STORAGE_FILE = os.path.join(ROOT, '.storage_config.json')


def _default_clips_dir() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'static', 'clips'))


def _default_config() -> Dict[str, Any]:
    return {
        'mode': 'local',  # 'local' | 'onedrive'
        'local_path': _default_clips_dir(),
        'oneDriveLink': None,
    }


def read_storage_config() -> Dict[str, Any]:
    try:
        if os.path.exists(STORAGE_FILE):
            with open(STORAGE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    # escreve default se não existir
    cfg = _default_config()
    try:
        write_storage_config(cfg)
    except Exception:
        pass
    return cfg


def write_storage_config(cfg: Dict[str, Any]) -> None:
    # validação mínima
    if not isinstance(cfg, dict):
        raise ValueError('config must be a dict')
    # assegura keys mínimas
    base = _default_config()
    base.update(cfg)
    # garantir caminho absoluto para local_path
    if base.get('local_path'):
        base['local_path'] = os.path.abspath(base['local_path'])
    # persistir atômico simples
    tmp = STORAGE_FILE + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(base, f, indent=2, ensure_ascii=False)
    os.replace(tmp, STORAGE_FILE)


def get_clips_dir() -> str:
    """Retorna o diretório onde os clipes devem ser gravados.

    Se o modo for 'local' e o local_path existir (ou puder ser criado), usa esse caminho.
    Caso contrário, retorna o diretório padrão dentro de backend/static/clips.
    """
    cfg = read_storage_config()
    if cfg.get('mode') == 'local' and cfg.get('local_path'):
        try:
            path = os.path.abspath(cfg.get('local_path'))
            os.makedirs(path, exist_ok=True)
            return path
        except Exception:
            # fallback para padrão
            pass
    # fallback para a pasta static/clips dentro do projeto (la onde o StaticFiles monta)
    default = _default_clips_dir()
    os.makedirs(default, exist_ok=True)
    return default


def get_storage_public_link() -> str | None:
    cfg = read_storage_config()
    return cfg.get('oneDriveLink')
