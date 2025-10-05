import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
GLOBAL_CONFIG = CONFIG_DIR / "global_config.json"
CURRENT_DAY = None 

def set_current_day(day: str):
    global CURRENT_DAY
    CURRENT_DAY = day

def _resolve_config_path() -> Path:
    if CURRENT_DAY:
        local_config = BASE_DIR / "post" / CURRENT_DAY / "config.json"
        if local_config.exists():
            return local_config
        else:
            return local_config

    selected_day = get_nested("selected_day", path=GLOBAL_CONFIG)
    if selected_day:
        local_config = BASE_DIR / "post" / selected_day / "config.json"
        if local_config.exists():
            return local_config

    return GLOBAL_CONFIG


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(data: dict, path: Path):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get_nested(key_path: str, default=None, path: Path = None):
    # ⚠️ selected_day siempre se lee del global
    if key_path == "selected_day":
        path = GLOBAL_CONFIG
    else:
        path = path or _resolve_config_path()

    data = _read_json(path)
    keys = key_path.split(".")
    for k in keys:
        if isinstance(data, dict) and k in data:
            data = data[k]
        else:
            return default
    return data


def set_nested(key_path: str, value, path: Path = None):
    # ⚠️ selected_day siempre se guarda en el global
    if key_path == "selected_day":
        path = GLOBAL_CONFIG
    else:
        path = path or _resolve_config_path()

    data = _read_json(path)
    keys = key_path.split(".")
    d = data
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value
    _write_json(data, path)
