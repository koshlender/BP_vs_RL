from __future__ import annotations
import json, random
from pathlib import Path
from typing import Any
try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

def load_config(path: str | Path) -> dict[str, Any]:
    text=Path(path).read_text(encoding='utf-8')
    if str(path).endswith('.json'):
        return json.loads(text)
    if yaml is None:
        raise RuntimeError('PyYAML is required to load YAML configs; use configs/chapter_4_5.json in minimal environments')
    return yaml.safe_load(text)

def set_seed(seed: int) -> None:
    random.seed(seed)

def write_json(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2, sort_keys=True)
