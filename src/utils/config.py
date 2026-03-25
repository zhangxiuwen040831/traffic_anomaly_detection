
import yaml
import os
from typing import Dict, Any

def load_config(config_path: str = 'config/base.yaml', override_path: str = None) -> Dict[str, Any]:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if override_path and os.path.exists(override_path):
        with open(override_path, 'r', encoding='utf-8') as f:
            override = yaml.safe_load(f)
        config = deep_merge(config, override)
    
    return config

def deep_merge(base: Dict, override: Dict) -> Dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
