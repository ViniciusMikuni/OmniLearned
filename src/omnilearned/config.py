"""YAML > explicit CLI > command defaults."""

from pathlib import Path

import yaml


def resolve(values):
    values = dict(values)
    path = values.pop("config", None)
    if not path:
        return values
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping of CLI option names to values")
    aliases = {
        "output_dir": "outdir",
        "input_dir": "indir",
        "k": "K",
        "model_size": "size",
    }
    normalized = {}
    for key, value in config.items():
        key = str(key).replace("-", "_")
        key = aliases.get(key, key)
        if key not in values:
            raise ValueError(f"Unknown config option: {key}")
        if key in normalized:
            raise ValueError(f"Duplicate config option: {key}")
        expected = type(values[key])
        if expected is float and type(value) is int:
            value = float(value)
        if expected in (bool, int, float, str) and type(value) is not expected:
            raise ValueError(f"Config option {key} requires {expected.__name__}")
        normalized[key] = value
    values.update(normalized)
    return values
