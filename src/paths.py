"""Central path configuration, loaded from configs/paths.yaml.

Replaces the absolute C:\\Users\\liisl\\Documents\\Kaugseire\\Puuliik\\...
paths hardcoded throughout scripts/, src/, and experiments/ before the
2026-07-16 reorg moved the underlying data into this repo's data/,
models/, and outputs/ trees. Import PATHS and use its attributes
(e.g. PATHS.intermediate / "foo.csv") instead of hardcoding a base dir.
"""

from pathlib import Path
from types import SimpleNamespace

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_FILE = REPO_ROOT / "configs" / "paths.yaml"

with open(_CONFIG_FILE, "r", encoding="utf-8") as _f:
    _raw = yaml.safe_load(_f)

PATHS = SimpleNamespace(**{key: REPO_ROOT / value for key, value in _raw.items()})
