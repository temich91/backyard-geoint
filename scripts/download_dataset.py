#!/usr/bin/env python3
"""Скрипт загрузки датасета с Mapillary."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.build_dataset import main

if __name__ == "__main__":
    main()
