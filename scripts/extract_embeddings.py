#!/usr/bin/env python3
"""Скрипт извлечения дескрипторов (эмбеддингов) из изображений."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.extract import main

if __name__ == "__main__":
    main()
