#!/usr/bin/env python3
"""Скрипт определения местоположения по фотографии."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.main import main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "locate":
        sys.argv.insert(1, "locate")
    elif len(sys.argv) == 1:
        sys.argv.append("locate")
        sys.argv.append("--help")
    main()
