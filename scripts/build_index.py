#!/usr/bin/env python3
"""Скрипт построения индекса FAISS."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.search.__main__ import main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] not in ("build", "query"):
        sys.argv.insert(1, "build")
    elif len(sys.argv) == 1:
        sys.argv.append("build")
        sys.argv.append("--help")
    main()
