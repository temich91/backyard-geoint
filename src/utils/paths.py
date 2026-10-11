# Вместо хардкода путей импортировать переменные отсюда

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "src"
DATA = ROOT / "data"
