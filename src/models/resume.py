"""Пакетная запись векторов на диск с возобновлением по подтверждённым строкам."""

import hashlib
import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import BinaryIO

import numpy as np

SAVE_EVERY_BATCHES = 16
DTYPE = np.dtype("<f4")


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def progress_paths(out: Path) -> tuple[Path, Path]:
    return out.with_name(out.name + ".part"), out.with_name(out.name + ".resume.json")


def _commit(state_path: Path, source: dict, rows: int) -> None:
    stage = state_path.with_name(state_path.name + ".tmp")
    with stage.open("w", encoding="utf-8") as stream:
        json.dump({"source": source, "rows": rows}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    stage.replace(state_path)


def _write_header(stream: BinaryIO, total: int, dimension: int) -> None:
    np.lib.format.write_array_header_1_0(stream, {
        "shape": (total, dimension), "descr": DTYPE.str, "fortran_order": False,
    })


def _seek_committed(stream: BinaryIO, total: int, rows: int) -> int:
    if np.lib.format.read_magic(stream) != (1, 0):
        raise ValueError("Неизвестный формат файла возобновления")
    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    if len(shape) != 2 or shape[0] != total or shape[1] <= 0 or fortran_order or dtype != DTYPE:
        raise ValueError("Неверная форма или тип векторов в файле возобновления")
    position = stream.tell() + rows * shape[1] * dtype.itemsize
    size = os.fstat(stream.fileno()).st_size
    if size < position or (rows == total and size != position):
        raise ValueError("Файл векторов не соответствует сохранённому прогрессу")
    stream.seek(position)
    return shape[1]


def save_resumable(
    out: Path, source: dict, total: int, make_batches: Callable[[int], Iterable[np.ndarray]],
    empty_dimension: Callable[[], int],
) -> tuple[int, int]:
    """Записать .npy; make_batches(offset) выдаёт батчи с указанной строки CSV."""
    partial, state_path = progress_paths(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or state.get("source") != source:
            raise ValueError(f"Входные данные изменились для {out}. Укажите новый путь вывода.")
        rows = state.get("rows")
        if type(rows) is not int or not 0 <= rows <= total:
            raise ValueError(f"Некорректное число сохранённых строк: {state_path}")
    elif out.exists() or partial.exists():
        raise ValueError(f"Нет состояния для {out}. Укажите новый путь вывода.")
    else:
        rows = 0
        _commit(state_path, source, rows)

    if out.exists():
        if rows != total or partial.exists():
            raise ValueError(f"Файл {out} не соответствует состоянию извлечения")
        with out.open("rb") as stream:
            dimension = _seek_committed(stream, total, rows)
        print(f"Готовые векторы: {out} ({rows}/{total})", flush=True)
        return total, dimension

    with partial.open("r+b" if rows else "wb") as stream:
        dimension = _seek_committed(stream, total, rows) if rows else None
        if rows:
            print(f"Продолжаем {out.name} с {rows}/{total} сохранённых строк", flush=True)
        if rows < total:
            for i, vectors in enumerate(make_batches(rows), 1):
                if (vectors.ndim != 2 or not len(vectors) or rows + len(vectors) > total
                        or vectors.shape[1] == 0 or dimension not in (None, vectors.shape[1])):
                    raise ValueError("Батч векторов не соответствует форме выходного файла")
                if dimension is None:
                    dimension = vectors.shape[1]
                    _write_header(stream, total, dimension)
                stream.write(vectors.astype(DTYPE, copy=False).tobytes())
                rows += len(vectors)
                if i % SAVE_EVERY_BATCHES == 0 or rows == total:
                    # Сначала данные, затем отметка прогресса. Неподтверждённый хвост перезапишется.
                    stream.flush()
                    os.fsync(stream.fileno())
                    _commit(state_path, source, rows)
        elif dimension is None:  # Пустой CSV.
            dimension = empty_dimension()
            _write_header(stream, total, dimension)
            stream.flush()
            os.fsync(stream.fileno())
        if rows != total or dimension is None:
            raise RuntimeError(f"Извлечено {rows} из {total} векторов")

    partial.replace(out)
    return total, dimension
