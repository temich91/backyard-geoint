"""Точный поиск мест по готовым дескрипторам; без зависимости от PyTorch."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Mapping

import faiss
import numpy as np

FORMAT_VERSION = 1


def load_embeddings(path: str | Path) -> np.ndarray:
    """Загрузить обычный числовой .npy, не разрешая pickle."""
    path = Path(path)
    if path.suffix.lower() != ".npy":
        raise ValueError("Ожидается файл векторов .npy")
    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray):
        array.close()
        raise ValueError("Ожидается массив .npy, а не архив .npz")
    return array


def _validate_metadata(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result, seen = [], set()
    for i, source in enumerate(rows):
        if not isinstance(source, Mapping):
            raise ValueError(f"Метаданные строки {i}: ожидается словарь")
        row = copy.deepcopy(dict(source))
        for key in ("image_id", "path", "lat", "lon"):
            if row.get(key) is None or str(row[key]).strip() == "":
                raise ValueError(f"Метаданные строки {i}: отсутствует {key}")
        row["image_id"] = str(row["image_id"])
        row["path"] = str(row["path"])
        if row["image_id"] in seen:
            raise ValueError(f"Повторяющийся image_id: {row['image_id']}")
        seen.add(row["image_id"])
        for key, limit in (("lat", 90), ("lon", 180)):
            try:
                value = float(row[key])
            except (TypeError, ValueError, OverflowError) as e:
                raise ValueError(f"Метаданные строки {i}: некорректное {key}") from e
            if not math.isfinite(value) or abs(value) > limit:
                raise ValueError(f"Метаданные строки {i}: некорректное {key}")
            row[key] = value
        if row.get("compass_angle") not in (None, ""):
            try:
                value = float(row["compass_angle"])
            except (TypeError, ValueError, OverflowError) as e:
                raise ValueError(f"Метаданные строки {i}: некорректное compass_angle") from e
            if not math.isfinite(value):
                raise ValueError(f"Метаданные строки {i}: некорректное compass_angle")
            row["compass_angle"] = value
        try:
            json.dumps(row, allow_nan=False)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Метаданные строки {i} должны быть JSON-совместимыми") from e
        result.append(row)
    if not result:
        raise ValueError("База метаданных пуста")
    return result


def read_metadata(path: str | Path) -> list[dict[str, Any]]:
    """CSV из build_dataset.py: сохраняет порядок строк, ID остаются строками."""
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return _validate_metadata(csv.DictReader(f))


def _normalize(vectors: np.ndarray, *, query: bool = False) -> np.ndarray:
    array = np.asarray(vectors)
    if query and array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError("Ожидаются векторы формы (N, D), D > 0")
    if not query and array.shape[0] == 0:
        raise ValueError("База векторов пуста")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError("Векторы должны содержать вещественные числа")
    with np.errstate(over="ignore", invalid="ignore"):
        matrix = np.array(array, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(matrix).all():
        raise ValueError("Векторы содержат NaN, Inf или числа вне диапазона float32")
    # Аккумуляция в float64 предотвращает переполнение/потерю нормы float32.
    norms = np.sqrt(np.einsum("ij,ij->i", matrix, matrix, dtype=np.float64))
    if (norms == 0).any():
        raise ValueError("Нулевые векторы недопустимы для косинусного поиска")
    matrix /= norms[:, None]
    return matrix


def _descriptor_id(value: str | None) -> str | None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError("descriptor_id должен быть непустой строкой или None")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SearchResult:
    rank: int
    index_id: int
    similarity: float
    metadata: dict[str, Any]

    @property
    def cosine_distance(self) -> float:
        """1 - cosine similarity, НЕ географическое расстояние в метрах."""
        return 1.0 - self.similarity

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank, "index_id": self.index_id,
            "similarity": self.similarity, "cosine_distance": self.cosine_distance,
            "metadata": copy.deepcopy(self.metadata),
        }


class PlaceIndex:
    """Неизменяемый набор векторов и метаданных в порядке внутренних ID FAISS.

    Создание: build() / from_files(). Загрузка сохранённого набора: load().
    Метка descriptor_id обозначает одинаковые веса и предобработку у базы и запросов.
    """

    def __init__(
        self, index: faiss.IndexFlatIP, metadata: list[dict[str, Any]], descriptor_id: str | None,
    ):
        self._index = index
        self._metadata = metadata
        self._descriptor_id = descriptor_id

    @property
    def size(self) -> int:
        return self._index.ntotal

    @property
    def dimension(self) -> int:
        return self._index.d

    @property
    def descriptor_id(self) -> str | None:
        return self._descriptor_id

    @classmethod
    def build(
        cls, embeddings: np.ndarray, metadata: Iterable[Mapping[str, Any]],
        *, descriptor_id: str | None = None,
    ) -> PlaceIndex:
        descriptor_id = _descriptor_id(descriptor_id)
        vectors = _normalize(embeddings)
        rows = _validate_metadata(metadata)
        if len(rows) != len(vectors):
            raise ValueError(f"Число векторов ({len(vectors)}) не совпадает с числом строк CSV ({len(rows)})")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, rows, descriptor_id)

    @classmethod
    def from_files(
        cls, embeddings: str | Path, metadata: str | Path,
        *, descriptor_id: str | None = None,
    ) -> PlaceIndex:
        return cls.build(load_embeddings(embeddings), read_metadata(metadata), descriptor_id=descriptor_id)

    def search(
        self, queries: np.ndarray, k: int = 5, *, descriptor_id: str | None = None,
    ) -> list[list[SearchResult]]:
        """Топ-k для каждого запроса; допускается один вектор (D,) или матрица (Q,D)."""
        if isinstance(k, bool) or not isinstance(k, Integral) or k <= 0:
            raise ValueError("k должен быть положительным целым числом")
        descriptor_id = _descriptor_id(descriptor_id)
        if descriptor_id is not None and descriptor_id != self.descriptor_id:
            raise ValueError("descriptor_id запроса не совпадает с descriptor_id индекса")
        vectors = _normalize(queries, query=True)
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Размерность запроса {vectors.shape[1]}, а индекса {self.dimension}")
        if len(vectors) == 0:
            return []
        similarities, ids = self._index.search(vectors, min(int(k), self.size))
        return [
            [SearchResult(
                rank=rank, index_id=int(index_id),
                similarity=float(np.clip(score, -1.0, 1.0)),
                metadata=copy.deepcopy(self._metadata[int(index_id)]),
            ) for rank, (score, index_id) in enumerate(zip(scores, indices), 1)]
            for scores, indices in zip(similarities, ids)
        ]

    def save(self, directory: str | Path) -> None:
        """Атомарно опубликовать новый каталог. Существующие данные не перезаписываются."""
        directory = Path(directory)
        if directory.exists() or directory.is_symlink():
            raise FileExistsError(f"Каталог индекса уже существует: {directory}. Укажите новый путь.")
        directory.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
        try:
            # Python открывает путь сам: это работает и с кириллицей в путях Windows.
            with (stage / "index.faiss").open("wb") as stream:
                faiss.write_index(self._index, faiss.PyCallbackIOWriter(stream.write))
            (stage / "metadata.json").write_text(
                json.dumps(self._metadata, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8",
            )
            manifest = {
                "format_version": FORMAT_VERSION, "index_type": "IndexFlatIP", "metric": "cosine",
                "dimension": self.dimension, "size": self.size, "descriptor_id": self.descriptor_id,
                "sha256": {name: _sha256(stage / name) for name in ("index.faiss", "metadata.json")},
            }
            (stage / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            stage.rename(directory)
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    @classmethod
    def load(cls, directory: str | Path) -> PlaceIndex:
        """Загрузить наш формат, проверив контрольные суммы, размерность и привязку строк.

        Загружайте только доверенные артефакты: SHA256 защищает от случайной порчи,
        а не от намеренно подменённых файлов FAISS.
        """
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or type(manifest.get("format_version")) is not int
            or manifest["format_version"] != FORMAT_VERSION
        ):
            raise ValueError("Неподдерживаемый формат сохранённого индекса")
        if manifest.get("metric") != "cosine" or manifest.get("index_type") != "IndexFlatIP":
            raise ValueError("Ожидается косинусный индекс IndexFlatIP")
        checksums = manifest.get("sha256")
        if not isinstance(checksums, dict):
            raise ValueError("Нет контрольных сумм файлов индекса")
        for name in ("index.faiss", "metadata.json"):
            if checksums.get(name) != _sha256(directory / name):
                raise ValueError(f"Не совпадает SHA256 файла {name}: индекс повреждён или файлы перепутаны")
        rows = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("metadata.json должен содержать список строк")
        rows = _validate_metadata(rows)
        with (directory / "index.faiss").open("rb") as stream:
            index = faiss.read_index(faiss.PyCallbackIOReader(stream.read))
        if not isinstance(index, faiss.IndexFlatIP) or index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("Файл FAISS не является IndexFlatIP")
        if type(manifest.get("dimension")) is not int or manifest["dimension"] != index.d:
            raise ValueError("Размерность индекса не совпадает с manifest.json")
        if (
            type(manifest.get("size")) is not int
            or manifest["size"] != index.ntotal
            or len(rows) != index.ntotal
        ):
            raise ValueError("Число векторов индекса не совпадает с метаданными/manifest.json")
        return cls(index, rows, _descriptor_id(manifest.get("descriptor_id")))
