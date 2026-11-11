"""backyard-geoint: единая точка входа для Visual Place Recognition (VPR).

Команды:
  locate       - определение геолокации по фотографии (поиск в индексе)
  extract      - извлечение дескрипторов (эмбеддингов) из датасета
  build-index  - построение и сохранение индекса FAISS
  evaluate     - расчет метрик качества (Recall@k, ошибка в метрах)
  train        - запуск обучения модели MixVPR
  pipeline     - запуск полного цикла (извлечение -> индекс -> оценка)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Корень проекта в sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.metrics import haversine_distance
from src.eval.evaluate import format_evaluation_report, run_evaluation
from src.models.extract import extract_embeddings, extract_image, load_model
from src.search.index import PlaceIndex


def osm_url(lat: float, lon: float) -> str:
    """Генерация ссылки на OpenStreetMap для визуализации точки."""
    return f"https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map=18/{lat:.6f}/{lon:.6f}"


def locate_image(
    image_path: str | Path,
    index_path: str | Path,
    checkpoint_path: str | Path | None = None,
    top_k: int = 5,
    device: str | None = None,
    gt_coords: tuple[float, float] | None = None,
    descriptor_id: str | None = None,
    pretrained: bool = True,
) -> dict[str, Any]:
    """Определение координат одного изображения через модель и индекс FAISS."""
    import torch
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    index = PlaceIndex.load(index_path)
    model = load_model(
        checkpoint_path=checkpoint_path,
        descriptor_dim=index.dimension,
        device=dev,
        pretrained=pretrained,
    )

    query_vec = extract_image(model, image_path, device=dev)
    results = index.search(query_vec, k=top_k, descriptor_id=descriptor_id)

    matches = results[0] if results else []
    best_candidate = matches[0] if matches else None

    output: dict[str, Any] = {
        "image": str(image_path),
        "index": str(index_path),
        "top_k": top_k,
        "matches": [m.to_dict() for m in matches],
    }

    if best_candidate:
        meta = best_candidate.metadata
        output["prediction"] = {
            "lat": meta["lat"],
            "lon": meta["lon"],
            "image_id": meta["image_id"],
            "similarity": best_candidate.similarity,
            "osm_link": osm_url(meta["lat"], meta["lon"]),
        }
        if gt_coords:
            gt_lat, gt_lon = gt_coords
            dist_m = haversine_distance(gt_lat, gt_lon, meta["lat"], meta["lon"])
            output["prediction"]["ground_truth"] = {"lat": gt_lat, "lon": gt_lon}
            output["prediction"]["error_meters"] = dist_m

    return output


def _cmd_locate(args):
    gt = (args.gt_lat, args.gt_lon) if (args.gt_lat is not None and args.gt_lon is not None) else None
    result = locate_image(
        image_path=args.image,
        index_path=args.index,
        checkpoint_path=args.checkpoint,
        top_k=args.top_k,
        device=args.device,
        gt_coords=gt,
        descriptor_id=args.descriptor_id,
        pretrained=not args.no_pretrained,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    pred = result.get("prediction")
    if not pred:
        print("Кандидаты не найдены в индексе.")
        return

    print("=" * 64)
    print(f"  РЕЗУЛЬТАТ ПОИСКА МЕСТА ДЛЯ: {Path(args.image).name}")
    print("=" * 64)
    print(f"Предсказанные координаты : {pred['lat']:.6f}, {pred['lon']:.6f}")
    print(f"Косинусное сходство      : {pred['similarity']:.4f}")
    print(f"ID ближайшего кадра в базе: {pred['image_id']}")
    print(f"Ссылка на карту (OSM)    : {pred['osm_link']}")
    if "error_meters" in pred:
        print(f"Расстояние до истины (GT): {pred['error_meters']:.2f} м")

    print("-" * 64)
    print("Топ кандидатов:")
    for match in result["matches"]:
        meta = match["metadata"]
        print(f"  #{match['rank']} | Сходство: {match['similarity']:.4f} | GPS: {meta['lat']:.6f}, {meta['lon']:.6f} | ID: {meta['image_id']}")
    print("=" * 64)


def _cmd_pipeline(args):
    """Сквозной пайплайн: извлечение векторов -> индекс -> оценка качества."""
    import torch
    dev = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    artifacts_dir = Path(args.artifacts)
    emb_dir = artifacts_dir / "embeddings"
    index_dir = artifacts_dir / "index"
    results_dir = artifacts_dir / "results"
    emb_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.root)
    db_csv = f"{args.views}/{args.database_csv}"
    q_csv = f"{args.views}/{args.queries_csv}"

    db_npy = emb_dir / "database.npy"
    q_npy = emb_dir / "queries.npy"
    eval_json = results_dir / "evaluation.json"

    print("\n[1/3] Загрузка модели и извлечение дескрипторов...")
    model = load_model(
        checkpoint_path=args.checkpoint,
        descriptor_dim=args.descriptor_dim,
        device=dev,
        pretrained=not args.no_pretrained,
    )

    from src.data.vpr_dataset import ImageDataset
    db_dataset = ImageDataset(root, db_csv)
    print(f"  База: {len(db_dataset)} изображений")
    db_emb = extract_embeddings(model, db_dataset, device=dev, batch_size=args.batch_size, num_workers=args.workers)
    np.save(db_npy, db_emb, allow_pickle=False)

    q_dataset = ImageDataset(root, q_csv)
    print(f"  Запросы: {len(q_dataset)} изображений")
    q_emb = extract_embeddings(model, q_dataset, device=dev, batch_size=args.batch_size, num_workers=args.workers)
    np.save(q_npy, q_emb, allow_pickle=False)

    print(f"\n[2/3] Построение FAISS-индекса в {index_dir}...")
    import shutil
    if index_dir.exists():
        shutil.rmtree(index_dir, ignore_errors=True)
    index = PlaceIndex.from_files(db_npy, root / db_csv, descriptor_id=args.descriptor_id)
    index.save(index_dir)

    print(f"\n[3/3] Оценка качества на тестовых запросах...")
    radii = tuple(float(r.strip()) for r in args.radii.split(",") if r.strip())
    metrics = run_evaluation(
        index_path=index_dir,
        queries_embeddings_path=q_npy,
        queries_metadata_path=root / q_csv,
        top_k=args.top_k,
        descriptor_id=args.descriptor_id,
        radii_m=radii,
    )

    report = format_evaluation_report(metrics)
    print(report)

    eval_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Итоговый отчет сохранен в: {eval_json}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="backyard-geoint: Visual Place Recognition CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # 1. locate
    p_locate = subparsers.add_parser("locate", help="Определить координаты для одного изображения")
    p_locate.add_argument("--image", type=Path, required=True, help="Путь к файлу изображения")
    p_locate.add_argument("--index", type=Path, required=True, help="Каталог сохраненного FAISS-индекса")
    p_locate.add_argument("--checkpoint", type=Path, default=None, help="Путь к весам MixVPR (.pt)")
    p_locate.add_argument("--top-k", type=int, default=5, help="Число ближайших кандидатов")
    p_locate.add_argument("--gt-lat", type=float, default=None, help="Истинная широта (для проверки ошибки)")
    p_locate.add_argument("--gt-lon", type=float, default=None, help="Истинная долгота (для проверки ошибки)")
    p_locate.add_argument("--descriptor-id", default=None, help="Метка дескриптора")
    p_locate.add_argument("--device", default=None, help="cuda или cpu")
    p_locate.add_argument("--no-pretrained", action="store_true", help="Не загружать предобученные веса ImageNet")
    p_locate.add_argument("--json", action="store_true", help="Вывод в формате JSON")
    p_locate.set_defaults(func=_cmd_locate)

    # 2. pipeline
    p_pipe = subparsers.add_parser("pipeline", help="Запустить сквозной цикл (извлечение -> индекс -> оценка)")
    p_pipe.add_argument("--root", type=Path, required=True, help="Папка датасета (напр. data/district)")
    p_pipe.add_argument("--views", default="views", help="Подпапка со сплитами (views или views_spatial)")
    p_pipe.add_argument("--database-csv", default="database.csv", help="Имя CSV базы")
    p_pipe.add_argument("--queries-csv", default="queries.csv", help="Имя CSV запросов")
    p_pipe.add_argument("--checkpoint", type=Path, default=None, help="Чекпоинт модели (.pt)")
    p_pipe.add_argument("--artifacts", type=Path, default=Path("artifacts"), help="Каталог для артефактов")
    p_pipe.add_argument("--top-k", type=int, default=10, help="Кандидатов при оценке")
    p_pipe.add_argument("--radii", default="25,50", help="Радиусы попадания в метрах")
    p_pipe.add_argument("--batch-size", type=int, default=32, help="Размер батча для извлечения")
    p_pipe.add_argument("--workers", type=int, default=0, help="Воркеры DataLoader")
    p_pipe.add_argument("--descriptor-dim", type=int, default=4096, help="Размерность дескриптора")
    p_pipe.add_argument("--descriptor-id", default=None, help="Метка дескриптора")
    p_pipe.add_argument("--device", default=None, help="cuda или cpu")
    p_pipe.add_argument("--no-pretrained", action="store_true", help="Не загружать предобученные веса ImageNet")
    p_pipe.set_defaults(func=_cmd_pipeline)

    # 3. train
    p_train = subparsers.add_parser("train", help="Обучить модель MixVPR")
    p_train.set_defaults(func=lambda args: _forward_to_module("src.training.train", argv))

    # 4. extract
    p_extract = subparsers.add_parser("extract", help="Извлечь дескрипторы из изображений")
    p_extract.set_defaults(func=lambda args: _forward_to_module("src.models.extract", argv))

    # 5. build-index
    p_index = subparsers.add_parser("build-index", help="Построить индекс FAISS")
    p_index.set_defaults(func=lambda args: _forward_to_module("src.search.__main__", ["build"] + (argv[2:] if argv and len(argv) > 2 else [])))

    # 6. evaluate
    p_eval = subparsers.add_parser("evaluate", help="Рассчитать Recall@k и ошибки локализации")
    p_eval.set_defaults(func=lambda args: _forward_to_module("src.eval.evaluate", argv))

    parsed, remaining = parser.parse_known_args(argv)
    if parsed.subcommand in ("train", "extract", "build-index", "evaluate"):
        parsed.func(parsed)
    else:
        args = parser.parse_args(argv)
        args.func(args)


def _forward_to_module(module_name: str, argv: list[str] | None):
    import importlib
    mod = importlib.import_module(module_name)
    sub_argv = argv[2:] if argv and len(argv) > 2 else sys.argv[2:]
    mod.main(sub_argv)


if __name__ == "__main__":
    main()
