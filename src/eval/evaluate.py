"""CLI и логика оценки качества поиска мест (Visual Place Recognition)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.search.index import PlaceIndex, load_embeddings, read_metadata
from src.eval.metrics import evaluate_results


def format_evaluation_report(results: dict[str, Any]) -> str:
    """Форматирование отчета оценки качества в текстовую таблицу."""
    num_queries = results.get("num_queries", 0)
    lines = [
        "=" * 64,
        "  ОЦЕНКА КАЧЕСТВА ПОИСКА МЕСТ (VISUAL PLACE RECOGNITION)",
        "=" * 64,
        f"Всего запросов: {num_queries}",
        "-" * 64,
    ]

    recall = results.get("recall", {})
    for radius_key, k_metrics in recall.items():
        radius_name = radius_key.replace("_", " ")
        lines.append(f"Попадание в радиус ({radius_name}):")
        for metric_name, value in sorted(k_metrics.items()):
            lines.append(f"  {metric_name:<15}: {value * 100:>6.2f}%")
        lines.append("-" * 64)

    loc_err = results.get("localization_error", {})
    lines.append("Географическая ошибка первого кандидата (top-1):")
    lines.append(f"  Средняя ошибка:   {loc_err.get('mean_m', 0.0):>8.2f} м")
    lines.append(f"  Медианная ошибка: {loc_err.get('median_m', 0.0):>8.2f} м")
    lines.append(f"  25-й перцентиль:  {loc_err.get('p25_m', 0.0):>8.2f} м")
    lines.append(f"  75-й перцентиль:  {loc_err.get('p75_m', 0.0):>8.2f} м")
    lines.append(f"  90-й перцентиль:  {loc_err.get('p90_m', 0.0):>8.2f} м")
    lines.append(f"  Мин / Макс:       {loc_err.get('min_m', 0.0):.1f} м / {loc_err.get('max_m', 0.0):.1f} м")
    lines.append("=" * 64)

    return "\n".join(lines)


def run_evaluation(
    index_path: str | Path,
    queries_embeddings_path: str | Path,
    queries_metadata_path: str | Path,
    top_k: int = 10,
    descriptor_id: str | None = None,
    radii_m: tuple[float, ...] = (25.0, 50.0),
) -> dict[str, Any]:
    """Выполнить поиск и рассчитать метрики по сохраненным файлам."""
    index = PlaceIndex.load(index_path)
    embeddings = load_embeddings(queries_embeddings_path)
    metadata = read_metadata(queries_metadata_path)

    if len(metadata) != len(embeddings):
        raise ValueError(
            f"Число строк метаданных ({len(metadata)}) не совпадает "
            f"с числом векторов запросов ({len(embeddings)})"
        )

    search_matches = index.search(
        embeddings,
        k=top_k,
        descriptor_id=descriptor_id,
    )

    metrics = evaluate_results(
        queries_metadata=metadata,
        search_matches=search_matches,
        radii_m=radii_m,
        k_values=(1, 5, min(10, top_k)),
    )
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Оценка качества VPR (Recall@k, ошибка локализации)")
    parser.add_argument("--index", type=Path, required=True, help="Каталог сохраненного FAISS-индекса")
    parser.add_argument("--queries-embeddings", type=Path, required=True, help="Файл .npy с дескрипторами запросов")
    parser.add_argument("--queries-metadata", type=Path, required=True, help="CSV с метаданными запросов (lat, lon)")
    parser.add_argument("--top-k", type=int, default=10, help="Количество кандидатов для каждого запроса")
    parser.add_argument("--radii", type=str, default="25,50", help="Радиусы критерия истинности через запятую (метры)")
    parser.add_argument("--descriptor-id", type=str, default=None, help="Метка дескриптора для проверки совместимости")
    parser.add_argument("--output", type=Path, default=None, help="Файл для сохранения результатов в формате JSON")

    args = parser.parse_args(argv)
    try:
        radii = tuple(float(r.strip()) for r in args.radii.split(",") if r.strip())
        metrics = run_evaluation(
            index_path=args.index,
            queries_embeddings_path=args.queries_embeddings,
            queries_metadata_path=args.queries_metadata,
            top_k=args.top_k,
            descriptor_id=args.descriptor_id,
            radii_m=radii,
        )

        report = format_evaluation_report(metrics)
        print(report)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nРезультаты сохранены в: {args.output}")

    except (OSError, ValueError, RuntimeError) as e:
        sys.stderr.write(f"Ошибка: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
