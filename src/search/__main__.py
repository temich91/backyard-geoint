"""CLI: python -m src.search build|query --help."""

import argparse
import json
from pathlib import Path

from .index import PlaceIndex, load_embeddings


def main(argv=None):
    parser = argparse.ArgumentParser(description="FAISS-поиск мест по готовым векторам .npy")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="Построить индекс по векторам базы и CSV")
    build.add_argument("--embeddings", type=Path, required=True, help="Векторы базы (N,D), .npy")
    build.add_argument("--metadata", type=Path, required=True, help="CSV в том же порядке, что векторы")
    build.add_argument("--out", type=Path, required=True, help="Новый каталог индекса")
    build.add_argument("--descriptor-id", help="Метка весов и предобработки (рекомендуется)")

    query = commands.add_parser("query", help="Найти топ-k для одного или нескольких векторов")
    query.add_argument("--index", type=Path, required=True, help="Каталог сохранённого индекса")
    query.add_argument("--queries", type=Path, required=True, help="Вектор (D,) или матрица (Q,D), .npy")
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--descriptor-id", help="Проверить совпадение метки с индексом")
    query.add_argument("--output", type=Path, help="Записать JSON в файл вместо stdout")
    args = parser.parse_args(argv)

    try:
        if args.command == "build":
            index = PlaceIndex.from_files(args.embeddings, args.metadata, descriptor_id=args.descriptor_id)
            index.save(args.out)
            print(json.dumps({
                "index": str(args.out), "size": index.size, "dimension": index.dimension,
                "metric": "cosine", "descriptor_id": index.descriptor_id,
            }, ensure_ascii=False))
        else:
            index = PlaceIndex.load(args.index)
            results = index.search(load_embeddings(args.queries), args.top_k, descriptor_id=args.descriptor_id)
            payload = {
                "index": str(args.index), "descriptor_id": index.descriptor_id,
                "queries": [
                    {"query_index": i, "matches": [match.to_dict() for match in matches]}
                    for i, matches in enumerate(results)
                ],
            }
            text = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text, encoding="utf-8")
            else:
                print(text, end="")
    except (OSError, ValueError, RuntimeError) as e:
        parser.exit(2, f"Ошибка: {e}\n")


if __name__ == "__main__":
    main()
