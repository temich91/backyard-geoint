"""Извлечение дескрипторов MixVPR из изображений."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.vpr_dataset import ImageDataset, image_loader, make_transform
from src.models.mixvpr import ResNet50MixVPR


def load_model(
    checkpoint_path: str | Path | None = None,
    descriptor_dim: int = 4096,
    device: torch.device | str = "cpu",
    pretrained: bool = True,
) -> ResNet50MixVPR:
    """Загрузить модель ResNet50MixVPR из чекпоинта или с весами ImageNet."""
    device = torch.device(device)
    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Файл чекпоинта не найден: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get("model_state", checkpoint)
        dim = checkpoint.get("descriptor_dim", descriptor_dim)
        model = ResNet50MixVPR(pretrained=False, descriptor_dim=dim)
        model.load_state_dict(state_dict)
    else:
        model = ResNet50MixVPR(pretrained=pretrained, descriptor_dim=descriptor_dim)

    model.to(device)
    model.eval()
    return model


def extract_embeddings(
    model: nn.Module,
    dataset_or_loader: Union[ImageDataset, DataLoader],
    device: torch.device | str = "cpu",
    batch_size: int = 32,
    num_workers: int = 0,
    show_progress: bool = True,
) -> np.ndarray:
    """Извлечь дескрипторы для датасета изображений в виде массива (N, D).

    Дескрипторы нормализованы по L2 и строго соответствуют порядку строк в датасете.
    """
    device = torch.device(device)
    model.eval()

    if isinstance(dataset_or_loader, DataLoader):
        loader = dataset_or_loader
    else:
        loader = image_loader(dataset_or_loader, batch_size=batch_size, num_workers=num_workers)

    all_embeddings: list[np.ndarray] = []
    iterator = tqdm(loader, desc="Извлечение дескрипторов", leave=False) if show_progress else loader

    with torch.no_grad():
        for batch in iterator:
            if isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch
            images = images.to(device, non_blocking=True)
            descriptors = model(images)
            all_embeddings.append(descriptors.detach().cpu().to(torch.float32).numpy())

    if not all_embeddings:
        return np.empty((0, model.aggregator.channel_proj.out_features * model.aggregator.row_proj.out_features), dtype=np.float32)

    return np.vstack(all_embeddings)


def extract_image(
    model: nn.Module,
    image_input: str | Path | Image.Image,
    device: torch.device | str = "cpu",
    transform=None,
) -> np.ndarray:
    """Извлечь один L2-нормализованный дескриптор формы (D,) для изображения."""
    device = torch.device(device)
    model.eval()

    if isinstance(image_input, (str, Path)):
        with Image.open(image_input) as img:
            image = img.convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise TypeError(f"Неподдерживаемый тип изображения: {type(image_input)}")

    if transform is None:
        transform = make_transform(train=False)

    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(tensor).squeeze(0).cpu().to(torch.float32).numpy()
    return emb


def main(argv=None):
    parser = argparse.ArgumentParser(description="Извлечение векторов признаков MixVPR из изображений")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Путь к весам модели (.pt)")
    parser.add_argument("--root", type=Path, required=True, help="Корневая папка датасета")
    parser.add_argument("--csv", type=str, required=True, help="Относительный или полный путь к CSV")
    parser.add_argument("--out", type=Path, required=True, help="Путь для сохранения .npy файла")
    parser.add_argument("--batch-size", type=int, default=32, help="Размер батча")
    parser.add_argument("--workers", type=int, default=0, help="Количество воркеров DataLoader")
    parser.add_argument("--descriptor-dim", type=int, default=4096, help="Размерность дескриптора")
    parser.add_argument("--device", default=None, help="cuda или cpu (по умолчанию автоопределение)")
    parser.add_argument("--no-pretrained", action="store_true", help="Не загружать предобученные веса ImageNet при отсутствии чекпоинта")

    args = parser.parse_args(argv)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    try:
        model = load_model(
            checkpoint_path=args.checkpoint,
            descriptor_dim=args.descriptor_dim,
            device=device,
            pretrained=not args.no_pretrained,
        )

        dataset = ImageDataset(args.root, args.csv)
        print(f"Загружен датасет: {len(dataset)} изображений из {args.csv}")

        embeddings = extract_embeddings(
            model=model,
            dataset_or_loader=dataset,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.workers,
            show_progress=True,
        )

        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out, embeddings, allow_pickle=False)

        summary = {
            "output": str(args.out),
            "count": int(embeddings.shape[0]),
            "dimension": int(embeddings.shape[1]),
            "device": str(device),
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError) as e:
        sys.stderr.write(f"Ошибка: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
