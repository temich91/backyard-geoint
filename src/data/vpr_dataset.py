#!/usr/bin/env python3
"""
vpr_dataset.py — классы Dataset для PyTorch поверх датасета из build_dataset.py.

  PlacesDataset — для обучения (MixVPR и подобные): один элемент = одно место (класс из
                  train.csv) и K его фото, по возможности из разных поездок. Батч из P мест
                  даёт P*K фото, у фото одного места одинаковая метка.
  ImageDataset  — одно фото без меток: прогнать database.csv / queries.csv через сеть,
                  получить векторы для FAISS и посчитать Recall.

Подготовка фото: обрезать низ кадра (там приборная панель), привести к одному размеру,
нормализовать как для сетей, обученных на ImageNet. При обучении — лёгкие аугментации
(яркость, контраст, небольшой случайный кроп). Отражение по горизонтали не используется:
оно меняет местами левую и правую сторону улицы.

Пример:
    from vpr_dataset import PlacesDataset, ImageDataset, make_transform, places_loader
    train = PlacesDataset("data/district", "views/train.csv", img_per_place=4,
                          transform=make_transform(train=True))
    loader = places_loader(train, places_per_batch=32, num_workers=4)
    for images, labels in loader:     # images: (32*4, 3, H, W), labels: (32*4,)
        ...

На Windows DataLoader с num_workers > 0 работает только внутри
`if __name__ == "__main__":`.

Проверка и просмотр того, что получает сеть:
    python vpr_dataset.py --root data\\district --views views
    (сохранит preview_train.jpg с одним батчем)

Установка: pip install torch torchvision pillow
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================================ подготовка фото

class CropBottom:
    """Отрезает нижнюю долю кадра (приборную панель, капот)."""

    def __init__(self, frac):
        self.frac = frac

    def __call__(self, img):
        w, h = img.size
        return img.crop((0, 0, w, round(h * (1 - self.frac)))) if self.frac > 0 else img

    def __repr__(self):
        return f"CropBottom({self.frac})"


def make_transform(train, size=(320, 320), crop_bottom=0.33):
    """Фото → тензор 3×H×W. size — (высота, ширина), 320×320 как в MixVPR.

    crop_bottom: панель у разных машин занимает от ~10 до ~40% высоты кадра; треть убирает
    её в большинстве кадров, а внизу кадра всё равно в основном асфальт.
    """
    steps = [CropBottom(crop_bottom)]
    if train:
        steps += [
            T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.75, 1.33), antialias=True),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        ]
    else:
        steps += [T.Resize(size, antialias=True)]
    steps += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(steps)


def denormalize(x):
    """Обратно к картинке 0..1 — только для просмотра."""
    mean = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(-1, 1, 1)
    return (x * std + mean).clamp(0, 1)


# ============================================================================ чтение таблиц

def read_rows(root, csv_name):
    """Строки CSV; lat/lon/compass_angle — числа, path — полный путь к файлу."""
    root = Path(root)
    with open(root / csv_name, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["file"] = root / r["path"]
        for k in ("lat", "lon", "compass_angle"):
            r[k] = float(r[k])
        if r.get("class_idx", "") != "":
            r["class_idx"] = int(r["class_idx"])
    return rows


def load_image(path):
    with Image.open(path) as im:
        return im.convert("RGB")


# ============================================================================ датасеты

class PlacesDataset(Dataset):
    """Обучение: элемент = одно место и img_per_place его фото.

    Фото выбираются заново при каждом обращении — каждую эпоху сеть видит разные
    сочетания. Сначала берётся по одному фото из разных поездок (разная погода,
    разные машины), и только если поездок меньше img_per_place — повторно из тех же.
    """

    def __init__(self, root, csv_name="views/train.csv", img_per_place=4, transform=None):
        self.rows = read_rows(root, csv_name)
        self.img_per_place = img_per_place
        self.transform = transform or make_transform(train=True)
        by_place = defaultdict(lambda: defaultdict(list))
        for r in self.rows:
            by_place[r["class_idx"]][r["sequence_id"]].append(r)
        # места, где фото меньше img_per_place, пропускаем
        self.places = [(idx, dict(seqs)) for idx, seqs in sorted(by_place.items())
                       if sum(map(len, seqs.values())) >= img_per_place]
        self.skipped = len(by_place) - len(self.places)

    def __len__(self):
        return len(self.places)

    def pick(self, seqs):
        order = list(seqs)
        random.shuffle(order)
        chosen, used = [], set()
        while len(chosen) < self.img_per_place:
            for s in order:
                free = [r for r in seqs[s] if id(r) not in used]
                if free and len(chosen) < self.img_per_place:
                    r = random.choice(free)
                    used.add(id(r))
                    chosen.append(r)
        return chosen

    def __getitem__(self, i):
        label, seqs = self.places[i]
        imgs = [self.transform(load_image(r["file"])) for r in self.pick(seqs)]
        return torch.stack(imgs), torch.full((self.img_per_place,), label, dtype=torch.long)


def collate_places(batch):
    """[(K×3×H×W, K), ...] из P мест → (P*K×3×H×W, P*K)."""
    images, labels = zip(*batch)
    return torch.cat(images), torch.cat(labels)


def places_loader(dataset, places_per_batch=32, num_workers=4, shuffle=True):
    return DataLoader(dataset, batch_size=places_per_batch, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_places, drop_last=shuffle, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)


class ImageDataset(Dataset):
    """Поиск и оценка: одно фото без меток, по порядку строк CSV.

    Номер элемента = номер строки, поэтому i-й вектор в FAISS соответствует self.rows[i]
    (там lat, lon, image_id и остальное).
    """

    def __init__(self, root, csv_name, transform=None):
        self.rows = read_rows(root, csv_name)
        self.transform = transform or make_transform(train=False)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.transform(load_image(self.rows[i]["file"])), i


def image_loader(dataset, batch_size=64, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())


# ============================================================================ проверка

def main():
    ap = argparse.ArgumentParser(description="Проверка датасетов и превью батча")
    ap.add_argument("--root", type=Path, default=Path("data") / "district")
    ap.add_argument("--views", default="views", help="views (по поездкам) или views_spatial")
    ap.add_argument("--img-per-place", type=int, default=4)
    ap.add_argument("--places", type=int, default=8, help="Мест в батче для превью")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    train = PlacesDataset(args.root, f"{args.views}/train.csv", args.img_per_place)
    print(f"train: {len(train.rows)} фото, {len(train)} мест (пропущено мест с < "
          f"{args.img_per_place} фото: {train.skipped})")
    loader = places_loader(train, args.places, num_workers=args.workers)
    images, labels = next(iter(loader))
    print(f"батч: images {tuple(images.shape)}, labels {labels.tolist()}")

    from torchvision.utils import save_image
    out = Path(f"preview_{Path(args.views).name}_train.jpg")
    save_image(denormalize(images), out, nrow=args.img_per_place)
    print(f"превью: {out} (одна строка = одно место)")

    names = (["database.csv", "queries.csv"] if args.views == "views"
             else ["val_database.csv", "val_queries.csv"])
    for name in names:
        ds = ImageDataset(args.root, f"{args.views}/{name}")
        x, i = ds[0]
        print(f"{name}: {len(ds)} фото, тензор {tuple(x.shape)}, "
              f"строка {i}: lat {ds.rows[i]['lat']:.6f}, lon {ds.rows[i]['lon']:.6f}")


if __name__ == "__main__":
    main()
