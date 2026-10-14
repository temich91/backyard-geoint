#!/usr/bin/env python3
"""
train.py — скрипт обучения MixVPR на базе датасета из src/data/vpr_dataset.py.

Использует:
  - ResNet-50 + MixVPR из src/models/mixvpr.py
  - Загрузчик PlacesDataset (батчи из P мест по K фото каждого)
  - Multi-Similarity Loss (канонический лосс для MixVPR и GSV-Cities)
  - Раздельный learning rate для бэкбона (layer3) и агрегатора MixVPR

Запуск:
  python src/training/train.py --root data/district --epochs 10 --batch-places 16
"""

import argparse
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path при прямом запуске
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from tqdm import tqdm

from src.data.vpr_dataset import PlacesDataset, places_loader, make_transform
from src.models.mixvpr import ResNet50MixVPR


class MultiSimilarityLoss(nn.Module):
    """Multi-Similarity (MS) Loss в чистом PyTorch.

    Стягивает вместе дескрипторы одного и того же места (positives) и
    расталкивает дескрипторы разных мест (negatives).
    """

    def __init__(self, alpha: float = 2.0, beta: float = 50.0, base: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.base = base

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # embeddings: [B, D] (уже L2-нормализованы моделью)
        # labels: [B]
        sim_mat = torch.matmul(embeddings, embeddings.t())
        b_size = embeddings.size(0)

        # Маски: диагональ исключаем (сравнение фото с самим собой)
        labels_mat = labels.view(-1, 1) == labels.view(1, -1)
        diag = torch.eye(b_size, dtype=torch.bool, device=embeddings.device)
        pos_mask = labels_mat & ~diag
        neg_mask = ~labels_mat

        loss_list = []
        for i in range(b_size):
            pos_pair = sim_mat[i][pos_mask[i]]
            neg_pair = sim_mat[i][neg_mask[i]]

            if pos_pair.numel() == 0 or neg_pair.numel() == 0:
                continue

            # Майнинг информативных пар по критерию Multi-Similarity:
            neg_mining = neg_pair[neg_pair + 0.1 > pos_pair.min()]
            pos_mining = pos_pair[pos_pair - 0.1 < neg_pair.max()]

            if neg_mining.numel() == 0:
                neg_mining = neg_pair
            if pos_mining.numel() == 0:
                pos_mining = pos_pair

            # Защита от переполнения exp при умножении на beta
            pos_exp = torch.clamp(-self.alpha * (pos_mining - self.base), max=50.0)
            neg_exp = torch.clamp(self.beta * (neg_mining - self.base), max=50.0)

            pos_loss = (1.0 / self.alpha) * torch.log1p(torch.sum(torch.exp(pos_exp)))
            neg_loss = (1.0 / self.beta) * torch.log1p(torch.sum(torch.exp(neg_exp)))
            loss_list.append(pos_loss + neg_loss)

        if not loss_list:
            return torch.tensor(0.0, requires_grad=True, device=embeddings.device)
        return torch.mean(torch.stack(loss_list))


# ============================================================================ Обучение одной эпохи
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    steps = 0

    pbar = tqdm(loader, desc="Обучение", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        embeddings = model(images)
        loss = criterion(embeddings, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(steps, 1)


# ============================================================================ Главный запуск
def main():
    parser = argparse.ArgumentParser(description="Обучение модели MixVPR")
    parser.add_argument("--root", type=Path, default=Path("data") / "district", help="Путь к датасету")
    parser.add_argument("--views", default="views", help="Папка с views (views или views_spatial)")
    parser.add_argument("--epochs", type=int, default=10, help="Количество эпох")
    parser.add_argument("--batch-places", type=int, default=16, help="Количество мест в батче (P)")
    parser.add_argument("--img-per-place", type=int, default=4, help="Фото на каждое место (K)")
    parser.add_argument("--lr-backbone", type=float, default=1e-5, help="Скорость обучения для layer3 ResNet-50")
    parser.add_argument("--lr-agg", type=float, default=1e-4, help="Скорость обучения для MixVPR")
    parser.add_argument("--descriptor-dim", type=int, default=4096, help="Размерность итогового дескриптора")
    parser.add_argument("--workers", type=int, default=2, help="Количество воркеров DataLoader")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"), help="Куда сохранять веса")
    parser.add_argument("--device", default=None, help="cuda или cpu (по умолчанию определяется автоматически)")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Используемое устройство: {device}")

    # 1. Загрузка данных
    train_csv = f"{args.views}/train.csv"
    train_dataset = PlacesDataset(
        root=args.root,
        csv_name=train_csv,
        img_per_place=args.img_per_place,
        transform=make_transform(train=True, size=(320, 320)),
    )
    print(f"Обучающая выборка: {len(train_dataset.rows)} фото, {len(train_dataset)} уникальных мест")

    train_loader = places_loader(
        train_dataset,
        places_per_batch=args.batch_places,
        num_workers=args.workers,
        shuffle=True,
    )

    # 2. Модель: готовая ResNet-50 + агрегатор MixVPR
    model = ResNet50MixVPR(
        pretrained=True,
        freeze_blocks=2,             # Заморожены conv1, layer1, layer2; layer3 обучается
        descriptor_dim=args.descriptor_dim,
        mix_depth=4,
    ).to(device)

    # 3. Оптимизатор с разделением скоростей обучения
    backbone_params = [p for p in model.layer3.parameters() if p.requires_grad]
    aggregator_params = list(model.aggregator.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": aggregator_params, "lr": args.lr_agg},
    ], weight_decay=1e-4)

    criterion = MultiSimilarityLoss(alpha=2.0, beta=50.0, base=0.5)

    # 4. Цикл обучения
    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    print("\nНачало обучения:")
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Эпоха [{epoch:02d}/{args.epochs:02d}] — Средний Loss: {avg_loss:.4f}")

        # Сохранение чекпоинта
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        checkpoint_data = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": avg_loss,
            "descriptor_dim": args.descriptor_dim,
        }

        last_path = args.out_dir / "mixvpr_last.pt"
        torch.save(checkpoint_data, last_path)

        if is_best:
            best_path = args.out_dir / "mixvpr_best.pt"
            torch.save(checkpoint_data, best_path)
            print(f"  * Сохранен лучший чекпоинт: {best_path}")

    print(f"\nОбучение завершено. Итоговые веса сохранены в {args.out_dir}")


if __name__ == "__main__":
    main()
