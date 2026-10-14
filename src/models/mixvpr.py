#!/usr/bin/env python3
"""
mixvpr.py — модель MixVPR на базе предобученной ResNet-50.

Архитектура:
  1. Бэкбон: готовая ResNet-50 из torchvision, обрезанная до layer3.
     При входе 320x320 выдает карты признаков [B, 1024, 20, 20].
  2. Агрегатор MixVPR: стек FeatureMixer блоков (перемешивание внутри карт)
     + канальная и строчная проекции -> L2-нормализованный дескриптор места [B, 4096].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FeatureMixerLayer(nn.Module):
    """Один блок перемешивания информации внутри отдельной карты признаков."""

    def __init__(self, in_dim: int, mlp_ratio: int = 1):
        super().__init__()
        mid_dim = int(in_dim * mlp_ratio)
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, in_dim),
        )

        # Стандартная инициализация весов
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, N]
        return x + self.mix(x)


class MixVPR(nn.Module):
    """Агрегатор признаков MixVPR.

    Принимает карты признаков [B, C, H, W] и формирует L2-нормализованный дескриптор.
    """

    def __init__(
        self,
        in_channels: int = 1024,
        in_h: int = 20,
        in_w: int = 20,
        out_channels: int = 1024,
        mix_depth: int = 4,
        mlp_ratio: int = 1,
        out_rows: int = 4,
    ):
        super().__init__()
        hw = in_h * in_w

        # Каскад блоков Feature Mixer
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
            for _ in range(mix_depth)
        ])

        # Канальная проекция (смешивание по каналам)
        self.channel_proj = nn.Linear(in_channels, out_channels)

        # Строчная проекция (сжатие пространства с hw до out_rows)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W], например [B, 1024, 20, 20]
        x = x.flatten(2)  # [B, 1024, 400]
        x = self.mix(x)   # [B, 1024, 400]

        # 1. Канальное перемешивание
        x = x.permute(0, 2, 1)       # [B, 400, 1024]
        x = self.channel_proj(x)      # [B, 400, out_channels]
        x = x.permute(0, 2, 1)       # [B, out_channels, 400]

        # 2. Пространственное сжатие
        x = self.row_proj(x)          # [B, out_channels, out_rows]

        # 3. Разворачивание в один вектор и L2-нормализация
        x = x.flatten(1)              # [B, out_channels * out_rows]
        return F.normalize(x, p=2, dim=-1)


class ResNet50MixVPR(nn.Module):
    """Полная модель: готовая предобученная ResNet-50 + агрегатор MixVPR.

    Принимает изображения [B, 3, 320, 320] и возвращает дескрипторы [B, descriptor_dim].
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_blocks: int = 2,
        descriptor_dim: int = 4096,
        mix_depth: int = 4,
    ):
        super().__init__()

        # Загружаем готовую ResNet-50 из torchvision
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Используем слои до layer3 включительно (layer4, avgpool, fc отбрасываем)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        # Заморозка начальных слоев бэкбона
        if freeze_blocks >= 0:
            for p in self.conv1.parameters():
                p.requires_grad = False
            for p in self.bn1.parameters():
                p.requires_grad = False
        if freeze_blocks >= 1:
            for p in self.layer1.parameters():
                p.requires_grad = False
        if freeze_blocks >= 2:
            for p in self.layer2.parameters():
                p.requires_grad = False

        out_rows = 4
        out_channels = descriptor_dim // out_rows

        # Агрегатор MixVPR
        self.aggregator = MixVPR(
            in_channels=1024,
            in_h=20,
            in_w=20,
            out_channels=out_channels,
            mix_depth=mix_depth,
            out_rows=out_rows,
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Прогон через бэкбон: [B, 3, 320, 320] -> [B, 1024, 20, 20]."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(x)
        return self.aggregator(feats)


if __name__ == "__main__":
    # Быстрая проверка формы тензоров
    model = ResNet50MixVPR(pretrained=False, descriptor_dim=4096)
    dummy_input = torch.randn(2, 3, 320, 320)
    out = model(dummy_input)
    print(f"Входной тензор:   {tuple(dummy_input.shape)}")
    print(f"Выходной вектор:  {tuple(out.shape)}")
    print(f"L2-норма выхода:  {torch.norm(out[0], p=2).item():.4f}")
