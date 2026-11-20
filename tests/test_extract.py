import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.models.extract import extract_embeddings, extract_image, load_model
from src.data.vpr_dataset import ImageDataset


class DummyModel(nn.Module):
    """Легковесная модель для быстрых тестов извлечения."""
    def __init__(self, out_dim=16):
        super().__init__()
        self.conv = nn.Conv2d(3, out_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feat = self.pool(self.conv(x)).flatten(1)
        return torch.nn.functional.normalize(feat, p=2, dim=-1)


class ExtractTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="test_extract_"))
        self.model = DummyModel(out_dim=32)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_extract_single_image(self):
        img_path = self.tmpdir / "sample.jpg"
        im = Image.new("RGB", (640, 480), color=(100, 150, 200))
        im.save(img_path)

        vec = extract_image(self.model, img_path, device="cpu")
        self.assertEqual(vec.ndim, 1)
        self.assertEqual(vec.shape[0], 32)
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_extract_dataset(self):
        # Создаем 3 тестовых изображения и CSV
        for i in range(3):
            im = Image.new("RGB", (320, 240), color=(i * 40, 50, 100))
            im.save(self.tmpdir / f"img_{i}.jpg")

        csv_path = self.tmpdir / "db.csv"
        csv_path.write_text(
            "image_id,path,lat,lon\n"
            "101,img_0.jpg,54.75,56.05\n"
            "102,img_1.jpg,54.76,56.06\n"
            "103,img_2.jpg,54.77,56.07\n",
            encoding="utf-8"
        )

        dataset = ImageDataset(self.tmpdir, "db.csv")
        embeddings = extract_embeddings(self.model, dataset, device="cpu", batch_size=2, show_progress=False)

        self.assertEqual(embeddings.shape, (3, 32))
        norms = np.linalg.norm(embeddings, axis=1)
        for n in norms:
            self.assertAlmostEqual(n, 1.0, places=4)

    def test_load_model_without_checkpoint(self):
        model = load_model(checkpoint_path=None, descriptor_dim=128, device="cpu", pretrained=False)
        self.assertEqual(model.aggregator.channel_proj.out_features * model.aggregator.row_proj.out_features, 128)
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
