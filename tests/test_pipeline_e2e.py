import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.main import locate_image, main as main_cli
from src.eval.metrics import haversine_distance
from src.eval.evaluate import run_evaluation
from src.models.extract import extract_embeddings
from src.search.index import PlaceIndex


class TinyModel(nn.Module):
    """Мини-модель для сквозного тестирования."""
    def __init__(self, dim=16):
        super().__init__()
        self.conv = nn.Conv2d(3, dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # Свойства как у ResNet50MixVPR
        self.aggregator = type("Agg", (), {
            "channel_proj": type("Linear", (), {"out_features": dim})(),
            "row_proj": type("Linear", (), {"out_features": 1})(),
        })()

    def forward(self, x):
        feat = self.pool(self.conv(x)).flatten(1)
        return torch.nn.functional.normalize(feat, p=2, dim=-1)


class EndToEndPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="test_e2e_"))
        self.root = self.tmpdir / "dataset"
        self.views = self.root / "views"
        self.views.mkdir(parents=True)

        # Создаем 4 тестовых фото
        for i in range(4):
            im = Image.new("RGB", (320, 240), color=(i * 60, 100, 150))
            im.save(self.root / f"img_{i}.jpg")

        # База из 3 изображений
        (self.views / "database.csv").write_text(
            "image_id,path,lat,lon,compass_angle\n"
            "db0,img_0.jpg,54.7600,56.0500,10.0\n"
            "db1,img_1.jpg,54.7700,56.0600,20.0\n"
            "db2,img_2.jpg,54.7800,56.0700,30.0\n",
            encoding="utf-8"
        )

        # Запрос из 1 изображения (img_3 с координатами рядом с db0)
        (self.views / "queries.csv").write_text(
            "image_id,path,lat,lon,compass_angle\n"
            "q0,img_3.jpg,54.7601,56.0501,15.0\n",
            encoding="utf-8"
        )

        self.model = TinyModel(dim=16)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_full_retrieval_and_locate_flow(self):
        # 1. Извлекаем эмбеддинги базы
        from src.data.vpr_dataset import ImageDataset
        db_ds = ImageDataset(self.root, "views/database.csv")
        db_embs = extract_embeddings(self.model, db_ds, device="cpu", show_progress=False)
        self.assertEqual(db_embs.shape, (3, 16))

        # 2. Строим индекс FAISS
        index_dir = self.tmpdir / "faiss_index"
        index = PlaceIndex.build(db_embs, db_ds.rows, descriptor_id="tiny-16")
        index.save(index_dir)
        self.assertTrue((index_dir / "index.faiss").exists())

        # 3. Извлекаем эмбеддинги запросов
        q_ds = ImageDataset(self.root, "views/queries.csv")
        q_embs = extract_embeddings(self.model, q_ds, device="cpu", show_progress=False)
        q_npy = self.tmpdir / "queries.npy"
        np.save(q_npy, q_embs)

        # 4. Оцениваем Recall
        metrics = run_evaluation(
            index_path=index_dir,
            queries_embeddings_path=q_npy,
            queries_metadata_path=self.views / "queries.csv",
            top_k=2,
            descriptor_id="tiny-16",
            radii_m=(50.0,),
        )
        self.assertEqual(metrics["num_queries"], 1)
        self.assertIn("within_50m", metrics["recall"])

        # 5. Тестируем locate_image
        # Для теста пропатчим load_model чтобы возвращала нашу TinyModel
        import src.main as main_mod
        orig_load_model = main_mod.load_model
        try:
            main_mod.load_model = lambda *args, **kwargs: self.model
            loc_res = locate_image(
                image_path=self.root / "img_3.jpg",
                index_path=index_dir,
                checkpoint_path=None,
                top_k=2,
                device="cpu",
                gt_coords=(54.7601, 56.0501),
                descriptor_id="tiny-16",
            )
            self.assertIn("prediction", loc_res)
            self.assertEqual(len(loc_res["matches"]), 2)
            self.assertIn("osm_link", loc_res["prediction"])
            self.assertIn("error_meters", loc_res["prediction"])
            self.assertTrue(loc_res["prediction"]["error_meters"] >= 0)

            # 6. Тестируем CLI locate
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                main_cli([
                    "locate",
                    "--image", str(self.root / "img_3.jpg"),
                    "--index", str(index_dir),
                    "--descriptor-id", "tiny-16",
                    "--top-k", "2",
                    "--json",
                ])
            cli_json = json.loads(f.getvalue())
            self.assertIn("prediction", cli_json)
            self.assertEqual(cli_json["prediction"]["lat"], loc_res["prediction"]["lat"])
        finally:
            main_mod.load_model = orig_load_model

    def test_train_epoch_and_loss(self):
        from src.data.vpr_dataset import PlacesDataset, places_loader, make_transform
        from src.training.train import MultiSimilarityLoss, train_epoch

        # Создаем мини-датасет для обучения: 2 места по 2 фото
        (self.views / "train.csv").write_text(
            "image_id,path,lat,lon,compass_angle,sequence_id,class_idx\n"
            "t0,img_0.jpg,54.7600,56.0500,10.0,seq_0,0\n"
            "t1,img_1.jpg,54.7600,56.0500,20.0,seq_1,0\n"
            "t2,img_2.jpg,54.7800,56.0700,30.0,seq_2,1\n"
            "t3,img_3.jpg,54.7800,56.0700,40.0,seq_3,1\n",
            encoding="utf-8"
        )

        train_ds = PlacesDataset(
            root=self.root,
            csv_name="views/train.csv",
            img_per_place=2,
            transform=make_transform(train=False, size=(32, 32)),
        )
        self.assertEqual(len(train_ds), 2)

        loader = places_loader(train_ds, places_per_batch=2, num_workers=0, shuffle=False)
        criterion = MultiSimilarityLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        loss = train_epoch(self.model, loader, criterion, optimizer, device="cpu")
        self.assertIsInstance(loss, float)
        self.assertTrue(loss >= 0.0)

    def test_pipeline_subcommand(self):
        import src.main as main_mod
        orig_load_model = main_mod.load_model
        try:
            main_mod.load_model = lambda *args, **kwargs: self.model
            artifacts_dir = self.tmpdir / "test_pipe_artifacts"
            main_cli([
                "pipeline",
                "--root", str(self.root),
                "--views", "views",
                "--artifacts", str(artifacts_dir),
                "--descriptor-id", "tiny-16",
                "--descriptor-dim", "16",
                "--device", "cpu",
                "--no-pretrained",
            ])
            self.assertTrue((artifacts_dir / "embeddings" / "database.npy").exists())
            self.assertTrue((artifacts_dir / "embeddings" / "queries.npy").exists())
            self.assertTrue((artifacts_dir / "index" / "index.faiss").exists())
            self.assertTrue((artifacts_dir / "results" / "evaluation.json").exists())
            eval_data = json.loads((artifacts_dir / "results" / "evaluation.json").read_text(encoding="utf-8"))
            self.assertEqual(eval_data["num_queries"], 1)
        finally:
            main_mod.load_model = orig_load_model


if __name__ == "__main__":
    unittest.main()
