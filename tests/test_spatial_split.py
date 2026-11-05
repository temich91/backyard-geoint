"""Проверки основного пространственного протокола на синтетических данных."""

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.data import build_dataset as loader


class SpatialSplitTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(redirect_stdout(io.StringIO()))
        self.grid = loader.Grid((0, 0, 0.1, 0.1))
        self.args = SimpleNamespace(
            seed=42, val_block=500.0, val_buffer=50.0, val_frac=0.15,
            class_cell=50.0, min_sequences=4, pos_radius=25.0,
            pos_angle=90.0, spatial_views=True,
        )
        self.split_of = loader.block_splitter(self.grid, self.args)

    def coordinates(self, x, y):
        return (self.grid.min_lat + y / self.grid.m_lat,
                self.grid.min_lon + x / self.grid.m_lon)

    def prepare_metadata(self):
        # По одному месту в train и val, вдали от границ блоков.
        places = {}
        for bx in range(20):
            lat, lon = self.coordinates(bx * 500 + 275, 275)
            places.setdefault(self.split_of(lat, lon), (lat, lon))
        self.assertIn("train", places)
        self.assertIn("val", places)
        rows = []
        for split in ("train", "val"):
            lat, lon = places[split]
            for i in range(5):
                image_id = len(rows) + 1
                path = f"{image_id}.jpg"
                (self.root / path).write_bytes(b"image placeholder")
                rows.append({
                    "path": path, "image_id": image_id,
                    "role": "ref" if i < 4 else "query",
                    "lat": lat, "lon": lon, "compass_angle": 90.0,
                    "captured_at": "", "sequence_id": f"trip-{image_id}",
                    "n_positives": 999,
                })
        loader.write_csv(self.root / "metadata.csv", rows, list(rows[0]))

    def read_view(self, name):
        with (self.root / "views_spatial" / name).open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def test_spatial_views_separate_classes_and_build_validation_retrieval_pair(self):
        self.prepare_metadata()
        loader.build_views(self.root, self.grid, self.args)
        train = self.read_view("train.csv")
        val = self.read_view("val.csv")
        database = self.read_view("val_database.csv")
        queries = self.read_view("val_queries.csv")
        classes = self.read_view("classes.csv")

        self.assertEqual(len(train), 4)
        self.assertEqual(len(val), 4)
        self.assertTrue({r["class"] for r in train}.isdisjoint(r["class"] for r in val))
        self.assertTrue({r["class_idx"] for r in train}.isdisjoint(r["class_idx"] for r in val))
        self.assertEqual({r["split"] for r in classes}, {"train", "val"})
        self.assertEqual({r["image_id"] for r in val}, {r["image_id"] for r in database})
        self.assertEqual(len(queries), 1)
        self.assertEqual(queries[0]["n_positives"], "4")  # пересчитано, а не скопировано 999
        self.assertTrue({r["sequence_id"] for r in database}.isdisjoint(
            r["sequence_id"] for r in queries))
        for expected, rows in (("train", train), ("val", val + database + queries)):
            for row in rows:
                self.assertEqual(self.split_of(float(row["lat"]), float(row["lon"])), expected)

    def test_buffer_excludes_photos_near_train_val_boundary(self):
        for bx in range(1, 20):
            left = self.split_of(*self.coordinates(bx * 500 - 250, 275))
            right = self.split_of(*self.coordinates(bx * 500 + 250, 275))
            if left != right:
                break
        else:
            self.fail("Не найдена граница между train и val в тестовых блоках")
        self.assertIsNone(self.split_of(*self.coordinates(bx * 500 - 25, 275)))
        self.assertIsNone(self.split_of(*self.coordinates(bx * 500 + 25, 275)))
        self.assertEqual(self.split_of(*self.coordinates(bx * 500 - 75, 275)), left)
        self.assertEqual(self.split_of(*self.coordinates(bx * 500 + 75, 275)), right)

    def test_scan_excludes_test_district_and_its_buffer(self):
        argv = [
            "build_dataset.py", "--bbox", "0", "0", "0.1", "0.1", "--token", "test-token",
            "--exclude-bbox", "0.03", "0.03", "0.05", "0.05",
            "--exclude-buffer", "100", "--spatial-views",
        ]
        with patch("sys.argv", argv), patch.object(loader, "load_dotenv"):
            args = loader.parse_args()
        points = []
        # Внутри тестового района, в его буфере и за пределами обоих.
        for image_id, (lat, lon) in enumerate(((0.04, 0.04), (0.04, 0.0501), (0.08, 0.08)), 1):
            points.append(({
                "id": image_id, "compass_angle": 90.0, "sequence_id": f"ref-{image_id}",
                "quality_score": 0.9,
            }, lat, lon))
        with patch.object(loader, "iter_tiles", return_value=[((0, 0), b"tile")]), \
                patch.object(loader, "tile_points", return_value=points), \
                patch.object(loader, "role_of", return_value="ref"):
            ref_cells, q_cells, stats, _ = loader.scan(args, self.grid, self.root)
        self.assertTrue(args.spatial_views)
        self.assertEqual(stats["исключённая область"], 2)
        self.assertEqual(stats["прошли фильтры"], 1)
        self.assertEqual([img.id for cell in ref_cells.values() for img in cell.values()], [3])
        self.assertFalse(q_cells)

    def test_rebuilding_spatial_views_is_repeatable(self):
        self.prepare_metadata()
        loader.build_views(self.root, self.grid, self.args)
        before = {p.name: p.read_bytes() for p in (self.root / "views_spatial").glob("*.csv")}
        loader.build_views(self.root, self.grid, self.args)
        after = {p.name: p.read_bytes() for p in (self.root / "views_spatial").glob("*.csv")}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
