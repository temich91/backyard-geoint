import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.eval.metrics import (
    compute_localization_errors,
    compute_recall_at_k,
    evaluate_results,
    haversine_distance,
)
from src.eval.evaluate import format_evaluation_report, run_evaluation
from src.search.index import PlaceIndex


class EvalMetricsTests(unittest.TestCase):
    def test_haversine_same_point(self):
        self.assertAlmostEqual(haversine_distance(55.7558, 37.6173, 55.7558, 37.6173), 0.0, places=3)

    def test_haversine_known_distance(self):
        # 1 градус широты примерно 111.139 км
        d = haversine_distance(54.0, 56.0, 55.0, 56.0)
        self.assertTrue(111000.0 < d < 112000.0, f"Distance {d} outside expected 111-112 km")

    def test_haversine_symmetry(self):
        d1 = haversine_distance(54.75, 56.05, 54.78, 56.08)
        d2 = haversine_distance(54.78, 56.08, 54.75, 56.05)
        self.assertAlmostEqual(d1, d2, places=5)

    def test_recall_at_k(self):
        gt = [(54.7500, 56.0500), (54.7600, 56.0600)]
        # Запрос 0: первый кандидат в 10 м, второй далеко
        # Запрос 1: первый кандидат далеко (1000 м), второй в 20 м
        p0_close = (54.75005, 56.0500)  # ~5.5 м
        p0_far = (54.8000, 56.0500)
        p1_far = (54.8000, 56.0600)
        p1_close = (54.7601, 56.0600)  # ~11 м

        retrieved = [
            [p0_close, p0_far],
            [p1_far, p1_close],
        ]

        # При k=1: запрос 0 успешен, запрос 1 нет -> 50%
        # При k=2: оба успешны -> 100%
        recalls = compute_recall_at_k(gt, retrieved, k_values=(1, 2), radius_m=50.0)
        self.assertAlmostEqual(recalls[1], 0.5)
        self.assertAlmostEqual(recalls[2], 1.0)

    def test_localization_errors(self):
        gt = [(54.7500, 56.0500), (54.7600, 56.0600)]
        retrieved = [
            [(54.7500, 56.0500)],  # ошибка 0
            [(54.7609, 56.0600)],  # ошибка ~100 м
        ]
        errors = compute_localization_errors(gt, retrieved)
        self.assertAlmostEqual(errors["min_m"], 0.0, places=1)
        self.assertTrue(90.0 < errors["max_m"] < 110.0)
        self.assertAlmostEqual(errors["mean_m"], errors["max_m"] / 2.0, places=1)

    def test_evaluate_results_end_to_end(self):
        queries_meta = [
            {"image_id": "q1", "lat": 54.75, "lon": 56.05},
            {"image_id": "q2", "lat": 54.76, "lon": 56.06},
        ]
        matches = [
            [{"metadata": {"image_id": "db1", "lat": 54.7501, "lon": 56.05}}],
            [{"metadata": {"image_id": "db2", "lat": 54.77, "lon": 56.06}}],
        ]
        res = evaluate_results(queries_meta, matches, radii_m=(25.0, 50.0), k_values=(1,))
        self.assertEqual(res["num_queries"], 2)
        self.assertIn("within_25m", res["recall"])
        self.assertIn("within_50m", res["recall"])
        report = format_evaluation_report(res)
        self.assertIn("recall@1", report)


class EvalPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="test_eval_"))

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_run_evaluation_with_index(self):
        # 1. Создаем синтетическую базу
        db_vecs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        db_meta = [
            {"image_id": "1", "path": "1.jpg", "lat": 54.75, "lon": 56.05},
            {"image_id": "2", "path": "2.jpg", "lat": 54.76, "lon": 56.06},
            {"image_id": "3", "path": "3.jpg", "lat": 54.77, "lon": 56.07},
        ]
        index = PlaceIndex.build(db_vecs, db_meta, descriptor_id="test-dim3")
        index_dir = self.tmpdir / "index"
        index.save(index_dir)

        # 2. Создаем синтетические запросы
        q_vecs = np.array([
            [0.99, 0.05, 0.0],   # похож на db1
            [0.0, 0.99, 0.05],   # похож на db2
        ], dtype=np.float32)
        q_npy = self.tmpdir / "q.npy"
        np.save(q_npy, q_vecs)

        q_csv = self.tmpdir / "q.csv"
        q_csv.write_text(
            "image_id,path,lat,lon\n"
            "q1,q1.jpg,54.7501,56.0500\n"
            "q2,q2.jpg,54.7601,56.0600\n",
            encoding="utf-8"
        )

        metrics = run_evaluation(
            index_path=index_dir,
            queries_embeddings_path=q_npy,
            queries_metadata_path=q_csv,
            top_k=2,
            descriptor_id="test-dim3",
            radii_m=(25.0, 50.0),
        )

        self.assertEqual(metrics["num_queries"], 2)
        # Обе точки попали в пределах 25м
        self.assertAlmostEqual(metrics["recall"]["within_25m"]["recall@1"], 1.0)
        self.assertAlmostEqual(metrics["recall"]["within_50m"]["recall@1"], 1.0)


if __name__ == "__main__":
    unittest.main()
