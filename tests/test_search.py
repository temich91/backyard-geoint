"""FAISS-поиск: численная корректность, привязка GPS, сохранение и CLI. Без PyTorch."""

import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np

from src.search import PlaceIndex, load_embeddings, read_metadata


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.vectors = np.array([[3.0, 0.0], [0.0, 2.0], [-1.0, 0.0]], dtype=np.float64)
        self.rows = [
            {"image_id": str(9007199254740993 + i), "path": f"images/{i:02d}/{i}.jpg",
             "lat": 54.77 + i * 0.001, "lon": 56.06 + i * 0.001,
             "compass_angle": 90, "sequence_id": f"trip-{i}", "captured_at": "2021-01-01"}
            for i in range(3)
        ]
        self.index = PlaceIndex.build(self.vectors, self.rows, descriptor_id="model-a:crop033:320")

    def write_inputs(self):
        vectors = self.root / "database.npy"
        metadata = self.root / "database.csv"
        np.save(vectors, self.vectors)
        with metadata.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        return vectors, metadata

    def saved_index(self):
        bundle = self.root / "bundle"
        self.index.save(bundle)
        return bundle

    def edit_manifest(self, bundle, **changes):
        path = bundle / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.update(changes)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_cosine_ranking_returns_corresponding_metadata(self):
        matches = self.index.search([10, 1], k=3)[0]
        self.assertEqual([m.index_id for m in matches], [0, 1, 2])
        self.assertEqual([m.rank for m in matches], [1, 2, 3])
        self.assertEqual(matches[0].metadata, self.rows[0])
        self.assertAlmostEqual(matches[0].similarity, 10 / np.sqrt(101), places=6)
        self.assertAlmostEqual(matches[0].cosine_distance, 1 - matches[0].similarity)
        self.assertAlmostEqual(matches[2].similarity, -10 / np.sqrt(101), places=6)
        self.assertIn("lat", matches[0].to_dict()["metadata"])

    def test_batched_search_agrees_with_numpy_brute_force(self):
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(31, 16))
        queries = rng.normal(size=(7, 16))
        rows = [dict(self.rows[0], image_id=str(i)) for i in range(len(vectors))]
        index = PlaceIndex.build(vectors, rows)
        expected = (queries / np.linalg.norm(queries, axis=1, keepdims=True)) @ (
            vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).T
        for scores, matches in zip(expected, index.search(queries, k=8)):
            ids = np.argsort(-scores)[:8]
            self.assertEqual([m.index_id for m in matches], ids.tolist())
            np.testing.assert_allclose([m.similarity for m in matches], scores[ids], atol=1e-6)

    def test_k_is_clamped_to_database_size(self):
        matches = self.index.search([0, 1], k=100)[0]
        self.assertEqual(len(matches), 3)
        self.assertTrue(all(0 <= m.index_id < 3 for m in matches))

    def test_invalid_k_is_rejected(self):
        for k in (0, -1, 1.5, True):
            with self.subTest(k=k), self.assertRaisesRegex(ValueError, "k должен"):
                self.index.search([1, 0], k=k)

    def test_empty_query_batch_returns_empty_results(self):
        self.assertEqual(self.index.search(np.empty((0, 2))), [])

    def test_wrong_query_dimension_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Размерность"):
            self.index.search([1, 2, 3])

    def test_zero_and_nonfinite_vectors_are_rejected(self):
        for vector in ([0, 0], [np.nan, 1], [1, np.inf], [1e100, 0]):
            with self.subTest(vector=vector):
                with self.assertRaises(ValueError):
                    self.index.search(vector)
                with self.assertRaises(ValueError):
                    PlaceIndex.build(np.array([vector]), [self.rows[0]])

    def test_wrong_shapes_and_non_numeric_arrays_are_rejected(self):
        arrays = (np.ones(2), np.ones((1, 1, 2)), np.ones((2, 0)), np.ones((0, 2)),
                  np.array([["1", "2"]]), np.array([[1j, 2j]]), np.array([[True, False]]))
        for array in arrays:
            with self.subTest(shape=array.shape, dtype=array.dtype), self.assertRaises(ValueError):
                PlaceIndex.build(array, [self.rows[0]])

    def test_normalization_does_not_modify_inputs(self):
        vectors = self.vectors.copy()
        queries = np.array([[10.0, 1.0]])
        original_queries = queries.copy()
        PlaceIndex.build(vectors, self.rows).search(queries)
        np.testing.assert_array_equal(vectors, self.vectors)
        np.testing.assert_array_equal(queries, original_queries)

    def test_noncontiguous_vectors_are_supported(self):
        vectors = np.ones((3, 4), dtype=np.float64)[:, ::2]
        self.assertFalse(vectors.flags.c_contiguous)
        index = PlaceIndex.build(vectors, self.rows)
        self.assertEqual(index.size, 3)
        self.assertAlmostEqual(index.search([1, 1], k=1)[0][0].similarity, 1.0, places=6)

    def test_very_small_and_large_float32_values_are_normalized(self):
        for scale in (1e-30, 1e30):
            with self.subTest(scale=scale):
                index = PlaceIndex.build(np.array([[scale, 0]]), [self.rows[0]])
                self.assertAlmostEqual(index.search([scale, 0])[0][0].similarity, 1.0, places=6)

    def test_metadata_count_must_match_vectors(self):
        with self.assertRaisesRegex(ValueError, "Число векторов"):
            PlaceIndex.build(self.vectors, self.rows[:2])

    def test_duplicate_image_ids_are_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[1]["image_id"] = rows[0]["image_id"]
        with self.assertRaisesRegex(ValueError, "Повторяющийся image_id"):
            PlaceIndex.build(self.vectors, rows)

    def test_metadata_requires_valid_coordinates_and_identifiers(self):
        for key, value in (("image_id", ""), ("path", None), ("lat", 91), ("lon", -181),
                           ("lat", "bad"), ("lon", float("nan")), ("compass_angle", "bad")):
            rows = copy.deepcopy(self.rows)
            rows[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                PlaceIndex.build(self.vectors, rows)

    def test_metadata_is_copied_including_search_results(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["extra"] = {"note": "original"}
        index = PlaceIndex.build(self.vectors, rows)
        rows[0]["extra"]["note"] = "changed input"
        result = index.search([1, 0], k=1)[0][0]
        self.assertEqual(result.metadata["extra"]["note"], "original")
        result.metadata["extra"]["note"] = "changed result"
        self.assertEqual(index.search([1, 0], k=1)[0][0].metadata["extra"]["note"], "original")

    def test_csv_and_npy_files_preserve_row_order_and_large_ids(self):
        vectors, metadata = self.write_inputs()
        rows = read_metadata(metadata)
        self.assertEqual(rows, self.rows)
        index = PlaceIndex.from_files(vectors, metadata)
        self.assertEqual(index.search([0, 1], k=1)[0][0].metadata["image_id"], "9007199254740994")
        np.testing.assert_array_equal(load_embeddings(vectors), self.vectors)

    def test_pickle_and_npz_inputs_are_rejected(self):
        path = self.root / "objects.npy"
        np.save(path, np.array([{"unsafe": True}], dtype=object))
        with self.assertRaises(ValueError):
            load_embeddings(path)
        archive = self.root / "archive.npz"
        np.savez(archive, embeddings=self.vectors)
        with self.assertRaisesRegex(ValueError, ".npy"):
            load_embeddings(archive)

    def test_descriptor_id_is_checked_when_provided(self):
        self.index.search([1, 0], descriptor_id="model-a:crop033:320")
        with self.assertRaisesRegex(ValueError, "descriptor_id"):
            self.index.search([1, 0], descriptor_id="different-model")
        with self.assertRaisesRegex(ValueError, "descriptor_id"):
            PlaceIndex.build(self.vectors, self.rows, descriptor_id=" ")

    def test_saved_index_preserves_results_and_descriptor_id(self):
        bundle = self.saved_index()
        self.assertEqual({p.name for p in bundle.iterdir()}, {"index.faiss", "metadata.json", "manifest.json"})
        loaded = PlaceIndex.load(bundle)
        self.assertEqual(loaded.size, 3)
        self.assertEqual(loaded.dimension, 2)
        self.assertEqual(loaded.descriptor_id, self.index.descriptor_id)
        before = [[m.to_dict() for m in group] for group in self.index.search([[1, 0], [0, 1]])]
        after = [[m.to_dict() for m in group] for group in loaded.search([[1, 0], [0, 1]])]
        self.assertEqual(before, after)

    def test_index_bundle_can_be_stored_in_a_unicode_path(self):
        bundle = self.root / "индексы" / "проверка"
        self.index.save(bundle)
        restored = PlaceIndex.load(bundle)
        self.assertEqual(restored.search([1, 0], k=1)[0][0].metadata, self.rows[0])

    def test_save_refuses_to_overwrite_existing_directory(self):
        bundle = self.saved_index()
        before = (bundle / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.index.save(bundle)
        self.assertEqual(before, (bundle / "manifest.json").read_bytes())
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(FileExistsError):
            self.index.save(empty)

    def test_failed_save_does_not_leave_a_partial_bundle(self):
        with patch("src.search.index.faiss.write_index", side_effect=RuntimeError("disk failure")):
            with self.assertRaisesRegex(RuntimeError, "disk failure"):
                self.index.save(self.root / "bundle")
        self.assertFalse((self.root / "bundle").exists())
        self.assertFalse(list(self.root.glob(".bundle-*")))

    def test_corrupted_files_are_rejected_before_faiss_deserialization(self):
        for name in ("index.faiss", "metadata.json"):
            with self.subTest(name=name):
                bundle = self.root / name.replace(".", "-")
                self.index.save(bundle)
                with (bundle / name).open("ab") as f:
                    f.write(b"corruption")
                with patch("src.search.index.faiss.read_index") as read_index:
                    with self.assertRaisesRegex(ValueError, "SHA256"):
                        PlaceIndex.load(bundle)
                    read_index.assert_not_called()

    def test_manifest_version_shape_and_metric_are_validated(self):
        cases = ({"format_version": 999}, {"format_version": True}, {"dimension": 99},
                 {"size": 99}, {"metric": "l2"}, {"sha256": None})
        for i, changes in enumerate(cases):
            with self.subTest(changes=changes):
                bundle = self.root / f"bundle-{i}"
                self.index.save(bundle)
                self.edit_manifest(bundle, **changes)
                with self.assertRaises(ValueError):
                    PlaceIndex.load(bundle)

    def test_metadata_row_count_is_validated_even_with_matching_checksum(self):
        bundle = self.saved_index()
        path = bundle / "metadata.json"
        path.write_text(json.dumps(self.rows[:2]), encoding="utf-8")
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        manifest["sha256"]["metadata.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.edit_manifest(bundle, sha256=manifest["sha256"])
        with self.assertRaisesRegex(ValueError, "Число векторов"):
            PlaceIndex.load(bundle)

    def test_wrong_faiss_index_type_is_rejected_even_with_matching_checksum(self):
        bundle = self.saved_index()
        path = bundle / "index.faiss"
        wrong_index = faiss.IndexFlatL2(2)
        wrong_index.add(self.vectors.astype(np.float32))
        faiss.write_index(wrong_index, str(path))
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        manifest["sha256"]["index.faiss"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.edit_manifest(bundle, sha256=manifest["sha256"])
        with self.assertRaisesRegex(ValueError, "IndexFlatIP"):
            PlaceIndex.load(bundle)

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-m", "src.search", *map(str, args)],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
            env={**os.environ, "OMP_NUM_THREADS": "1"}, timeout=30,
        )

    def test_cli_build_and_query_json(self):
        vectors, metadata = self.write_inputs()
        bundle = self.root / "cli-index"
        built = self.run_cli("build", "--embeddings", vectors, "--metadata", metadata,
                             "--out", bundle, "--descriptor-id", "model-cli")
        self.assertEqual(built.returncode, 0, built.stderr)
        self.assertEqual(json.loads(built.stdout)["size"], 3)
        queries = self.root / "queries.npy"
        np.save(queries, np.array([[10, 1], [0, 2]]))
        result = self.run_cli("query", "--index", bundle, "--queries", queries,
                              "--top-k", 2, "--descriptor-id", "model-cli")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(len(payload["queries"]), 2)
        first = payload["queries"][0]["matches"][0]
        self.assertEqual(first["metadata"]["image_id"], self.rows[0]["image_id"])
        self.assertEqual(first["metadata"]["lat"], self.rows[0]["lat"])
        self.assertEqual(len(payload["queries"][1]["matches"]), 2)
        output = self.root / "results" / "matches.json"
        saved = self.run_cli("query", "--index", bundle, "--queries", queries, "--output", output)
        self.assertEqual(saved.returncode, 0, saved.stderr)
        self.assertEqual(saved.stdout, "")
        self.assertEqual(len(json.loads(output.read_text(encoding="utf-8"))["queries"]), 2)

    def test_cli_errors_are_actionable_without_tracebacks(self):
        bundle = self.saved_index()
        queries = self.root / "queries.npy"
        np.save(queries, np.ones((1, 7)))
        result = self.run_cli("query", "--index", bundle, "--queries", queries)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Размерность", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        result = self.run_cli("query", "--index", bundle, "--queries", queries, "--top-k", 0)
        self.assertEqual(result.returncode, 2)
        self.assertIn("k должен", result.stderr)


if __name__ == "__main__":
    unittest.main()
