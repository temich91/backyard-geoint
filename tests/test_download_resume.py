"""Регрессии докачки Mapillary; без токена и сетевых запросов.

Запуск: python -m unittest discover -s tests -v
Нужны только зависимости сбора датасета, PyTorch не используется.
"""

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.data import build_dataset as loader


class DownloadResumeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.args = SimpleNamespace(workers=2, token="test-token", max_side=512)
        self.output = self.enterContext(redirect_stdout(io.StringIO()))
        self.enterContext(patch.object(loader, "tqdm", side_effect=lambda it, **kw: it))

    def cached(self, image_id, legacy=False):
        if legacy:
            path = self.root / "images" / str(image_id)[-2:] / f"{image_id}.jpg"
        else:
            path = self.root / f"{image_id}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"already downloaded")
        return path

    def image(self, image_id):
        return {
            "id": image_id, "lat": 54.77, "lon": 56.06,
            "compass_angle": 90.0, "sequence_id": f"seq-{image_id}",
        }

    def metadata(self):
        with (self.root / "metadata.csv").open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    def save_download(self, img, path, *args):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"new download")

    def test_new_images_use_flat_layout(self):
        self.assertEqual(loader.image_path(self.root, 12345), self.root / "12345.jpg")

    def test_existing_legacy_image_is_reused(self):
        path = self.cached(12345, legacy=True)
        self.assertEqual(loader.image_path(self.root, 12345), path)

    def test_flat_image_has_priority_when_both_exist(self):
        self.cached(12345, legacy=True)
        path = self.cached(12345)
        self.assertEqual(loader.image_path(self.root, 12345), path)

    def test_temporary_image_is_not_a_cache_hit(self):
        (self.root / "12345.tmp.jpg").write_bytes(b"incomplete")
        self.assertEqual(loader.image_path(self.root, 12345), self.root / "12345.jpg")

    def test_fully_cached_plan_rebuilds_metadata_without_api_or_downloads(self):
        flat = self.cached(12345)
        legacy = self.cached(23456, legacy=True)
        plan = [("ref", self.image(12345)), ("query", self.image(23456))]
        with patch.object(loader, "get_thumb_url") as probe, \
                patch.object(loader, "download_one") as download:
            loader.download_all(plan, self.root, self.args)
        probe.assert_not_called()
        download.assert_not_called()
        rows = {row["image_id"]: row for row in self.metadata()}
        self.assertEqual(rows["12345"]["path"], flat.relative_to(self.root).as_posix())
        self.assertEqual(rows["23456"]["path"], legacy.relative_to(self.root).as_posix())
        self.assertIn("осталось скачать: 0", self.output.getvalue())

    def test_only_missing_images_are_queued_and_cached_rows_are_preserved(self):
        self.cached(12345, legacy=True)
        self.cached(23456)
        plan = [("ref", self.image(12345)), ("ref", self.image(23456)),
                ("query", self.image(34567))]
        with patch.object(loader, "get_thumb_url", return_value="test-url") as probe, \
                patch.object(loader, "download_one", side_effect=self.save_download) as download:
            loader.download_all(plan, self.root, self.args)
        self.assertEqual(download.call_count, 1)
        self.assertEqual(download.call_args.args[0]["id"], 34567)
        self.assertEqual(probe.call_args.args[1], 34567)
        self.assertEqual(len(self.metadata()), 3)
        self.assertEqual((self.root / "34567.jpg").read_bytes(), b"new download")
        self.assertIn("осталось скачать: 1", self.output.getvalue())

    def test_failed_image_is_retried_without_redownloading_successes(self):
        self.cached(12345, legacy=True)
        plan = [("ref", self.image(12345)), ("query", self.image(23456))]
        with patch.object(loader, "get_thumb_url", return_value="test-url"), \
                patch.object(loader, "download_one", side_effect=OSError("interrupted")):
            loader.download_all(plan, self.root, self.args)
        self.assertEqual(len(self.metadata()), 1)
        self.assertIn("23456", (self.root / "failed.txt").read_text(encoding="utf-8"))
        with patch.object(loader, "get_thumb_url", return_value="test-url"), \
                patch.object(loader, "download_one", side_effect=self.save_download) as download:
            loader.download_all(plan, self.root, self.args)
        self.assertEqual(download.call_count, 1)
        self.assertEqual(download.call_args.args[0]["id"], 23456)
        self.assertEqual(len(self.metadata()), 2)
        self.assertFalse((self.root / "failed.txt").exists())

    def test_moving_entire_dataset_preserves_both_cache_layouts(self):
        self.cached(12345, legacy=True)
        self.cached(23456)
        (self.root / "tiles").mkdir()
        (self.root / "tiles" / "0_0.pbf").write_bytes(b"tile")
        moved = self.root / "new-parent" / "raw"
        moved.parent.mkdir()
        # Имитация переноса data/geo -> data/raw без изменения внутренней структуры.
        old = self.root / "old"
        old.mkdir()
        for name in ("images", "23456.jpg", "tiles"):
            (self.root / name).rename(old / name)
        old.rename(moved)
        plan = [("ref", self.image(12345)), ("query", self.image(23456))]
        with patch.object(loader, "get_thumb_url") as probe, \
                patch.object(loader, "download_one") as download:
            loader.download_all(plan, moved, self.args)
        probe.assert_not_called()
        download.assert_not_called()
        with (moved / "metadata.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(all((moved / row["path"]).is_file() for row in rows))
        self.assertEqual((moved / "tiles" / "0_0.pbf").read_bytes(), b"tile")

    def test_download_one_skips_an_existing_image(self):
        path = self.cached(12345, legacy=True)
        api, cdn = Mock(), Mock()
        loader.download_one(self.image(12345), path, api, cdn, self.args)
        api.get.assert_not_called()
        cdn.get.assert_not_called()

    def test_real_jpeg_download_is_resized_and_not_requested_again(self):
        _, encoded = loader.cv2.imencode(".jpg", loader.np.zeros((16, 32, 3), dtype=loader.np.uint8))
        api, cdn = Mock(), Mock()
        api.get.return_value = SimpleNamespace(
            status_code=200, json=lambda: {"thumb_1024_url": "https://example.test/image.jpg"},
        )
        cdn.get.return_value = SimpleNamespace(status_code=200, content=encoded.tobytes())
        self.args.max_side = 8
        plan = [("ref", self.image(12345))]
        with patch.object(loader, "make_session", side_effect=[api, cdn, api, cdn]):
            loader.download_all(plan, self.root, self.args)
            path = self.root / "12345.jpg"
            self.assertEqual(loader.cv2.imread(str(path)).shape[:2], (4, 8))
            self.assertFalse((self.root / "12345.tmp.jpg").exists())
            self.assertEqual(cdn.get.call_count, 1)
            api.reset_mock()
            cdn.reset_mock()
            loader.download_all(plan, self.root, self.args)
        api.get.assert_not_called()
        cdn.get.assert_not_called()
        self.assertEqual(len(self.metadata()), 1)

    def test_cached_tiles_are_processed_in_stable_order(self):
        cache_dir = self.root / "tiles"
        cache_dir.mkdir()
        (cache_dir / "0_0.pbf").write_bytes(b"first")
        (cache_dir / "1_0.pbf").write_bytes(b"second")
        session = Mock()
        with patch.object(loader, "lonlat_to_tile", side_effect=[(0, 0), (1, 0)]), \
                patch.object(loader, "make_session", return_value=session), \
                patch.object(loader, "as_completed", side_effect=lambda fs: reversed(list(fs))):
            tiles = list(loader.iter_tiles((0, 0, 1, 1), "test-token", cache_dir, 2))
        self.assertEqual(tiles, [((0, 0), b"first"), ((1, 0), b"second")])
        session.get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
