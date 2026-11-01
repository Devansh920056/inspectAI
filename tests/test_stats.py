import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from inspectai.data.layout import index_category  # noqa: E402
from inspectai.data.loading import load_image  # noqa: E402
from inspectai.data.stats import category_statistics, dataset_statistics, to_markdown, write_report  # noqa: E402
from synthetic import make_synthetic_mvtec  # noqa: E402


class StatsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.expected = make_synthetic_mvtec(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_counts(self):
        c = category_statistics(self.root, "carpet")
        e = self.expected["carpet"]
        self.assertEqual(c["train"]["n_images"], e["n_train"])
        self.assertEqual(c["train"]["n_defective"], 0)
        self.assertEqual(c["test"]["n_normal"], e["n_test_good"])
        self.assertEqual(c["test"]["n_defective"], e["n_defective"])
        self.assertEqual(c["defect_types"], e["defects"])
        self.assertTrue(c["validation"]["ok"])

    def test_inventory(self):
        c = category_statistics(self.root, "hazelnut")
        self.assertEqual(c["images"]["formats"], {"PNG": 6 + 2 + 5})
        self.assertEqual(c["images"]["modes"], {"RGB": 13})
        self.assertEqual(c["images"]["sizes"], {"80x80": 13})

    def test_mask_area_fraction(self):
        c = category_statistics(self.root, "bottle")
        a = c["masks"]["anomaly_area_fraction"]
        self.assertAlmostEqual(a["mean"], self.expected["bottle"]["box_fraction"], places=6)
        self.assertEqual(c["masks"]["n_masks"], 5)
        self.assertEqual(c["masks"]["n_empty_masks"], 0)

    def test_pixel_stats_match_direct_computation_on_train_only(self):
        c = category_statistics(self.root, "bottle")
        arr = np.concatenate([
            np.asarray(load_image(s.image_path), dtype=np.float64).reshape(-1, 3) / 255.0
            for s in index_category(self.root, "bottle", "train")
        ])
        np.testing.assert_allclose(c["train_pixel_stats"]["mean"], arr.mean(axis=0), atol=1e-5)
        np.testing.assert_allclose(c["train_pixel_stats"]["std"], arr.std(axis=0), atol=1e-5)
        self.assertEqual(c["train_pixel_stats"]["n_images"], 6)

    def test_missing_category_gives_error_entry(self):
        c = category_statistics(self.root, "wood")
        self.assertIn("error", c)

    def test_root_label(self):
        stats = dataset_statistics(self.root, ["bottle"], pixel_stats=False, root_label="data/mvtec_ad")
        self.assertEqual(stats["root"], "data/mvtec_ad")
        self.assertIn("`data/mvtec_ad`", to_markdown(stats))

    def test_report_files(self):
        stats = dataset_statistics(self.root, ["bottle", "hazelnut", "carpet"])
        j, m = write_report(stats, self.root / "out")
        self.assertEqual(json.loads(j.read_text())["categories"]["bottle"]["test"]["n_defective"], 5)
        text = m.read_text()
        for name in ("bottle", "hazelnut", "carpet"):
            self.assertIn(f"## {name}", text)
        self.assertEqual(text, to_markdown(stats))


if __name__ == "__main__":
    unittest.main()
