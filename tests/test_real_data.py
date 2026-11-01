"""Checks against the real MVTec AD files. Skipped if the data is not present.

Set INSPECTAI_DATA_ROOT to override the default location (<repo>/data/mvtec_ad).
Expected counts are the published MVTec AD benchmark numbers; a mismatch means
the download is incomplete or modified.
"""

import os
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from inspectai.data import index_category, validate_category  # noqa: E402

ROOT = Path(os.environ.get("INSPECTAI_DATA_ROOT", REPO / "data" / "mvtec_ad"))

PUBLISHED = {
    # category: (train_good, test_good, {defect: count}, image_side_px)
    "bottle": (209, 20, {"broken_large": 20, "broken_small": 22, "contamination": 21}, 900),
    "hazelnut": (391, 40, {"crack": 18, "cut": 17, "hole": 18, "print": 17}, 1024),
    "carpet": (280, 28, {"color": 19, "cut": 17, "hole": 17, "metal_contamination": 17, "thread": 19}, 1024),
}


@unittest.skipUnless(ROOT.is_dir(), f"MVTec AD not found at {ROOT}")
class RealDataTests(unittest.TestCase):
    def test_counts_match_published_benchmark(self):
        for cat, (n_train, n_good, defects, _) in PUBLISHED.items():
            with self.subTest(category=cat):
                train = index_category(ROOT, cat, "train")
                test = index_category(ROOT, cat, "test")
                self.assertEqual(len(train), n_train)
                self.assertEqual(sum(s.label == 0 for s in test), n_good)
                got = {}
                for s in test:
                    if s.is_defective:
                        got[s.defect_type] = got.get(s.defect_type, 0) + 1
                self.assertEqual(got, defects)

    def test_no_missing_corrupt_or_mismatched_files(self):
        for cat in PUBLISHED:
            with self.subTest(category=cat):
                report = validate_category(ROOT, cat, check_images=True)
                self.assertTrue(report.ok, report.to_dict())

    def test_image_side_lengths(self):
        from inspectai.data import read_image_info

        for cat, (*_, side) in PUBLISHED.items():
            with self.subTest(category=cat):
                info = read_image_info(index_category(ROOT, cat, "train")[0].image_path, deep=False)
                self.assertEqual(info.size, (side, side))


if __name__ == "__main__":
    unittest.main()
