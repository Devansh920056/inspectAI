import hashlib
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from inspectai.data import (  # noqa: E402
    CorruptImageError, DatasetError, index_category, load_image, load_mask,
    read_image_info, validate_category,
)
from synthetic import make_synthetic_mvtec  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.expected = make_synthetic_mvtec(self.root)

    def tearDown(self):
        self._tmp.cleanup()


class IndexTests(Base):
    def test_counts_and_labels(self):
        e = self.expected["bottle"]
        train = index_category(self.root, "bottle", "train")
        test = index_category(self.root, "bottle", "test")
        self.assertEqual(len(train), e["n_train"])
        self.assertTrue(all(s.label == 0 and s.mask_path is None for s in train))
        self.assertEqual(sum(s.label == 0 for s in test), e["n_test_good"])
        self.assertEqual(sum(s.label == 1 for s in test), e["n_defective"])

    def test_defective_samples_have_masks_and_normal_do_not(self):
        for s in index_category(self.root, "carpet", "test"):
            self.assertEqual(s.mask_path is not None, s.is_defective)
            if s.mask_path is not None:
                self.assertTrue(s.mask_path.is_file())

    def test_defect_types(self):
        types = {s.defect_type for s in index_category(self.root, "hazelnut", "test")}
        self.assertEqual(types, {"good", "broken", "stain"})

    def test_deterministic_order(self):
        a = [s.image_path for s in index_category(self.root, "bottle", "test")]
        self.assertEqual(a, sorted(a, key=lambda p: (p.parent.name, p.name)))

    def test_hidden_files_ignored(self):
        (self.root / "bottle" / "train" / "good" / "._000.png").write_bytes(b"junk")
        (self.root / "bottle" / "train" / "good" / ".DS_Store").write_bytes(b"junk")
        self.assertEqual(len(index_category(self.root, "bottle", "train")), self.expected["bottle"]["n_train"])

    def test_missing_category(self):
        with self.assertRaises(DatasetError):
            index_category(self.root, "wood", "train")

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            index_category(self.root, "not_a_category", "train")
        with self.assertRaises(ValueError):
            index_category(self.root, "bottle", "val")

    def test_train_with_defect_folder_is_rejected(self):
        (self.root / "bottle" / "train" / "broken").mkdir()
        with self.assertRaises(DatasetError):
            index_category(self.root, "bottle", "train")

    def test_train_never_lists_test_images(self):
        train = {s.image_path for s in index_category(self.root, "bottle", "train")}
        test = {s.image_path for s in index_category(self.root, "bottle", "test")}
        self.assertFalse(train & test)
        self.assertTrue(all("/train/" in str(p) for p in train))


class LoadingTests(Base):
    def test_load_image_is_rgb_and_leaves_file_unchanged(self):
        path = index_category(self.root, "bottle", "train")[0].image_path
        before = sha(path)
        img = load_image(path)
        self.assertEqual(img.mode, "RGB")
        img.putpixel((0, 0), (1, 2, 3))  # mutate the in-memory copy
        self.assertEqual(sha(path), before)

    def test_grayscale_converted_to_rgb(self):
        p = self.root / "gray.png"
        Image.new("L", (10, 10), 7).save(p)
        self.assertEqual(load_image(p).mode, "RGB")

    def test_load_mask_single_channel_values(self):
        s = next(s for s in index_category(self.root, "bottle", "test") if s.is_defective)
        m = load_mask(s.mask_path)
        self.assertEqual(m.mode, "L")
        self.assertEqual(set(np.asarray(m).ravel().tolist()), {0, 255})

    def test_read_image_info(self):
        path = index_category(self.root, "hazelnut", "train")[0].image_path
        info = read_image_info(path)
        self.assertEqual((info.width, info.height, info.mode, info.format), (80, 80, "RGB", "PNG"))

    def test_corrupt_file_detected(self):
        p = self.root / "bad.png"
        p.write_bytes(b"this is not a png")
        with self.assertRaises(CorruptImageError):
            read_image_info(p)
        with self.assertRaises(CorruptImageError):
            load_image(p)

    def test_truncated_file_detected(self):
        src = index_category(self.root, "bottle", "train")[0].image_path
        p = self.root / "trunc.png"
        p.write_bytes(src.read_bytes()[:-40])
        with self.assertRaises(CorruptImageError):
            read_image_info(p, deep=True)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_image(self.root / "nope.png")


class ValidationTests(Base):
    def test_clean_dataset_is_ok(self):
        r = validate_category(self.root, "bottle")
        self.assertTrue(r.ok, r.to_dict())
        self.assertEqual(r.n_images_checked, 6 + 2 + 5)

    def test_missing_mask_reported(self):
        (self.root / "bottle" / "ground_truth" / "broken" / "000_mask.png").unlink()
        r = validate_category(self.root, "bottle")
        self.assertEqual(len(r.missing_masks), 1)
        self.assertFalse(r.ok)

    def test_orphan_mask_reported(self):
        gt = self.root / "bottle" / "ground_truth" / "broken"
        shutil.copy(gt / "000_mask.png", gt / "099_mask.png")
        self.assertEqual(len(validate_category(self.root, "bottle").orphan_masks), 1)

    def test_corrupt_image_reported(self):
        victim = self.root / "bottle" / "train" / "good" / "000.png"
        victim.write_bytes(b"garbage")
        r = validate_category(self.root, "bottle")
        self.assertEqual([p for p, _ in r.corrupt_images], [victim])

    def test_mask_size_mismatch_reported(self):
        mp = self.root / "bottle" / "ground_truth" / "stain" / "000_mask.png"
        Image.new("L", (10, 10), 0).save(mp)
        self.assertEqual(len(validate_category(self.root, "bottle").size_mismatches), 1)

    def test_missing_category_reported_not_raised(self):
        r = validate_category(self.root, "wood")
        self.assertIsNotNone(r.structural_error)
        self.assertFalse(r.ok)


if __name__ == "__main__":
    unittest.main()
