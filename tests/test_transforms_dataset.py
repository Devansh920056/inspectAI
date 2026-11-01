import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from inspectai.data import DatasetError  # noqa: E402
from inspectai.data.dataloader import build_dataloader, move_batch  # noqa: E402
from inspectai.data.dataset import MVTecDataset  # noqa: E402
from inspectai.data.transforms import (  # noqa: E402
    IMAGENET_MEAN, IMAGENET_STD, build_image_transform, build_mask_transform, denormalize,
)
from synthetic import make_synthetic_mvtec  # noqa: E402


class TransformTests(unittest.TestCase):
    def test_image_transform_shape_dtype_range(self):
        img = Image.fromarray(np.full((50, 70, 3), 255, dtype=np.uint8))
        t = build_image_transform(32, normalize=False)(img)
        self.assertEqual(tuple(t.shape), (3, 32, 32))
        self.assertEqual(t.dtype, torch.float32)
        self.assertTrue(torch.allclose(t, torch.ones_like(t)))

    def test_normalization_math(self):
        img = Image.fromarray(np.full((8, 8, 3), 255, dtype=np.uint8))
        t = build_image_transform(8)(img)
        for c in range(3):
            self.assertAlmostEqual(t[c, 0, 0].item(), (1 - IMAGENET_MEAN[c]) / IMAGENET_STD[c], places=5)

    def test_custom_mean_std(self):
        img = Image.fromarray(np.full((8, 8, 3), 51, dtype=np.uint8))  # 0.2
        t = build_image_transform(8, mean=(0.2, 0.2, 0.2), std=(0.5, 0.5, 0.5))(img)
        self.assertTrue(torch.allclose(t, torch.zeros_like(t), atol=1e-6))

    def test_denormalize_roundtrip(self):
        rng = np.random.default_rng(0)
        img = Image.fromarray(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8))
        plain = build_image_transform(16, normalize=False)(img)
        norm = build_image_transform(16)(img)
        self.assertTrue(torch.allclose(denormalize(norm), plain, atol=1e-5))

    def test_mask_transform_binary_and_nearest(self):
        m = np.zeros((40, 40), dtype=np.uint8)
        m[10:20, 10:20] = 255
        t = build_mask_transform(20)(Image.fromarray(m))
        self.assertEqual(tuple(t.shape), (1, 20, 20))
        self.assertEqual(t.dtype, torch.float32)
        self.assertEqual(set(t.unique().tolist()), {0.0, 1.0})

    def test_rectangular_target_size(self):
        img = Image.new("RGB", (40, 40))
        self.assertEqual(tuple(build_image_transform((24, 32))(img).shape), (3, 24, 32))


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.expected = make_synthetic_mvtec(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_train_dataset_normal_only_zero_masks(self):
        ds = MVTecDataset(self.root, "bottle", "train", image_size=32)
        self.assertEqual(len(ds), 6)
        self.assertEqual(ds.num_defective, 0)
        for i in range(len(ds)):
            item = ds[i]
            self.assertEqual(item["label"].item(), 0)
            self.assertEqual(item["mask"].sum().item(), 0.0)
            self.assertEqual(item["defect_type"], "good")

    def test_test_dataset_labels_and_masks_consistent(self):
        ds = MVTecDataset(self.root, "hazelnut", "test", image_size=40)
        self.assertEqual((ds.num_normal, ds.num_defective), (2, 5))
        for i in range(len(ds)):
            item = ds[i]
            self.assertEqual(tuple(item["image"].shape), (3, 40, 40))
            self.assertEqual(tuple(item["mask"].shape), (1, 40, 40))
            self.assertEqual(item["image"].dtype, torch.float32)
            self.assertEqual(item["label"].dtype, torch.int64)
            # image-level label and pixel-level mask agree but are separate fields
            self.assertEqual(item["label"].item() == 1, item["mask"].sum().item() > 0)

    def test_mask_area_survives_resize(self):
        ds = MVTecDataset(self.root, "carpet", "test", image_size=48)  # 96 -> 48
        item = next(ds[i] for i in range(len(ds)) if ds.samples[i].is_defective)
        frac = item["mask"].mean().item()
        self.assertAlmostEqual(frac, self.expected["carpet"]["box_fraction"], delta=0.01)

    def test_no_leakage_between_train_and_test(self):
        tr = {s.image_path for s in MVTecDataset(self.root, "bottle", "train", image_size=16).samples}
        te = {s.image_path for s in MVTecDataset(self.root, "bottle", "test", image_size=16).samples}
        self.assertFalse(tr & te)

    def test_originals_unchanged_by_loading(self):
        files = sorted((self.root / "bottle").rglob("*.png"))
        before = {f: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
        ds = MVTecDataset(self.root, "bottle", "test", image_size=16)
        for i in range(len(ds)):
            ds[i]
        after = {f: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
        self.assertEqual(before, after)

    def test_missing_mask_raises_at_construction(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_synthetic_mvtec(root, categories=("bottle",))
            (root / "bottle" / "ground_truth" / "broken" / "000_mask.png").unlink()
            with self.assertRaises(DatasetError):
                MVTecDataset(root, "bottle", "test")

    def test_verify_images_catches_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_synthetic_mvtec(root, categories=("bottle",))
            (root / "bottle" / "train" / "good" / "000.png").write_bytes(b"garbage")
            MVTecDataset(root, "bottle", "train")  # lazy: not detected without verify
            with self.assertRaises(DatasetError):
                MVTecDataset(root, "bottle", "train", verify_images=True)


class DataLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        make_synthetic_mvtec(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_batch_shapes_and_types(self):
        ds = MVTecDataset(self.root, "bottle", "test", image_size=32)
        batch = next(iter(build_dataloader(ds, batch_size=4)))
        self.assertEqual(tuple(batch["image"].shape), (4, 3, 32, 32))
        self.assertEqual(tuple(batch["label"].shape), (4,))
        self.assertEqual(tuple(batch["mask"].shape), (4, 1, 32, 32))
        self.assertIsInstance(batch["defect_type"], list)
        self.assertEqual(len(batch["path"]), 4)

    def test_last_partial_batch_and_drop_last(self):
        ds = MVTecDataset(self.root, "bottle", "train", image_size=16)  # 6 images
        self.assertEqual([b["image"].shape[0] for b in build_dataloader(ds, batch_size=4)], [4, 2])
        self.assertEqual([b["image"].shape[0] for b in build_dataloader(ds, batch_size=4, drop_last=True)], [4])

    def test_shuffle_defaults(self):
        tr = MVTecDataset(self.root, "bottle", "train", image_size=16)
        te = MVTecDataset(self.root, "bottle", "test", image_size=16)
        self.assertIsInstance(build_dataloader(tr).sampler, torch.utils.data.RandomSampler)
        self.assertIsInstance(build_dataloader(te).sampler, torch.utils.data.SequentialSampler)

    def test_seeded_shuffle_is_reproducible(self):
        ds = MVTecDataset(self.root, "bottle", "train", image_size=16)
        order = lambda seed: [p for b in build_dataloader(ds, batch_size=2, seed=seed) for p in b["path"]]
        self.assertEqual(order(3), order(3))
        self.assertEqual(sorted(order(3)), sorted(order(4)))

    def test_multiprocess_workers(self):
        ds = MVTecDataset(self.root, "bottle", "train", image_size=16)
        batches = list(build_dataloader(ds, batch_size=3, num_workers=2))
        self.assertEqual(sum(b["image"].shape[0] for b in batches), 6)

    def test_move_batch_keeps_strings(self):
        ds = MVTecDataset(self.root, "bottle", "test", image_size=16)
        b = move_batch(next(iter(build_dataloader(ds, batch_size=2))), "cpu")
        self.assertIsInstance(b["path"], list)
        self.assertEqual(b["image"].device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
