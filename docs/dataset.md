# MVTec AD dataset documentation

## What it is

MVTec AD is a benchmark for **unsupervised anomaly detection** in industrial inspection:
15 categories (10 objects, 5 textures), each with high-resolution images of defect-free
and defective items. InspectAI uses three categories first: `bottle`, `hazelnut` (objects) and
`carpet` (texture).

**Licence:** CC BY-NC-SA 4.0 — non-commercial use, attribution required, share-alike.
The data is not redistributed in this repo (`data/` is git-ignored).

## Getting the data

1. Open the official dataset page: <https://www.mvtec.com/research-teaching/datasets/mvtec-ad>
   (the original MVTec AD, not MVTec AD 2), read and accept the licence, and download
   either the full archive or the per-category archives for `bottle`, `hazelnut` and `carpet`.
2. Extract so the layout is `data/mvtec_ad/<category>/...`:
   ```bash
   mkdir -p data/mvtec_ad
   tar -xf bottle.tar.xz   -C data/mvtec_ad
   tar -xf hazelnut.tar.xz -C data/mvtec_ad
   tar -xf carpet.tar.xz   -C data/mvtec_ad
   ```
3. Check: `data/mvtec_ad/bottle/train/good/` must exist. Keep the data elsewhere if you prefer and set
   `INSPECTAI_DATA_ROOT=/path/to/mvtec_ad`.
4. Validate and generate the report: `python scripts/dataset_report.py --figures`.

## Folder structure

```
<root>/<category>/
    train/good/                  normal images only
    test/good/                   normal images
    test/<defect_type>/          defective images
    ground_truth/<defect_type>/  masks, <image_stem>_mask.png, one per defective test image
```

* Training data contains **no defects**. Defects occur only in `test/`. The model must learn "normal"
  and detect deviations (the one-class / unsupervised setting).
* **Image-level label** (normal vs defective) comes from the folder name. **Pixel-level label** comes from
  the mask PNG (0 = normal pixel, 255 = defective pixel). Normal images have no mask.
* Test sets are dominated by defective images, unlike training.

## Expected sizes (published benchmark values)

These come from the MVTec AD paper/website and are what `tests/test_real_data.py` checks. The authoritative
numbers for *your* copy are in `reports/dataset_statistics.md` once you run the report.

| Category | Train (normal) | Test normal | Test defective | Defect types | Image size |
|---|---:|---:|---:|---|---|
| bottle | 209 | 20 | 63 | broken_large 20, broken_small 22, contamination 21 | 900×900 |
| hazelnut | 391 | 40 | 70 | crack 18, cut 17, hole 18, print 17 | 1024×1024 |
| carpet | 280 | 28 | 89 | color 19, cut 17, hole 17, metal_contamination 17, thread 19 | 1024×1024 |

## Code map

| Module | Purpose |
|---|---|
| `inspectai.data.loading` | `load_image` (RGB copy), `load_mask` (single channel), `read_image_info`; raises `CorruptImageError` |
| `inspectai.data.layout` | `index_category` → `Sample` records; `validate_category` → `ValidationReport` |
| `inspectai.data.stats` | counts, size/format inventory, mask-area stats, train-only mean/std, JSON + Markdown report |
| `inspectai.data.transforms` | image and mask transform builders, `denormalize`, `TRANSFORM_DOCS` |
| `inspectai.data.dataset` | `MVTecDataset` |
| `inspectai.data.dataloader` | `build_dataloader`, `move_batch` |
| `inspectai.data.viz` | sample and batch plots |

```python
from inspectai.data.dataset import MVTecDataset
from inspectai.data.dataloader import build_dataloader

train = MVTecDataset("data/mvtec_ad", "bottle", split="train", image_size=256)
loader = build_dataloader(train, batch_size=16)          # shuffled, seeded
batch = next(iter(loader))
batch["image"].shape   # (16, 3, 256, 256) float32, ImageNet-normalised
batch["label"].shape   # (16,)             int64, image-level (all 0 in train)
batch["mask"].shape    # (16, 1, 256, 256) float32 in {0,1}, pixel-level
```

## Preprocessing (documented transformations)

| Step | Image | Mask |
|---|---|---|
| Load | Pillow, fully decoded, converted to RGB in memory | Pillow, single channel `L` |
| Resize | `image_size × image_size`, bilinear + antialias | same size, **nearest-neighbour** |
| To tensor | `(H,W,C)` uint8 0–255 → `(C,H,W)` float32 0–1 | `> 0` → `{0.0, 1.0}`, shape `(1,H,W)` |
| Normalise | `(x − mean)/std`, ImageNet statistics by default | — |

* Default `image_size` is 256. These three categories have square images, so resizing does not distort the
  aspect ratio; verify this for other categories before reusing the defaults.
* ImageNet statistics are the default because pretrained backbones expect them. Dataset-specific statistics
  (computed from `train/good` only) are in the report and can be passed to `build_image_transform`.
* No data augmentation is applied.
* Original files are never written to; loaders return in-memory copies.

## Integrity and leakage rules

* `MVTecDataset(split="train")` only reads `<category>/train/good`; `index_category` rejects a train split that
  contains anything other than `good`.
* There is no code path that mixes `train` and `test`. Do not tune thresholds or hyper-parameters on
  `test`; a validation set, if needed, must be carved out of `train/good` in a later stage.
* Normalisation statistics and any other fitted preprocessing use **train images only**.
* Validation (`validate_category`, `scripts/dataset_report.py`) fully decodes every image and mask and reports:
  missing directories, missing masks, orphan masks, corrupted/truncated files, mask/image size mismatches.
  `MVTecDataset` raises `DatasetError` at construction for missing masks (and for corrupted files with
  `verify_images=True`).
* macOS junk files (`.DS_Store`, `._*.png`) are ignored.

## Tests

`python -m unittest discover tests` — unit tests use a small **synthetic** dataset in the MVTec folder layout
(`tests/synthetic.py`), never real data. `tests/test_real_data.py` checks the actual files (published counts,
no corruption, image sizes) and is skipped automatically if the data is absent.
