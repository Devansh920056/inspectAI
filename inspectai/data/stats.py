"""Dataset statistics and report generation (torch-free).

All pixel statistics used for normalisation are computed on the *train* split
only, so no test information reaches preprocessing.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .layout import GOOD, Sample, index_category, validate_category
from .loading import CorruptImageError, load_image, load_mask, read_image_info


def _size_key(info) -> str:
    return f"{info.width}x{info.height}"


def _inventory(paths: list[Path]) -> dict:
    """Header-only inventory of formats / modes / sizes."""
    formats, modes, sizes, unreadable = Counter(), Counter(), Counter(), 0
    for p in paths:
        try:
            info = read_image_info(p, deep=False)
        except (CorruptImageError, FileNotFoundError):
            unreadable += 1
            continue
        formats[str(info.format)] += 1
        modes[info.mode] += 1
        sizes[_size_key(info)] += 1
    return {"formats": dict(formats), "modes": dict(modes), "sizes": dict(sizes), "unreadable": unreadable}


def _summary(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def train_pixel_stats(train_samples: list[Sample]) -> dict:
    """Per-channel mean/std over all *train* pixels, on a [0, 1] scale (RGB)."""
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    count = 0
    for s in train_samples:
        arr = np.asarray(load_image(s.image_path, "RGB"), dtype=np.float64).reshape(-1, 3) / 255.0
        total += arr.sum(axis=0)
        total_sq += (arr ** 2).sum(axis=0)
        count += arr.shape[0]
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean ** 2, 0.0))
    return {
        "mean": [round(float(x), 5) for x in mean],
        "std": [round(float(x), 5) for x in std],
        "n_images": len(train_samples),
        "source": "train/good only, pixel values scaled to [0, 1], RGB order",
    }


def category_statistics(root: str | Path, category: str, pixel_stats: bool = True) -> dict:
    root = Path(root)
    report = validate_category(root, category, check_images=True)
    out: dict = {"category": category, "validation": report.to_dict()}
    if report.structural_error:
        out["error"] = report.structural_error
        return out

    train = index_category(root, category, "train")
    test = index_category(root, category, "test")
    test_normal = [s for s in test if s.label == 0]
    test_defective = [s for s in test if s.label == 1]
    defect_counts = Counter(s.defect_type for s in test_defective)

    out["train"] = {"n_images": len(train), "n_normal": len(train), "n_defective": 0}
    out["test"] = {
        "n_images": len(test),
        "n_normal": len(test_normal),
        "n_defective": len(test_defective),
        "defect_ratio": round(len(test_defective) / len(test), 4),
    }
    out["defect_types"] = dict(sorted(defect_counts.items()))
    out["images"] = _inventory([s.image_path for s in train + test])

    # Pixel-level: fraction of each image that is defective
    fractions: dict[str, list[float]] = {}
    mask_paths = []
    for s in test_defective:
        if s.mask_path is not None and s.mask_path.is_file():
            mask_paths.append(s.mask_path)
            try:
                m = np.asarray(load_mask(s.mask_path)) > 0
            except CorruptImageError:
                continue
            fractions.setdefault(s.defect_type, []).append(float(m.mean()))
    all_fr = [f for v in fractions.values() for f in v]
    out["masks"] = {
        "n_masks": len(mask_paths),
        "inventory": _inventory(mask_paths),
        "anomaly_area_fraction": _summary(all_fr),
        "anomaly_area_fraction_by_defect": {k: _summary(v) for k, v in sorted(fractions.items())},
        "n_empty_masks": sum(1 for f in all_fr if f == 0.0),
    }
    if pixel_stats:
        out["train_pixel_stats"] = train_pixel_stats(train)
    return out


def dataset_statistics(
    root: str | Path,
    categories: list[str] | tuple[str, ...],
    pixel_stats: bool = True,
    root_label: str | None = None,
) -> dict:
    """``root_label`` is the path recorded in the report (default: ``root`` as given);
    pass a repo-relative label to keep machine-specific absolute paths out of reports."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": root_label if root_label is not None else str(root),
        "categories": {c: category_statistics(root, c, pixel_stats) for c in categories},
    }


def _fmt_counter(d: dict) -> str:
    return ", ".join(f"{k} ({v})" for k, v in d.items()) if d else "-"


def to_markdown(stats: dict) -> str:
    cats = stats["categories"]
    lines = [
        "# MVTec AD statistics report",
        "",
        f"Generated: {stats['generated_at']}  ",
        f"Data root: `{stats['root']}`",
        "",
        "## Summary",
        "",
        "| Category | Train (normal) | Test normal | Test defective | Defect types | Image size(s) | Mean defect area | Problems |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for name, c in cats.items():
        if "error" in c:
            lines.append(f"| {name} | - | - | - | - | - | - | ERROR |")
            continue
        area = c["masks"]["anomaly_area_fraction"]
        area_s = f"{area['mean'] * 100:.2f}%" if area else "-"
        sizes = ", ".join(c["images"]["sizes"])
        lines.append(
            f"| {name} | {c['train']['n_normal']} | {c['test']['n_normal']} | {c['test']['n_defective']} "
            f"| {len(c['defect_types'])} | {sizes} | {area_s} | {c['validation']['n_problems']} |"
        )
    for name, c in cats.items():
        lines += ["", f"## {name}", ""]
        if "error" in c:
            lines.append(f"**Error:** {c['error']}")
            continue
        lines += [
            f"- Train: {c['train']['n_images']} images (all normal)",
            f"- Test: {c['test']['n_images']} images — {c['test']['n_normal']} normal, "
            f"{c['test']['n_defective']} defective ({c['test']['defect_ratio'] * 100:.1f}% defective)",
            f"- Defect types: {_fmt_counter(c['defect_types'])}",
            f"- Image formats: {_fmt_counter(c['images']['formats'])}; modes: {_fmt_counter(c['images']['modes'])}; "
            f"sizes: {_fmt_counter(c['images']['sizes'])}",
            f"- Masks: {c['masks']['n_masks']} (formats {_fmt_counter(c['masks']['inventory']['formats'])}, "
            f"modes {_fmt_counter(c['masks']['inventory']['modes'])}), empty masks: {c['masks']['n_empty_masks']}",
        ]
        by = c["masks"]["anomaly_area_fraction_by_defect"]
        if by:
            lines += ["", "| Defect type | Images | Min area | Median | Mean | Max |", "|---|---:|---:|---:|---:|---:|"]
            for d, s in by.items():
                lines.append(
                    f"| {d} | {c['defect_types'][d]} | {s['min'] * 100:.2f}% | {s['median'] * 100:.2f}% "
                    f"| {s['mean'] * 100:.2f}% | {s['max'] * 100:.2f}% |"
                )
        if "train_pixel_stats" in c:
            p = c["train_pixel_stats"]
            lines += ["", f"- Train pixel mean (RGB, [0,1]): {p['mean']}", f"- Train pixel std (RGB, [0,1]): {p['std']}"]
        v = c["validation"]
        lines += ["", f"- Validation: {'OK' if v['ok'] else str(v['n_problems']) + ' problem(s)'} "
                      f"({v['n_images_checked']} images fully decoded)"]
        if not v["ok"]:
            for key in ("missing_masks", "orphan_masks", "corrupt_images", "size_mismatches"):
                if v[key]:
                    lines.append(f"  - {key}: {len(v[key])} (see JSON report)")
    return "\n".join(lines) + "\n"


def write_report(stats: dict, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "dataset_statistics.json"
    md_path = out_dir / "dataset_statistics.md"
    json_path.write_text(json.dumps(stats, indent=2) + "\n")
    md_path.write_text(to_markdown(stats))
    return json_path, md_path
