#!/usr/bin/env python3
"""Validate MVTec AD and write the dataset statistics report.

Outputs reports/dataset_statistics.{json,md} (and optionally sample figures).
Exit code 0 = dataset valid, 1 = dataset missing or validation problems.

Usage:
    python scripts/dataset_report.py                       # bottle hazelnut carpet
    python scripts/dataset_report.py --categories bottle --figures
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from inspectai.data.layout import CATEGORIES, FOCUS_CATEGORIES  # noqa: E402
from inspectai.data.stats import dataset_statistics, write_report  # noqa: E402


def main() -> int:
    default_root = Path(os.environ.get("INSPECTAI_DATA_ROOT", REPO_ROOT / "data" / "mvtec_ad"))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=default_root, help=f"MVTec AD root (default: {default_root})")
    ap.add_argument("--categories", nargs="+", default=list(FOCUS_CATEGORIES), choices=CATEGORIES)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "reports")
    ap.add_argument("--no-pixel-stats", action="store_true", help="skip train-set mean/std (faster)")
    ap.add_argument("--figures", action="store_true", help="also save sample figures (needs matplotlib)")
    args = ap.parse_args()

    if not args.root.is_dir():
        print(f"Dataset root not found: {args.root}\nSee docs/dataset.md for download and placement instructions.")
        return 1

    try:  # record a repo-relative path rather than a machine-specific absolute one
        label = args.root.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        label = str(args.root)
    stats = dataset_statistics(args.root, args.categories, pixel_stats=not args.no_pixel_stats, root_label=label)
    json_path, md_path = write_report(stats, args.out)
    print(f"Wrote {json_path}\nWrote {md_path}\n")

    exit_code = 0
    for name, c in stats["categories"].items():
        if "error" in c:
            print(f"[FAIL] {name}: {c['error']}")
            exit_code = 1
            continue
        v = c["validation"]
        flag = "OK  " if v["ok"] else "FAIL"
        print(f"[{flag}] {name:<9} train={c['train']['n_images']:>4}  test_normal={c['test']['n_normal']:>3}  "
              f"test_defective={c['test']['n_defective']:>3}  defect_types={len(c['defect_types'])}  "
              f"sizes={list(c['images']['sizes'])}  problems={v['n_problems']}")
        exit_code |= 0 if v["ok"] else 1

    if args.figures:
        import matplotlib

        matplotlib.use("Agg")
        from inspectai.data.viz import plot_category_samples

        fig_dir = args.out / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for name, c in stats["categories"].items():
            if "error" in c:
                continue
            fig = plot_category_samples(args.root, name)
            fig.savefig(fig_dir / f"{name}_samples.png", dpi=110)
            print(f"Wrote {fig_dir / (name + '_samples.png')}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
