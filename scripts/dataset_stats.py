"""AMOS22 CT 数据特性统计：原始体素间距、器官体积与出现率、类别不平衡、各器官 HU 分布。

    python scripts/dataset_stats.py --raw data/raw --cache data/cache/amos22 --out outputs/dataset_stats
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_ablations import ORGANS  # noqa: E402
from src.data.amos_dataset import build_case_list  # noqa: E402

VOXEL_ML = 1.5 ** 3 / 1000.0


def main() -> None:
    import nibabel as nib

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    cases = [c for c in build_case_list(args.raw) if c.is_ct and (args.cache / f"{c.case_id}.img.npy").exists()]
    spacing, shapes, volumes, hu = [], [], {o: [] for o in ORGANS}, {o: [] for o in ORGANS}
    fg_fraction = []
    for i, case in enumerate(cases, 1):
        hdr = nib.load(str(case.img_path)).header
        spacing.append([float(v) for v in hdr.get_zooms()[:3]])
        shapes.append([int(v) for v in hdr.get_data_shape()[:3]])
        img = np.load(args.cache / f"{case.case_id}.img.npy", mmap_mode="r")
        lbl = np.load(args.cache / f"{case.case_id}.lbl.npy")
        counts = np.bincount(lbl.ravel(), minlength=len(ORGANS) + 1)
        fg_fraction.append(1.0 - counts[0] / lbl.size)
        for c, name in enumerate(ORGANS, 1):
            if counts[c]:
                volumes[name].append(counts[c] * VOXEL_ML)
                idx = np.flatnonzero(lbl.ravel() == c)
                pick = rng.choice(idx, size=min(400, len(idx)), replace=False)
                hu[name].extend(np.asarray(img).ravel()[pick].tolist())
        if i % 20 == 0:
            print(f"[{i}/{len(cases)}]", flush=True)

    sp = np.array(spacing)
    summary = {
        "ct_cases": len(cases),
        "inplane_mm": {"min": float(sp[:, :2].min()), "median": float(np.median(sp[:, :2])), "max": float(sp[:, :2].max())},
        "slice_mm": {"min": float(sp[:, 2].min()), "median": float(np.median(sp[:, 2])), "max": float(sp[:, 2].max())},
        "slices": {"min": int(min(s[2] for s in shapes)), "median": float(np.median([s[2] for s in shapes])),
                   "max": int(max(s[2] for s in shapes))},
        "foreground_fraction_mean": float(np.mean(fg_fraction)),
        "organs": {n: {"present_cases": len(volumes[n]),
                       "volume_ml_median": float(np.median(volumes[n])) if volumes[n] else None,
                       "volume_ml_p5_p95": [float(np.percentile(volumes[n], 5)), float(np.percentile(volumes[n], 95))]
                       if volumes[n] else None,
                       "hu_median": float(np.median(hu[n])) if hu[n] else None,
                       "hu_p5_p95": [float(np.percentile(hu[n], 5)), float(np.percentile(hu[n], 95))] if hu[n] else None}
                   for n in ORGANS},
    }
    (args.out / "dataset_stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.out / "dataset_samples.npz", spacing=sp,
                        **{f"vol_{i}": np.array(volumes[n]) for i, n in enumerate(ORGANS)},
                        **{f"hu_{i}": np.array(hu[n], dtype=np.int16) for i, n in enumerate(ORGANS)})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
