"""把 AMOS22 的 NIfTI 预处理成可内存映射的 .npy 缓存。

做三件事（对应设计文档里的 DICOM 管线第 5 步，这里作用于 NIfTI）：
    1. 抗锯齿重采样到 1.5mm 等向（原数据层厚 5mm，是层内分辨率的 ~9 倍）
    2. 影像用线性插值、标签用最近邻（标签线性插值会产生不存在的类别值）
    3. 存成 .npy 供 mmap 随机读取 —— 训练时按 patch 读取，不用载入整卷

用法:
    python scripts/build_cache.py --workers 6
    python scripts/build_cache.py --limit 8        # 先做 8 例试水
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.amos_dataset import build_case_list  # noqa: E402


def read_image_nib(path: Path) -> sitk.Image:
    """用 nibabel 读 NIfTI 再转成 SimpleITK 图像。

    为什么不用 sitk.ReadImage：SimpleITK 在 Windows 下按 ANSI 代码页处理路径，
    遇到中文目录（如「医学图像分割」）会报 "Unable to open ... for reading"。
    nibabel 走 Python 的文件 API，Unicode 路径没问题。
    """
    nii = nib.load(str(path))
    data = np.asanyarray(nii.dataobj)                 # nibabel: (X, Y, Z)
    if data.ndim == 4:                                # 丢掉多余时间维
        data = data[..., 0]
    data = np.ascontiguousarray(np.transpose(data, (2, 1, 0)))   # -> (Z, Y, X)

    img = sitk.GetImageFromArray(data)
    aff = np.asarray(nii.affine, dtype=float)
    spacing = np.sqrt((aff[:3, :3] ** 2).sum(axis=0))
    dirs = aff[:3, :3] / spacing
    img.SetSpacing(tuple(float(s) for s in spacing))
    img.SetOrigin(tuple(float(v) for v in aff[:3, 3]))
    img.SetDirection(tuple(float(v) for v in dirs.flatten()))
    return img


def resample_to_isotropic(img: sitk.Image, labels: sitk.Image | None, spacing: float):
    """重采样到等向 spacing；影像线性插值，标签最近邻插值。"""
    orig = np.array(img.GetSpacing(), dtype=float)
    size = np.array(img.GetSize(), dtype=float)
    new_size = np.maximum(np.round(orig * size / spacing).astype(int), 1)

    def _run(image: sitk.Image, interp: int) -> sitk.Image:
        rf = sitk.ResampleImageFilter()
        rf.SetSize([int(v) for v in new_size])
        rf.SetOutputSpacing([float(spacing)] * 3)
        rf.SetOutputOrigin(img.GetOrigin())
        rf.SetOutputDirection(img.GetDirection())
        rf.SetInterpolator(interp)
        rf.SetDefaultPixelValue(-1024.0)
        return rf.Execute(image)

    img_rs = _run(img, sitk.sitkLinear)
    lbl_rs = _run(labels, sitk.sitkNearestNeighbor) if labels is not None else None
    return img_rs, lbl_rs


def process_one(case, out_dir: Path, spacing: float) -> str:
    cid = case.case_id
    img_out = out_dir / f"{cid}.img.npy"
    lbl_out = out_dir / f"{cid}.lbl.npy"
    if img_out.exists() and lbl_out.exists():
        return f"skip {cid}"

    t0 = time.time()
    img = read_image_nib(case.img_path)
    lbl = read_image_nib(case.lbl_path)

    img_rs, lbl_rs = resample_to_isotropic(img, lbl, spacing)

    arr = sitk.GetArrayFromImage(img_rs)                     # [Z, Y, X] float32
    arr = np.clip(arr, -1024, 3071).astype(np.int16)         # CT 有效 HU 范围
    lab = sitk.GetArrayFromImage(lbl_rs).astype(np.uint8)    # [Z, Y, X]

    np.save(img_out, arr, allow_pickle=False)
    np.save(lbl_out, lab, allow_pickle=False)
    dt = time.time() - t0
    return f"ok {cid} shape={arr.shape} {dt:.1f}s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacing", type=float, default=1.5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 例（0=全部）")
    ap.add_argument("--ct-only", action="store_true", help="只处理 CT（编号 < 500）")
    ap.add_argument("--root", type=Path, default=ROOT / "data" / "processed" / "amos22")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "cache" / "amos22")
    args = ap.parse_args()

    root = args.root
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = build_case_list(root)
    if args.ct_only:
        cases = [c for c in cases if c.is_ct]
    if args.limit:
        cases = cases[: args.limit]
    print(f"待处理 {len(cases)} 例 -> {out_dir}  (spacing={args.spacing}mm)")

    t0 = time.time()
    if args.workers <= 1:
        for c in tqdm(cases, ncols=80):
            process_one(c, out_dir, args.spacing)
    else:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            it = pool.starmap_async(process_one, [(c, out_dir, args.spacing) for c in cases])
            # 简单进度：轮询文件数
            total = len(cases)
            while not it.ready():
                done = len(list(out_dir.glob("*.img.npy")))
                print(f"\r  {done}/{total} 完成", end="", flush=True)
                time.sleep(5)
            res = it.get()
            print()
            for r in res:
                if r.startswith("ok"):
                    print("  " + r)

    n_img = len(list(out_dir.glob("*.img.npy")))
    sz = sum(f.stat().st_size for f in out_dir.glob("*.npy")) / 1024**3
    print(f"\n完成: {n_img} 例, 缓存 {sz:.1f} GB, 耗时 {(time.time()-t0)/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
