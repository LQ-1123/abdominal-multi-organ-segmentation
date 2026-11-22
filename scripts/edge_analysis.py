"""用课本边缘算子（Sobel、Canny）分析各器官边界在 CT 上的可见程度。

对 fold0 验证集 CT 的轴位切片（每 4 层取 1 层）：
1. 按模型同款窗 [-150, 250] HU 映射到 [0, 1]；
2. Sobel 梯度幅值：统计真值器官边界上的平均梯度（边界对比度）；
3. Canny 边缘：真值边界像素中，距 Canny 边缘不超过 1 像素（1.5 mm）的比例（边缘召回）。

两项都只用真值，不依赖模型。输出逐器官表和一张示例图，用来解释为什么胰腺、
肾上腺、十二指肠这类边界在灰度上“看不出来”的器官 Dice 偏低。

    python scripts/edge_analysis.py --archive F:/BaiduNetdiskDownload/fold0_val.zip \
        --out outputs/edge_analysis
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_ablations import ORGANS, archive_pairs, load_case  # noqa: E402
from src.preprocess.classic import window  # noqa: E402

CANNY_SIGMA = 2.0
STEP = 4


def boundary(mask2d: np.ndarray) -> np.ndarray:
    return mask2d & ~ndi.binary_erosion(mask2d)


def main() -> None:
    from skimage.feature import canny

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/edge_analysis")
    ap.add_argument("--dice-summary", type=Path,
                    default=ROOT / "outputs/ablation_fold0/postprocess_summary.csv")
    ap.add_argument("--example", default="amos_0005")
    ap.add_argument("--figure-only", action="store_true", help="只重画示例图")
    args = ap.parse_args()
    if args.figure_only:
        with ZipFile(args.archive) as archive:
            image, label = load_case(archive, archive_pairs(archive)[args.example])
        args.out.mkdir(parents=True, exist_ok=True)
        figure(args.out / "edge_example.png", window(image), label)
        return
    args.out.mkdir(parents=True, exist_ok=True)

    grad_sum = np.zeros(len(ORGANS) + 1)
    hit = np.zeros(len(ORGANS) + 1)
    total = np.zeros(len(ORGANS) + 1)
    with ZipFile(args.archive) as archive:
        pairs = archive_pairs(archive)
        ids = sorted(c for c in pairs if int(c.rsplit("_", 1)[1]) < 500)
        for n, cid in enumerate(ids, 1):
            image, label = load_case(archive, pairs[cid])
            x = window(image)
            zs = np.nonzero(label.any(axis=(1, 2)))[0][::STEP]
            for z in zs:
                plane, lab = x[z], label[z]
                gm = np.hypot(ndi.sobel(plane, 0), ndi.sobel(plane, 1))
                edges = ndi.binary_dilation(canny(plane, sigma=CANNY_SIGMA), iterations=1)
                for c in np.unique(lab):
                    if c == 0:
                        continue
                    b = boundary(lab == c)
                    k = int(b.sum())
                    total[c] += k
                    hit[c] += int((b & edges).sum())
                    grad_sum[c] += float(gm[b].sum())
            print(f"[{n}/{len(ids)}] {cid}", flush=True)
            if cid == args.example:
                example = (x, label)

    dice = {}
    if args.dice_summary.exists():
        with args.dice_summary.open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r["method"] == "none" and r["organ"] in ORGANS:
                    dice[r["organ"]] = float(r["dice"])
    rows = []
    for c, name in enumerate(ORGANS, 1):
        rows.append({"organ": name, "boundary_pixels": int(total[c]),
                     "canny_recall": hit[c] / total[c] if total[c] else None,
                     "sobel_mean": grad_sum[c] / total[c] if total[c] else None,
                     "dice_raw": dice.get(name)})
    with (args.out / "edge_by_organ.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    stats = {"cases": len(ids), "slice_step": STEP, "canny_sigma": CANNY_SIGMA}
    if dice:
        for key in ("canny_recall", "sobel_mean"):
            rho, p = spearmanr([r[key] for r in rows], [r["dice_raw"] for r in rows])
            stats[f"spearman_{key}_vs_dice"] = {"rho": float(rho), "p": float(p)}
    (args.out / "edge_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    figure(args.out / "edge_example.png", *example)


def figure(path: Path, x: np.ndarray, label: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.feature import canny

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    # 选胰腺（10）面积最大的一层，同一层里通常也有肝、脾、肾、胃
    z = int(np.argmax((label == 10).sum(axis=(1, 2))))
    # 缓存的轴位切片背侧朝上；上下翻转成常规阅片方向（腹侧朝上、患者右侧在图像左侧）
    plane, lab = np.flipud(x[z]), np.flipud(label[z])
    # 裁掉体外的黑边和检查床
    body = ndi.binary_fill_holes(ndi.binary_opening(plane > 0.1, iterations=3))
    parts, _ = ndi.label(body)
    sizes = np.bincount(parts.ravel())
    sizes[0] = 0
    rows, cols = np.nonzero(parts == sizes.argmax())
    box = (slice(max(rows.min() - 6, 0), rows.max() + 7), slice(max(cols.min() - 6, 0), cols.max() + 7))
    plane, lab = plane[box], lab[box]
    gm = np.hypot(ndi.sobel(plane, 0), ndi.sobel(plane, 1))
    edges = canny(plane, sigma=CANNY_SIGMA)
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6), layout="constrained")
    ax[0].imshow(plane, cmap="gray", vmin=0, vmax=1)
    ax[0].set_title("CT 轴位（窗宽 400 / 窗位 50）")
    ax[1].imshow(gm, cmap="magma", vmax=np.percentile(gm, 99.5))
    ax[1].set_title("Sobel 梯度幅值")
    ax[2].imshow(edges, cmap="gray")
    ax[2].set_title(f"Canny 边缘（σ = {CANNY_SIGMA:g}）")
    ax[3].imshow(plane, cmap="gray", vmin=0, vmax=1)
    ax[3].imshow(np.ma.masked_where(~edges, edges), cmap=ListedColormap(["#ff9f1c"]), alpha=0.9)
    for c, color in ((10, "cyan"), (6, "lime"), (13, "deepskyblue")):
        if (lab == c).any():
            ax[3].contour(lab == c, levels=[0.5], colors=color, linewidths=1.2)
    ax[3].set_title("Canny 边缘（橙）与真值边界：肝（绿）胰（青）十二指肠（蓝）", fontsize=10)
    for a in ax:
        a.axis("off")
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
