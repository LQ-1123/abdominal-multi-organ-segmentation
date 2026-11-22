"""第二批报告/PPT 图：数据特性、预处理效果、频谱、后处理与预处理对照、边缘分析。

每张图只依赖对应实验的输出文件，缺哪个就跳过哪张。

    python scripts/make_report_figs2.py --archive F:/BaiduNetdiskDownload/fold0_val.zip
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_ablations import ORGANS  # noqa: E402
from src.preprocess import classic  # noqa: E402

OUT = ROOT / "outputs/report_figs"
plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, GREY, RED = "#2f6db5", "#e08a1e", "#3a9a5b", "#8a8a8a", "#c0392b"
METHOD_CN = {"none": "无处理", "gaussian": "高斯", "median": "中值", "butterworth": "Butterworth", "clahe": "CLAHE"}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fnum(v):
    return float(v) if v not in (None, "") else None


def body_box(plane: np.ndarray, pad: int = 6):
    from scipy import ndimage as ndi

    body = ndi.binary_fill_holes(ndi.binary_opening(plane > 0.1, iterations=3))
    parts, _ = ndi.label(body)
    sizes = np.bincount(parts.ravel())
    sizes[0] = 0
    r, c = np.nonzero(parts == sizes.argmax())
    return slice(max(r.min() - pad, 0), r.max() + pad + 1), slice(max(c.min() - pad, 0), c.max() + pad + 1)


# ----------------------------------------------------------------------------- 数据特性
def fig_data(stats_dir: Path) -> None:
    npz = stats_dir / "dataset_samples.npz"
    if not npz.exists():
        print("跳过 fig_data：缺少", npz)
        return
    d = np.load(npz)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained",
                           gridspec_kw={"width_ratios": [0.8, 1.6, 1.6]})
    thick = d["spacing"][:, 2]
    ax[0].hist(thick, bins=np.arange(0, 6, 0.25), color=BLUE, edgecolor="white")
    ax[0].set_xlabel("原始层厚 / mm")
    ax[0].set_ylabel("病例数")
    ax[0].set_title("(a) 层厚分布", loc="left")
    order = sorted(range(len(ORGANS)), key=lambda i: -np.median(d[f"vol_{i}"]))
    ax[1].boxplot([d[f"vol_{i}"] for i in order], vert=True, showfliers=False, widths=0.6,
                  medianprops={"color": ORANGE})
    ax[1].set_yscale("log")
    ax[1].set_xticks(range(1, len(order) + 1), [ORGANS[i] for i in order], rotation=60, ha="right")
    ax[1].set_ylabel("体积 / mL")
    ax[1].set_title("(b) 器官体积", loc="left")
    ax[2].axhspan(-150, 250, color="#e6e6e6", zorder=0)
    ax[2].boxplot([d[f"hu_{i}"] for i in order], showfliers=False, widths=0.6, medianprops={"color": ORANGE})
    ax[2].set_xticks(range(1, len(order) + 1), [ORGANS[i] for i in order], rotation=60, ha="right")
    ax[2].set_ylabel("CT 值 / HU")
    ax[2].set_ylim(-250, 450)
    ax[2].set_title("(c) 器官 CT 值（灰带为输入窗）", loc="left")
    fig.savefig(OUT / "fig_data.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- 预处理效果
def fig_preprocess_visual(archive: Path, case: str) -> None:
    from zipfile import ZipFile

    from scripts.run_ablations import archive_pairs, load_case

    with ZipFile(archive) as z:
        image, label = load_case(z, archive_pairs(z)[case])
    zc = int(np.argmax((label == 10).sum(axis=(1, 2))))
    sub = image[max(zc - 8, 0): zc + 9]          # 三维滤波只需目标层附近的一段
    k = zc - max(zc - 8, 0)
    planes = {m: np.flipud(classic.apply(sub, m)[k]) for m in classic.METHODS}
    lab = np.flipud(label[zc])
    box = body_box(planes["none"])
    ys, xs = np.nonzero(lab == 10)
    cy, cx = int(ys.mean()), int(xs.mean())
    zoom = (slice(cy - 40, cy + 40), slice(cx - 55, cx + 55))
    fig, ax = plt.subplots(2, 5, figsize=(16, 5.0), layout="constrained",
                           gridspec_kw={"height_ratios": [1.35, 1]})
    titles = {"none": "原图（仅窗宽窗位）", "gaussian": "3D 高斯 σ=1 mm", "median": "3D 中值 3×3×3",
              "butterworth": "Butterworth 低通 $D_0$=0.2", "clahe": "轴位 CLAHE"}
    for j, m in enumerate(classic.METHODS):
        ax[0, j].imshow(planes[m][box], cmap="gray", vmin=0, vmax=1)
        ax[0, j].set_title(titles[m])
        ax[1, j].imshow(planes[m][zoom], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax[1, j].contour(lab[zoom] == 10, levels=[0.5], colors="cyan", linewidths=0.9)
        for a in ax[:, j]:
            a.set_xticks([])
            a.set_yticks([])
    ax[1, 0].set_ylabel("胰腺区域放大\n（青线为真值）")
    fig.savefig(OUT / "fig_pre_visual.png", dpi=180)
    plt.close(fig)

    # 频谱：原图与 Butterworth 滤波后轴位切片的二维幅度谱，及一维传递函数
    from scipy import fft

    f0 = np.log1p(np.abs(fft.fftshift(fft.fft2(planes["none"] - planes["none"].mean()))))
    f1 = np.log1p(np.abs(fft.fftshift(fft.fft2(planes["butterworth"] - planes["butterworth"].mean()))))
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.0), layout="constrained")
    vmax = np.percentile(f0, 99.9)
    ax[0].imshow(f0, cmap="viridis", vmax=vmax)
    ax[0].set_title("原图幅度谱 log(1+|F|)")
    ax[1].imshow(f1, cmap="viridis", vmax=vmax)
    ax[1].set_title("Butterworth 低通后幅度谱")
    for a in ax[:2]:
        a.set_xticks([])
        a.set_yticks([])
    d = np.linspace(0, 1 / (2 * classic.SPACING_MM), 300)
    for n, ls in ((1, ":"), (2, "-"), (4, "--")):
        ax[2].plot(d, 1 / (1 + (d / classic.BUTTERWORTH_CUTOFF) ** (2 * n)), ls, color=BLUE if n == 2 else GREY,
                   label=f"n = {n}" + ("（本文）" if n == 2 else ""))
    ax[2].plot(d, (d <= classic.BUTTERWORTH_CUTOFF).astype(float), color=ORANGE, lw=1, label="理想低通")
    ax[2].axvline(classic.BUTTERWORTH_CUTOFF, color=GREY, lw=0.8)
    ax[2].set_xlabel("频率 D / (周期·mm$^{-1}$)")
    ax[2].set_ylabel("H(D)")
    ax[2].set_title("传递函数，$D_0$ = 0.2，奈奎斯特频率 0.33")
    ax[2].legend(frameon=False)
    fig.savefig(OUT / "fig_pre_spectrum.png", dpi=180)
    plt.close(fig)


# ----------------------------------------------------------------------------- 后处理对照
def fig_postprocess(summary_csv: Path) -> None:
    if not summary_csv.exists():
        print("跳过 fig_post：缺少", summary_csv)
        return
    rows = read_csv(summary_csv)
    names = {
        "none": "无后处理",
        "cc": "连通域",
        "morph_cc": "开闭+连通域",
        "morph_sel_cc": "选择性开闭+连通域",
        "cc_solid": "连通域+实质最大块",
        "morph_sel_cc_solid": "选择性开闭+实质最大块",
    }
    total = {r["method"]: r for r in rows if r["organ"].startswith("全部")}
    methods = [m for m in names if m in total]
    fig, ax = plt.subplots(1, 4, figsize=(17.0, 4.5), layout="constrained",
                           gridspec_kw={"width_ratios": [1.15, 1.15, 1.15, 2.15]})
    colors = [GREY, BLUE, ORANGE, GREEN, "#5b8cc4", "#1f7a4c"][: len(methods)]
    for a, key, label in ((ax[0], "dice", "mDice"), (ax[1], "hd95_mm", "平均 HD95 / mm"),
                          (ax[2], "components_per_case", "每例连通域数")):
        vals = [fnum(total[m][key]) for m in methods]
        bars = a.bar(range(len(methods)), vals, color=colors)
        a.set_xticks(range(len(methods)), [names[m] for m in methods], rotation=35, ha="right", fontsize=8.5)
        a.set_title(label, loc="left")
        for b, v in zip(bars, vals):
            a.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}" if key == "dice" else f"{v:.1f}",
                   ha="center", va="bottom", fontsize=7.5)
        if key == "dice":
            a.set_ylim(min(vals) - 0.008, max(vals) + 0.006)
        else:
            a.set_ylim(0, max(vals) * 1.15)
    per = {(r["method"], r["organ"]): fnum(r["dice"]) for r in rows if not r["organ"].startswith("全部")}
    x = np.arange(len(ORGANS))
    for k, (m, c) in enumerate((("morph_cc", ORANGE), ("morph_sel_cc", GREEN))):
        if m not in total:
            continue
        delta = [100 * (per[(m, o)] - per[("cc", o)]) for o in ORGANS]
        ax[3].bar(x + (k - 0.5) * 0.4, delta, width=0.4, color=c, label=f"{names[m]} − 连通域")
    ax[3].axhline(0, color="black", lw=0.6)
    ax[3].set_xticks(x, ORGANS, rotation=60, ha="right")
    ax[3].set_ylabel("Dice 变化 / 百分点")
    ax[3].set_title("开闭运算对各器官 Dice 的影响（相对仅连通域）", loc="left")
    ax[3].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "fig_post.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- 碎块示意
def fig_fragments(archive: Path, pred_dir: Path, case: str) -> None:
    from zipfile import ZipFile

    from matplotlib.colors import ListedColormap

    from scripts.run_ablations import archive_pairs, load_case
    from src.infer.postprocess import SOLID_ORGANS, clean_label

    f = pred_dir / f"{case}.npz"
    if not f.exists():
        print("跳过 fig_fragments：缺少", f)
        return
    pred = np.load(f)["pred"]
    with ZipFile(archive) as z:
        image, _ = load_case(z, archive_pairs(z)[case])
    cleaned, _ = clean_label(pred, largest_only=SOLID_ORGANS)
    removed = (pred > 0) & (cleaned == 0)
    # 冠状面投影：沿前后方向（Y 轴）取最大，上下翻转成头在上
    bg = np.flipud(classic.window(image).mean(axis=1))
    lab_raw = np.flipud(pred.max(axis=1))
    lab_cln = np.flipud(cleaned.max(axis=1))
    rem = np.flipud(removed.any(axis=1))
    cmap = plt.get_cmap("tab20", 20)
    fig, ax = plt.subplots(1, 2, figsize=(9, 6.4), layout="constrained")
    for a, lab, ttl in ((ax[0], lab_raw, f"原始预测：{int(sum(np.unique(pred) > 0))} 类，碎块散布"),
                        (ax[1], lab_cln, "连通域清理后")):
        a.imshow(bg, cmap="gray", vmin=0, vmax=np.percentile(bg, 99.5), aspect="equal")
        a.imshow(np.ma.masked_equal(lab, 0), cmap=cmap, vmin=0.5, vmax=20.5, alpha=0.55, interpolation="nearest")
        a.set_title(ttl)
        a.axis("off")
    ax[0].imshow(np.ma.masked_where(~rem, rem), cmap=ListedColormap(["#e31a1c"]), alpha=1.0, interpolation="nearest")
    fig.suptitle(f"{case}：被清理掉的碎块（红色，{int(removed.sum())} 个体素）", fontsize=11)
    fig.savefig(OUT / "fig_fragments.png", dpi=180)
    plt.close(fig)


# ----------------------------------------------------------------------------- 预处理对照
def fig_preprocess(train_side: Path) -> None:
    res = {}
    for m in classic.METHODS:
        for ep in ("0", "20"):
            f = train_side / f"{m}_ft{ep}" / "summary.json"
            if f.exists():
                res[(m, ep)] = json.loads(f.read_text(encoding="utf-8"))
    if not res:
        print("跳过 fig_pre：缺少训练端结果")
        return
    fig, ax = plt.subplots(1, 2, figsize=(13.2, 4.4), layout="constrained")
    x = np.arange(len(classic.METHODS))
    for k, (ep, label, c) in enumerate((("0", "推理端直接加处理（原模型）", GREY), ("20", "带处理微调 20 轮后", BLUE))):
        dice = [res[(m, ep)]["mean_dice"] if (m, ep) in res else np.nan for m in classic.METHODS]
        hd = [res[(m, ep)]["mean_hd95_mm"] if (m, ep) in res else np.nan for m in classic.METHODS]
        b0 = ax[0].bar(x + (k - 0.5) * 0.39, dice, width=0.39, color=c, label=label)
        b1 = ax[1].bar(x + (k - 0.5) * 0.39, hd, width=0.39, color=c, label=label)
        for b, v in zip(b0, dice):
            if not np.isnan(v):
                ax[0].text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7.0)
        for b, v in zip(b1, hd):
            if not np.isnan(v):
                ax[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
    lo = np.nanmin([r["mean_dice"] for r in res.values()])
    hi = np.nanmax([r["mean_dice"] for r in res.values()])
    hd_hi = np.nanmax([r["mean_hd95_mm"] for r in res.values()])
    ax[0].set_ylim(lo - 0.014, hi + 0.010)
    ax[1].set_ylim(0, hd_hi * 1.18)
    for a, t in ((ax[0], "mDice（59 例验证 CT）"), (ax[1], "平均 HD95 / mm")):
        a.set_xticks(x, [METHOD_CN[m] for m in classic.METHODS])
        a.set_title(t, loc="left")
    ax[1].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.savefig(OUT / "fig_pre.png", dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- 边缘分析
def fig_edge(edge_csv: Path) -> None:
    if not edge_csv.exists():
        print("跳过 fig_edge：缺少", edge_csv)
        return
    rows = read_csv(edge_csv)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for a, key, label in ((ax[0], "canny_recall", "Canny 边缘召回率"), (ax[1], "sobel_mean", "边界平均 Sobel 梯度")):
        xs = [float(r[key]) for r in rows]
        ys = [float(r["dice_raw"]) for r in rows]
        a.scatter(xs, ys, color=BLUE, s=22)
        for r, xv, yv in zip(rows, xs, ys):
            a.annotate(r["organ"], (xv, yv), textcoords="offset points", xytext=(4, 3), fontsize=8)
        a.set_xlabel(label)
        a.set_ylabel("器官 Dice（无后处理）")
    fig.savefig(OUT / "fig_edge_corr.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, default=Path("F:/BaiduNetdiskDownload/fold0_val.zip"))
    ap.add_argument("--case", default="amos_0016")
    ap.add_argument("--only", nargs="*", default=[])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = {
        "data": lambda: fig_data(ROOT / "outputs/dataset_stats"),
        "pre_visual": lambda: fig_preprocess_visual(args.archive, args.case),
        "fragments": lambda: fig_fragments(args.archive, ROOT / "outputs/ablation_fold0_post_v3/pred", "amos_0063"),
        "post": lambda: fig_postprocess(ROOT / "outputs/ablation_fold0_post_v3/postprocess_summary.csv"),
        "pre": lambda: fig_preprocess(ROOT / "outputs/train_side"),
        "edge": lambda: fig_edge(ROOT / "outputs/edge_analysis/edge_by_organ.csv"),
    }
    for name, job in jobs.items():
        if not args.only or name in args.only:
            job()


if __name__ == "__main__":
    main()
