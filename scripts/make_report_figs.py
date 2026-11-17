"""生成报告/PPT 用的图表。

产出
----
    outputs/figs/fig1_training_curve.png    V1 vs V2 训练曲线（fold0 同期对比）
    outputs/figs/fig2_organ_dice.png        分器官 Dice 对比（横向柱状图）
    outputs/figs/fig3_fold_dice.png         各折 mDice 分布
    outputs/figs/fig4_improvement.png       各器官提升幅度

为什么单独写一个脚本
------------------
训练日志是 CSV，报告需要的是矢量质量的图。把"数据处理"和"画图"分开，
换数据（V1/V2/V3）只改参数，不用重写绘图逻辑。

用法
----
    python scripts/make_report_figs.py
    python scripts/make_report_figs.py --v1 outputs --v2 outputs_v2
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "outputs" / "figs"

ORGAN_CN = {
    "spleen": "脾脏", "right_kidney": "右肾", "left_kidney": "左肾",
    "gallbladder": "胆囊", "esophagus": "食管", "liver": "肝脏", "stomach": "胃",
    "aorta": "主动脉", "postcava": "下腔静脉", "pancreas": "胰腺",
    "right_adrenal": "右肾上腺", "left_adrenal": "左肾上腺",
    "duodenum": "十二指肠", "bladder": "膀胱", "prostate_uterus": "前列腺/子宫",
}


def setup_font() -> None:
    """让 matplotlib 能显示中文。"""
    for name in ("Microsoft YaHei", "SimHei", "SimSun"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [name]
            break
        except Exception:  # noqa: BLE001
            continue
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 200
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.25


def read_log(p: Path) -> tuple[list[int], list[float]]:
    e, d = [], []
    if not p.exists():
        return e, d
    with open(p, encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) >= 4:
                try:
                    e.append(int(row[0]))
                    d.append(float(row[3]))
                except ValueError:
                    continue
    return e, d


def smooth(y: list[float], k: int = 5) -> np.ndarray:
    if len(y) < k:
        return np.array(y)
    arr = np.convolve(y, np.ones(k) / k, mode="valid")
    pad = np.full(k - 1, np.nan)
    return np.concatenate([pad, arr])


def fig_training_curve(v1_dir: Path, v2_dir: Path) -> None:
    e1, d1 = read_log(v1_dir / "fold0" / "train_log.csv")
    e2, d2 = read_log(v2_dir / "fold0" / "train_log.csv")
    if not d1 and not d2:
        print("  跳过 fig1：缺少训练日志")
        return

    fig, ax = plt.subplots(figsize=(9, 4.6))
    if d1:
        ax.plot(e1, d1, alpha=0.22, color="#888888", lw=1)
        ax.plot(e1, smooth(d1), color="#3b6ea5", lw=2, label=f"V1（96³，弱增强，100ep）")
    if d2:
        ax.plot(e2, d2, alpha=0.22, color="#e8a33d", lw=1)
        ax.plot(e2, smooth(d2), color="#d1495b", lw=2, label="V2（112³，强增强，120ep）")

    ax.set_xlabel("训练轮次 (epoch)")
    ax.set_ylabel("验证 patch Dice")
    ax.set_title("训练曲线对比（fold 0 验证集，细线=原始值，粗线=5 轮滑动平均）")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(0, 0.85)
    fig.tight_layout()
    out = FIGS / "fig1_training_curve.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✅ {out.name}")


def load_organ_dice(d: Path) -> dict:
    p = d / "fold0" / "val_organ_dice.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))["organ_dice"]
    return {k: v for k, v in d.items() if k in ORGAN_CN}      # 去掉 'mean' 等非器官键


def fig_organ_dice(v1_dir: Path, v2_dir: Path) -> None:
    a = load_organ_dice(v1_dir)
    b = load_organ_dice(v2_dir)
    if not a and not b:
        print("  跳过 fig2：缺少分器官结果")
        return

    organs = [k for k in ORGAN_CN if k in a or k in b]
    organs.sort(key=lambda k: a.get(k, b.get(k, 0)))
    va = [a.get(k, 0) for k in organs]
    vb = [b.get(k, 0) for k in organs]

    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    y = np.arange(len(organs))
    h = 0.38
    ax.barh(y - h / 2, va, h, label="V1", color="#8fa8c8")
    ax.barh(y + h / 2, vb, h, label="V2", color="#d1495b")
    ax.set_yticks(y)
    ax.set_yticklabels([ORGAN_CN[k] for k in organs])
    ax.set_xlabel("Dice (fold 0 全卷滑窗评估)")
    ax.set_xlim(0, 1.0)
    ax.set_title("分器官分割精度：V1 与 V2 对比")
    for i, (x1, x2) in enumerate(zip(va, vb)):
        ax.text(x1 + 0.01, i - h / 2, f"{x1:.3f}", va="center", fontsize=7.5, color="#33475b")
        ax.text(x2 + 0.01, i + h / 2, f"{x2:.3f}", va="center", fontsize=7.5, color="#7a2230")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = FIGS / "fig2_organ_dice.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✅ {out.name}")


def fig_fold_dice(v1_dir: Path, v2_dir: Path) -> None:
    def folds(d: Path) -> tuple[list[int], list[float]]:
        xs, ys = [], []
        for p in sorted(d.glob("fold*/val_organ_dice.json")):
            j = json.loads(p.read_text(encoding="utf-8"))
            xs.append(int(j["fold"]))
            ys.append(j["organ_dice"]["mean"])
        return xs, ys

    x1, y1 = folds(v1_dir)
    x2, y2 = folds(v2_dir)
    if not y1 and not y2:
        print("  跳过 fig3：缺少折结果")
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if y1:
        ax.scatter(x1, y1, s=70, color="#3b6ea5", label=f"V1  平均 {np.mean(y1):.4f}", zorder=3)
        ax.axhline(np.mean(y1), color="#3b6ea5", ls="--", lw=1, alpha=0.6)
    if y2:
        ax.scatter(x2, y2, s=70, color="#d1495b", label=f"V2  平均 {np.mean(y2):.4f}", zorder=3)
        ax.axhline(np.mean(y2), color="#d1495b", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("折 (fold)")
    ax.set_ylabel("15 器官平均 Dice")
    ax.set_xticks(sorted(set(x1 + x2)))
    ax.set_title("各折交叉验证结果")
    ax.legend()
    fig.tight_layout()
    out = FIGS / "fig3_fold_dice.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✅ {out.name}")


def fig_improvement(v1_dir: Path, v2_dir: Path) -> None:
    a = load_organ_dice(v1_dir)
    b = load_organ_dice(v2_dir)
    if not a or not b:
        print("  跳过 fig4：需要两个版本的分器官结果")
        return

    diffs = [(k, b[k] - a[k]) for k in a if k in b]
    diffs.sort(key=lambda t: t[1])
    names = [ORGAN_CN[k] for k, _ in diffs]
    vals = [v for _, v in diffs]

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    colors = ["#2a9d8f" if v > 0 else "#e76f51" for v in vals]
    ax.barh(np.arange(len(vals)), vals, color=colors)
    ax.set_yticks(np.arange(len(vals)))
    ax.set_yticklabels(names)
    ax.axvline(0, color="#333", lw=1)
    ax.set_xlabel("Dice 提升（V2 − V1）")
    ax.set_title("V2 相对 V1 的分器官提升\n提升集中在难分割的小器官")
    for i, v in enumerate(vals):
        ax.text(v + 0.002, i, f"{v:+.3f}", va="center", fontsize=8)
    fig.tight_layout()
    out = FIGS / "fig4_improvement.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ✅ {out.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default="outputs")
    ap.add_argument("--v2", default="outputs_v2")
    args = ap.parse_args()

    setup_font()
    FIGS.mkdir(parents=True, exist_ok=True)
    v1, v2 = ROOT / args.v1, ROOT / args.v2
    print(f"生成图表 -> {FIGS}")
    fig_training_curve(v1, v2)
    fig_organ_dice(v1, v2)
    fig_fold_dice(v1, v2)
    fig_improvement(v1, v2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
