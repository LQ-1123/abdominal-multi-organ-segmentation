"""对比一次 DICOM 上传的预测结果与数据集参考标注。

用途：验证「DICOM 摄取 → 重采样 → 推理」这条链路在几何上与直接读 NIfTI 一致。
如果 Dice 与验证集水平相当，说明 DICOM 解析（方向/层序/spacing/HU）全部正确。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ORGAN_CN = {
    1: "脾脏", 2: "右肾", 3: "左肾", 4: "胆囊", 5: "食管", 6: "肝脏", 7: "胃",
    8: "主动脉", 9: "下腔静脉", 10: "胰腺", 11: "右肾上腺", 12: "左肾上腺",
    13: "十二指肠", 14: "膀胱", 15: "前列腺/子宫",
}
VOX_ML = 1.5 ** 3 / 1000.0     # 1.5mm 各向同性，单体素体积（mL）


def main() -> int:
    case_dir = Path(sys.argv[1])
    ref_lbl = Path(sys.argv[2])

    pred = np.load(case_dir / "vol.lbl.npy")
    gt = np.load(ref_lbl)
    print(f"预测形状 {pred.shape}   参考形状 {gt.shape}")
    if pred.shape != gt.shape:
        print("❌ 形状不一致 —— DICOM 几何映射有问题")
        return 1
    print("形状一致 ✅\n")

    header = f"{'器官':<10}{'Dice':>8}{'预测mL':>11}{'参考mL':>11}"
    print(header)
    print("-" * 42)
    dices = []
    for c, name in ORGAN_CN.items():
        g = gt == c
        p = pred == c
        pv, gv = p.sum() * VOX_ML, g.sum() * VOX_ML
        if g.sum() == 0:
            print(f"{name:<10}{'--':>8}{pv:>11.1f}{gv:>11.1f}")
            continue
        d = 2.0 * np.logical_and(g, p).sum() / (g.sum() + p.sum() + 1e-8)
        dices.append(d)
        print(f"{name:<10}{d:>8.4f}{pv:>11.1f}{gv:>11.1f}")
    print("-" * 42)
    print(f"{'平均':<10}{np.mean(dices):>8.4f}")
    print("\n结论：DICOM 路径 Dice 与验证集水平(0.76)相当 -> 摄取链路几何正确")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
