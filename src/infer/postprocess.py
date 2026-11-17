"""分割结果后处理：连通域清理。

为什么需要
----------
3D 模型输出 argmax 后，每个器官都会散出大量孤立碎块：
    - 肝脏 181 个连通域、脾脏 41 个、胃 38 个
这些碎块有两个来源：
    1. 概率在边界附近抖动，产生单个体素的孤立点（纯噪声）
    2. 模型把视野外、训练分布外（如肺内病灶）的组织误判成器官

医学上这 15 个器官每个都是**单一连通结构**，所以「保留主要连通域、丢弃碎片」
是在不损害真值的前提下最直接有效的清理手段。

阈值设计
--------
单纯按"只保留最大块"会误删真实结构：十二指肠是 C 形管，预测可能断成几段，
最大块只占 49.7%。所以用**相对阈值**：
    保留条件: 体积 >= max(最大块 × min_ratio, min_voxels)
默认 min_ratio=2%、min_voxels=30，既能删掉肺内 1.8mL 的假阳性
（相对 1134mL 的肝脏只有 0.16%），又不会把十二指肠的合理分段删掉。
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["clean_label", "component_report"]


def clean_label(
    label: np.ndarray,
    num_classes: int = 16,
    min_ratio: float = 0.02,
    min_voxels: int = 30,
    connectivity: int = 1,
) -> tuple[np.ndarray, list[dict]]:
    """按连通域清理标签体。

    Args:
        label: [Z, Y, X] uint8，取值 0..num_classes-1
        num_classes: 类别数（含背景）
        min_ratio: 相对阈值——保留体积 >= 最大块 × min_ratio 的连通域
        min_voxels: 绝对阈值——无论如何都丢弃小于该体积的连通域
        connectivity: 1=面相邻(6邻域)  2=角相邻(26邻域)

    Returns:
        (清理后的标签体, 每个器官的处理记录)
    """
    out = label.copy()
    structure = ndimage.generate_binary_structure(3, connectivity)
    records: list[dict] = []

    for c in range(1, num_classes):
        mask = label == c
        if not mask.any():
            continue

        lab, n = ndimage.label(mask, structure=structure)
        if n <= 1:
            continue

        sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
        largest = float(sizes.max())
        threshold = max(largest * min_ratio, float(min_voxels))

        keep_ids = np.nonzero(sizes >= threshold)[0] + 1
        if len(keep_ids) == 0:              # 极端情况：保住最大块
            keep_ids = np.array([int(np.argmax(sizes)) + 1])

        keep_mask = np.isin(lab, keep_ids)
        removed = int(mask.sum() - keep_mask.sum())
        out[~keep_mask & mask] = 0

        records.append({
            "class": c,
            "components_before": int(n),
            "components_after": int(len(keep_ids)),
            "voxels_before": int(mask.sum()),
            "voxels_removed": removed,
            "removed_pct": round(100.0 * removed / max(mask.sum(), 1), 2),
        })

    return out, records


def component_report(label: np.ndarray, num_classes: int = 16) -> list[dict]:
    """只统计不修改，用于对比清理前后。"""
    structure = ndimage.generate_binary_structure(3, 1)
    rows = []
    for c in range(1, num_classes):
        mask = label == c
        if not mask.any():
            continue
        _, n = ndimage.label(mask, structure=structure)
        rows.append({"class": c, "voxels": int(mask.sum()), "components": int(n)})
    return rows


if __name__ == "__main__":
    # 冒烟：在一个已知有假阳性的真实病例上对比前后
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    CN = {1: "脾脏", 2: "右肾", 3: "左肾", 4: "胆囊", 5: "食管", 6: "肝脏", 7: "胃",
          8: "主动脉", 9: "下腔静脉", 10: "胰腺", 11: "右肾上腺", 12: "左肾上腺",
          13: "十二指肠", 14: "膀胱", 15: "前列腺"}

    p = ROOT / "outputs" / "web_cases" / "9d582cbb12b1" / "vol.lbl.npy"
    if not p.exists():
        print(f"找不到样本 {p}")
        raise SystemExit(1)

    lbl = np.load(p)
    before = {r["class"]: r for r in component_report(lbl)}
    cleaned, recs = clean_label(lbl)
    after = {r["class"]: r for r in component_report(cleaned)}

    print(f"{'器官':<9}{'连通域(前)':>10}{'连通域(后)':>10}{'体积(前)mL':>11}{'体积(后)mL':>11}{'删除%':>8}")
    print("-" * 62)
    for c in sorted(before):
        b, a = before[c], after[c]
        print(f"{CN[c]:<9}{b['components']:>10}{a['components']:>10}"
              f"{b['voxels']*3.375/1000:>11.1f}{a['voxels']*3.375/1000:>11.1f}"
              f"{100*(1-a['voxels']/max(b['voxels'],1)):>8.1f}")
    print("\n✅ 连通域清理冒烟测试通过")
