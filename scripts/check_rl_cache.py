"""校验 RL 缓存里的概率是否合理（初始掩膜 Dice 应接近验证集水平）。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.refine import dice_slice  # noqa: E402

HARD = [10, 13, 11, 12]
CN = {10: "胰腺", 13: "十二指肠", 11: "右肾上腺", 12: "左肾上腺"}


def main() -> int:
    files = sorted((ROOT / "data" / "rl_cache").glob("*.npz"))
    print(f"缓存病例 {len(files)} 个\n")
    for f in files[:3]:
        d = np.load(f)
        prob, gt = d["prob"], d["gt"]
        print(f"{f.stem}  prob{prob.shape} 取值范围[{prob.min()},{prob.max()}]  "
              f"gt 类别={np.unique(gt).tolist()}")
        for k, organ in enumerate(HARD):
            dices = []
            step = max(1, gt.shape[0] // 40)
            for z in range(0, gt.shape[0], step):
                g = gt[z] == organ
                if g.sum() <= 30:
                    continue
                p = prob[k, z] > 127
                dices.append(dice_slice(p, g))
            if dices:
                print(f"   {CN[organ]:<8} 切片 Dice 平均 {np.mean(dices):.4f}  ({len(dices)} 层)")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
