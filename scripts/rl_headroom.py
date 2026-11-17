"""RL 精修的"余量实验"：在不同质量的初始掩膜上测 oracle 上界。

要回答的问题
------------
上一步实验发现：在 3D U-Net 的 argmax 掩膜上，oracle（上帝视角、每步穷举最优动作）
只能提升 +0.005 Dice —— 低于 PPO 的噪声水平，所以策略学不到东西。

这引出一个更本质的问题：
    **是这个动作空间没用，还是这个基线没有余量？**

设计三组对照：
    argmax      : prob > 0.5            （模型给的实际输出，强基线）
    prob > 0.9  : 更严格阈值，掩膜偏小   （模拟"保守"的模型）
    erode3      : argmax 腐蚀 3 体素     （模拟"系统性偏小"的模型）

如果 argmax 上 oracle 几乎不涨、而差基线上涨幅明显，
就证明：动作空间有效，RL 精修能救弱模型，但救不了已经很强的模型。
这是一个可以写进报告的、诚实且有信息量的结论。
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl.refine import RefineEnv, dice_slice  # noqa: E402

HARD_ORGANS = [10, 13, 11, 12]
ORGAN_CN = {10: "胰腺", 13: "十二指肠", 11: "右肾上腺", 12: "左肾上腺"}
CACHE = ROOT / "data" / "rl_cache"
VOX = ROOT / "data" / "cache" / "amos22"


def make_init(prob: np.ndarray, mode: str) -> np.ndarray:
    if mode == "argmax":
        return prob > 0.5
    if mode == "prob90":
        return prob > 0.9
    if mode == "erode3":
        from scipy.ndimage import binary_erosion
        m = prob > 0.5
        return binary_erosion(m, iterations=3)
    if mode == "dilate3":
        from scipy.ndimage import binary_dilation
        m = prob > 0.5
        return binary_dilation(m, iterations=3)
    raise ValueError(mode)


def oracle_episode(env: RefineEnv) -> float:
    """每步穷举 5 个动作，选 Dice 最高者。返回最终 Dice。"""
    env.reset()
    for _ in range(env.steps):
        best_a, best_dice, best_mask = 0, -1.0, None
        for a in range(5):
            snap = copy.deepcopy(env)
            snap.step(a)
            d = dice_slice(snap.mask, snap.gt)
            if d > best_dice:
                best_a, best_dice, best_mask = a, d, snap.mask
        env.mask = best_mask
        env.t += 1
        if env.t >= env.steps:
            break
    return dice_slice(env.mask, env.gt)


def main() -> int:
    modes = ["argmax", "prob90", "erode3", "dilate3"]
    files = sorted(CACHE.glob("*.npz"))
    if not files:
        print("没有缓存，先跑 scripts/train_rl.py --precompute")
        return 1

    rng = np.random.RandomState(2024)
    data: dict = {}
    for f in files:
        d = np.load(f)
        data[f.stem] = (np.load(VOX / f"{f.stem}.img.npy", mmap_mode="r"), d["prob"], d["gt"])

    results = {m: {"before": [], "after": []} for m in modes}
    per_organ = {m: {o: [] for o in HARD_ORGANS} for m in modes}
    target_eps = 120

    for mode in modes:
        got = 0
        attempts = 0
        while got < target_eps and attempts < target_eps * 40:
            attempts += 1
            cid = list(data)[rng.randint(len(data))]
            img, prob_all, gt_all = data[cid]
            z = rng.randint(0, gt_all.shape[0])
            k = rng.randint(len(HARD_ORGANS))
            organ = HARD_ORGANS[k]
            gt = gt_all[z] == organ
            prob = prob_all[k, z].astype(np.float32) / 255.0
            init = make_init(prob, mode)
            if gt.sum() < 50 or init.sum() < 20:
                continue
            env = RefineEnv(img[z], prob, gt, init, steps=20, seed=int(rng.randint(1 << 30)))
            b = env.initial_dice
            a = oracle_episode(env)
            results[mode]["before"].append(b)
            results[mode]["after"].append(a)
            per_organ[mode][organ].append((b, a))
            got += 1

    print(f"\n{'初始掩膜':<10}{'样本':>6}{'精修前':>10}{'oracle后':>11}{'余量 Δ':>10}")
    print("-" * 50)
    for mode in modes:
        b = results[mode]["before"]
        a = results[mode]["after"]
        if not b:
            continue
        print(f"{mode:<10}{len(b):>6}{np.mean(b):>10.4f}{np.mean(a):>11.4f}"
              f"{np.mean(a)-np.mean(b):>+10.4f}")

    print(f"\n{'初始掩膜':<10}" + "".join(f"{ORGAN_CN[o]:>11}" for o in HARD_ORGANS))
    print("-" * 56)
    for mode in modes:
        row = f"{mode:<10}"
        for o in HARD_ORGANS:
            v = per_organ[mode][o]
            if v:
                b = np.mean([x[0] for x in v])
                a = np.mean([x[1] for x in v])
                row += f"{a-b:>+11.4f}"
            else:
                row += f"{'-':>11}"
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
