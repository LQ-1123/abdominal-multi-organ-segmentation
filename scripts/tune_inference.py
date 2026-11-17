"""推理期调参实验：阈值 / 温度 / 集成 对 Dice 的实际影响。

要回答的问题
------------
「能否通过调温度、阈值等数值来微调结果？」

三个关键事实需要实测确认：
  1. **温度对 argmax 完全无效** —— softmax(logits/T) 是单调变换，
     取 argmax 后结果一模一样。温度只能改变"概率数值"，改变不了分割图。
     只有当附带"低置信度丢弃"规则时，温度才间接产生影响。
  2. **阈值有效但收益有限** —— argmax 等价于"取最大类"，加一个
     max_prob >= t 的闸门后，t 增大则掩膜收缩、t 减小则膨胀。
     对系统性偏小/偏大的器官有用，但对已收敛的模型收益通常 <1 个点。
  3. **每个器官的最优阈值不同** —— 肾上腺/十二指肠常被低估（低阈值更好），
     肝脏边界偏松（高阈值更好）。这是唯一有实操价值的调参方向。

用法:
    python scripts/tune_inference.py --cases 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infer.predictor import Predictor, find_checkpoints  # noqa: E402
from data.amos_dataset import make_folds  # noqa: E402

CACHE = ROOT / "data" / "cache" / "amos22"
ORGAN = ["脾脏", "右肾", "左肾", "胆囊", "食管", "肝脏", "胃", "主动脉",
         "下腔静脉", "胰腺", "右肾上腺", "左肾上腺", "十二指肠", "膀胱", "前列腺"]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else float(2 * (pred & gt).sum() / s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=4)
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    device = "cuda"
    # 只用 fold0 的模型，并在 fold0 的验证集上评估 —— 这些病例它没见过
    ckpts = [find_checkpoints(ROOT / "outputs_v2")[args.fold]]
    print(f"使用 fold{args.fold} 单模型（该折验证集对其为完全未见数据）")
    pred = Predictor(ckpts, patch=(112, 112, 112), device=device)

    folds = make_folds(sorted(p.stem.replace(".img", "")
                              for p in CACHE.glob("*.img.npy")), n_folds=5, seed=42)
    _, val_ids = folds[args.fold]
    cases = list(val_ids)[: args.cases]
    print(f"评估病例: {cases}\n")

    # 收集 {阈值: {器官: [dice...]}}
    per_thr: dict[float, dict[int, list[float]]] = {t: {} for t in THRESHOLDS}
    # 温度实验：只验证 argmax 是否改变
    argmax_changed = 0

    for cid in cases:
        img = np.load(CACHE / f"{cid}.img.npy")
        gt = np.load(CACHE / f"{cid}.lbl.npy")
        # 注意: predict_proba 只返回最大类别概率 [Z,Y,X]，
        # 需要完整 16 通道概率体才能做阈值扫描 -> 用 logits 自己算 softmax
        logits = pred.predict_logits(img, verbose=False)       # [16,Z,Y,X]
        lg = logits.astype(np.float32)
        e = np.exp(lg - lg.max(0, keepdims=True))
        prob = e / e.sum(0, keepdims=True)
        base = prob.argmax(0)

        # --- 温度实验：T=0.5 与 T=2.0 的 argmax 与 T=1.0 比较 ---
        for T in (0.5, 2.0):
            shifted = np.log(prob + 1e-9) / T
            alt = shifted.argmax(0)
            argmax_changed += int((alt != base).sum())

        maxp = prob.max(0)
        for t in THRESHOLDS:
            keep = maxp >= t
            seg = np.where(keep, base, 0)
            for c in range(1, 16):
                g = gt == c
                if g.sum() == 0:
                    continue
                per_thr[t].setdefault(c, []).append(dice(seg == c, g))
        print(f"  {cid} 完成")

    # ---------------- 报告 1：温度 ----------------
    print(f"\n=== 温度实验 ===")
    print(f"  T=0.5 / T=2.0 与 T=1.0 的 argmax 差异体素总数: {argmax_changed}")
    print("  -> 结论: 温度不改变 argmax，无法用于调整分割结果")

    # ---------------- 报告 2：全局阈值 ----------------
    print(f"\n=== 全局阈值扫描（15 器官平均 Dice）===")
    print(f"{'阈值':>6}{'平均Dice':>11}{'相对 0.5':>11}")
    print("-" * 30)
    base_dice = None
    best_t, best_d = None, -1
    for t in THRESHOLDS:
        vals = [d for arr in per_thr[t].values() for d in arr]
        m = float(np.mean(vals))
        if t == 0.5:
            base_dice = m
        if m > best_d:
            best_t, best_d = t, m
        print(f"{t:>6.1f}{m:>11.4f}{'':>11}")
    print(f"\n最佳全局阈值: {best_t}  ({best_d:.4f})")
    vals05 = [d for arr in per_thr[0.5].values() for d in arr]
    print(f"默认 argmax(=0.5) 平均: {np.mean(vals05):.4f}")
    print(f"最大可提升: {best_d - np.mean(vals05):+.4f}")

    # ---------------- 报告 3：分器官最优阈值 ----------------
    print(f"\n=== 分器官最优阈值 ===")
    print(f"{'器官':<10}{'0.5':>8}{'最优阈值':>9}{'该阈值Dice':>11}{'提升':>9}")
    print("-" * 50)
    result = {}
    for c in range(1, 16):
        d05 = np.mean(per_thr[0.5].get(c, [0]))
        if not per_thr[0.5].get(c):
            continue
        best_c, best_v = None, -1
        for t in THRESHOLDS:
            v = np.mean(per_thr[t].get(c, [0]))
            if v > best_v:
                best_c, best_v = t, v
        result[ORGAN[c - 1]] = {"t_opt": best_c, "dice_opt": round(best_v, 4),
                                "dice_05": round(float(d05), 4),
                                "gain": round(float(best_v - d05), 4)}
        print(f"{ORGAN[c-1]:<10}{d05:>8.4f}{best_c:>9.1f}{best_v:>11.4f}{best_v-d05:>+9.4f}")

    total_gain = np.mean([v["gain"] for v in result.values()])
    print("-" * 50)
    print(f"分器官调阈值后的平均提升: {total_gain:+.4f}")

    out = ROOT / "outputs" / "tune_inference.json"
    out.write_text(json.dumps({"global": {"t_best": best_t, "dice": best_d,
                                          "dice_05": float(np.mean(vals05))},
                               "per_organ": result,
                               "temperature_argmax_diff": argmax_changed},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
