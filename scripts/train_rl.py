"""强化学习边界精修的完整训练/评估流程。

流程
----
1. **预计算**（只需一次）：用 fold0 的分割模型对 fold0 的验证病例做滑窗推理，
   把「难分割器官」的软概率量化为 uint8 存盘。这样训练时读盘即可，
   不必每个回合都跑一次 3D 推理。
2. **训练**：采样 (病例, 切片, 器官) 构造环境，跑 12 步的回合，PPO 更新策略。
3. **评估**：对比精修前后的切片 Dice。

为什么只用难器官
----------------
肝脏 Dice 已 0.94，边界几乎不需要修；真正值得修的是胰腺、十二指肠、肾上腺。
把 RL 集中在 4 个难器官上，既符合"用强化学习解决监督学习解决不好的问题"的动机，
也让训练数据量可控。

用法
----
    python scripts/train_rl.py --precompute --n-cases 12     # 第一次：建缓存
    python scripts/train_rl.py --iters 300                    # 训练
    python scripts/train_rl.py --eval-only                    # 只评估
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

HARD_ORGANS = [10, 13, 11, 12]        # 胰腺 / 十二指肠 / 右肾上腺 / 左肾上腺
ORGAN_CN = {10: "胰腺", 13: "十二指肠", 11: "右肾上腺", 12: "左肾上腺"}
CACHE = ROOT / "data" / "rl_cache"


# ============================================================ 1. 预计算
def precompute(n_cases: int, patch: int = 96) -> None:
    import torch
    from data.amos_dataset import build_case_list, make_folds
    from infer.predictor import Predictor, find_checkpoints

    cache_dir = ROOT / "data" / "cache" / "amos22"
    cases = build_case_list(ROOT / "data" / "processed" / "amos22")
    ids = [c.case_id for c in cases]
    have = {p.stem.replace(".img", "") for p in cache_dir.glob("*.img.npy")}
    ids = [i for i in ids if i in have]

    # fold0 的验证集 = 模型没训练过的病例
    _, val_ids = make_folds(ids, n_folds=5, seed=42)[0]
    use = val_ids[:n_cases]
    print(f"将预计算 {len(use)} 个 fold0 验证病例: {use}")

    ckpt = [ROOT / "outputs" / "fold0" / "best.pth"]
    predictor = Predictor(ckpt, patch=(patch,) * 3)

    CACHE.mkdir(parents=True, exist_ok=True)
    for i, cid in enumerate(use):
        out = CACHE / f"{cid}.npz"
        if out.exists():
            print(f"  [{i+1}/{len(use)}] {cid} 已存在，跳过")
            continue
        t0 = time.time()
        img = np.load(cache_dir / f"{cid}.img.npy")
        lbl = np.load(cache_dir / f"{cid}.lbl.npy")
        logits = predictor.predict_logits(img, verbose=False)     # [16, Z, Y, X]
        # 关键：必须用「完整 16 类 softmax」里的对应通道，不能只在这 4 类内部归一化。
        # 否则每个器官的概率被严重高估（例如胰腺 0.5 的阈值会覆盖大片肝胃区域）。
        # 为了省内存，用两遍流式算分母：先求逐体素最大 logit，再累加 exp。
        mx = np.full(img.shape, -1e4, dtype=np.float32)
        for c in range(logits.shape[0]):
            np.maximum(mx, logits[c].astype(np.float32), out=mx)
        denom = np.zeros(img.shape, dtype=np.float32)
        for c in range(logits.shape[0]):
            denom += np.exp(logits[c].astype(np.float32) - mx)
        prob = np.stack([np.exp(logits[c].astype(np.float32) - mx) / denom for c in HARD_ORGANS])
        prob_u8 = np.clip(prob * 255, 0, 255).astype(np.uint8)
        np.savez_compressed(out, prob=prob_u8, gt=lbl.astype(np.uint8))
        print(f"  [{i+1}/{len(use)}] {cid} 完成 {time.time()-t0:.1f}s "
              f"prob {prob_u8.shape} {out.stat().st_size/1024**2:.0f} MB")
    print(f"缓存目录: {CACHE}")


# ============================================================ 2. 数据采样
def load_case_data(cid: str):
    d = np.load(CACHE / f"{cid}.npz")
    img = np.load(ROOT / "data" / "cache" / "amos22" / f"{cid}.img.npy", mmap_mode="r")
    return img, d["prob"], d["gt"]


def sample_episode(rng, case_ids, data_cache):
    """随机取 (病例, 切片, 器官)，且要求该层该器官 GT 与预测都非空。"""
    from rl.refine import RefineEnv

    for _ in range(200):
        cid = case_ids[rng.randint(len(case_ids))]
        if cid not in data_cache:
            data_cache[cid] = load_case_data(cid)
        img, prob_all, gt_all = data_cache[cid]
        z = rng.randint(0, gt_all.shape[0])
        k = rng.randint(len(HARD_ORGANS))
        organ = HARD_ORGANS[k]

        gt = gt_all[z] == organ
        prob = prob_all[k, z].astype(np.float32) / 255.0
        init = prob > 0.5

        if gt.sum() < 30 or init.sum() < 30:
            continue
        env = RefineEnv(img[z], prob, gt, init, steps=16, seed=int(rng.randint(1 << 30)))
        return env, organ
    return None, None


# ============================================================ 3. 训练
def train(iters: int, n_cases: int, lr: float, device: str, log_every: int = 20) -> None:
    import torch
    from rl.refine import PPO, PolicyNet, Transition

    files = sorted(CACHE.glob("*.npz"))
    if not files:
        print("❌ 没有缓存，先跑 --precompute")
        return
    case_ids = [f.stem for f in files][:n_cases]
    print(f"训练病例 {len(case_ids)} 个: {case_ids}")

    rng = np.random.RandomState(0)
    net = PolicyNet()
    agent = PPO(net, lr=lr, device=device)
    print(f"策略网络 {sum(p.numel() for p in net.parameters())/1e3:.1f}K 参数，设备 {device}")

    data_cache: dict = {}
    base_dices, refined_dices = [], []
    t0 = time.time()

    EPISODES_PER_UPDATE = 8       # 单条轨迹的 PPO 方差极大，必须成批更新

    for it in range(1, iters + 1):
        trajs = []
        for _ in range(EPISODES_PER_UPDATE):
            env, organ = sample_episode(rng, case_ids, data_cache)
            if env is None:
                continue
            obs = env.reset()
            traj = []
            for _ in range(env.steps):
                a, logp, v = agent.act(obs.state)
                nobs, r, done, _ = env.step(a)
                traj.append(Transition(obs.state, a, logp, v, r, done))
                obs = nobs
                if done:
                    break
            trajs.append(traj)
            base_dices.append(env.initial_dice)
            refined_dices.append(env.final_dice)
        if not trajs:
            continue
        stats = agent.update(trajs)

        if it % log_every == 0:
            b = float(np.mean(base_dices[-log_every:]))
            a_ = float(np.mean(refined_dices[-log_every:]))
            el = time.time() - t0
            print(f"  it {it:4d} | loss {stats['loss']:+.4f} | entropy {stats['entropy']:.3f} "
                  f"| Dice {b:.4f} → {a_:.4f} (Δ{a_-b:+.4f}) | {el:.0f}s")

    out = ROOT / "outputs" / "rl_policy.pth"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "hard_organs": HARD_ORGANS}, out)
    print(f"\n策略已保存 {out}")
    print(f"全程平均: 精修前 {np.mean(base_dices):.4f} → 精修后 {np.mean(refined_dices):.4f} "
          f"(Δ{np.mean(refined_dices)-np.mean(base_dices):+.4f})")


# ============================================================ 4. 评估
def evaluate(n_cases: int, device: str, episodes_per_case: int = 40) -> None:
    import torch
    from rl.refine import PolicyNet, RefineEnv

    ck = torch.load(ROOT / "outputs" / "rl_policy.pth", map_location=device, weights_only=True)
    net = PolicyNet().to(device)
    net.load_state_dict(ck["model"])
    net.eval()

    files = sorted(CACHE.glob("*.npz"))[:n_cases]
    rng = np.random.RandomState(123)
    data_cache: dict = {}
    results = {o: {"before": [], "after": []} for o in HARD_ORGANS}

    with torch.no_grad():
        for _ in range(n_cases * episodes_per_case):
            env, organ = sample_episode(rng, [f.stem for f in files], data_cache)
            if env is None:
                continue
            obs = env.reset()
            before = env.initial_dice
            for _ in range(env.steps):
                x = torch.from_numpy(obs.state[None]).to(device)
                logits, _ = net(x)
                a = int(logits.argmax(1).item())
                obs, r, done, _ = env.step(a)
                if done:
                    break
            results[organ]["before"].append(before)
            results[organ]["after"].append(env.final_dice)

    print(f"\n{'器官':<10}{'样本':>6}{'精修前':>10}{'精修后':>10}{'Δ':>10}")
    print("-" * 46)
    all_b, all_a = [], []
    for o in HARD_ORGANS:
        b, a = results[o]["before"], results[o]["after"]
        if not b:
            continue
        mb, ma = float(np.mean(b)), float(np.mean(a))
        all_b += b
        all_a += a
        print(f"{ORGAN_CN[o]:<10}{len(b):>6}{mb:>10.4f}{ma:>10.4f}{ma-mb:>+10.4f}")
    print("-" * 46)
    print(f"{'合计平均':<10}{len(all_b):>6}{np.mean(all_b):>10.4f}{np.mean(all_a):>10.4f}"
          f"{np.mean(all_a)-np.mean(all_b):>+10.4f}")

    out = ROOT / "outputs" / "rl_eval.json"
    out.write_text(json.dumps({"per_organ": {ORGAN_CN[o]: {
        "n": len(results[o]["before"]),
        "before": float(np.mean(results[o]["before"])) if results[o]["before"] else None,
        "after": float(np.mean(results[o]["after"])) if results[o]["after"] else None,
    } for o in HARD_ORGANS}}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果写入 {out}")


def oracle_eval(n_cases: int, episodes: int = 150, seed: int = 7) -> None:
    """决定性实验：上帝视角 oracle —— 每步穷举 5 个动作，选当前 Dice 最高的。

    - oracle 提升明显  -> 任务可学，RL 训练方法有问题，值得继续调
    - oracle 几乎不涨  -> argmax 掩膜已在局部最优，任何策略都不可能超过它，
                          这是动作空间/奖励设计的上限，应如实写进报告
    """
    import copy

    files = sorted(CACHE.glob("*.npz"))[:n_cases]
    rng = np.random.RandomState(seed)
    data_cache: dict = {}
    before_all, after_all, oracle_noop = [], [], 0
    total_steps = 0

    for _ in range(episodes):
        env, organ = sample_episode(rng, case_ids=[f.stem for f in files], data_cache=data_cache)
        if env is None:
            continue
        env.reset()
        b = env.initial_dice
        for _ in range(env.steps):
            best_a, best_dice, best_mask = 0, -1.0, None
            for a in range(5):
                snapshot = copy.deepcopy(env)
                _, _, done, info = snapshot.step(a)
                if info["dice"] > best_dice:
                    best_a, best_dice, best_mask = a, info["dice"], snapshot.mask
            total_steps += 1
            if best_a == 0:
                oracle_noop += 1
            env.mask = best_mask
            # 推进步骤计数
            env.t += 1
            if env.t >= env.steps:
                break
        before_all.append(b)
        after_all.append(env.final_dice)

    print(f"\n=== Oracle 上界实验（{len(before_all)} 回合, {total_steps} 步）===")
    print(f"  精修前 Dice      : {np.mean(before_all):.4f}")
    print(f"  oracle 精修后    : {np.mean(after_all):.4f}")
    print(f"  oracle 增益 Δ    : {np.mean(after_all)-np.mean(before_all):+.4f}")
    print(f"  oracle 选择'不动'的比例: {oracle_noop/total_steps*100:.1f}%")
    print(f"\n解读: 若 Δ 接近 0 且'不动'占比很高 -> argmax 已在局部最优，")
    print(f"      在当前动作空间下不存在可学的策略上界。")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precompute", action="store_true")
    ap.add_argument("--n-cases", type=int, default=12)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--oracle", action="store_true")
    args = ap.parse_args()

    if args.precompute:
        precompute(args.n_cases)
    if args.oracle:
        oracle_eval(args.n_cases)
        return 0
    if args.eval_only:
        evaluate(args.n_cases, args.device)
    elif not args.precompute:
        train(args.iters, args.n_cases, args.lr, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
