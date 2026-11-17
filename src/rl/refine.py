"""强化学习边界精修：把分割结果当作"可编辑的初始猜测"，用 agent 逐步修正。

问题定义
--------
3D U-Net 输出的是**概率图 + argmax**，边界处往往是"平均意义上的最优"，
在器官与邻近组织对比度低的地方会系统性偏移（比如胰腺和十二指肠的交界）。

我们把它建模成一个马尔可夫决策过程（MDP）：

    状态 s_t   = 当前切片以候选边界点为中心的局部观测
                 (CT 灰度块, 当前掩膜块, 模型概率块)  形状 3×33×33
    动作 a_t   ∈ { 收缩2, 收缩1, 保持, 扩张1, 扩张2 }    沿局部法向往内/外移动
    奖励 r_t   = (Dice_after - Dice_before) × 100       即时的切片级 Dice 增量
    回合结束   = 执行满 T 步，或步数用尽

为什么这样设计是合理的
----------------------
1. **奖励直接是评测指标**。监督学习优化的是软 Dice 代理损失，这里直接优化 Dice 本身，
   属于"指标对齐"（metric-aligned）训练 —— 这是 RL 在分割里最站得住脚的理由。
2. **局部动作 + 全局奖励**。agent 每次只动一小块，但奖励看整层 Dice，
   所以它必须学会"这一步修了边界，会不会把别处搞坏"。
3. **可以做任意后处理**。训练好的策略可以套在**任何**分割模型输出上（V1/V2/别人的），
   是一个独立的精修模块，不是替代品。

已知局限（需要在报告里写清楚）
------------------------------
- 动作空间是离散的局部膨胀/腐蚀，表达能力弱于逐点回归边界
- 单步奖励不含长期规划，本质接近 contextual bandit 的推广
- 只在 2D 切片上操作，不保证 3D 一致性（相邻层可能不连续）
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HU_MIN, HU_MAX = -150.0, 250.0
PATCH = 33          # 局部观测边长（奇数，便于取中心）
HALF = PATCH // 2
N_ACTIONS = 5       # 收缩2/收缩1/保持/扩张1/扩张2
ACTION_OFFSETS = np.array([-2, -1, 0, 1, 2], dtype=np.int32)


# ============================================================== 几何工具
def _disk(radius: int) -> np.ndarray:
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx) <= radius * radius


def _stamp(mask: np.ndarray, y: int, x: int, radius: int, value: int) -> None:
    """在 (y,x) 处盖一个圆形印章（原地修改）。"""
    r = int(np.ceil(radius))
    y0, y1 = max(0, y - r), min(mask.shape[0], y + r + 1)
    x0, x1 = max(0, x - r), min(mask.shape[1], x + r + 1)
    if y0 >= y1 or x0 >= x1:
        return
    d = _disk(radius)[(y0 - (y - r)):(y1 - (y - r)), (x0 - (x - r)):(x1 - (x - r))]
    if value:
        mask[y0:y1, x0:x1] |= d
    else:
        mask[y0:y1, x0:x1] &= ~d


def boundary_points(mask: np.ndarray, n: int = 64, rng: np.random.RandomState | None = None) -> np.ndarray:
    """取掩膜的边界点（形态学边界），随机或等距采样 n 个。"""
    from scipy.ndimage import binary_erosion

    if not mask.any():
        return np.zeros((0, 2), dtype=np.int32)
    edge = mask & ~binary_erosion(mask, iterations=1)
    ys, xs = np.nonzero(edge)
    if len(ys) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    idx = np.arange(len(ys))
    if rng is not None and len(ys) > n:
        idx = rng.choice(idx, size=n, replace=False)
    elif len(ys) > n:
        idx = idx[:: max(1, len(ys) // n)][:n]
    return np.stack([ys[idx], xs[idx]], axis=1)


def dice_slice(pred: np.ndarray, gt: np.ndarray) -> float:
    s = pred.sum() + gt.sum()
    if s == 0:
        return 1.0
    return float(2.0 * (pred & gt).sum() / s)


# ============================================================== 环境
@dataclass
class StepObs:
    state: np.ndarray        # [3, PATCH, PATCH] float32
    point: tuple[int, int]   # 当前边界点
    done: bool
    info: dict


class RefineEnv:
    """单层单器官的边界精修环境。

    Args:
        ct_slice:   [H, W] float32 原始 HU
        prob_slice: [H, W] float32 该器官的概率 (0..1)
        gt_slice:   [H, W] bool   仅训练时提供（用于算奖励）
        init_mask:  [H, W] bool   模型给的初始掩膜
        steps:      一个回合的编辑步数
    """

    def __init__(self, ct_slice, prob_slice, gt_slice, init_mask, steps: int = 12, seed: int | None = None):
        self.ct = ((np.clip(ct_slice, HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)
        self.prob = prob_slice.astype(np.float32)
        self.gt = gt_slice.astype(bool) if gt_slice is not None else None
        self.mask = init_mask.astype(bool).copy()
        self.steps = steps
        self.rng = np.random.RandomState(seed)
        self.t = 0
        self.initial_dice = dice_slice(self.mask, self.gt) if self.gt is not None else 0.0
        self.history: list[dict] = []

    # ---------------- 观测
    def _observe(self, y: int, x: int) -> np.ndarray:
        H, W = self.ct.shape
        y0, y1 = y - HALF, y + HALF + 1
        x0, x1 = x - HALF, x + HALF + 1
        pad_y0, pad_y1 = max(0, -y0), max(0, y1 - H)
        pad_x0, pad_x1 = max(0, -x0), max(0, x1 - W)
        yy0, yy1 = max(0, y0), min(H, y1)
        xx0, xx1 = max(0, x0), min(W, x1)

        def _crop(arr):
            out = np.zeros((PATCH, PATCH), dtype=np.float32)
            piece = arr[yy0:yy1, xx0:xx1].astype(np.float32)
            out[HALF - (y - yy0):HALF - (y - yy0) + piece.shape[0],
                HALF - (x - xx0):HALF - (x - xx0) + piece.shape[1]] = piece
            return out, (pad_y0, pad_y1, pad_x0, pad_x1)

        im, _ = _crop(self.ct)
        mk, _ = _crop(self.mask)
        pr, _ = _crop(self.prob)
        return np.stack([im, mk, pr], axis=0)

    # ---------------- 交互
    def reset(self) -> StepObs:
        self.mask = self.mask.copy()
        self.t = 0
        pts = boundary_points(self.mask, n=1, rng=self.rng)
        if len(pts) == 0:
            ys, xs = np.nonzero(self.mask)
            if len(ys) == 0:
                y, x = self.ct.shape[0] // 2, self.ct.shape[1] // 2
            else:
                i = self.rng.randint(len(ys))
                y, x = int(ys[i]), int(xs[i])
        else:
            y, x = int(pts[0][0]), int(pts[0][1])
        self._cur = (y, x)
        return StepObs(self._observe(y, x), (y, x), False, {})

    # ---------------------------------------------------------- 法向
    def _normal(self, y: int, x: int) -> np.ndarray:
        """估计边界点外法向（指向掩膜外）。

        用掩膜/背景两条距离变换的梯度取法向：
            d_out = 到掩膜的距离（外部大）
            d_in  = 到背景的距离（内部大）
        外法向 ∝ ∇d_out，在边界处方向稳定。
        """
        from scipy.ndimage import distance_transform_edt, sobel

        d_out = distance_transform_edt(~self.mask)
        gy = sobel(d_out, axis=0, mode="nearest")
        gx = sobel(d_out, axis=1, mode="nearest")
        v = np.array([gy[y, x], gx[y, x]], dtype=np.float32)
        nrm = np.linalg.norm(v)
        if nrm < 1e-6:
            # 退化情况：用中心指向外的方向
            cy, cx = np.array(self.mask.shape) / 2
            v = np.array([y - cy, x - cx], dtype=np.float32)
            nrm = np.linalg.norm(v) + 1e-6
        return v / nrm

    def _shift_boundary(self, y: int, x: int, offset: int, radius: int = 7) -> None:
        """把局部边界沿法向推到偏移 offset 处。

        在半径 radius 的窗口内，按「沿法向的有符号距离 d」重设掩膜：
            d <= offset  → 掩膜内
            d >  offset  → 掩膜外
        这样移动的是一个**轮廓段**（沿法向平移），而不是打一个圆包。
        """
        n = self._normal(y, x)
        H, W = self.mask.shape
        y0, y1 = max(0, y - radius), min(H, y + radius + 1)
        x0, x1 = max(0, x - radius), min(W, x + radius + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d = (yy - y) * n[0] + (xx - x) * n[1]
        self.mask[y0:y1, x0:x1] = d <= float(offset)

    def step(self, action: int) -> tuple[StepObs, float, bool, dict]:
        y, x = self._cur
        d = int(ACTION_OFFSETS[action])
        before = dice_slice(self.mask, self.gt) if self.gt is not None else 0.0

        if d != 0:
            self._shift_boundary(y, x, d)

        after = dice_slice(self.mask, self.gt) if self.gt is not None else before
        reward = (after - before) * 100.0        # 放大量级便于 PPO 学习

        self.t += 1
        self.history.append({"point": (y, x), "action": action, "dice": after, "reward": reward})

        # 下一步换个边界点（若已无边界则结束）
        pts = boundary_points(self.mask, n=1, rng=self.rng)
        if len(pts) == 0 or self.t >= self.steps:
            done = True
            ny, nx = y, x
        else:
            ny, nx = int(pts[0][0]), int(pts[0][1])
            done = False
        self._cur = (ny, nx)
        return StepObs(self._observe(ny, nx), (ny, nx), done, {"dice": after}), reward, done, {
            "dice": after, "mask": self.mask if done else None,
        }

    @property
    def final_dice(self) -> float:
        return dice_slice(self.mask, self.gt) if self.gt is not None else 0.0


# ============================================================== 策略网络
class PolicyNet(nn.Module):
    """小 CNN：3×33×33 局部观测 → 5 个动作的 logits + 状态价值。"""

    def __init__(self, in_ch: int = 3, n_actions: int = N_ACTIONS, width: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.GroupNorm(4, width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(4, width), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.GroupNorm(4, width * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.GroupNorm(4, width * 2), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        feat = width * 2
        self.pi = nn.Linear(feat, n_actions)
        self.v = nn.Linear(feat, 1)

    def forward(self, x: torch.Tensor):
        h = self.body(x).flatten(1)
        return self.pi(h), self.v(h).squeeze(-1)


# ============================================================== PPO
@dataclass
class Transition:
    state: np.ndarray
    action: int
    logp: float
    value: float
    reward: float
    done: bool


class PPO:
    """简化版 PPO-clip（单环境采样，回合内做 advantage 标准化）。

    说明：为了在课程设计的算力预算内跑通，这里没有做并行环境、GAE 与价值裁剪，
    只用「回合累积回报 + 回报标准化」的近似。训练报告里会如实写明这一点。
    """

    def __init__(self, net: PolicyNet, lr: float = 3e-4, gamma: float = 0.95,
                 clip: float = 0.2, epochs: int = 4, device: str = "cpu"):
        self.net = net.to(device)
        self.device = device
        self.opt = torch.optim.Adam(net.parameters(), lr=lr)
        self.gamma = gamma
        self.clip = clip
        self.epochs = epochs

    @torch.no_grad()
    def act(self, state: np.ndarray) -> tuple[int, float, float]:
        x = torch.from_numpy(state[None]).to(self.device)
        logits, v = self.net(x)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(v.item())

    def update(self, trajs: list[list[Transition]]) -> dict:
        states, actions, logps_old, returns = [], [], [], []
        for traj in trajs:
            ret = 0.0
            rets_slice = []
            for tr in reversed(traj):
                ret = tr.reward + self.gamma * ret * (0.0 if tr.done else 1.0)
                rets_slice.append(ret)
            rets_slice.reverse()
            for tr, r in zip(traj, rets_slice):
                states.append(tr.state)
                actions.append(tr.action)
                logps_old.append(tr.logp)
                returns.append(r)

        if not states:
            return {"loss": 0.0, "entropy": 0.0}

        S = torch.from_numpy(np.stack(states)).to(self.device)
        A = torch.tensor(actions, device=self.device)
        LP_old = torch.tensor(logps_old, dtype=torch.float32, device=self.device)
        R = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if len(R) > 1:
            R = (R - R.mean()) / (R.std() + 1e-8)

        stats = {}
        for _ in range(self.epochs):
            logits, v = self.net(S)
            dist = torch.distributions.Categorical(logits=logits)
            lp = dist.log_prob(A)
            ratio = torch.exp(lp - LP_old)
            s1 = ratio * R
            s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * R
            policy_loss = -torch.min(s1, s2).mean()
            value_loss = F.mse_loss(v, R)
            entropy = dist.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()
            stats = {"loss": float(loss.item()), "policy_loss": float(policy_loss.item()),
                     "value_loss": float(value_loss.item()), "entropy": float(entropy.item())}
        return stats


if __name__ == "__main__":
    # 冒烟：构造一个"偏小"的初始掩膜，看 agent 能否学会往外扩
    rng = np.random.RandomState(0)
    H = W = 64
    yy, xx = np.mgrid[0:H, 0:W]
    gt = ((yy - 32) ** 2 + (xx - 32) ** 2) <= 14 ** 2
    init = ((yy - 32) ** 2 + (xx - 32) ** 2) <= 11 ** 2        # 系统性偏小 3 像素
    ct = np.where(gt, 40.0, -200.0) + rng.randn(H, W) * 20     # 原图边界清楚
    prob = gt.astype(np.float32) * 0.6 + init.astype(np.float32) * 0.35

    env = RefineEnv(ct, prob, gt, init, steps=10, seed=1)
    obs = env.reset()
    print(f"初始 Dice = {env.initial_dice:.4f}")
    net = PolicyNet()
    agent = PPO(net, device="cpu")
    print(f"策略网络参数量: {sum(p.numel() for p in net.parameters())/1e3:.1f}K")

    traj = []
    for t in range(10):
        a, logp, v = agent.act(obs.state)
        nobs, r, done, info = env.step(a)
        traj.append(Transition(obs.state, a, logp, v, r, done))
        obs = nobs
        if done:
            break
    print(f"随机策略一轮后 Dice = {env.final_dice:.4f}")
    st = agent.update([traj])
    print(f"PPO 更新: {st}")
    print("✅ RL 模块冒烟测试通过（结构可跑通，实际效果需在真实数据上训练）")
