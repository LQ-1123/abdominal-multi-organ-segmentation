"""分割损失函数。

核心问题：**缺失器官必须被屏蔽**
------------------------------
AMOS 里器官缺失是常态——盆腔外的前列腺、视野外的胃。如果直接把所有类别都算进
Dice，缺失类别的 Dice 分母为 0，会产生 NaN 或把整体指标严重拉低。

正确做法（本文件的实现）：
    - 每一批数据都带有 ``present[C]`` 掩膜，标记该 batch 里哪些类别真实存在
    - Dice 只在 present 的类别上求平均
    - CrossEntropy 对全部体素计算（背景也在内），因为 CE 不需要分母

为什么是 Dice + CE 混合：
    - 纯 CE：类别不平衡时小器官（肾上腺、十二指肠）被背景淹没
    - 纯 Dice：训练初期梯度不稳定
    - 混合：CE 提供稳定梯度，Dice 直接优化重叠度 —— 医学分割的标准配方
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DiceCELoss", "dice_score"]


def dice_score(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    present: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """返回每个类别的 Dice（未平均）。缺失类别返回 NaN，便于外部忽略。"""
    probs = torch.softmax(logits, dim=1)                     # [B, C, ...]
    target_1h = F.one_hot(target.clamp(0, num_classes - 1), num_classes).movedim(-1, 1).float()

    dims = list(range(2, probs.ndim))
    inter = (probs * target_1h).sum(dim=dims)                # [B, C]
    denom = probs.sum(dim=dims) + target_1h.sum(dim=dims)
    dice = (2 * inter + eps) / (denom + eps)                 # [B, C]

    if present is not None:
        dice = dice.masked_fill(~present, float("nan"))
    return dice


class DiceCELoss(nn.Module):
    """Dice + CrossEntropy 混合损失，按 batch 内存在性屏蔽缺失类别。

    Args:
        num_classes: 类别数（含背景）
        include_background: Dice 是否计入背景类（默认 False，医学分割惯例）
        weight_dice: Dice 权重
        weight_ce: CE 权重
        class_weights: 可选的类别权重（进一步缓解不平衡），长度 num_classes
    """

    def __init__(
        self,
        num_classes: int = 16,
        include_background: bool = False,
        weight_dice: float = 1.0,
        weight_ce: float = 1.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        present: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ---- CrossEntropy ----
        ce = F.cross_entropy(logits, target.clamp(0, self.num_classes - 1), weight=self.class_weights)

        # ---- Dice ----
        dice = dice_score(logits, target, self.num_classes, present)   # [B, C]，缺失处为 NaN
        if not self.include_background:
            dice = dice[:, 1:]                                          # 丢掉背景列
        # nanmean 自动忽略缺失类别；若整批都缺（极端情况）则记为 0
        dice_mean = torch.nanmean(dice) if torch.isnan(dice).any() else dice.mean()
        if torch.isnan(dice_mean):
            dice_mean = torch.zeros((), device=logits.device)
        dice_loss = 1.0 - dice_mean

        return self.weight_dice * dice_loss + self.weight_ce * ce


if __name__ == "__main__":
    torch.manual_seed(0)
    C = 16
    logits = torch.randn(2, C, 8, 8, 8, requires_grad=True)
    target = torch.randint(0, C, (2, 8, 8, 8))
    present = torch.zeros(2, C, dtype=torch.bool)
    for b in range(2):
        for c in torch.unique(target[b]):
            present[b, c] = True
    present[:, 0] = True

    loss_fn = DiceCELoss(C)
    loss = loss_fn(logits, target, present)
    print(f"loss = {loss.item():.4f}")
    loss.backward()
    print(f"grad ok: {logits.grad is not None and torch.isfinite(logits.grad).all()}")

    d = dice_score(logits.detach(), target, C, present)
    print(f"Dice 矩阵 {tuple(d.shape)}，缺失类别数 = {int(torch.isnan(d).sum())}")
    print("✅ 损失函数冒烟测试通过")
