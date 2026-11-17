"""3D U-Net for multi-organ abdominal CT segmentation.

设计要点
--------
1. ``out_ch`` 可配置：从第一天就为「单器官 → 多器官」预留，切换只需改一个数字。
2. ``deep_supervision``：多器官小数据下能稳定梯度、加速收敛。
3. 只用 PyTorch 原生算子，不依赖 MONAI，便于移植与答辩讲解。
4. 归一化用 InstanceNorm3d（医学分割常用，batch 很小时比 BN 稳）。

用法
----
>>> m = UNet3D(in_ch=1, out_ch=9)          # 背景 + 8 器官
>>> y = m(torch.randn(2, 1, 96, 96, 96))   # -> [2, 9, 96, 96, 96]
>>> m = UNet3D(in_ch=1, out_ch=9, deep_supervision=True)
>>> ys = m(x)                              # -> list[Tensor], 由细到粗
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["UNet3D", "ConvBlock", "build_model"]


class ConvBlock(nn.Module):
    """(Conv3d -> InstanceNorm -> LeakyReLU) x 2"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    """3D U-Net with configurable class count and optional deep supervision.

    Args:
        in_ch: 输入通道数（CT 单通道=1）。
        out_ch: 输出通道数 = 1 + 器官数。
        features: 各层级通道数，长度决定深度（默认 5 层）。
        deep_supervision: 是否输出多尺度辅助预测（训练用，推理取 [0]）。
        dropout: 解码器 dropout 概率（小数据可略微提升泛化）。
    """

    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 2,
        features: Sequence[int] = (32, 64, 128, 256, 512),
        deep_supervision: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(features) < 2:
            raise ValueError("features 至少需要 2 层")
        self.out_ch = out_ch
        self.deep_supervision = deep_supervision
        self.n_stages = len(features)

        # ---- Encoder ----
        self.stem = ConvBlock(in_ch, features[0])
        self.down_convs = nn.ModuleList(
            ConvBlock(features[i - 1], features[i]) for i in range(1, self.n_stages)
        )

        # ---- Decoder ----
        self.up_trans = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(self.n_stages - 1, 0, -1):
            self.up_trans.append(nn.ConvTranspose3d(features[i], features[i - 1], kernel_size=2, stride=2))
            self.up_convs.append(ConvBlock(features[i], features[i - 1]))

        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        # ---- Heads：推理只用主头；深监督额外挂 1/2、1/4、1/8 分辨率辅助头 ----
        self.main_head = nn.Conv3d(features[0], out_ch, kernel_size=1)
        if deep_supervision:
            self.aux_heads = nn.ModuleList(
                nn.Conv3d(features[0], out_ch, kernel_size=1) for _ in range(self.n_stages - 1)
            )
        else:
            self.aux_heads = None

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        # Encoder：保存每一级特征用于 skip connection
        skips: list[torch.Tensor] = []
        feat = self.stem(x)
        skips.append(feat)
        for conv in self.down_convs:
            feat = conv(F.max_pool3d(feat, kernel_size=2, stride=2))
            skips.append(feat)

        # Decoder
        feat = skips[-1]
        for idx, (up, conv) in enumerate(zip(self.up_trans, self.up_convs)):
            feat = up(feat)
            skip = skips[-(idx + 2)]
            # 尺寸兜底：奇数尺寸输入时对齐到 skip 的空间大小
            if feat.shape[2:] != skip.shape[2:]:
                feat = F.interpolate(feat, size=skip.shape[2:], mode="trilinear", align_corners=False)
            feat = conv(torch.cat([feat, skip], dim=1))
            feat = self.dropout(feat)

        pred = self.main_head(feat)
        if not self.deep_supervision or self.aux_heads is None:
            return pred

        preds = [pred]
        for aux in self.aux_heads:
            preds.append(aux(feat))
            # 退化到半分辨率，模拟深监督的多尺度
            feat = F.avg_pool3d(feat, kernel_size=2, stride=2)
        return preds


def build_model(cfg: dict) -> UNet3D:
    """从配置字典构建模型。

    >>> build_model({"in_ch": 1, "out_ch": 16, "features": [32, 64, 128, 256, 512]})
    """
    return UNet3D(
        in_ch=cfg.get("in_ch", 1),
        out_ch=cfg["out_ch"],
        features=tuple(cfg.get("features", (32, 64, 128, 256, 512))),
        deep_supervision=cfg.get("deep_supervision", False),
        dropout=cfg.get("dropout", 0.0),
    )


if __name__ == "__main__":
    # 冒烟测试：形状 + 参数量 + 显存粗估
    for out_ch in (2, 9, 16):
        net = UNet3D(in_ch=1, out_ch=out_ch)
        n = sum(p.numel() for p in net.parameters())
        x = torch.randn(1, 1, 96, 96, 96)
        with torch.no_grad():
            y = net(x)
        print(f"out_ch={out_ch:<3} params={n/1e6:6.2f}M  in={tuple(x.shape)} -> out={tuple(y.shape)}")
