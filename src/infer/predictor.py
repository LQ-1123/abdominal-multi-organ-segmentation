"""推理引擎：滑窗推理 + 5 折集成。

流程
----
    Volume(任意来源) → 归一化 → [逐折滑窗推理 → softmax 概率] → 平均 → argmax → LabelMap

为什么必须滑窗
--------------
重采样到 1.5mm 后单卷约 300×292×292 = 2500 万体素，整卷塞进显存会 OOM。
滑窗把体积切成 96³ 小块逐块推理，再用高斯权重融合边界（避免拼接接缝）。

为什么做 5 折集成
----------------
5 个 fold 各自在不同训练子集上过拟合了不同的噪声，概率平均相当于一次免费的
模型集成，通常能提升 2~3 个 Dice 点，是性价比最高的一招。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from models.unet3d import UNet3D  # noqa: E402

HU_MIN, HU_MAX = -150.0, 250.0
NUM_CLASSES = 16


class Predictor:
    """多折集成分割预测器。"""

    def __init__(
        self,
        ckpt_paths: list[str | Path],
        patch: tuple[int, int, int] = (96, 96, 96),
        overlap: float = 0.5,
        num_classes: int = NUM_CLASSES,
        device: str | None = None,
        sw_batch_size: int = 2,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.patch = patch
        self.overlap = overlap
        self.num_classes = num_classes
        self.sw_batch_size = sw_batch_size   # 小显存(<6GB)建议设 1，峰值可省约 500MB

        self.models = []
        for p in ckpt_paths:
            ckpt = torch.load(str(p), map_location=self.device, weights_only=True)
            a = ckpt.get("args", {})
            m = UNet3D(
                in_ch=1,
                out_ch=a.get("out_ch", num_classes),
                features=tuple(a.get("features", (32, 64, 128, 256, 512))),
                # 必须与训练时一致：深监督头虽然推理用不到，但权重键要对得上
                deep_supervision=True,
            ).to(self.device)
            m.load_state_dict(ckpt["model"])
            m.eval()
            self.models.append(m)
        print(f"[Predictor] 已加载 {len(self.models)} 个模型到 {self.device}")

    # ------------------------------------------------------------------ 归一化
    @staticmethod
    def normalize(vol_hu: np.ndarray) -> np.ndarray:
        x = np.clip(vol_hu.astype(np.float32), HU_MIN, HU_MAX)
        return ((x - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)

    @torch.no_grad()
    def predict_logits(self, vol_hu: np.ndarray, verbose: bool = True) -> np.ndarray:
        """输入原始 HU 体积 [Z,Y,X]，返回集成后的 logits [C,Z,Y,X] (float16)。"""
        from monai.inferers import sliding_window_inference

        x = torch.from_numpy(self.normalize(vol_hu))[None, None].to(self.device)
        acc = None
        for i, m in enumerate(self.models):
            # 深监督模型返回 list，滑窗推理只取主输出
            def _pred(inp, _m=m):
                o = _m(inp)
                return o[0] if isinstance(o, (list, tuple)) else o

            logits = sliding_window_inference(
                x, self.patch, sw_batch_size=self.sw_batch_size, predictor=_pred,
                overlap=self.overlap, mode="gaussian", progress=False,
            )
            if acc is None:
                acc = logits.to(torch.float16)
            else:
                acc = acc + logits.to(torch.float16)
            del logits
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            if verbose:
                print(f"  fold {i + 1}/{len(self.models)} 完成")
        acc /= len(self.models)
        return acc[0].cpu().numpy()

    def predict(self, vol_hu: np.ndarray, verbose: bool = True,
                clean: bool = True) -> np.ndarray:
        """返回标签图 uint8 [Z,Y,X]，取值 0..15。

        clean=True 时做连通域清理：模型在训练分布外（如胸部层面）会把肺内病灶
        误判成肝脏，argmax 也会散出大量孤立碎块。这两 15 个器官每个都是单一
        连通结构，因此保留主要连通域是安全且必要的后处理。
        """
        logits = self.predict_logits(vol_hu, verbose=verbose)
        label = logits.argmax(axis=0).astype(np.uint8)
        if clean:
            from infer.postprocess import clean_label
            label, recs = clean_label(label, num_classes=self.num_classes)
            removed = sum(r["voxels_removed"] for r in recs)
            if verbose:
                print(f"  连通域清理: {len(recs)} 个器官，移除 {removed} 个体素")
        return label

    def predict_proba(self, vol_hu: np.ndarray, verbose: bool = True) -> np.ndarray:
        """返回最大类别概率 [Z,Y,X] float32，可用于置信度热力图。"""
        logits = self.predict_logits(vol_hu, verbose=verbose)
        e = np.exp(logits.astype(np.float32) - logits.max(0, keepdims=True))
        p = e / e.sum(0, keepdims=True)
        return p.max(0)


def find_checkpoints(outputs_dir: str | Path = "outputs") -> list[Path]:
    """按 fold 顺序收集所有 best.pth。"""
    d = Path(outputs_dir)
    ckpts = []
    for k in range(10):
        p = d / f"fold{k}" / "best.pth"
        if p.exists():
            ckpts.append(p)
    if not ckpts:
        raise FileNotFoundError(f"{d} 下没有找到 fold*/best.pth")
    return ckpts


if __name__ == "__main__":
    # 冒烟：用一例真实数据跑通
    cache = ROOT / "data" / "cache" / "amos22"
    cid = "amos_0001"
    vol = np.load(cache / f"{cid}.img.npy")
    print(f"输入体积: {vol.shape}, HU=[{vol.min()},{vol.max()}]")

    ckpts = find_checkpoints(ROOT / "outputs")
    pred = Predictor(ckpts[:1], patch=(96, 96, 96))       # 冒烟只用 1 折
    lbl = pred.predict(vol, verbose=False)
    print(f"输出标签: {lbl.shape}, 类别={np.unique(lbl).tolist()}")

    gt = np.load(cache / f"{cid}.lbl.npy")
    dices = []
    for c in range(1, NUM_CLASSES):
        g, p = gt == c, lbl == c
        if g.sum() == 0:
            continue
        dices.append(2 * (g & p).sum() / (g.sum() + p.sum() + 1e-8))
    print(f"该例 mDice = {np.mean(dices):.4f}  (训练集病例，仅作管线验证)")
    print("✅ 推理引擎冒烟测试通过")
