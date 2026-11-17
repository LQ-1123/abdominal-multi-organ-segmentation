"""用真实体积 + 真实权重量一次完整滑窗推理的显存峰值。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from infer.predictor import Predictor  # noqa: E402


def run(ckpt: Path, patch, sw_bs, overlap) -> None:
    vol = np.load(ROOT / "data/cache/amos22/amos_0001.img.npy")
    torch.cuda.reset_peak_memory_stats()
    p = Predictor([ckpt], patch=patch, overlap=overlap)
    # 把滑窗 batch 从 2 改小（直接改属性不行，predict_logits 里写死 sw_bs=2，
    # 这里手动内联同样流程以可控参数测量）
    from monai.inferers import sliding_window_inference
    x = torch.from_numpy(Predictor.normalize(vol))[None, None].cuda()
    with torch.no_grad(), torch.autocast("cuda"):
        def _pred(inp):
            o = p.models[0](inp)
            return o[0] if isinstance(o, (list, tuple)) else o
        logits = sliding_window_inference(
            x, patch, sw_batch_size=sw_bs, predictor=_pred,
            overlap=overlap, mode="gaussian", progress=False)
    lbl = logits.argmax(1).cpu().numpy()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  体积={vol.shape} patch={patch} sw_bs={sw_bs} overlap={overlap} "
          f"-> 峰值={peak:.0f}MB  类别={np.unique(lbl).tolist()[:6]}")
    del p, x, logits, lbl
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ckpt = ROOT / "outputs_v2/fold0/best.pth"
    print(f"权重: {ckpt}")
    run(ckpt, (96, 96, 96), 2, 0.5)
    run(ckpt, (96, 96, 96), 1, 0.5)
    run(ckpt, (64, 64, 64), 1, 0.5)
