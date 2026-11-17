"""估算 4GB 显存笔记本上跑剩余任务的可行性：实测各配置的显存峰值。

只测前向/反向一步的峰值（torch.cuda.max_memory_allocated），
峰值与所在显卡的容量无关，可直接作为目标机器的需求估计。
另测 CPU 单 patch 前向耗时，作为无 GPU 时的参照。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.unet3d import UNet3D  # noqa: E402

FEAT = (32, 64, 128, 256)          # 与训练一致（train_amos.py 默认）
OUT_CH = 16


def peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**2


def bench_infer(patch: int, amp: bool, sw_bs: int) -> None:
    """单模型推理峰值（模拟 MONAI 滑窗一次喂 sw_bs 个 patch）。"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = UNet3D(in_ch=1, out_ch=OUT_CH, features=FEAT, deep_supervision=True).cuda().eval()
    params_mb = sum(p.numel() for p in m.parameters()) * 4 / 1024**2
    x = torch.randn(sw_bs, 1, patch, patch, patch, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", enabled=amp):
        y = m(x)
        y = y[0] if isinstance(y, (list, tuple)) else y
        torch.cuda.synchronize()
    print(f"推理  patch={patch}³  amp={amp}  sw_bs={sw_bs}  "
          f"参数={params_mb:.0f}MB(fp32)  峰值={peak_mb():.0f}MB")
    del m, x, y
    torch.cuda.empty_cache()


def bench_train(patch: int, bs: int, amp: bool) -> None:
    """一步 训练峰值（前向+反向+Adam 更新）。"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = UNet3D(in_ch=1, out_ch=OUT_CH, features=FEAT, deep_supervision=True).cuda().train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    x = torch.randn(bs, 1, patch, patch, patch, device="cuda")
    y = torch.randint(0, OUT_CH, (bs, patch, patch, patch), device="cuda")
    with torch.autocast("cuda", enabled=amp):
        out = m(x)
        outs = out if isinstance(out, (list, tuple)) else [out]
        loss = sum(o.float().mean() for o in outs)   # 占位损失，梯度全通，量级与真实训练相当
        opt.zero_grad()
        loss.backward()
    opt.step()
    torch.cuda.synchronize()
    print(f"训练  patch={patch}³  bs={bs}  amp={amp}  峰值={peak_mb():.0f}MB")
    del m, opt, x, y, out, loss
    torch.cuda.empty_cache()


def bench_ensemble(patch: int, amp: bool) -> None:
    """5 折集成：5 个模型同时驻留 + 单 patch 前向（predictor 的用法）。"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ms = [UNet3D(in_ch=1, out_ch=OUT_CH, features=FEAT, deep_supervision=True).cuda().eval()
          for _ in range(5)]
    x = torch.randn(1, 1, patch, patch, patch, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", enabled=amp):
        y = ms[0](x)
        y = y[0] if isinstance(y, (list, tuple)) else y
        torch.cuda.synchronize()
    print(f"集成×5  patch={patch}³  amp={amp}  峰值={peak_mb():.0f}MB")
    del ms, x, y
    torch.cuda.empty_cache()


def bench_cpu(patch: int) -> None:
    """CPU 单 patch 前向耗时（本机桌面 CPU，笔记本大约慢 1.5-2.5 倍）。"""
    m = UNet3D(in_ch=1, out_ch=OUT_CH, features=FEAT, deep_supervision=True).eval()
    x = torch.randn(1, 1, patch, patch, patch)
    with torch.no_grad():
        m(x)                                  # 预热
        t0 = time.perf_counter()
        m(x)
        dt = time.perf_counter() - t0
    print(f"CPU前向  patch={patch}³  单patch耗时={dt:.1f}s  "
          f"(线程={torch.get_num_threads()})")


if __name__ == "__main__":
    torch.manual_seed(0)
    print(f"torch={torch.__version__}  device={torch.cuda.get_device_name(0)}")
    for p in (64, 96, 128):
        bench_infer(p, amp=True, sw_bs=1)
    bench_infer(96, amp=False, sw_bs=1)
    bench_infer(96, amp=True, sw_bs=2)       # predictor 当前默认
    for p, bs in ((96, 1), (96, 2), (128, 2)):
        bench_train(p, bs, amp=True)
    bench_ensemble(96, amp=True)
    bench_ensemble(64, amp=True)
    for p in (64, 96):
        bench_cpu(p)
