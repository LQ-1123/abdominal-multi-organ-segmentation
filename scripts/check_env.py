"""环境自检：验证 PyTorch / CUDA / sm_120 与关键依赖。

用法:
    .\\.venv\\Scripts\\python.exe scripts\\check_env.py

约定:
    [ OK ]   必需项通过
    [WARN]   可选项缺失（不影响训练）
    [FAIL]   必需项失败（退出码 1）
"""

from __future__ import annotations

import platform
import sys

OK, WARN, BAD = "[ OK ]", "[WARN]", "[FAIL]"

failures: list[str] = []
warnings: list[str] = []


def line(tag: str, msg: str) -> None:
    print(f"{tag} {msg}")


def check(mod: str, required: bool = True) -> bool:
    """检查模块可导入；required=False 时缺失只算警告。"""
    try:
        m = __import__(mod)
        line(OK, f"{mod:<22} {getattr(m, '__version__', 'ok')}")
        return True
    except Exception as exc:  # noqa: BLE001
        if required:
            line(BAD, f"{mod:<22} {type(exc).__name__}: {exc}")
            failures.append(mod)
        else:
            line(WARN, f"{mod:<22} 未安装（可选）")
            warnings.append(mod)
        return False


print("=" * 68)
print("  Medical Image Segmentation - Environment Self-Check")
print("=" * 68)
print(f"Python : {sys.version.split()[0]}  ({sys.executable})")
print(f"OS     : {platform.system()} {platform.release()}")

# ---------------- 1. PyTorch / GPU ----------------
print("\n[ 1 ] PyTorch & GPU")
try:
    import torch

    line(OK, f"{'torch':<22} {torch.__version__}")
    line(OK, f"{'torch.version.cuda':<22} {torch.version.cuda}")

    cuda_ok = torch.cuda.is_available()
    line(OK if cuda_ok else BAD, f"{'cuda.is_available':<22} {cuda_ok}")
    if not cuda_ok:
        failures.append("cuda")

    if cuda_ok:
        cap = torch.cuda.get_device_capability(0)
        line(OK, f"{'device':<22} {torch.cuda.get_device_name(0)}")
        line(OK, f"{'compute capability':<22} sm_{cap[0]}{cap[1]}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        line(OK, f"{'VRAM':<22} {vram:.1f} GB")

        # 关键验证：能否在 sm_120 上真正执行 3D 卷积
        try:
            x = torch.randn(2, 1, 32, 32, 32, device="cuda")
            y = torch.nn.Conv3d(1, 8, 3, padding=1).cuda()(x)
            torch.cuda.synchronize()
            line(OK, f"{'3D conv on sm_120':<22} {tuple(y.shape)}  <- kernel works")
        except Exception as exc:  # noqa: BLE001
            line(BAD, f"{'3D conv on sm_120':<22} {type(exc).__name__}: {exc}")
            line(BAD, f"{'':<27} torch lacks sm_120 kernels, install cu128+ build")
            failures.append("sm_120_kernel")
except Exception as exc:  # noqa: BLE001
    line(BAD, f"torch import failed       {exc}")
    failures.append("torch")

# ---------------- 2. 必需依赖 ----------------
print("\n[ 2 ] Required dependencies")
for mod in [
    "numpy", "scipy", "pandas", "matplotlib", "PIL",
    "pydicom", "SimpleITK", "nibabel", "monai",
    "skimage", "cv2", "fastapi", "uvicorn", "yaml",
]:
    check(mod, required=True)

# ---------------- 3. 可选依赖 ----------------
print("\n[ 3 ] Optional (missing is fine)")
for mod in ["highdicom", "pyvista", "tensorboard"]:
    check(mod, required=False)
print(f"{WARN} {'pydicom-seg':<22} 已弃用（SEG 导出改用 highdicom）")

# ---------------- 结论 ----------------
print("\n" + "=" * 68)
if failures:
    line(BAD, f"{len(failures)} required item(s) failed: {', '.join(failures)}")
    sys.exit(1)
if warnings:
    line(WARN, f"{len(warnings)} optional item(s) missing: {', '.join(warnings)}")
line(OK, "Environment ready for training.")
print("=" * 68)
