"""课本里的传统预处理，作用在 1.5 mm 等向的 HU 体积上。

所有方法最终都输出模型输入：窗宽窗位 [-150, 250] HU 线性映射到 [0, 1] 的 float32。
推理侧消融（scripts/run_ablations.py）和训练侧微调（scripts/finetune_preprocess.py）
共用这里的实现，保证两边的“高斯滤波”“CLAHE”是同一个操作。

| 方法 | 课本章节 | 作用位置 |
|---|---|---|
| gaussian | 空间域平滑：高斯低通 | 窗限前，HU 空间 |
| median | 空间域平滑：中值滤波（统计排序） | 窗限前，HU 空间 |
| butterworth | 频率域：Butterworth 低通 | 窗限前，HU 空间，3D FFT |
| clahe | 图像增强：限制对比度自适应直方图均衡 | 窗限后，逐轴位切片 |
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

HU_MIN, HU_MAX = -150.0, 250.0
SPACING_MM = 1.5
METHODS = ("none", "gaussian", "median", "butterworth", "clahe")
GAUSSIAN_SIGMA_MM = 1.0
MEDIAN_SIZE = 3
BUTTERWORTH_CUTOFF = 0.2   # cycles/mm；1.5 mm 采样的奈奎斯特频率约 0.333
BUTTERWORTH_ORDER = 2
CLAHE_KERNEL = (64, 64)
CLAHE_CLIP = 0.01


def window(hu: np.ndarray) -> np.ndarray:
    x = np.clip(hu.astype(np.float32, copy=False), HU_MIN, HU_MAX)
    return (x - HU_MIN) / (HU_MAX - HU_MIN)


def butterworth_lowpass(image: np.ndarray, cutoff: float = BUTTERWORTH_CUTOFF,
                        order: int = BUTTERWORTH_ORDER, spacing: float = SPACING_MM) -> np.ndarray:
    """3D 傅里叶变换后乘 H = 1 / (1 + (D/D0)^(2n))，再逆变换。

    D 是频率点到原点的距离（cycles/mm）。先减去均值再变换，直流分量不受边界延拓影响。
    """
    from scipy import fft

    x = image.astype(np.float32)
    offset = float(x.mean())
    spectrum = fft.rfftn(x - offset, workers=-1)
    freqs = [fft.fftfreq(n, d=spacing) for n in x.shape[:-1]] + [fft.rfftfreq(x.shape[-1], d=spacing)]
    d2 = np.zeros(spectrum.shape, dtype=np.float32)
    for axis, f in enumerate(freqs):
        shape = [1] * x.ndim
        shape[axis] = len(f)
        d2 += (f.astype(np.float32) ** 2).reshape(shape)
    spectrum *= 1.0 / (1.0 + (d2 / cutoff ** 2) ** order)
    del d2
    return fft.irfftn(spectrum, s=x.shape, workers=-1).astype(np.float32) + offset


def clahe_axial(normalized: np.ndarray) -> np.ndarray:
    from skimage.exposure import equalize_adapthist

    out = normalized.copy()
    for z, plane in enumerate(out):
        if np.ptp(plane):
            out[z] = equalize_adapthist(plane, kernel_size=CLAHE_KERNEL, clip_limit=CLAHE_CLIP)
    return out


def apply(image_hu: np.ndarray, method: str, spacing: float = SPACING_MM) -> np.ndarray:
    """HU 体积 -> 模型输入 [0, 1]。"""
    if method == "none":
        return window(image_hu)
    if method == "gaussian":
        return window(ndi.gaussian_filter(image_hu, sigma=GAUSSIAN_SIGMA_MM / spacing, output=np.float32))
    if method == "median":
        return window(ndi.median_filter(image_hu, size=MEDIAN_SIZE))
    if method == "butterworth":
        return window(butterworth_lowpass(image_hu, spacing=spacing))
    if method == "clahe":
        return clahe_axial(window(image_hu))
    raise ValueError(f"未知预处理方法：{method}")


def to_cache_hu(normalized: np.ndarray) -> np.ndarray:
    """把 [0, 1] 的模型输入映射回窗内 HU 并取整，存成 int16 缓存。

    读取时再用 window() 即可还原，量化误差不超过 1/800。
    """
    return np.rint(normalized * (HU_MAX - HU_MIN) + HU_MIN).astype(np.int16)
