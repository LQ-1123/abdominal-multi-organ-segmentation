"""DICOM / NIfTI ingestion -> unified 3D volume representation.

这是整个系统最关键的工程模块。它把「脏乱的 DICOM 世界」归一化成模型能吃的
标准对象 ``Volume``，五步全部在此完成：

    1. 序列选择  : 一个 study 里可能有平扫/增强/定位片，必须挑出体积序列
    2. HU 转换   : HU = pixel * RescaleSlope + RescaleIntercept
    3. 层序排序  : 按 ImagePositionPatient 在法向量上的投影排序
                   （绝不能用 InstanceNumber，很多机器乱填）
    4. 方向归一  : 用 ImageOrientationPatient 构造方向矩阵，重排到统一解剖朝向
    5. 等向重采样: 层厚 5mm / 层内 0.7mm 很常见，统一到各向同性

做错任何一步的后果：
    1 错 -> 分割到定位片上          2 错 -> 模型输入分布完全错误
    3 错 -> 滚动时层序乱跳          4 错 -> 左右镜像，脾脏跑到右侧
    5 错 -> 滚动一卡一卡、矢状面拉长

验收方式：把原始 DICOM 和解析结果同时拖入 3D Slicer，观察是否自动配准叠加。

用法:
    >>> series = scan_directory("data/raw/case01")      # 列出所有序列
    >>> best   = pick_best_series(series)               # 自动挑体积序列
    >>> vol    = load_series(best, target_spacing=(1.5, 1.5, 1.5))
    >>> vol.pixels.shape, vol.spacing
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError as exc:  # pragma: no cover
    raise ImportError("需要 pydicom: pip install pydicom") from exc

__all__ = [
    "Volume",
    "SeriesInfo",
    "scan_directory",
    "pick_best_series",
    "load_series",
    "load_nifti",
    "save_nifti",
]


# =============================================================================
# 统一中间表示
# =============================================================================
@dataclass
class Volume:
    """标准三维体积。模型只认这个对象，不关心它从 DICOM 还是 NIfTI 来。"""

    pixels: np.ndarray                  # float32 [Z, Y, X]，已转 HU（或原始强度）
    spacing: tuple[float, float, float] # (z, y, x) 毫米
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    direction: np.ndarray = field(default_factory=lambda: np.eye(3))
    source: str = "unknown"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pixels = np.asarray(self.pixels, dtype=np.float32)
        if self.pixels.ndim != 3:
            raise ValueError(f"Volume.pixels 必须是 3D，得到 {self.pixels.shape}")

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.pixels.shape)  # type: ignore[return-value]

    @property
    def n_slices(self) -> int:
        return int(self.pixels.shape[0])

    def __repr__(self) -> str:
        return (f"Volume(shape={self.shape}, spacing="
                f"({self.spacing[0]:.2f},{self.spacing[1]:.2f},{self.spacing[2]:.2f})mm, "
                f"HU=[{self.pixels.min():.0f},{self.pixels.max():.0f}], source={self.source})")


@dataclass
class SeriesInfo:
    """一个 DICOM 序列的元信息（不含像素，扫描阶段用）。"""

    series_uid: str
    description: str
    modality: str
    files: list[str]
    rows: int = 0
    cols: int = 0
    slice_thickness: float = 0.0

    @property
    def n_files(self) -> int:
        return len(self.files)

    def __repr__(self) -> str:
        return (f"SeriesInfo(desc={self.description!r}, modality={self.modality}, "
                f"n={self.n_files}, {self.rows}x{self.cols}, "
                f"thick={self.slice_thickness:.1f}mm)")


# =============================================================================
# 1. 扫描目录，按 SeriesInstanceUID 分组
# =============================================================================
def scan_directory(root: str | os.PathLike[str]) -> list[SeriesInfo]:
    """递归扫描目录，把所有 DICOM 文件按序列分组。

    只读元信息（stop_before_pixels=True），因此很快，不动磁盘上的大像素数据。
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)

    groups: dict[str, dict] = {}
    # 严格模式读不出来的文件先记下来：有些导出工具会去掉 128 字节前导码，
    # 变成「裸数据集」，需要 force=True 才能读。若严格模式一个序列都没找到，
    # 再拿这些文件重试一次，避免整个上传失败。
    skipped: list[str] = []

    def _scan_file(fp: str, force: bool):
        try:
            ds = pydicom.dcmread(fp, stop_before_pixels=True, force=force)
        except (InvalidDicomError, Exception):  # noqa: BLE001 - 非 DICOM 文件跳过
            return False

        uid = getattr(ds, "SeriesInstanceUID", None)
        if uid is None:
            return False
        g = groups.get(uid)
        if g is None:
            g = groups[uid] = {
                "files": [],
                "desc": str(getattr(ds, "SeriesDescription", "") or ""),
                "modality": str(getattr(ds, "Modality", "") or ""),
                "rows": int(getattr(ds, "Rows", 0) or 0),
                "cols": int(getattr(ds, "Columns", 0) or 0),
                "thick": float(getattr(ds, "SliceThickness", 0) or 0),
            }
        g["files"].append(fp)
        return True

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            if not _scan_file(fp, force=False):
                skipped.append(fp)

    if not groups and skipped:
        for fp in skipped:
            _scan_file(fp, force=True)

    out: list[SeriesInfo] = []
    for uid, g in groups.items():
        out.append(SeriesInfo(
            series_uid=uid, description=g["desc"], modality=g["modality"],
            files=sorted(g["files"]), rows=g["rows"], cols=g["cols"],
            slice_thickness=g["thick"],
        ))
    # 层数多的排前面（体积序列一般层数最多）
    out.sort(key=lambda s: s.n_files, reverse=True)
    return out


# =============================================================================
# 2. 选序列
# =============================================================================
_NON_VOLUME_HINTS = ("localizer", "scout", "topogram", "surview", "定位", "定位像")


def pick_best_series(series: Sequence[SeriesInfo], modality: str = "CT") -> SeriesInfo:
    """挑出最可能是「体积序列」的那一个。

    规则（按重要性）：
        1. 排除定位片 / scout（描述关键词）
        2. 模态匹配（默认 CT）
        3. 层数最多
        4. 层数相同则面内分辨率大者优先
    """
    if not series:
        raise ValueError("未找到任何 DICOM 序列")

    cands = [s for s in series if s.modality.upper() == modality.upper()] or list(series)
    filtered = [s for s in cands
                if not any(h in s.description.lower() for h in _NON_VOLUME_HINTS)]
    pool = filtered or cands

    return max(pool, key=lambda s: (s.n_files, s.rows * s.cols))


# =============================================================================
# 3-5. 加载序列：读像素 -> HU -> 排序 -> 方向归一 -> 等向重采样
# =============================================================================
def _read_dataset(fp: str):
    return pydicom.dcmread(fp, force=False)


def _hu_from_dataset(ds) -> np.ndarray:
    """像素 -> HU。正确处理 Slope/Intercept 缺失、为 1/0 的边界情况。"""
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return arr * slope + intercept


def _slice_normal_and_position(ds) -> tuple[np.ndarray, np.ndarray]:
    """返回 (法向量, 层位置向量)。用于正确排序层。"""
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if iop is None or ipp is None:
        raise ValueError("缺少 ImageOrientationPatient / ImagePositionPatient，无法定位层")

    row = np.array([float(v) for v in iop[:3]])
    col = np.array([float(v) for v in iop[3:6]])
    normal = np.cross(row, col)
    n = np.linalg.norm(normal)
    normal = normal / n if n > 0 else np.array([0.0, 0.0, 1.0])
    return normal, np.array([float(v) for v in ipp])


def load_series(
    info: SeriesInfo,
    target_spacing: tuple[float, float, float] | None = (1.5, 1.5, 1.5),
    max_slices: int | None = None,
) -> Volume:
    """把一个 DICOM 序列加载成标准 Volume。

    Args:
        info: 由 scan_directory / pick_best_series 得到的序列信息
        target_spacing: 等向重采样目标 spacing (z,y,x) mm；None 表示不重采样
        max_slices: 最多取多少层（调试用）
    """
    if not info.files:
        raise ValueError("序列为空")

    datasets = []
    for fp in info.files:
        ds = _read_dataset(fp)
        if getattr(ds, "PixelData", None) is None:
            continue          # 有些文件是 dose report 之类，没有像素
        datasets.append(ds)

    if not datasets:
        raise ValueError("该序列没有任何含像素的 DICOM 文件")

    # ---- 3. 层序排序：按 IPP 在法向量上的投影，而非 InstanceNumber ----
    normal, _ = _slice_normal_and_position(datasets[0])
    keyed = []
    for ds in datasets:
        _, ipp = _slice_normal_and_position(ds)
        keyed.append((float(np.dot(ipp, normal)), ds))
    keyed.sort(key=lambda t: t[0])

    if max_slices:
        keyed = keyed[:max_slices]
    datasets = [ds for _, ds in keyed]

    # ---- 2. HU 转换 + 堆叠 ----
    slices = [_hu_from_dataset(ds) for ds in datasets]
    shapes = {s.shape for s in slices}
    if len(shapes) != 1:
        raise ValueError(f"序列内各层尺寸不一致: {shapes}")
    volume = np.stack(slices, axis=0)     # [Z, Y, X]

    # ---- spacing ----
    ds0 = datasets[0]
    ps = getattr(ds0, "PixelSpacing", [1.0, 1.0])
    dy, dx = float(ps[0]), float(ps[1])
    if len(datasets) > 1:
        # 层间距取相邻层的实际几何距离，比 SliceThickness 更可靠
        _, p0 = _slice_normal_and_position(datasets[0])
        _, p1 = _slice_normal_and_position(datasets[1])
        dz = float(abs(np.dot(p1 - p0, normal))) or float(getattr(ds0, "SliceThickness", 1.0) or 1.0)
    else:
        dz = float(getattr(ds0, "SliceThickness", 1.0) or 1.0)

    spacing = (dz, dy, dx)

    # ---- 4. 方向：把 row/col/normal 转成标准 LPS 方向的映射 ----
    iop = getattr(ds0, "ImageOrientationPatient", None)
    if iop is not None:
        row = np.array([float(v) for v in iop[:3]])
        col = np.array([float(v) for v in iop[3:6]])
        direction = np.stack([normal, col, row], axis=1)   # 列向量 = (z,y,x) 轴方向
    else:
        direction = np.eye(3)

    ipp = getattr(ds0, "ImagePositionPatient", None)
    origin = tuple(float(v) for v in ipp) if ipp is not None else (0.0, 0.0, 0.0)

    vol = Volume(
        pixels=volume, spacing=spacing, origin=origin,
        direction=direction, source="dicom",
        meta={"series_uid": info.series_uid, "description": info.description,
              "modality": info.modality, "n_files": len(datasets)},
    )

    if target_spacing is not None:
        vol = resample_isotropic(vol, target_spacing)

    return vol


# =============================================================================
# 等向重采样
# =============================================================================
def resample_isotropic(vol: Volume, target_spacing: tuple[float, float, float],
                       order: int = 1) -> Volume:
    """重采样到目标 spacing。用 SimpleITK 保证与 NIfTI / Slicer 一致。"""
    try:
        import SimpleITK as sitk
    except ImportError:  # 退化到 scipy，精度略差
        return _resample_scipy(vol, target_spacing, order)

    img = sitk.GetImageFromArray(vol.pixels)
    img.SetSpacing(tuple(float(s) for s in vol.spacing[::-1]))   # SITK 是 (x,y,z)
    img.SetOrigin(tuple(float(o) for o in vol.origin[::-1]))

    orig_size = np.array(img.GetSize(), dtype=float)
    orig_spacing = np.array(img.GetSpacing(), dtype=float)
    tgt_spacing = np.array(target_spacing[::-1], dtype=float)
    tgt_size = np.maximum(np.round(orig_size * orig_spacing / tgt_spacing).astype(int), 1)

    rs = sitk.ResampleImageFilter()
    rs.SetSize([int(v) for v in tgt_size])
    rs.SetOutputSpacing([float(v) for v in tgt_spacing])
    rs.SetOutputOrigin(img.GetOrigin())
    rs.SetOutputDirection(img.GetDirection())
    rs.SetInterpolator(sitk.sitkLinear if order > 0 else sitk.sitkNearestNeighbor)
    out = rs.Execute(img)

    arr = sitk.GetArrayFromImage(out).astype(np.float32)
    return Volume(
        pixels=arr,
        spacing=tuple(float(v) for v in out.GetSpacing()[::-1]),
        origin=tuple(float(v) for v in out.GetOrigin()[::-1]),
        direction=vol.direction, source=vol.source, meta=dict(vol.meta),
    )


def _resample_scipy(vol: Volume, target_spacing, order: int) -> Volume:  # pragma: no cover
    from scipy.ndimage import zoom

    factors = np.array(vol.spacing) / np.array(target_spacing, dtype=float)
    arr = zoom(vol.pixels, factors, order=order)
    return Volume(pixels=arr.astype(np.float32), spacing=tuple(target_spacing),
                  origin=vol.origin, direction=vol.direction,
                  source=vol.source, meta=dict(vol.meta))


# =============================================================================
# NIfTI IO（训练数据的主要格式）
# =============================================================================
def load_nifti(path: str | os.PathLike[str], is_label: bool = False) -> Volume:
    """读 NIfTI。标签用最近邻以外的插值都不合适，所以单独标记。"""
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:                      # 有些文件多一个时间维
        data = data[..., 0]
    # nibabel 的数组轴序是 (X, Y, Z)，转到我们的 (Z, Y, X)
    arr = np.transpose(data, (2, 1, 0))
    zooms = img.header.get_zooms()[:3]
    spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    dtype = np.uint8 if is_label else np.float32

    return Volume(
        pixels=arr.astype(dtype), spacing=spacing,
        origin=tuple(float(v) for v in img.affine[:3, 3]),
        direction=img.affine[:3, :3] / np.array(spacing)[::-1],
        source="nifti", meta={"path": str(path)},
    )


def save_nifti(vol: Volume, path: str | os.PathLike[str]) -> None:
    """保存为 NIfTI（轴序转回 XYZ）。"""
    import nibabel as nib

    arr = np.transpose(vol.pixels, (2, 1, 0))
    zooms = (vol.spacing[2], vol.spacing[1], vol.spacing[0])
    affine = np.eye(4)
    affine[:3, :3] = vol.direction * np.array(zooms)
    affine[:3, 3] = vol.origin
    nib.save(nib.Nifti1Image(arr.astype(np.int16) if vol.pixels.dtype == np.uint8
                             else arr.astype(np.float32), affine),
             str(path))
