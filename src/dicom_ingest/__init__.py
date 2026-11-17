"""DICOM / NIfTI 摄取模块。

对外只暴露统一的入口与原语：

    from dicom_ingest import Volume, scan_directory, pick_best_series, load_series
    from dicom_ingest import load_nifti, save_nifti, resample_isotropic
"""

from .core import (
    SeriesInfo,
    Volume,
    load_nifti,
    load_series,
    pick_best_series,
    resample_isotropic,
    save_nifti,
    scan_directory,
)

__all__ = [
    "Volume",
    "SeriesInfo",
    "scan_directory",
    "pick_best_series",
    "load_series",
    "load_nifti",
    "save_nifti",
    "resample_isotropic",
]
