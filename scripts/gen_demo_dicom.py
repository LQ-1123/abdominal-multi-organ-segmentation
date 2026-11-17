"""生成一个「假的但完全合规」的 DICOM CT 序列，用于在没有真实数据时测通全链路。

为什么需要它：
    公开数据集都是 NIfTI，但系统要演示"上传 DICOM"。这个脚本把合成的三维体
    写成标准 DICOM 序列，并且故意加入一个「定位片」序列作为干扰项，用来验证
    序列选择逻辑是否正确。

生成内容：
    - 主序列 CT_ABDOMEN : 80 层 256x256，层厚 3mm，含 2 个椭球"器官"
    - 干扰序列 LOCALIZER: 2 层 512x512 的定位片（应当被 pick_best_series 排除）

用法:
    python scripts/gen_demo_dicom.py [输出目录]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

# ----------------------------------------------------------------------------- 合成体数据
def make_volume(nz: int = 80, ny: int = 256, nx: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """返回 (HU 体积, 标签体积)。标签: 1=器官A(类脾), 2=器官B(类肾)。"""
    zz, yy, xx = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32)
    cz, cy, cx = nz / 2, ny / 2, nx / 2

    hu = np.full((nz, ny, nx), -1000.0, dtype=np.float32)          # 空气
    label = np.zeros((nz, ny, nx), dtype=np.uint8)

    # 体表 / 脂肪 / 软组织分层（只是为了图像看起来像 CT）
    body = (((yy - cy) / (ny * 0.42)) ** 2 + ((xx - cx) / (nx * 0.42)) ** 2) <= 1.0
    hu[body] = 40.0                                                # 软组织
    inner = (((yy - cy) / (ny * 0.38)) ** 2 + ((xx - cx) / (nx * 0.38)) ** 2) <= 1.0
    hu[inner] = 60.0

    def ellipsoid(cz_, cy_, cx_, rz, ry, rx):
        return (((zz - cz_) / rz) ** 2 + ((yy - cy_) / ry) ** 2
                + ((xx - cx_) / rx) ** 2) <= 1.0

    # 器官 A：偏左上（脾脏位置），约 90 HU
    a = ellipsoid(cz, cy - ny * 0.16, cx - nx * 0.18, nz * 0.22, ny * 0.10, nx * 0.12)
    hu[a] = 95.0
    label[a] = 1

    # 器官 B：偏右侧（肾脏位置），约 110 HU，略小
    b = ellipsoid(cz, cy + ny * 0.18, cx + nx * 0.16, nz * 0.16, ny * 0.08, nx * 0.07)
    hu[b] = 112.0
    label[b] = 2

    return hu, label


# ----------------------------------------------------------------------------- DICOM 写出
def _base_ds(path: Path, series_uid: str, study_uid: str, frame_uid: str,
             rows: int, cols: int, pixel_spacing: tuple[float, float],
             orientation: tuple[float, ...], position: tuple[float, ...],
             thickness: float, instance: int, desc: str, modality: str = "CT") -> FileDataset:
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = CTImageStorage
    fm.MediaStorageSOPInstanceUID = frame_uid
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    fm.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=fm, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = frame_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Modality = modality
    ds.SeriesDescription = desc
    ds.PatientName = "DEMO^SYNTHETIC"
    ds.PatientID = "DEMO-0001"
    ds.PatientBirthDate = ""
    ds.PatientSex = "O"
    ds.StudyDate = "20260101"
    ds.StudyTime = "120000"
    ds.StudyID = "1"
    ds.SeriesNumber = 1 if desc.startswith("CT_") else 2
    ds.InstanceNumber = instance
    ds.Rows, ds.Columns = rows, cols
    ds.PixelSpacing = [float(pixel_spacing[0]), float(pixel_spacing[1])]
    ds.ImageOrientationPatient = [float(v) for v in orientation]
    ds.ImagePositionPatient = [float(v) for v in position]
    ds.SliceThickness = float(thickness)
    ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
    ds.PatientPosition = "HFS"
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1          # signed
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = -1024.0       # 关键：HU = pixel + (-1024)
    return ds


def write_series(out_dir: Path, hu: np.ndarray, spacing_mm: tuple[float, float, float],
                 desc: str = "CT_ABDOMEN") -> Path:
    d = out_dir / desc
    d.mkdir(parents=True, exist_ok=True)
    nz, ny, nx = hu.shape
    dz, dy, dx = spacing_mm
    series_uid = generate_uid()
    study_uid = generate_uid()
    orientation = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)   # 轴位：row=+X, col=+Y

    for z in range(nz):
        ds = _base_ds(d / f"img_{z:04d}.dcm", series_uid, study_uid, generate_uid(),
                      ny, nx, (dy, dx), orientation,
                      (0.0, 0.0, float(z) * dz),        # IPP 沿 +Z 递增，与升序排序约定一致
                      dz, instance=z + 1, desc=desc)
        # HU -> 存储像素（逆变换），并裁剪到 int16
        stored = np.clip(hu[z] - ds.RescaleIntercept, 0, 65535).astype(np.uint16)
        ds.PixelData = stored.astype("<u2").tobytes()
        ds.save_as(d / f"img_{z:04d}.dcm", enforce_file_format=True)
    print(f"  写入 {nz} 层 -> {d}")
    return d


def write_localizer(out_dir: Path) -> Path:
    """干扰项：2 层的定位片，应当被序列选择逻辑排除。"""
    d = out_dir / "LOCALIZER"
    d.mkdir(parents=True, exist_ok=True)
    series_uid, study_uid = generate_uid(), generate_uid()
    img = np.full((512, 512), -1024.0, dtype=np.float32)
    img[100:400, 80:420] = 30.0
    for z in range(2):
        ds = _base_ds(d / f"loc_{z}.dcm", series_uid, study_uid, generate_uid(),
                      512, 512, (0.8, 0.8), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                      (0.0, 0.0, float(z) * 5.0), 5.0, instance=z + 1,
                      desc="LOCALIZER topogram")
        stored = np.clip(img - ds.RescaleIntercept, 0, 65535).astype(np.uint16)
        ds.PixelData = stored.astype("<u2").tobytes()
        ds.save_as(d / f"loc_{z}.dcm", enforce_file_format=True)
    print(f"  写入干扰项 LOCALIZER -> {d}")
    return d


# ----------------------------------------------------------------------------- 自测
def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/demo/synthetic_dicom")
    out.mkdir(parents=True, exist_ok=True)

    print("生成合成 CT 序列 ...")
    hu, label = make_volume()
    write_series(out, hu, spacing_mm=(3.0, 1.2, 1.2))
    write_localizer(out)

    # 保存标签供训练/评估对照
    np.savez_compressed(out / "labels.npz", label=label, spacing=np.array((3.0, 1.2, 1.2)))
    print(f"  标签已存 -> {out / 'labels.npz'}")

    # ---------------- 回读验证 ----------------
    print("\n回读验证 DICOM 摄取模块 ...")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from dicom_ingest import load_series, pick_best_series, scan_directory

    series = scan_directory(out)
    print(f"  扫描到 {len(series)} 个序列:")
    for s in series:
        print(f"    - {s}")

    best = pick_best_series(series)
    print(f"  自动选择: {best.description!r} ({best.n_files} 层)")
    assert "LOCALIZER" not in best.description.upper(), "序列选择错误：选中了定位片"
    assert best.n_files == 80, f"层数不对: {best.n_files}"

    vol = load_series(best, target_spacing=None)
    print(f"  {vol}")
    assert vol.shape == hu.shape, f"形状不符: {vol.shape} vs {hu.shape}"
    diff = np.abs(vol.pixels - hu).max()
    print(f"  HU 重建误差: {diff:.3f}")
    assert diff < 1.0, "HU 转换有误"
    assert abs(vol.spacing[0] - 3.0) < 1e-3, f"层厚不对: {vol.spacing}"

    # 等向重采样
    vol_iso = load_series(best, target_spacing=(1.5, 1.5, 1.5))
    print(f"  等向重采样后: {vol_iso}")

    print("\n✅ M1 DICOM 摄取模块验证通过：序列选择 / HU / 层序 / spacing 全部正确")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
