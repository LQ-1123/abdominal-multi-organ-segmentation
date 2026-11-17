"""把 AMOS 的 NIfTI 病例转换成「真实的」DICOM 序列。

用途
----
系统要求"用户上传 DICOM"，但公开训练数据是 NIfTI。这个脚本把验证集病例
转成标准 DICOM 序列，用来：
    1. 端到端验证「上传 DICOM → 分割 → 展示」的完整链路
    2. 作为演示素材（真实解剖结构，不是人造器官）

写出的 DICOM 带完整几何信息（ImagePositionPatient / ImageOrientationPatient /
PixelSpacing / RescaleSlope / RescaleIntercept），3D Slicer 可直接打开。

用法:
    python scripts/nifti_to_dicom.py amos_0004 --out data/demo/amos_0004_dicom
    python scripts/nifti_to_dicom.py amos_0004 --out ... --also-localizer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

ROOT = Path(__file__).resolve().parents[1]


def build_series(nii_path: Path, out_dir: Path, desc: str = "CT ABDOMEN") -> Path:
    nii = nib.load(str(nii_path))
    data = np.asanyarray(nii.dataobj)               # (X, Y, Z) HU
    if data.ndim == 4:
        data = data[..., 0]

    aff = np.asarray(nii.affine, dtype=float)
    spacing = np.sqrt((aff[:3, :3] ** 2).sum(axis=0))    # (sx, sy, sz)
    dirs = aff[:3, :3] / spacing
    origin = aff[:3, 3]

    # DICOM 约定：行方向 = 列索引的余弦 = dirs[:,0]；列方向 = dirs[:,1]
    # 法向量 = dirs[:,2]
    row_cos = dirs[:, 0]
    col_cos = dirs[:, 1]
    normal = np.cross(row_cos, col_cos)

    out_dir.mkdir(parents=True, exist_ok=True)
    study_uid = generate_uid()
    series_uid = generate_uid()

    nx, ny, nz = data.shape
    # 写之前检查：法向量应指向 +Z 或 -Z（轴位），否则提示
    if abs(abs(normal[2]) - 1.0) > 0.1:
        print(f"⚠️ 该病例不是轴位序列（法向量={normal}），DICOM 仍按原几何写出")

    print(f"转换 {nii_path.name}: {data.shape} spacing={np.round(spacing,3)} → {nz} 层 DICOM")

    for z in range(nz):
        sop_uid = generate_uid()
        fm = FileMetaDataset()
        fm.MediaStorageSOPClassUID = CTImageStorage
        fm.MediaStorageSOPInstanceUID = sop_uid
        fm.TransferSyntaxUID = ExplicitVRLittleEndian
        fm.ImplementationClassUID = generate_uid()

        ds = FileDataset("", {}, file_meta=fm, preamble=b"\0" * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = sop_uid
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.Modality = "CT"
        ds.SeriesDescription = desc
        ds.SeriesNumber = 1
        ds.InstanceNumber = z + 1
        ds.PatientName = "AMOS^DEMO"
        ds.PatientID = "AMOS-DEMO"
        ds.PatientBirthDate = ""
        ds.PatientSex = "O"
        ds.StudyDate = "20260101"
        ds.StudyTime = "120000"
        ds.StudyID = "1"
        ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
        ds.PatientPosition = "HFS"

        ds.Rows, ds.Columns = ny, nx
        ds.PixelSpacing = [float(spacing[1]), float(spacing[0])]
        ds.SliceThickness = float(spacing[2])
        ds.ImageOrientationPatient = [float(v) for v in (*row_cos, *col_cos)]

        ipp = origin + normal * (spacing[2] * z)
        ds.ImagePositionPatient = [float(v) for v in ipp]

        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 1          # signed

        # HU -> 存储像素；用 slope=1 / intercept=-1024，与真实 CT 惯例一致
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = -1024.0
        slice_hu = data[:, :, z].T.astype(np.float32)        # (Y, X)
        stored = np.clip(slice_hu - ds.RescaleIntercept, 0, 65535).astype(np.uint16)
        ds.PixelData = stored.astype("<u2").tobytes()

        ds.save_as(out_dir / f"img_{z:04d}.dcm", enforce_file_format=True)

    print(f"完成: {out_dir}  ({nz} 个 .dcm)")
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("case_id", help="如 amos_0004")
    ap.add_argument("--out", required=True)
    ap.add_argument("--desc", default="CT ABDOMEN")
    args = ap.parse_args()

    src = ROOT / "data" / "processed" / "amos22" / "imagesVa" / f"{args.case_id}.nii.gz"
    if not src.exists():
        src = ROOT / "data" / "processed" / "amos22" / "imagesTr" / f"{args.case_id}.nii.gz"
    if not src.exists():
        print(f"找不到病例 {args.case_id}")
        return 1

    build_series(src, Path(args.out), args.desc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
