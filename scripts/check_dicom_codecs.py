"""验证 DICOM 压缩像素数据的解码能力。

背景
----
真实医院 PACS 导出的 CT 多为压缩传输语法（JPEG Lossless / JPEG2000 / RLE），
pydicom 解析元信息不需要插件，但**解压像素**需要 pylibjpeg / GDCM。

本脚本做两件事：
  1. 列出当前环境实际能用的解码插件
  2. 把一例标准 DICOM 压缩成多种传输语法再读回，逐体素比对是否无损

用法:
    python scripts/check_dicom_codecs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pydicom
from pydicom.uid import (
    JPEG2000Lossless,
    JPEGLosslessSV1,
    RLELossless,
    UID,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "demo" / "amos_0004_dicom" / "img_0040.dcm"


def list_decoders() -> None:
    print("=== 已注册的解码插件 ===")
    try:
        from pydicom.pixels.decoders import DECODER_REGISTRY
    except Exception as exc:  # noqa: BLE001
        print(f"  无法读取注册表: {exc}")
        return

    seen = set()
    for uid, plugins in DECODER_REGISTRY.items():
        try:
            name = UID(uid).name
        except Exception:  # noqa: BLE001
            name = str(uid)
        for p in plugins:
            pname = p[0] if isinstance(p, (tuple, list)) else str(p)
            seen.add(pname)
    print(f"  插件: {sorted(seen)}")


def roundtrip(ds: pydicom.Dataset, uid: UID) -> str:
    """把数据集压缩成 uid 指定的传输语法，再读回比对像素。"""
    try:
        src = np.array(ds.pixel_array)
    except Exception as exc:  # noqa: BLE001
        return f"读取原件失败: {exc}"

    tmp = ROOT / "outputs" / "_codec_test.dcm"
    try:
        ds2 = ds.copy()
        ds2.compress(uid)
        ds2.save_as(tmp, enforce_file_format=True)
    except Exception as exc:  # noqa: BLE001
        return f"编码失败（需要编码器）: {type(exc).__name__}: {str(exc)[:90]}"

    try:
        back = pydicom.dcmread(tmp)
        arr = np.array(back.pixel_array)
    except Exception as exc:  # noqa: BLE001
        return f"❌ 解码失败: {type(exc).__name__}: {str(exc)[:90]}"
    finally:
        tmp.unlink(missing_ok=True)

    if arr.shape != src.shape:
        return f"❌ 形状不一致 {arr.shape} vs {src.shape}"
    same = np.array_equal(arr, src)
    return "✅ 无损往返成功" if same else "⚠️ 能解码但像素不一致"


def main() -> int:
    list_decoders()
    print()

    if not SAMPLE.exists():
        print(f"找不到测试样本 {SAMPLE}")
        return 1

    ds = pydicom.dcmread(str(SAMPLE))
    ts = getattr(ds.file_meta, "TransferSyntaxUID", None)
    print(f"=== 测试样本 ===\n  {SAMPLE.name}  原始传输语法: {ts.name if ts else '未知'}\n")

    cases = [
        ("JPEG Lossless (Process 14, SV1)", JPEGLosslessSV1),   # 用户遇到的这种
        ("JPEG 2000 Lossless", JPEG2000Lossless),               # 心血管/部分 PACS 常用
        ("RLE Lossless", RLELossless),                          # 部分 CT/MR 常用
    ]
    print("=== 压缩往返测试 ===")
    ok = 0
    for label, uid in cases:
        res = roundtrip(ds, uid)
        if res.startswith("✅"):
            ok += 1
        print(f"  {label:<34} {res}")

    print(f"\n可用: {ok}/{len(cases)}")
    if ok == 0:
        print("⚠️ 一个都不可用 —— 请安装 pylibjpeg 系列插件")
        return 1
    print("✅ 压缩 DICOM 解码能力验证通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
