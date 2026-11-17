"""查询当前环境实际支持哪些 DICOM 压缩传输语法（解码 + 编码）。"""

from __future__ import annotations

import pylibjpeg
from pylibjpeg.utils import get_pixel_data_decoders, get_pixel_data_encoders
from pydicom.uid import UID

# 真实 PACS 里最常见的几种压缩语法
COMMON = {
    "1.2.840.10008.1.2.4.70": "JPEG Lossless SV1 (Process 14)  ← 用户报错的那种",
    "1.2.840.10008.1.2.4.57": "JPEG Lossless (Process 14)",
    "1.2.840.10008.1.2.4.80": "JPEG-LS Lossless",
    "1.2.840.10008.1.2.4.90": "JPEG 2000 Lossless",
    "1.2.840.10008.1.2.4.91": "JPEG 2000",
    "1.2.840.10008.1.2.5": "RLE Lossless",
    "1.2.840.10008.1.2.1": "Explicit VR Little Endian (未压缩)",
    "1.2.840.10008.1.2": "Implicit VR Little Endian (未压缩)",
}


def main() -> int:
    dec = get_pixel_data_decoders()
    enc = get_pixel_data_encoders()
    print(f"pylibjpeg {pylibjpeg.__version__}")
    print(f"支持解码 {len(dec)} 种传输语法，支持编码 {len(enc)} 种\n")

    print(f"{'传输语法':<46}{'解码':>6}{'编码':>6}")
    print("-" * 60)
    ok_lossless = False
    for uid, label in COMMON.items():
        d = "OK" if uid in dec else "--"
        e = "OK" if uid in enc else "--"
        if uid == "1.2.840.10008.1.2.4.70" and uid in dec:
            ok_lossless = True
        print(f"{label:<46}{d:>6}{e:>6}")

    print()
    if ok_lossless:
        print("✅ JPEG Lossless 解码已可用 —— 你可以重新上传那个文件了")
        return 0
    print("❌ JPEG Lossless 仍不可解码")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
