"""解压大数据集 zip（跳过 macOS 垃圾文件，带进度输出）。

用法:
    python scripts/extract_zip.py data/raw/amos22.zip data/processed amos22/

为什么不用 Expand-Archive：PowerShell 的 Expand-Archive 对 20GB+ 的 zip
会出现内存暴涨和极慢的问题；Python 的 zipfile 流式解压稳定得多。
"""

from __future__ import annotations

import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils.safeio import bin_writer, ensure_under  # noqa: E402

SKIP_PARTS = {"__MACOSX", ".DS_Store"}

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if len(sys.argv) < 3:
        print("用法: python extract_zip.py <zip> <dest_dir> [prefix]")
        return 2

    zip_path = Path(sys.argv[1])
    prefix = sys.argv[3] if len(sys.argv) > 3 else ""
    # 解压目标必须位于项目目录内（本工具的实际用途是解到 data/processed 等项目路径）；
    # ensure_under 做规范化 + 越界拒绝，后续所有写入统一用常量 ROOT 再校验一遍
    try:
        dest = ensure_under(sys.argv[2], ROOT)
    except ValueError as exc:
        print(f"目标目录非法: {exc}")
        return 2
    dest.mkdir(parents=True, exist_ok=True)

    print(f"源: {zip_path}  ({zip_path.stat().st_size / 1024**3:.2f} GB)")
    print(f"目标: {dest.resolve()}")
    print(f"只解压前缀: {prefix!r}（空=全部）")

    t0 = time.time()
    n_done = n_skip = 0
    bytes_done = 0

    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.infolist() if not (set(m.filename.split("/")) & SKIP_PARTS)]
        if prefix:
            members = [m for m in members if m.filename.startswith(prefix)]
        total = len(members)
        total_bytes = sum(m.file_size for m in members)
        print(f"待解压 {total} 项, 解压后约 {total_bytes / 1024**3:.2f} GB\n")

        dest_resolved = dest.resolve()

        def _safe_target(name: str) -> Path | None:
            """解析成员路径；越出 dest（zip-slip）返回 None 跳过。"""
            target = (dest / name).resolve()
            if target == dest_resolved or not target.is_relative_to(dest_resolved):
                return None
            return target

        for i, m in enumerate(members, 1):
            if m.is_dir():
                tgt = _safe_target(m.filename)
                if tgt is not None:
                    tgt.mkdir(parents=True, exist_ok=True)
                n_skip += 1
                continue
            out = _safe_target(m.filename)
            if out is None:
                n_skip += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            # 已存在且大小一致 -> 跳过（支持断点续解压）
            if out.exists() and out.stat().st_size == m.file_size:
                n_skip += 1
                bytes_done += m.file_size
                continue
            with z.open(m) as src, bin_writer(out, root=ROOT) as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            n_done += 1
            bytes_done += m.file_size

            if i % 25 == 0 or i == total:
                el = time.time() - t0
                rate = bytes_done / el / 1024**2 if el > 0 else 0
                eta = (total_bytes - bytes_done) / (bytes_done / el) if bytes_done else 0
                print(f"  [{i}/{total}] {bytes_done/1024**3:.2f}/{total_bytes/1024**3:.2f} GB "
                      f"| {rate:.0f} MB/s | 剩余约 {eta/60:.1f} min")

    el = time.time() - t0
    print(f"\n完成: 新解压 {n_done} 项, 跳过 {n_skip} 项, 耗时 {el/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
