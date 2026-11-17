"""带路径约束的安全文件写入工具。

所有"路径可能由外部输入（CLI 参数 / HTTP 上传 / zip 成员名）拼出来"的写入，
统一经过这里：先做 root 逃逸校验（防父目录逃逸），再落盘。
项目内其余模块不要绕过本模块直接以写模式打开这类路径。

用法:
    from utils.safeio import ensure_under, json_dump, bin_writer

    json_dump(obj, out_dir / "summary.json", root=ROOT)
    with bin_writer(dst, root=up_dir) as f:
        shutil.copyfileobj(src, f)
"""

from __future__ import annotations

import csv
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


def ensure_under(path: str | Path, root: str | Path) -> Path:
    """校验 path 解析后位于 root 内，返回解析后的绝对路径；越界抛 ValueError。"""
    raw = Path(path)
    # 显式拒绝 .. 相对分量，杜绝父目录穿越
    if ".." in raw.parts:
        raise ValueError(f"路径含越界分量: {path!r}")
    p = raw.resolve()
    r = Path(root).resolve()
    if p != r and not p.is_relative_to(r):
        raise ValueError(f"路径越出 {r}: {path!r}")
    return p


def _target(path: str | Path, root: str | Path | None) -> Path:
    return ensure_under(path, root) if root is not None else Path(path)


@contextmanager
def text_writer(path: str | Path, *, root: str | Path | None = None,
                encoding: str = "utf-8", newline: str = "") -> Iterator[TextIO]:
    with _target(path, root).open("w", encoding=encoding, newline=newline) as f:
        yield f


@contextmanager
def bin_writer(path: str | Path, *, root: str | Path | None = None) -> Iterator:
    with _target(path, root).open("wb") as f:
        yield f


def json_dump(obj, path: str | Path, *, root: str | Path | None = None,
              indent: int = 2) -> None:
    p = _target(path, root)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")


@contextmanager
def csv_writer(path: str | Path, *, root: str | Path | None = None,
               bom: bool = False):
    """yield (file, csv.writer)。bom=True 用 utf-8-sig，便于 Excel 直接打开。"""
    enc = "utf-8-sig" if bom else "utf-8"
    with _target(path, root).open("w", newline="", encoding=enc) as f:
        yield f, csv.writer(f)


@contextmanager
def append_writer(path: str | Path, *, root: str | Path | None = None):
    """追加模式文本写入，同样先做 root 逃逸校验。"""
    with _target(path, root).open("a", newline="", encoding="utf-8") as f:
        yield f


def safe_extract_all(zf, dest: str | Path, skip_parts: set[str] = frozenset()) -> int:
    """把 ZipFile 安全解压到 dest：逐成员校验目标路径不越出 dest（防 zip-slip）。

    返回解压的成员数。skip_parts 可传 {"__MACOSX", ".DS_Store"} 之类垃圾条目。
    """
    import zipfile

    if not isinstance(zf, zipfile.ZipFile):
        raise TypeError("safe_extract_all 需要 zipfile.ZipFile")
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for m in zf.infolist():
        if set(m.filename.split("/")) & skip_parts:
            continue
        target = ensure_under(dest / m.filename, dest)
        if m.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(m) as src, target.open("wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
        n += 1
    return n
