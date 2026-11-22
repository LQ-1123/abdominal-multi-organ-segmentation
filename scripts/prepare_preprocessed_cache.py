"""为训练侧预处理对照生成各方法的缓存。

对 <base>/<case>.img.npy 逐例做 src/preprocess/classic.py 里的处理，把模型输入映射回
窗内 HU 存成 int16：<out>/<method>/<case>.img.npy。标签直接硬链接（不支持时复制）。
"none" 组直接用原缓存，不生成。

    python scripts/prepare_preprocessed_cache.py --base data/cache/amos22 \
        --out data/cache/pre --ids split.json --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocess import classic  # noqa: E402


def one(task: tuple[Path, Path, str, str]) -> str:
    base, out, method, cid = task
    dst = out / method / f"{cid}.img.npy"
    lbl = out / method / f"{cid}.lbl.npy"
    if not lbl.exists():
        try:
            os.link(base / f"{cid}.lbl.npy", lbl)
        except OSError:
            shutil.copyfile(base / f"{cid}.lbl.npy", lbl)
    if dst.exists():
        return f"skip {method} {cid}"
    t0 = time.time()
    image = np.load(base / f"{cid}.img.npy")
    cached = classic.to_cache_hu(classic.apply(image, method))
    tmp = dst.with_suffix(".tmp.npy")
    np.save(tmp, cached, allow_pickle=False)
    tmp.replace(dst)
    return f"ok {method} {cid} {time.time() - t0:.1f}s"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ids", type=Path, required=True, help="含 train_ids / val_ids 的 JSON")
    ap.add_argument("--methods", nargs="+", default=[m for m in classic.METHODS if m != "none"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    split = json.loads(args.ids.read_text(encoding="utf-8"))
    ids = sorted(set(split["train_ids"]) | set(split["val_ids"]))
    for m in args.methods:
        (args.out / m).mkdir(parents=True, exist_ok=True)
    tasks = [(args.base, args.out, m, cid) for m in args.methods for cid in ids]
    print(f"{len(ids)} 例 × {len(args.methods)} 种方法 = {len(tasks)} 个任务", flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, msg in enumerate(pool.imap_unordered(one, tasks, chunksize=1), 1):
            if i % 25 == 0 or i == len(tasks):
                print(f"[{i}/{len(tasks)}] {msg}  已用 {(time.time() - t0) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
