"""AMOS22 病例清单、按病例分折和 patch 训练数据集。

缓存格式（scripts/build_cache.py 生成）：
    <cache>/<case_id>.img.npy   int16 HU，[Z, Y, X]，1.5 mm 等向
    <cache>/<case_id>.lbl.npy   uint8 标签 0..15，[Z, Y, X]

用法
----
    python -m src.data.amos_dataset        # 打印可用病例数和第 0 折划分
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

HU_MIN, HU_MAX = -150.0, 250.0
CASE_RE = re.compile(r"(amos_\d{4})\.nii(\.gz)?$")


@dataclass(frozen=True)
class Case:
    case_id: str
    img_path: Path
    lbl_path: Path

    @property
    def is_ct(self) -> bool:
        # AMOS22 官方 readme：编号小于 500 的是 CT，其余是 MRI
        return int(self.case_id.rsplit("_", 1)[1]) < 500


def build_case_list(root: Path) -> list[Case]:
    """在 root 下找 images*/labels* 目录里同名的影像与标注，按编号排序返回。

    兼容官方目录（imagesTr/labelsTr、imagesVa/labelsVa）。只有影像没有标注的病例
    （如 imagesTs）不会进入清单。
    """
    root = Path(root)
    labels: dict[str, Path] = {}
    for d in root.glob("labels*"):
        for p in d.iterdir():
            m = CASE_RE.search(p.name)
            if m:
                labels[m.group(1)] = p
    cases = []
    for d in root.glob("images*"):
        for p in d.iterdir():
            m = CASE_RE.search(p.name)
            if m and m.group(1) in labels:
                cases.append(Case(m.group(1), p, labels[m.group(1)]))
    return sorted(cases, key=lambda c: c.case_id)


def make_folds(ids: list[str], n_folds: int = 5, seed: int = 42) -> list[tuple[list[str], list[str]]]:
    """按病例确定性分折：固定种子打乱后均分，第 k 份做验证，其余做训练。"""
    ordered = sorted(ids)
    perm = np.random.RandomState(seed).permutation(len(ordered))
    chunks = np.array_split(perm, n_folds)
    folds = []
    for k in range(n_folds):
        val = sorted(ordered[i] for i in chunks[k])
        val_set = set(val)
        folds.append(([c for c in ordered if c not in val_set], val))
    return folds


def window(arr: np.ndarray) -> np.ndarray:
    x = np.clip(arr.astype(np.float32), HU_MIN, HU_MAX)
    return (x - HU_MIN) / (HU_MAX - HU_MIN)


class AMOSDataset(Dataset):
    """每个病例随机取 samples_per_case 个 patch。

    - fg_sampling：以 fg_ratio 的概率把 patch 中心放在某个前景器官的随机体素上，
      其余情况在整卷内均匀取中心；
    - augment：三轴随机翻转；
    - strong_aug（V2）：亮度缩放 ±10%、亮度偏移 ±0.1、随机 Gamma 0.7–1.4。
    验证集（train=False）用按索引固定的随机数，每次取到同一批 patch。
    """

    def __init__(self, case_ids, cache_dir, patch=(128, 128, 128), samples_per_case: int = 4,
                 train: bool = True, augment: bool | None = None, strong_aug: bool = False,
                 fg_sampling: bool = True, fg_ratio: float = 0.67, num_classes: int = 16,
                 seed: int = 0, deterministic: bool = False) -> None:
        self.ids = list(case_ids)
        self.cache = Path(cache_dir)
        self.patch = tuple(int(p) for p in patch)
        self.spc = int(samples_per_case)
        self.train = train
        self.augment = train if augment is None else augment
        self.strong_aug = strong_aug and self.augment
        self.fg_sampling = fg_sampling
        self.fg_ratio = fg_ratio if train else 1.0
        self.num_classes = num_classes
        self.seed = seed
        # deterministic=True 时训练集也用 (seed, epoch, index) 定随机数：
        # 不同对照组在同一 epoch 取到完全相同的 patch 和增强，只差输入预处理。
        self.deterministic = deterministic
        self.epoch = 0
        self._fg: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.ids) * self.spc

    def _load(self, cid: str):
        img = np.load(self.cache / f"{cid}.img.npy", mmap_mode="r")
        lbl = np.load(self.cache / f"{cid}.lbl.npy", mmap_mode="r")
        return img, lbl

    def _foreground(self, cid: str, lbl) -> np.ndarray:
        # 每个病例的前景坐标只算一次；稀疏下采样，控制内存
        if cid not in self._fg:
            coords = np.argwhere(np.asarray(lbl[::2, ::2, ::2]) > 0) * 2
            self._fg[cid] = coords.astype(np.int32)
        return self._fg[cid]

    def _crop(self, arr, start, fill):
        out = np.full(self.patch, fill, dtype=arr.dtype)
        src, dst = [], []
        for s, p, n in zip(start, self.patch, arr.shape):
            a, b = max(s, 0), min(s + p, n)
            src.append(slice(a, b))
            dst.append(slice(a - s, b - s))
        out[tuple(dst)] = arr[tuple(src)]
        return out

    def __getitem__(self, index: int) -> dict:
        cid = self.ids[index // self.spc]
        if self.train and not self.deterministic:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng([self.seed, self.epoch, index])
        img, lbl = self._load(cid)
        shape = img.shape

        center = None
        if self.fg_sampling and rng.random() < self.fg_ratio:
            fg = self._foreground(cid, lbl)
            if len(fg):
                center = fg[rng.integers(len(fg))]
        if center is None:
            center = [rng.integers(0, max(n, 1)) for n in shape]
        start = [int(c) - p // 2 for c, p in zip(center, self.patch)]
        # 体积比 patch 大时，把窗口收回到体积内部，避免大面积填充
        start = [min(max(s, 0), max(n - p, 0)) if n >= p else (n - p) // 2
                 for s, p, n in zip(start, self.patch, shape)]

        x = window(self._crop(img, start, fill=-1024))
        y = self._crop(lbl, start, fill=0).astype(np.int64)

        if self.augment:
            for axis in range(3):
                if rng.random() < 0.5:
                    x, y = np.flip(x, axis), np.flip(y, axis)
            if self.strong_aug:
                if rng.random() < 0.5:
                    x = x * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)
                if rng.random() < 0.3:
                    x = np.clip(x, 0, 1) ** rng.uniform(0.7, 1.4)
            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)

        present = torch.zeros(self.num_classes, dtype=torch.bool)
        present[np.unique(y)] = True
        present[0] = True
        return {"image": torch.from_numpy(x.astype(np.float32))[None],
                "label": torch.from_numpy(y), "present": present, "case_id": cid}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    cases = build_case_list(root / "data" / "processed" / "amos22")
    cache = root / "data" / "cache" / "amos22"
    have = {p.name[:-8] for p in cache.glob("*.img.npy")}
    ids = [c.case_id for c in cases if c.case_id in have]
    print(f"有标注病例 {len(cases)} 例，其中已缓存 {len(ids)} 例")
    if ids:
        tr, va = make_folds(ids)[0]
        print(f"第 0 折：训练 {len(tr)} 例，验证 {len(va)} 例")
