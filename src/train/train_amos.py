"""AMOS22 多器官 3D 分割训练脚本（5 折交叉验证）。

功能
----
- 5 折交叉验证，**按病例划分**（防止同一病人 patch 泄漏到验证集）
- 128³ patch + 深监督 + AMP 混合精度
- 缺失类别屏蔽的 DiceCE 损失（见 src/train/losses.py）
- 每 epoch 用固定 patch 快速验证；每个 fold 结束做滑窗全卷评估（分器官 Dice）
- 断点续训、CSV 日志、最佳权重保存

用法
----
    # 快速冒烟（8 例, 2 epoch）
    python -m src.train.train_amos --smoke

    # 单折正式训练
    python -m src.train.train_amos --fold 0 --epochs 100

    # 全部 5 折
    python -m src.train.train_amos --fold -1 --epochs 100
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data.amos_dataset import AMOSDataset, build_case_list, make_folds  # noqa: E402
from models.unet3d import UNet3D  # noqa: E402
from train.losses import DiceCELoss, dice_score  # noqa: E402
from utils.safeio import append_writer, csv_writer, json_dump  # noqa: E402

ORGAN_NAMES = [
    "background", "spleen", "right_kidney", "left_kidney", "gallbladder",
    "esophagus", "liver", "stomach", "aorta", "postcava", "pancreas",
    "right_adrenal", "left_adrenal", "duodenum", "bladder", "prostate_uterus",
]


# ----------------------------------------------------------------------------- 工具
def make_loader(ds: AMOSDataset, batch: int, train: bool, workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=train,
        num_workers=workers,
        pin_memory=True,
        drop_last=train,
        persistent_workers=workers > 0,
    )


def deep_supervision_loss(loss_fn, outputs, label, num_classes: int) -> torch.Tensor:
    """深监督：主输出权重 1.0，辅助输出 0.5/0.25/0.125，标签逐级下采样。"""
    if not isinstance(outputs, (list, tuple)):
        present = _present_of(label, num_classes)
        return loss_fn(outputs, label, present)

    weights = [1.0, 0.5, 0.25, 0.125, 0.0625]
    total = 0.0
    tgt = label
    for i, out in enumerate(outputs):
        if i > 0:
            tgt = F.interpolate(
                label.unsqueeze(1).float(),
                size=out.shape[2:],
                mode="nearest",
            ).squeeze(1).long()
        present = _present_of(tgt, num_classes)
        total = total + weights[min(i, len(weights) - 1)] * loss_fn(out, tgt, present)
    return total


def _present_of(label: torch.Tensor, num_classes: int) -> torch.Tensor:
    """[B, ...] -> [B, C] bool。某个类别在这块 patch 里是否出现过。"""
    b = label.shape[0]
    present = torch.zeros(b, num_classes, dtype=torch.bool, device=label.device)
    for i in range(b):
        present[i, torch.unique(label[i])] = True
    present[:, 0] = True
    return present


@torch.no_grad()
def quick_val(model, ds, loss_fn, num_classes, device, max_batches: int = 20) -> tuple[float, float]:
    """固定 patch 上的快速验证（便宜，用于监控趋势）。"""
    model.eval()
    loader = make_loader(ds, batch=1, train=False, workers=0)
    losses, dices = [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        img = batch["image"].to(device, non_blocking=True)
        lbl = batch["label"].to(device, non_blocking=True)
        present = batch["present"].to(device, non_blocking=True)
        out = model(img)
        if isinstance(out, (list, tuple)):
            out = out[0]
        losses.append(loss_fn(out, lbl, present).item())
        d = dice_score(out, lbl, num_classes, present)[:, 1:]
        m = torch.nanmean(d).item()
        if not np.isnan(m):
            dices.append(m)
    model.train()
    return float(np.mean(losses)) if losses else float("nan"), float(np.mean(dices)) if dices else float("nan")


@torch.no_grad()
def full_volume_eval(model, val_ids, cache_dir, patch, num_classes, device) -> dict[str, float]:
    """滑窗推理整卷，按器官统计 Dice（缺失器官跳过）。"""
    from monai.inferers import sliding_window_inference

    model.eval()
    per_class_sum = np.zeros(num_classes)
    per_class_cnt = np.zeros(num_classes)

    for cid in val_ids:
        img = np.load(Path(cache_dir) / f"{cid}.img.npy", mmap_mode="r")
        lbl = np.load(Path(cache_dir) / f"{cid}.lbl.npy", mmap_mode="r")
        arr = (np.clip(np.asarray(img, dtype=np.float32), -150, 250) + 150) / 400.0
        x = torch.from_numpy(arr)[None, None].to(device)
        # 深监督模式下模型返回 list，滑窗推理只取主输出
        def _predict(inp):
            o = model(inp)
            return o[0] if isinstance(o, (list, tuple)) else o

        logits = sliding_window_inference(x, patch, sw_batch_size=2, predictor=_predict,
                                          overlap=0.5, mode="gaussian")
        pred = logits.argmax(1)[0].cpu().numpy()
        gt = np.asarray(lbl)

        for c in range(1, num_classes):
            g = gt == c
            if not g.any():
                continue                     # 该病例没有这个器官 -> 跳过
            p = pred == c
            inter = np.logical_and(g, p).sum()
            dice = 2.0 * inter / (g.sum() + p.sum() + 1e-8)
            per_class_sum[c] += dice
            per_class_cnt[c] += 1

    result = {}
    for c in range(1, num_classes):
        if per_class_cnt[c] > 0:
            result[ORGAN_NAMES[c]] = float(per_class_sum[c] / per_class_cnt[c])
    if result:
        result["mean"] = float(np.mean(list(result.values())))
    return result


# ----------------------------------------------------------------------------- 单折训练
def train_fold(fold: int, train_ids, val_ids, cache_dir, args, device) -> dict:
    out_dir = ROOT / args.out_dir / f"fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = AMOSDataset(train_ids, cache_dir, patch=(args.patch,) * 3,
                           samples_per_case=args.samples_per_case, train=True,
                           strong_aug=args.strong_aug,
                           num_classes=args.out_ch)
    val_ds = AMOSDataset(val_ids, cache_dir, patch=(args.patch,) * 3,
                         samples_per_case=args.val_samples, train=False,
                         augment=False, fg_sampling=True,
                         num_classes=args.out_ch)

    train_loader = make_loader(train_ds, args.batch, True, args.workers)
    val_loader = make_loader(val_ds, 1, False, 0)

    model = UNet3D(in_ch=1, out_ch=args.out_ch, features=tuple(args.features),
                   deep_supervision=args.deep_sup, dropout=args.dropout).to(device)
    n_param = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[fold {fold}] 训练 {len(train_ids)} 例 / 验证 {len(val_ids)} 例 | "
          f"参数 {n_param:.2f}M | 每 epoch {len(train_ds)} patch")

    loss_fn = DiceCELoss(args.out_ch, include_background=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    log_csv = out_dir / "train_log.csv"
    with csv_writer(log_csv, root=ROOT) as (f, w):
        w.writerow(["epoch", "train_loss", "val_loss", "val_dice", "lr", "sec"])

    best_dice, best_epoch = -1.0, -1
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        run_loss, n = 0.0, 0
        for batch in train_loader:
            img = batch["image"].to(device, non_blocking=True)
            lbl = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=args.amp):
                out = model(img)
                loss = deep_supervision_loss(loss_fn, out, lbl, args.out_ch)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
            scaler.step(opt)
            scaler.update()

            run_loss += loss.item()
            n += 1
        sched.step()

        train_loss = run_loss / max(n, 1)
        val_loss, val_dice = quick_val(model, val_ds, loss_fn, args.out_ch, device,
                                       max_batches=args.val_batches)
        dt = time.time() - t0
        lr = opt.param_groups[0]["lr"]
        print(f"[fold {fold}] ep {epoch:3d}/{args.epochs} | "
              f"loss {train_loss:.4f} | val {val_loss:.4f} | dice {val_dice:.4f} | "
              f"{dt:.0f}s | lr {lr:.2e}")
        with append_writer(log_csv, root=ROOT) as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.5f}", f"{val_loss:.5f}",
                                    f"{val_dice:.5f}", f"{lr:.2e}", f"{dt:.1f}"])

        if val_dice > best_dice:
            best_dice, best_epoch = val_dice, epoch
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "dice": val_dice,
                "args": vars(args),
                "fold": fold,
            }, out_dir / "best.pth")

    # ---- fold 结束：滑窗全卷评估 ----
    print(f"[fold {fold}] 最佳 patch dice {best_dice:.4f} @ep{best_epoch}，开始滑窗全卷评估 ...")
    ckpt = torch.load(out_dir / "best.pth", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    organ_dice = full_volume_eval(model, val_ids, cache_dir, (args.patch,) * 3, args.out_ch, device)
    json_dump({"fold": fold, "best_epoch": best_epoch, "patch_dice": best_dice,
               "organ_dice": organ_dice}, out_dir / "val_organ_dice.json", root=ROOT)
    print(f"[fold {fold}] 全卷评估:")
    for k, v in organ_dice.items():
        print(f"    {k:<16} {v:.4f}")

    return {"fold": fold, "patch_dice": best_dice, "best_epoch": best_epoch, "organ_dice": organ_dice}


# ----------------------------------------------------------------------------- 仅评估
def eval_fold_only(fold: int, val_ids, cache_dir, args, device) -> dict:
    """加载已训练的 best.pth 做滑窗全卷评估，不重新训练。

    用途：训练被中断、或想用不同 overlap 重算指标时，避免重跑训练。
    """
    out_dir = ROOT / args.out_dir / f"fold{fold}"
    ckpt_path = out_dir / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"找不到 {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    saved = ckpt.get("args", {})
    patch = int(saved.get("patch", args.patch))
    features = tuple(saved.get("features", args.features))
    out_ch = int(saved.get("out_ch", args.out_ch))

    print(f"[fold {fold}] 载入 best.pth (ep{ckpt.get('epoch')}, patch Dice {ckpt.get('dice'):.4f})")
    model = UNet3D(in_ch=1, out_ch=out_ch, features=features,
                   deep_supervision=True, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model"])

    print(f"[fold {fold}] 滑窗全卷评估 {len(val_ids)} 例，patch={patch}³ ...")
    organ_dice = full_volume_eval(model, val_ids, cache_dir, (patch,) * 3, out_ch, device)

    out = {"fold": fold, "best_epoch": ckpt.get("epoch"),
           "patch_dice": ckpt.get("dice"), "organ_dice": organ_dice}
    json_dump(out, out_dir / "val_organ_dice.json", root=ROOT)

    print(f"[fold {fold}] 全卷评估:")
    for k, v in organ_dice.items():
        print(f"    {k:<16} {v:.4f}")
    return out


# ----------------------------------------------------------------------------- 主入口
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0, help="-1 表示跑全部 5 折")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--features", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--out-ch", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--samples-per-case", type=int, default=4)
    ap.add_argument("--val-samples", type=int, default=4)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--deep-sup", action="store_true", default=True)
    ap.add_argument("--no-deep-sup", dest="deep_sup", action="store_false")
    ap.add_argument("--smoke", action="store_true", help="小规模冒烟测试")
    ap.add_argument("--limit-cases", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="outputs", help="产物目录（V2 用 outputs_v2）")
    ap.add_argument("--strong-aug", action="store_true", help="启用强增强：90°旋转/缩放/Gamma")
    ap.add_argument("--eval-only", action="store_true", help="跳过训练，只对已有 best.pth 做全卷评估")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.limit_cases = args.limit_cases or 8
        args.samples_per_case = 2
        args.val_samples = 1
        args.workers = 0
        args.patch = min(args.patch, 96)
        print("=== 冒烟模式 ===")

    # 产物目录必须位于项目内，防止 --out-dir 携带 ../ 造成路径穿越
    root_resolved = ROOT.resolve()
    out_root = (ROOT / args.out_dir).resolve()
    if not out_root.is_relative_to(root_resolved):
        raise SystemExit(f"--out-dir 必须位于项目目录内，收到: {args.out_dir!r}")
    args.out_dir = out_root.relative_to(root_resolved).as_posix()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    root = ROOT / "data" / "processed" / "amos22"
    cache_dir = ROOT / "data" / "cache" / "amos22"
    cases = build_case_list(root)
    ids = [c.case_id for c in cases]
    have = {p.stem.replace(".img", "") for p in Path(cache_dir).glob("*.img.npy")}
    ids = [i for i in ids if i in have]
    if args.limit_cases:
        ids = ids[: args.limit_cases]
    print(f"可用缓存病例: {len(ids)}")

    folds = make_folds(ids, n_folds=5, seed=42)
    which = range(5) if args.fold == -1 else [args.fold]

    summary = []
    for k in which:
        tr, va = folds[k]
        if args.eval_only:
            summary.append(eval_fold_only(k, va, cache_dir, args, device))
        else:
            summary.append(train_fold(k, tr, va, cache_dir, args, device))

    out = ROOT / args.out_dir / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json_dump(summary, out, root=ROOT)
    print(f"\n汇总已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
