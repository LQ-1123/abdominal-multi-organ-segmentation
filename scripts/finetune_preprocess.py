"""训练侧预处理对照：从同一个 V2 fold0 权重出发，用某种预处理后的输入微调，再在验证 CT 上评估。

各组共用种子、epoch 数、学习率、patch 采样和数据增强（AMOSDataset deterministic=True），
只有输入预处理不同。评估用最后一个 epoch 的权重，不按验证集挑 epoch。

    # 只评估原模型（推理端直接加预处理）
    python scripts/finetune_preprocess.py --method gaussian --epochs 0 ...
    # 微调 20 个 epoch 后评估
    python scripts/finetune_preprocess.py --method gaussian --epochs 20 ...

输出 <out>/<tag>/：per_case.csv（逐例逐器官 Dice/HD95）、summary.json、train_log.csv、
model.pth（只含权重）。
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_ablations import ORGANS, measure  # noqa: E402
from src.data.amos_dataset import AMOSDataset  # noqa: E402
from src.models.unet3d import UNet3D  # noqa: E402
from src.preprocess import classic  # noqa: E402
from src.train.losses import DiceCELoss  # noqa: E402
from src.train.train_amos import deep_supervision_loss  # noqa: E402


def method_dir(args, method: str) -> Path:
    return args.base if method == "none" else args.pre_root / method


@torch.no_grad()
def evaluate(model, ids, cache: Path, base: Path, patch: int, device, sw_batch: int) -> list[dict]:
    from monai.inferers import sliding_window_inference

    model.eval()

    def logits_for(x):
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out = model(x)
        return out[0] if isinstance(out, (list, tuple)) else out

    rows = []
    for i, cid in enumerate(ids, 1):
        t0 = time.time()
        x = classic.window(np.load(cache / f"{cid}.img.npy"))
        truth = np.load(base / f"{cid}.lbl.npy")
        logits = sliding_window_inference(torch.from_numpy(x)[None, None].to(device), (patch,) * 3,
                                          sw_batch_size=sw_batch, predictor=logits_for,
                                          overlap=0.5, mode="gaussian")
        pred = logits[0].argmax(0).to(torch.uint8).cpu().numpy()
        del logits
        rows.extend({"case_id": cid, **r} for r in measure(pred, truth, surfaces=True))
        print(f"  eval [{i}/{len(ids)}] {cid} {time.time() - t0:.1f}s", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=classic.METHODS, required=True)
    ap.add_argument("--base", type=Path, required=True, help="原始 HU 缓存目录")
    ap.add_argument("--pre-root", type=Path, required=True, help="prepare_preprocessed_cache.py 的输出目录")
    ap.add_argument("--split", type=Path, required=True, help="含 train_ids / val_ids 的 JSON")
    ap.add_argument("--init", type=Path, required=True, help="V2 fold0 best.pth")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--samples-per-case", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sw-batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    tag = f"{args.method}_ft{args.epochs}"
    out = args.out / tag
    if (out / "summary.json").exists():
        print(f"{tag} 已完成，跳过")
        return
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.init, map_location="cpu", weights_only=True)
    a = ckpt["args"]
    if ckpt.get("fold") != 0:
        raise SystemExit("初始权重必须是 fold0")
    patch = int(a["patch"])
    model = UNet3D(in_ch=1, out_ch=a["out_ch"], features=tuple(a["features"]),
                   deep_supervision=True, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model"])
    del ckpt

    split = json.loads(args.split.read_text(encoding="utf-8"))
    cache = method_dir(args, args.method)
    log_rows = []
    if args.epochs > 0:
        ds = AMOSDataset(split["train_ids"], cache, patch=(patch,) * 3,
                         samples_per_case=args.samples_per_case, train=True, strong_aug=True,
                         num_classes=a["out_ch"], seed=args.seed, deterministic=True)
        loss_fn = DiceCELoss(a["out_ch"], include_background=False).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.01)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        for epoch in range(1, args.epochs + 1):
            ds.epoch = epoch
            gen = torch.Generator().manual_seed(args.seed * 1000 + epoch)
            loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, generator=gen,
                                                 num_workers=args.workers, pin_memory=True, drop_last=True)
            model.train()
            t0, total, n = time.time(), 0.0, 0
            for batch in loader:
                img = batch["image"].to(device, non_blocking=True)
                lbl = batch["label"].to(device, non_blocking=True)
                with torch.autocast("cuda", enabled=device.type == "cuda"):
                    loss = deep_supervision_loss(loss_fn, model(img), lbl, a["out_ch"])
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
                scaler.step(opt)
                scaler.update()
                total += loss.item()
                n += 1
            sched.step()
            row = {"epoch": epoch, "train_loss": total / max(n, 1), "lr": opt.param_groups[0]["lr"],
                   "sec": time.time() - t0}
            log_rows.append(row)
            print(f"[{tag}] ep {epoch}/{args.epochs} loss {row['train_loss']:.4f} {row['sec']:.0f}s", flush=True)
        torch.save({"model": model.state_dict(), "args": a, "fold": 0, "method": args.method,
                    "epochs": args.epochs}, out / "model.pth")
        with (out / "train_log.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            w.writeheader()
            w.writerows(log_rows)

    rows = evaluate(model, split["val_ids"], cache, args.base, patch, device, args.sw_batch)
    with (out / "per_case.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["study", "method", *rows[0].keys()])
        w.writeheader()
        w.writerows({"study": "train_side", "method": tag, **r} for r in rows)

    organ = {}
    for c, name in enumerate(ORGANS, 1):
        d = [r["dice"] for r in rows if r["organ_id"] == c and r["dice"] is not None]
        h = [r["hd95_mm"] for r in rows if r["organ_id"] == c and r["hd95_mm"] is not None]
        organ[name] = {"dice": float(np.mean(d)) if d else None, "hd95_mm": float(np.mean(h)) if h else None}
    summary = {
        "tag": tag, "method": args.method, "epochs": args.epochs, "lr": args.lr, "seed": args.seed,
        "train_cases": len(split["train_ids"]), "val_cases": len(split["val_ids"]),
        "mean_dice": float(np.mean([v["dice"] for v in organ.values() if v["dice"] is not None])),
        "mean_hd95_mm": float(np.mean([v["hd95_mm"] for v in organ.values() if v["hd95_mm"] is not None])),
        "organs": organ, "final_train_loss": log_rows[-1]["train_loss"] if log_rows else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[{tag}] mean Dice {summary['mean_dice']:.4f}  HD95 {summary['mean_hd95_mm']:.2f} mm", flush=True)


if __name__ == "__main__":
    main()
