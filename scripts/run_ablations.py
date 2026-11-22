"""在交接的 AMOS22 fold0 验证缓存上跑配对消融，不重新训练。"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.infer.postprocess import SOLID_ORGANS, clean_label  # noqa: E402
from src.preprocess import classic  # noqa: E402


ORGANS = (
    "脾脏", "右肾", "左肾", "胆囊", "食管", "肝脏", "胃", "主动脉", "下腔静脉",
    "胰腺", "右肾上腺", "左肾上腺", "十二指肠", "膀胱", "前列腺/子宫",
)
SPACING_MM = 1.5
STRUCTURE = ndi.generate_binary_structure(3, 1)
METHODS = {
    "postprocess": {"none": "无后处理", "cc": "仅连通域", "morph_cc": "3D 开闭运算 + 连通域",
                    "morph_sel_cc": "选择性开闭 + 连通域",
                    "cc_solid": "连通域（实质器官只留最大块）",
                    "morph_sel_cc_solid": "选择性开闭 + 连通域（实质器官只留最大块）"},
    "preprocess": {"none": "无额外预处理", "gaussian": "3D 高斯去噪", "median": "3D 中值滤波",
                   "butterworth": "频域 Butterworth 低通", "clahe": "轴位 CLAHE"},
}
BUTTERWORTH_CUTOFF = classic.BUTTERWORTH_CUTOFF
BUTTERWORTH_ORDER = classic.BUTTERWORTH_ORDER


def archive_pairs(archive: ZipFile) -> dict[str, dict[str, str]]:
    pairs: dict[str, dict[str, str]] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        match = re.fullmatch(r"(amos_\d{4})\.(img|lbl)\.npy", Path(info.filename).name)
        if not match:
            continue
        case_id, kind = match.groups()
        entry = pairs.setdefault(case_id, {})
        if kind in entry:
            raise ValueError(f"{case_id} 有重复的 {kind} 文件")
        entry[kind] = info.filename
    if not pairs or any(set(p) != {"img", "lbl"} for p in pairs.values()):
        raise ValueError("压缩包内必须有配对的 amos_NNNN.img.npy / amos_NNNN.lbl.npy")
    return pairs


def sample(ids: list[str], n: int) -> list[str]:
    if n >= len(ids):
        return ids.copy()
    if n == 1:
        return ids[:1]
    return [ids[i] for i in np.linspace(0, len(ids) - 1, n, dtype=int)]


def load_case(archive: ZipFile, pair: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    with archive.open(pair["img"]) as file:
        image = np.load(file, allow_pickle=False)
    with archive.open(pair["lbl"]) as file:
        label = np.load(file, allow_pickle=False)
    if (image.ndim != 3 or label.shape != image.shape or image.dtype != np.int16
            or label.dtype != np.uint8 or int(label.max()) > len(ORGANS)
            or not label.any()):
        raise ValueError(f"配对体积不符合 AMOS 1.5mm 缓存格式：{pair}")
    return image, label


def morphology(label: np.ndarray, organs: set[int] | None = None) -> np.ndarray:
    """逐器官开后闭；仅允许向原背景填补，多个类别争抢位置时留作背景。

    organs 给定时，只对名单内器官做开闭，其余器官原样保留。
    """
    out = np.zeros_like(label)
    conflict = np.zeros(label.shape, dtype=bool)
    for organ, box in enumerate(ndi.find_objects(label, max_label=len(ORGANS)), start=1):
        if box is None:
            continue
        if organs is not None and organ not in organs:
            out[box][label[box] == organ] = organ
            continue
        # 组合操作的 3D 邻域；裁剪不改变整卷上以 0 为外部值的运算结果。
        region = tuple(slice(max(0, s.start - 2), min(dim, s.stop + 2))
                       for s, dim in zip(box, label.shape))
        local = label[region]
        mask = ndi.binary_opening(local == organ, structure=STRUCTURE)
        mask = ndi.binary_closing(mask, structure=STRUCTURE)
        result = out[region]
        fill = mask & (local == 0)
        conflict[region] |= fill & (result != 0)
        result[(mask & (local == organ)) | fill] = organ
    out[conflict] = 0
    return out


def preprocessed(image: np.ndarray, method: str) -> np.ndarray:
    return classic.apply(image, method, SPACING_MM)


def hd95_mm(pred: np.ndarray, truth: np.ndarray) -> float:
    """6 邻域表面体素中心，取两个方向各自 P95 中较大值。两边均须非空。"""
    def surface(mask: np.ndarray) -> np.ndarray:
        return np.argwhere(mask & ~ndi.binary_erosion(mask, structure=STRUCTURE))

    p, g = surface(pred), surface(truth)
    p_to_g = cKDTree(g).query(p)[0]
    g_to_p = cKDTree(p).query(g)[0]
    return SPACING_MM * float(max(np.percentile(p_to_g, 95), np.percentile(g_to_p, 95)))


def measure(pred: np.ndarray, truth: np.ndarray, surfaces: bool) -> list[dict]:
    p_boxes = ndi.find_objects(pred, max_label=len(ORGANS))
    g_boxes = ndi.find_objects(truth, max_label=len(ORGANS))
    rows = []
    for c, (p_box, g_box) in enumerate(zip(p_boxes, g_boxes), 1):
        boxes = [b for b in (p_box, g_box) if b is not None]
        dice, hd, components = None, None, 0
        if boxes:
            box = tuple(slice(min(s.start for s in axes), max(s.stop for s in axes))
                        for axes in zip(*boxes))
            p, g = pred[box] == c, truth[box] == c
            if g_box is not None:
                dice = float(2 * np.count_nonzero(p & g) / (p.sum() + g.sum()))
            if surfaces and p_box is not None:
                components = int(ndi.label(p, structure=STRUCTURE)[1])
                if g_box is not None:
                    hd = hd95_mm(p, g)
        rows.append({"organ_id": c, "organ": ORGANS[c - 1], "dice": dice, "hd95_mm": hd,
                     "components": components if surfaces else None,
                     "pred_missing": int(g_box is not None and p_box is None)})
    return rows


def predict(normalized: np.ndarray, engine) -> np.ndarray:
    import torch
    from monai.inferers import sliding_window_inference

    model = engine.models[0]

    def logits_for(window):
        with torch.autocast(engine.device.type, enabled=engine.device.type == "cuda"):
            logits = model(window)
        return logits[0] if isinstance(logits, (tuple, list)) else logits

    with torch.inference_mode():
        logits = sliding_window_inference(
            torch.from_numpy(normalized)[None, None], engine.patch,
            sw_batch_size=1, predictor=logits_for, overlap=0.5, mode="gaussian",
            sw_device=engine.device, device="cpu",
        )
        return logits[0].argmax(dim=0).to(torch.uint8).numpy()


def csv_out(path: Path, rows: list[dict]) -> None:
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def mean(values) -> float | None:
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def summarize(out: Path, study: str, case_rows: list[dict]) -> None:
    rows = [r for r in case_rows if r["study"] == study]
    if not rows:
        return
    csv_out(out / f"{study}_per_case.csv", rows)
    summary = []
    for method, name in METHODS[study].items():
        group = [r for r in rows if r["method"] == method]
        if not group:
            continue
        organs = []
        case_n = len({r["case_id"] for r in group})
        for c, organ in enumerate(ORGANS, 1):
            part = [r for r in group if r["organ_id"] == c]
            organs.append({
                "method": method, "name": name, "organ": organ, "cases": case_n,
                "dice": mean(r["dice"] for r in part), "dice_n": sum(r["dice"] is not None for r in part),
                "hd95_mm": mean(r["hd95_mm"] for r in part), "hd95_n": sum(r["hd95_mm"] is not None for r in part),
                "components_per_case": mean(r["components"] for r in part),
                "missed_organs": sum(r["pred_missing"] for r in part),
            })
        counts = [r["components_per_case"] for r in organs if r["components_per_case"] is not None]
        total = {
            "method": method, "name": name, "organ": "全部（器官宏平均）", "cases": case_n,
            "dice": mean(r["dice"] for r in organs), "dice_n": sum(r["dice_n"] for r in organs),
            "hd95_mm": mean(r["hd95_mm"] for r in organs), "hd95_n": sum(r["hd95_n"] for r in organs),
            "components_per_case": float(sum(counts)) if counts else None,
            "missed_organs": sum(r["missed_organs"] for r in organs),
        }
        summary.extend([total, *organs])
    baseline = {r["organ"]: r["dice"] for r in summary if r["method"] == "none"}
    for row in summary:
        d, b = row["dice"], baseline[row["organ"]]
        row["delta_dice_pp"] = 100.0 * (d - b) if d is not None and b is not None else None
    csv_out(out / f"{study}_summary.csv", summary)


def comparison_figure(path: Path, cid: str, truth: np.ndarray, panels: list[tuple]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = int(np.argmax(np.count_nonzero(truth, axis=(1, 2))))
    fig, axes = plt.subplots(2, len(panels), figsize=(3.1 * len(panels), 6.2), layout="constrained")
    for j, (name, image, labels) in enumerate(panels):
        plane = image[z]
        axes[0, j].imshow(plane, cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(name)
        axes[1, j].imshow(plane, cmap="gray", vmin=0, vmax=1)
        axes[1, j].imshow(np.ma.masked_equal(labels[z], 0), cmap="tab20", vmin=1, vmax=15, alpha=0.55)
        for ax in axes[:, j]:
            ax.axis("off")
    fig.suptitle(f"{cid}  |  axial slice {z}  |  input and predicted organs")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--archive", type=Path, required=True, help="原始 fold0_val.zip")
    cli.add_argument("--checkpoint", type=Path, default=ROOT / "outputs_v2/fold0/best.pth")
    cli.add_argument("--out", type=Path, default=ROOT / "outputs/ablation_fold0")
    cli.add_argument("--study", choices=("both", *METHODS), default="both")
    cli.add_argument("--pre-cases", type=int, default=12, help="在 CT 验证病例中等距采样；0=全量")
    cli.add_argument("--max-cases", type=int, default=0, help="限制两个实验的总病例池；0=不限制")
    cli.add_argument("--device", default="cuda")
    cli.add_argument("--patch", type=int, default=112, choices=(64, 96, 112, 128))
    cli.add_argument("--morph-selection", type=Path, default=ROOT / "configs/morph_selection.json",
                     help="ablation_stats.py select 生成的器官名单；只在开发集病例上选出")
    cli.add_argument("--save-pred", action="store_true", help="保存未经后处理的预测，便于之后离线比较后处理")
    args = cli.parse_args()
    if args.pre_cases < 0 or args.max_cases < 0:
        cli.error("病例数不能为负")
    if args.out.exists() and any(args.out.iterdir()):
        cli.error("输出目录已有内容；请指定空目录，避免混入其他运行的结果")

    import torch
    from src.infer.predictor import Predictor

    torch.set_num_threads(min(4, torch.get_num_threads()))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("fold") != 0 or checkpoint.get("args", {}).get("out_ch") != len(ORGANS) + 1:
        cli.error("必须使用 AMOS15 的 fold0 权重，不能用其他折或五折集成")
    epoch = checkpoint.get("epoch")
    del checkpoint
    selection = json.loads(args.morph_selection.read_text(encoding="utf-8"))
    morph_organs = {int(k) for k in selection["morph_organ_ids"]}

    with ZipFile(args.archive) as archive:
        pairs = archive_pairs(archive)
        # 官方压缩包内 readme.md：编号小于 500 的是 CT。
        case_pool = sorted(cid for cid in pairs if int(cid.rsplit("_", 1)[1]) < 500)
        if not case_pool:
            cli.error("压缩包没有 AMOS22 CT 病例")
        ids = sample(case_pool, args.max_cases or len(case_pool))
        post_ids = ids if args.study != "preprocess" else []
        pre_ids = sample(ids, args.pre_cases or len(ids)) if args.study != "postprocess" else []
        order = list(dict.fromkeys([*pre_ids, *post_ids]))
        meta = {
            "archive": args.archive.name, "archive_pairs": len(pairs), "ct_pairs": len(case_pool),
            "checkpoint": args.checkpoint.name, "checkpoint_fold": 0, "checkpoint_epoch": epoch,
            "spacing_zyx_mm": [SPACING_MM] * 3, "device": args.device, "cuda_amp": args.device.startswith("cuda"),
            "patch_zyx": [args.patch] * 3, "overlap": 0.5, "sw_batch_size": 1,
            "input_hu_window": [-150, 250], "postprocess_case_ids": post_ids, "preprocess_case_ids": pre_ids,
            "morph_selection": {"file": args.morph_selection.name, "organs": selection["morph_organs"],
                                "dev_case_ids": selection["dev_case_ids"],
                                "test_case_ids": selection["test_case_ids"]},
            "butterworth": {"cutoff_cycles_per_mm": BUTTERWORTH_CUTOFF, "order": BUTTERWORTH_ORDER},
            "completed_case_ids": [],
        }
        args.out.mkdir(parents=True, exist_ok=True)

        def write_meta() -> None:
            temp = args.out / "run.tmp"
            temp.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temp.replace(args.out / "run.json")

        write_meta()
        engine = Predictor([args.checkpoint], patch=(args.patch,) * 3, device=args.device, sw_batch_size=1)
        rows = []
        for i, cid in enumerate(order, 1):
            start = time.monotonic()
            image, truth = load_case(archive, pairs[cid])
            base_input = engine.normalize(image)
            base = predict(base_input, engine)
            print(f"[{i}/{len(order)}] {cid} raw inference: {time.monotonic() - start:.1f}s", flush=True)

            def add(study: str, method: str, label: np.ndarray) -> None:
                rows.extend({"study": study, "case_id": cid, "method": method, **r}
                            for r in measure(label, truth, surfaces=(study == "postprocess")))

            if cid in post_ids:
                cc, _ = clean_label(base)
                morph_cc, _ = clean_label(morphology(base))
                sel_morph = morphology(base, morph_organs)
                sel_cc, _ = clean_label(sel_morph)
                cc_solid, _ = clean_label(base, largest_only=SOLID_ORGANS)
                sel_cc_solid, _ = clean_label(sel_morph, largest_only=SOLID_ORGANS)
                labels = (base, cc, morph_cc, sel_cc, cc_solid, sel_cc_solid)
                for method, label in zip(METHODS["postprocess"], labels):
                    add("postprocess", method, label)
                if args.save_pred:
                    (args.out / "pred").mkdir(exist_ok=True)
                    np.savez_compressed(args.out / "pred" / f"{cid}.npz", pred=base)
                if i == 1:
                    comparison_figure(args.out / "postprocess_slices.png", cid, truth, [
                        ("Raw", base_input, base), ("Connected components", base_input, cc),
                        ("Open / close + components", base_input, morph_cc),
                    ])
                del cc, morph_cc, sel_cc, sel_morph, cc_solid, sel_cc_solid, labels
                summarize(args.out, "postprocess", rows)
            if cid in pre_ids:
                add("preprocess", "none", base)
                panels = [("No extra processing", base_input, base)]
                for method, title in (("gaussian", "Gaussian"), ("median", "Median"),
                                      ("butterworth", "Butterworth LPF"), ("clahe", "CLAHE")):
                    x = preprocessed(image, method)
                    alt = predict(x, engine)
                    add("preprocess", method, alt)
                    if i == 1:
                        panels.append((title, x, alt))
                if i == 1:
                    comparison_figure(args.out / "preprocess_slices.png", cid, truth, panels)
                del panels, x, alt
                summarize(args.out, "preprocess", rows)
            meta["completed_case_ids"].append(cid)
            write_meta()
            print(f"[{i}/{len(order)}] {cid} complete: {time.monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
