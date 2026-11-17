"""汇总 k 折交叉验证结果，并生成 V1/V2 对比表。

为什么需要这个脚本
------------------
训练脚本每折结束会单独写 ``<out_dir>/fold<k>/val_organ_dice.json``，
但 ``summary.json`` 每次运行只包含当次跑的折（会被覆盖）。
这里统一扫描所有 fold 目录，合并成完整的交叉验证汇总。

用法
----
    python scripts/summarize_cv.py                      # 汇总 outputs_v2
    python scripts/summarize_cv.py --dirs outputs outputs_v2   # 两个版本对比
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.safeio import csv_writer  # noqa: E402

ORGAN_ORDER = [
    "spleen", "right_kidney", "left_kidney", "gallbladder", "esophagus",
    "liver", "stomach", "aorta", "postcava", "pancreas",
    "right_adrenal", "left_adrenal", "duodenum", "bladder", "prostate_uterus",
]
ORGAN_CN = {
    "spleen": "脾脏", "right_kidney": "右肾", "left_kidney": "左肾",
    "gallbladder": "胆囊", "esophagus": "食管", "liver": "肝脏", "stomach": "胃",
    "aorta": "主动脉", "postcava": "下腔静脉", "pancreas": "胰腺",
    "right_adrenal": "右肾上腺", "left_adrenal": "左肾上腺",
    "duodenum": "十二指肠", "bladder": "膀胱", "prostate_uterus": "前列腺/子宫",
}


def collect(out_dir: Path) -> dict:
    """扫描 fold*/val_organ_dice.json，返回 {fold: record}。"""
    recs = {}
    for p in sorted(out_dir.glob("fold*/val_organ_dice.json")):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        recs[int(j["fold"])] = j
    return recs


def summarize(out_dir: Path, verbose: bool = True) -> dict:
    recs = collect(out_dir)
    if not recs:
        print(f"⚠️ {out_dir} 下没有找到 fold*/val_organ_dice.json")
        return {}

    folds = sorted(recs)
    means = [recs[k]["organ_dice"]["mean"] for k in folds]
    patch = [recs[k]["patch_dice"] for k in folds]

    per_organ: dict[str, list[float]] = {o: [] for o in ORGAN_ORDER}
    for k in folds:
        for o in ORGAN_ORDER:
            v = recs[k]["organ_dice"].get(o)
            if v is not None and v == v:      # 排除 NaN
                per_organ[o].append(v)

    result = {
        "version": out_dir.name,
        "n_folds": len(folds),
        "folds": folds,
        "mean_dice": float(sum(means) / len(means)),
        "std_dice": float(statistics.stdev(means)) if len(means) > 1 else 0.0,
        "per_fold": {k: {"patch_dice": recs[k]["patch_dice"],
                         "best_epoch": recs[k]["best_epoch"],
                         "organ_dice": recs[k]["organ_dice"]} for k in folds},
        "per_organ_mean": {o: (float(sum(v) / len(v)) if v else None)
                           for o, v in per_organ.items()},
        "mean_patch_dice": float(sum(patch) / len(patch)),
    }

    if verbose:
        print(f"\n=== {out_dir.name}  ({len(folds)} 折) ===")
        print(f"{'fold':>6}{'patch':>9}{'best_ep':>9}{'mDice':>10}")
        print("-" * 36)
        for k in folds:
            print(f"{k:>6}{recs[k]['patch_dice']:>9.4f}{recs[k]['best_epoch']:>9}"
                  f"{recs[k]['organ_dice']['mean']:>10.4f}")
        print("-" * 36)
        print(f"{'平均':>6}{result['mean_patch_dice']:>9.4f}{'':>9}"
              f"{result['mean_dice']:>10.4f}")
        print(f"标准差: {result['std_dice']:.4f}")
        if len(folds) < 5:
            print(f"⚠️ 目前只有 {len(folds)} 折，剩余折数尚未完成")

    out = out_dir / "summary.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def compare(r1: dict, r2: dict) -> None:
    if not r1 or not r2:
        return
    print(f"\n=== {r1['version']}  vs  {r2['version']} ===")
    print(f"{'器官':<14}{r1['version']:>10}{r2['version']:>10}{'Δ':>10}")
    print("-" * 46)
    for o in ORGAN_ORDER:
        a = r1["per_organ_mean"].get(o)
        b = r2["per_organ_mean"].get(o)
        if a is None or b is None:
            continue
        print(f"{ORGAN_CN[o]:<14}{a:>10.4f}{b:>10.4f}{b-a:>+10.4f}")
    print("-" * 46)
    print(f"{'平均':<14}{r1['mean_dice']:>10.4f}{r2['mean_dice']:>10.4f}"
          f"{r2['mean_dice']-r1['mean_dice']:>+10.4f}")

    # 导出 CSV 便于画图
    out = ROOT / "outputs" / "cv_compare.csv"
    with csv_writer(out, root=ROOT, bom=True) as (f, w):
        w.writerow(["器官", r1["version"], r2["version"], "提升"])
        for o in ORGAN_ORDER:
            a = r1["per_organ_mean"].get(o)
            b = r2["per_organ_mean"].get(o)
            if a is None or b is None:
                continue
            w.writerow([ORGAN_CN[o], f"{a:.4f}", f"{b:.4f}", f"{b-a:+.4f}"])
        w.writerow(["平均", f"{r1['mean_dice']:.4f}", f"{r2['mean_dice']:.4f}",
                    f"{r2['mean_dice']-r1['mean_dice']:+.4f}"])
    print(f"\n对比表已导出: {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["outputs_v2"],
                    help="要汇总的目录（多个则做对比）")
    args = ap.parse_args()

    # 目录参数必须位于项目内，防止携带 ../ 造成路径穿越
    root_resolved = ROOT.resolve()
    dirs = []
    for d in args.dirs:
        p = (ROOT / d).resolve()
        if not p.is_relative_to(root_resolved):
            raise SystemExit(f"--dirs 必须位于项目目录内，收到: {d!r}")
        dirs.append(p)

    results = []
    for d in dirs:
        results.append(summarize(d))
    if len(results) == 2:
        compare(results[0], results[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
