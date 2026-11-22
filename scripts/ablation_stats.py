"""消融结果的配对统计，以及选择性形态学的器官名单。

两个子命令：

select  在开发集（按病例编号排序后的偶数位病例）上比较“仅连通域”和
        “开闭运算 + 连通域”的逐器官平均 Dice，Dice 不下降的器官进入名单。
        名单只看开发集，测试集（奇数位病例）留给 run_ablations.py 复核。
stats   对 *_per_case.csv 做病例级配对比较：每例先对有真值的器官取平均，
        再与对照组配对，输出均值差、bootstrap 95% 置信区间和 Wilcoxon 符号秩检验 p 值。
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in ("dice", "hd95_mm", "components"):
            r[key] = float(r[key]) if r.get(key) not in (None, "") else None
        r["organ_id"] = int(r["organ_id"])
    return rows


def split_cases(case_ids: list[str]) -> tuple[list[str], list[str]]:
    ordered = sorted(set(case_ids))
    return ordered[0::2], ordered[1::2]


def organ_means(rows: list[dict], method: str, cases: set[str]) -> dict[int, float]:
    acc: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r["method"] == method and r["case_id"] in cases and r["dice"] is not None:
            acc[r["organ_id"]].append(r["dice"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def cmd_select(args) -> None:
    rows = [r for r in read_rows(args.per_case) if r["study"] == "postprocess"]
    dev, test = split_cases([r["case_id"] for r in rows])
    cc = organ_means(rows, "cc", set(dev))
    morph = organ_means(rows, "morph_cc", set(dev))
    names = {r["organ_id"]: r["organ"] for r in rows}
    chosen = sorted(k for k in cc if morph[k] - cc[k] >= 0)
    record = {
        "rule": "开发集上 Dice(开闭+连通域) - Dice(仅连通域) >= 0 的器官启用开闭运算",
        "dev_case_ids": dev, "test_case_ids": test,
        "dev_delta_pp": {names[k]: round(100 * (morph[k] - cc[k]), 3) for k in sorted(cc)},
        "morph_organ_ids": chosen, "morph_organs": [names[k] for k in chosen],
    }
    args.out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


def case_scores(rows: list[dict], method: str, key: str) -> dict[str, float]:
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["method"] == method and r[key] is not None:
            acc[r["case_id"]].append(r[key])
    if key == "components":
        return {k: float(np.sum(v)) for k, v in acc.items()}
    return {k: float(np.mean(v)) for k, v in acc.items()}


def paired(a: dict[str, float], b: dict[str, float], rng) -> dict:
    ids = sorted(set(a) & set(b))
    d = np.array([a[i] - b[i] for i in ids])
    boots = rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {"n": len(ids), "mean_delta": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "wilcoxon_p": p, "better_cases": int((d > 0).sum()), "worse_cases": int((d < 0).sum())}


def cmd_stats(args) -> None:
    rows = read_rows(args.per_case)
    if args.cases:
        keep = set(json.loads(args.cases.read_text(encoding="utf-8"))[args.cases_key])
        rows = [r for r in rows if r["case_id"] in keep]
    rng = np.random.default_rng(0)
    methods = list(dict.fromkeys(r["method"] for r in rows))
    out = []
    for key in ("dice", "hd95_mm", "components"):
        if all(r[key] is None for r in rows):
            continue
        scores = {m: case_scores(rows, m, key) for m in methods}
        for ref in args.reference:
            if ref not in scores:
                continue
            for m in methods:
                if m == ref:
                    continue
                res = paired(scores[m], scores[ref], rng)
                out.append({"metric": key, "method": m, "reference": ref, **res})
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = cli.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--per-case", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.set_defaults(func=cmd_select)
    t = sub.add_parser("stats")
    t.add_argument("--per-case", type=Path, required=True)
    t.add_argument("--reference", nargs="+", default=["none"])
    t.add_argument("--cases", type=Path, help="只统计 JSON 文件里列出的病例")
    t.add_argument("--cases-key", default="test_case_ids")
    t.add_argument("--out", type=Path)
    t.set_defaults(func=cmd_stats)
    args = cli.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
