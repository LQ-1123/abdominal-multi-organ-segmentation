#!/usr/bin/env bash
# Colab 上的训练侧预处理对照，全流程可重入（已完成的步骤自动跳过）。
#   项目代码放在 /content/proj，fold0 权重放在 /content/proj/outputs_v2/fold0/best.pth，
#   fold0 验证名单放在 /content/proj/configs/fold0_val_ids.json。
# 用法：bash scripts/colab_pipeline.sh [epochs] ["微调组"] ["推理端评估组"]
#   两个会话并行时各分几组，例如 20 "none gaussian median" ""
set -euo pipefail
EPOCHS=${1:-20}
FT_METHODS=${2:-"none gaussian median butterworth clahe"}
EVAL_METHODS=${3-"none gaussian median butterworth clahe"}
P=/content/proj
D=/content/data
cd "$P"
mkdir -p "$D" "$P/outputs/train_side"
NPROC=$(nproc)
W=$(( NPROC > 2 ? NPROC - 2 : 1 ))
echo "== $(date) CPU=$NPROC workers=$W"; nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /content | tail -1

if [ ! -f "$D/.deps" ]; then
  pip -q install monai nibabel SimpleITK scikit-image tqdm
  touch "$D/.deps"
fi

if [ ! -f "$D/.extracted" ]; then
  if [ ! -f "$D/amos22.zip.done" ]; then
    command -v aria2c >/dev/null || (apt-get -qq update && apt-get -qq install -y aria2 >/dev/null)
    aria2c -x 16 -s 16 -c -d "$D" -o amos22.zip \
      "https://zenodo.org/api/records/7155725/files/amos22.zip/content"
    touch "$D/amos22.zip.done"
  fi
  # 只解出有标注的 CT（编号 < 500）
  python - <<'EOF'
import re, zipfile, pathlib
src = pathlib.Path("/content/data/amos22.zip"); out = pathlib.Path("/content/data/raw")
with zipfile.ZipFile(src) as z:
    names = [n for n in z.namelist()
             if re.search(r"/(imagesTr|labelsTr|imagesVa|labelsVa)/amos_0([0-4]\d\d)\.nii\.gz$", n)]
    all_labeled = sorted({re.search(r"amos_\d{4}", n).group() for n in z.namelist()
                          if re.search(r"/labels(Tr|Va)/amos_\d{4}\.nii\.gz$", n)})
    for n in names:
        parts = pathlib.Path(n).parts
        dst = out / parts[-2] / parts[-1]
        dst.parent.mkdir(parents=True, exist_ok=True)
        with z.open(n) as f, open(dst, "wb") as g:
            while chunk := f.read(1 << 22):
                g.write(chunk)
pathlib.Path("/content/data/all_labeled_ids.txt").write_text("\n".join(all_labeled) + "\n")
print("extracted", len(names), "files; labeled cases in archive:", len(all_labeled))
EOF
  rm -f "$D/amos22.zip"
  touch "$D/.extracted"
fi

if [ ! -f "$D/.cached" ]; then
  python scripts/build_cache.py --ct-only --root "$D/raw" --out "$D/cache" --workers "$W"
  touch "$D/.cached"
fi

# 与交接包里的 fold0 缓存逐字节比对两例，确认这里重建的缓存和原训练用的一致
python - <<'EOF'
import hashlib, numpy as np
want = {"amos_0005.img.npy": "8aea520909807b8258e1b71d0020bbee",
        "amos_0005.lbl.npy": "4cea0938fe27e6065d2104d3f3ca8d26"}
for name, md5 in want.items():
    got = hashlib.md5(np.load(f"/content/data/cache/{name}").tobytes()).hexdigest()
    print(name, "match" if got == md5 else f"MISMATCH {got}")
EOF

python - <<'EOF'
import json, pathlib, sys
sys.path.insert(0, "/content/proj")
from src.data.amos_dataset import make_folds
val_all = json.loads(pathlib.Path("/content/proj/configs/fold0_val_ids.json").read_text())
cached = sorted(p.name[:-8] for p in pathlib.Path("/content/data/cache").glob("*.img.npy"))
val = [c for c in val_all if c in cached]
train = [c for c in cached if c not in set(val_all)]
labeled = pathlib.Path("/content/data/all_labeled_ids.txt").read_text().split()
reproduced = make_folds(labeled, 5, seed=42)[0][1] == sorted(val_all)
split = {"train_ids": train, "val_ids": val, "make_folds_reproduces_fold0": reproduced,
         "labeled_cases": len(labeled)}
pathlib.Path("/content/proj/outputs/train_side/split.json").write_text(json.dumps(split, indent=1))
print(f"train CT {len(train)}  val CT {len(val)}  make_folds reproduces fold0: {reproduced}")
EOF

NEED=$(echo $FT_METHODS $EVAL_METHODS | tr ' ' '\n' | grep -v '^none$' | sort -u | tr '\n' ' ' || true)
if [ -n "${NEED// /}" ]; then
  python scripts/prepare_preprocessed_cache.py --base "$D/cache" --out "$D/pre" \
    --ids outputs/train_side/split.json --workers "$W" --methods $NEED
fi

COMMON=(--base "$D/cache" --pre-root "$D/pre" --split outputs/train_side/split.json
        --init outputs_v2/fold0/best.pth --out outputs/train_side --workers "$W")
# 先跑最耗时、最关键的训练端对照，推理端评估放最后
for m in $FT_METHODS; do
  python scripts/finetune_preprocess.py --method "$m" --epochs "$EPOCHS" "${COMMON[@]}"
done
for m in $EVAL_METHODS; do
  python scripts/finetune_preprocess.py --method "$m" --epochs 0 "${COMMON[@]}"
done
echo "== ALL DONE $(date)"
