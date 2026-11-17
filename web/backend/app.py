"""Web 后端：DICOM/NIfTI 上传 → 分割 → 逐层 PNG 渲染。

架构
----
    上传 → 预处理(重采样1.5mm) → 5折集成滑窗推理 → 存缓存(.npy)
                                                   ↓
    前端滚动 ← 按需渲染 /slice/{z}.png ← 读取缓存(内存映射)

关键设计
--------
- **推理只在后台做一次**，结果落盘；前端每次滚动只读缓存 + 渲染，毫秒级响应。
- **切片索引与原数据 1:1 对应**：展示层数 = 重采样后的层数（不是原始层数），
  但保留 spacing 信息，体积计算准确。
- 进度用轮询（/status），避免 WebSocket 复杂度。

启动:
    uvicorn web.backend.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dicom_ingest import load_series, pick_best_series, scan_directory  # noqa: E402
from render.slice_renderer import ORGAN_NAMES_CN, ORGAN_ORDER, SliceRenderer  # noqa: E402
from utils.safeio import bin_writer, safe_extract_all  # noqa: E402

WORK = ROOT / "outputs" / "web_cases"
WORK.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 16
# 优先使用 V2 权重（112³ patch + 强增强，5 折 mDice 0.8056；V1 是 0.7665）
MODEL_DIR = ROOT / "outputs_v2" if (ROOT / "outputs_v2").exists() else ROOT / "outputs"
TARGET_SPACING = 1.5


# ============================================================ 病例状态
@dataclass
class Case:
    case_id: str
    dir: Path
    status: str = "queued"          # queued|preprocess|infer|done|error
    progress: float = 0.0
    message: str = ""
    name: str = ""
    shape: tuple = ()
    spacing: tuple = ()
    n_slices: int = 0
    organs: dict = field(default_factory=dict)     # {class_id: volume_ml}
    elapsed: float = 0.0
    error: str = ""
    best_z: int = 0                # 器官最丰富的层，前端首次打开停在这里
    mask_source: str = "model"     # model | ground_truth（区分预测与参考标注）
    warning: str = ""              # 非 CT 或层数异常时的提示

    @property
    def img_path(self) -> Path:
        return self.dir / "vol.img.npy"

    @property
    def lbl_path(self) -> Path:
        return self.dir / "vol.lbl.npy"

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id, "name": self.name, "status": self.status,
            "progress": round(self.progress, 3), "message": self.message,
            "shape": list(self.shape), "spacing": list(self.spacing),
            "n_slices": self.n_slices,
            "organs": {ORGAN_NAMES_CN[ORGAN_ORDER[int(k)]]: v for k, v in self.organs.items()},
            "organs_by_id": {int(k): v for k, v in self.organs.items()},
            "elapsed": round(self.elapsed, 1), "error": self.error,
            "best_slice": int(self.best_z),
            "mask_source": self.mask_source,
            "warning": self.warning,
        }


CASES: dict[str, Case] = {}


# ============================================================ 预处理
def _resample_isotropic(vol: np.ndarray, spacing: tuple, target: float = TARGET_SPACING,
                        is_label: bool = False) -> tuple[np.ndarray, tuple]:
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(vol.astype(np.int16 if not is_label else np.uint8))
    img.SetSpacing(tuple(float(s) for s in spacing[::-1]))     # SITK: x,y,z
    orig = np.array(img.GetSpacing())
    size = np.array(img.GetSize(), dtype=float)
    new_size = np.maximum(np.round(orig * size / target).astype(int), 1)

    rf = sitk.ResampleImageFilter()
    rf.SetSize([int(v) for v in new_size])
    rf.SetOutputSpacing([float(target)] * 3)
    rf.SetOutputOrigin(img.GetOrigin())
    rf.SetOutputDirection(img.GetDirection())
    rf.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    rf.SetDefaultPixelValue(-1024.0)
    out = rf.Execute(img)
    arr = sitk.GetArrayFromImage(out)
    return arr, (target, target, target)


def _load_any(path: Path) -> tuple[np.ndarray, tuple, str, str]:
    """返回 (HU体积[Z,Y,X], spacing(z,y,x), 来源描述, 模态)。"""
    if path.suffix.lower() in (".nii", ".gz"):
        import nibabel as nib
        nii = nib.load(str(path))
        data = np.asanyarray(nii.dataobj)
        if data.ndim == 4:
            data = data[..., 0]
        arr = np.ascontiguousarray(np.transpose(data, (2, 1, 0))).astype(np.float32)
        z = nii.header.get_zooms()[:3]
        return arr, (float(z[2]), float(z[1]), float(z[0])), f"NIfTI {path.name}", "NIFTI"

    series = scan_directory(path)
    if not series:
        raise ValueError("压缩包内没有找到可识别的 DICOM 序列")
    best = pick_best_series(series)
    vol = load_series(best, target_spacing=None)
    return (vol.pixels, vol.spacing,
            f"DICOM: {best.description} ({best.n_files} 层)",
            str(getattr(best, "modality", "") or ""))


# ============================================================ 后台任务
def process_case(case: Case, src: Path) -> None:
    t0 = time.time()
    try:
        case.status, case.progress, case.message = "preprocess", 0.05, "读取影像 ..."
        vol, spacing, desc, modality = _load_any(src)
        case.name = desc

        # --- 模态与几何检查：模型只在 CT 腹部数据上训练过 ---
        warns = []
        if modality and modality.upper() not in ("CT", ""):
            warns.append(
                f"检测到模态为 {modality.upper()}，但模型仅在 CT 上训练。"
                f"非 CT 的像素值不适用 HU 窗宽窗位归一化，分割结果不可信。"
            )
        if vol.shape[0] < 3:
            warns.append(
                f"该序列只有 {vol.shape[0]} 层，构不成三维体积。"
                f"DR/X 光等二维投影影像无法进行三维分割。"
            )
        if warns:
            case.warning = "；".join(warns)
            print(f"[case {case.case_id}] 警告: {case.warning}")
        case.message = f"重采样到 {TARGET_SPACING}mm 等向 ..."
        case.progress = 0.15

        vol_rs, sp = _resample_isotropic(vol, spacing)
        np.save(case.img_path, vol_rs.astype(np.int16))
        case.shape, case.spacing, case.n_slices = tuple(vol_rs.shape), sp, vol_rs.shape[0]

        case.status, case.progress, case.message = "infer", 0.25, "加载分割模型 ..."
        from infer.predictor import Predictor, find_checkpoints

        ckpts = find_checkpoints(MODEL_DIR)
        predictor = Predictor(ckpts, patch=(96, 96, 96))

        case.message = f"滑窗推理（{len(ckpts)} 折集成）..."
        lbl = predictor.predict(vol_rs, verbose=False)
        case.progress = 0.9
        np.save(case.lbl_path, lbl.astype(np.uint8))

        # 各器官体积（1.5mm 各向同性，单体素 = 3.375 mm³）
        # 一次 bincount 统计所有类别，避免 15 次全卷遍历
        counts = np.bincount(np.asarray(lbl).ravel(), minlength=NUM_CLASSES)
        vox_mm3 = TARGET_SPACING ** 3
        case.organs = {c: round(int(counts[c]) * vox_mm3 / 1000.0, 1)
                       for c in range(1, NUM_CLASSES) if counts[c] > 200}

        # 找器官最丰富的层（前端默认停在这里，避免停在无器官的空层）
        per_slice = (np.asarray(lbl) > 0).sum(axis=(1, 2))
        case.best_z = int(np.argmax(per_slice)) if per_slice.size else 0

        case.status, case.progress = "done", 1.0
        case.message = f"完成，识别 {len(case.organs)} 个器官"
        case.elapsed = time.time() - t0
        print(f"[case {case.case_id}] 完成 {case.shape} {case.elapsed:.1f}s 器官={len(case.organs)}")

    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        # 把底层库的英文异常翻译成用户能理解的说明
        low = str(exc).lower()
        if "decompress" in low and ("plugin" in low or "pixel data" in low):
            msg = ("该 DICOM 使用压缩像素数据，但当前环境缺少对应的解压插件。"
                   "请安装 pylibjpeg / pylibjpeg-libjpeg / pylibjpeg-openjpeg / pylibjpeg-rle "
                   f"后重试。（原始错误：{exc}）")
        elif "no pixel data" in low:
            msg = "这个文件里没有像素数据（可能是 DICOMDIR 或结构化报告），无法用于分割。"
        elif "only uncompressed" in low:
            msg = "像素数据格式异常，无法读取。"
        case.status, case.error = "error", msg
        traceback.print_exc()
    finally:
        if case.status != "error":
            case.elapsed = time.time() - t0
        # 清理上传临时目录
        try:
            if src.parent.name == "upload":
                shutil.rmtree(src.parent, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass


# ============================================================ FastAPI
app = FastAPI(title="腹部多器官分割 Demo")

FRONTEND = ROOT / "web" / "frontend"
if (FRONTEND / "static").exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    f = FRONTEND / "index.html"
    if not f.exists():
        return HTMLResponse("<h1>前端文件缺失</h1>", status_code=500)
    return HTMLResponse(f.read_text(encoding="utf-8"))


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> JSONResponse:
    cid = uuid.uuid4().hex[:12]
    up_dir = WORK / cid / "upload"
    up_dir.mkdir(parents=True, exist_ok=True)

    # 只取文件名部分，杜绝客户端用 ../ 之类的路径把文件写到病例目录之外
    fname = Path(file.filename or "upload.bin").name or "upload.bin"
    dst = up_dir / fname
    with bin_writer(dst, root=WORK) as f:
        shutil.copyfileobj(file.file, f)

    # 如果是 zip，先解压
    src = dst
    if dst.suffix.lower() == ".zip":
        extract_dir = up_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)
        try:
            with zipfile.ZipFile(dst) as z:
                safe_extract_all(z, extract_dir, skip_parts={"__MACOSX", ".DS_Store"})
            src = extract_dir
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(400, f"zip 解压失败: {exc}") from exc

    case = Case(case_id=cid, dir=WORK / cid)
    CASES[cid] = case
    threading.Thread(target=process_case, args=(case, src), daemon=True).start()
    return JSONResponse({"case_id": cid})


@app.get("/api/case/{case_id}/status")
def status(case_id: str) -> JSONResponse:
    c = CASES.get(case_id)
    if not c:
        raise HTTPException(404, "病例不存在")
    return JSONResponse(c.to_dict())


@app.get("/api/case/{case_id}/slice/{z}")
def slice_png(
    case_id: str, z: int,
    wl: float = Query(40.0), ww: float = Query(400.0),
    alpha: float = Query(0.45), mode: str = Query("both"),
    show_mask: bool = Query(True),
) -> Response:
    c = CASES.get(case_id)
    if not c or c.status != "done":
        raise HTTPException(404, "病例未就绪")
    img = np.load(c.img_path, mmap_mode="r")
    lbl = np.load(c.lbl_path, mmap_mode="r") if show_mask else None
    z = max(0, min(int(z), img.shape[0] - 1))

    r = SliceRenderer(wl=wl, ww=ww, alpha=alpha, mode=mode)
    present = sorted(set(int(v) for v in np.unique(lbl[z])) - {0}) if lbl is not None else None
    pil = r.render(img[z], lbl[z] if lbl is not None else None,
                   z=z, n_total=img.shape[0], present_organs=present)
    buf = io.BytesIO()
    pil.save(buf, format="PNG", optimize=True)
    return Response(content=buf.getvalue(), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=600"})


@app.get("/api/case/{case_id}/montage")
def montage(case_id: str, n: int = Query(24)) -> Response:
    """缩略图墙：等间隔取 n 层，拼成网格（报告/PPT 用）。"""
    c = CASES.get(case_id)
    if not c or c.status != "done":
        raise HTTPException(404, "病例未就绪")
    from PIL import Image

    img = np.load(c.img_path, mmap_mode="r")
    lbl = np.load(c.lbl_path, mmap_mode="r")
    total = img.shape[0]
    idx = np.linspace(0, total - 1, n).astype(int)
    r = SliceRenderer(mode="fill")
    thumbs = []
    for z in idx:
        present = sorted(set(int(v) for v in np.unique(lbl[z])) - {0})
        im = r.render(img[z], lbl[z], present_organs=present, scale=1)
        thumbs.append(im.crop((0, 0, min(im.width, im.height), min(im.width, im.height))))

    cols = 6
    rows = (len(thumbs) + cols - 1) // cols
    tw, th = thumbs[0].size
    canvas = Image.new("RGB", (cols * tw, rows * th), (12, 14, 18))
    for i, t in enumerate(thumbs):
        canvas.paste(t, ((i % cols) * tw, (i // cols) * th))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/api/palette")
def palette_api() -> JSONResponse:
    """返回器官配色（唯一数据源在 configs/labels.yaml）。

    前端不再硬编码颜色，避免出现"后端图例有颜色、前端图例是黑块"这类不同步问题。
    """
    from render.slice_renderer import ORGAN_NAMES_CN, ORGAN_ORDER, _load_palette

    pal = _load_palette()
    colors = {}
    for i, key in enumerate(ORGAN_ORDER):
        if i == 0:
            continue
        c = tuple(int(v) for v in pal[i])
        colors[str(i)] = {"hex": "#%02x%02x%02x" % c, "rgb": list(c),
                          "name": ORGAN_NAMES_CN.get(key, key)}
    return JSONResponse({"palette": colors})


@app.get("/api/demo_cases")
def demo_cases() -> JSONResponse:
    """列出可直接演示的验证集病例（无需上传）。"""
    cache = ROOT / "data" / "cache" / "amos22"
    items = []
    for p in sorted(cache.glob("*.img.npy"))[:8]:
        items.append({"id": p.name.replace(".img.npy", "")})
    return JSONResponse({"cases": items})


@app.post("/api/case/from_cache/{cid}")
def from_cache(cid: str) -> JSONResponse:
    """直接把缓存里的验证集病例变成可看病例（跳过上传与推理）。"""
    cache = ROOT / "data" / "cache" / "amos22"
    imp, lmp = cache / f"{cid}.img.npy", cache / f"{cid}.lbl.npy"
    if not imp.exists():
        raise HTTPException(404, "缓存中无此病例")
    case_id = uuid.uuid4().hex[:12]
    d = WORK / case_id
    d.mkdir(parents=True, exist_ok=True)
    # 用符号链接会跨盘问题，直接硬链接/复制
    import os
    for src, dst in ((imp, d / "vol.img.npy"), (lmp, d / "vol.lbl.npy")):
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)
    img = np.load(d / "vol.img.npy", mmap_mode="r")
    lbl = np.load(d / "vol.lbl.npy", mmap_mode="r")
    counts = np.bincount(np.asarray(lbl).ravel(), minlength=NUM_CLASSES)
    vox = TARGET_SPACING ** 3
    organs = {c: round(int(counts[c]) * vox / 1000.0, 1)
              for c in range(1, NUM_CLASSES) if counts[c] > 200}
    per_slice = (np.asarray(lbl) > 0).sum(axis=(1, 2))
    case = Case(case_id=case_id, dir=d, status="done", progress=1.0,
                message="验证集病例（显示的是数据集参考标注，非模型预测）", name=f"{cid} (验证集)",
                shape=tuple(img.shape), spacing=(TARGET_SPACING,) * 3,
                n_slices=img.shape[0], organs=organs, elapsed=0.0,
                best_z=int(np.argmax(per_slice)) if per_slice.size else 0,
                mask_source="ground_truth")
    CASES[case_id] = case
    return JSONResponse({"case_id": case_id})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
