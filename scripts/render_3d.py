"""3D 器官面绘制（marching cubes + VTK 离屏渲染）。

原理
----
    标签体 [Z,Y,X] uint8
        → 每个器官二值化 → vtkFlyingEdges3D 提取等值面（三角网格）
        → 平滑 + 简化（减少三角形数，提升观感）
        → 按调色板上色 → 光照 → 离屏渲染成 PNG

为什么用 VTK 而不是 matplotlib
----------------------------
matplotlib 的 3D 用画家算法排序，15 个器官互相嵌套时会出现**深度错误**
（后面的器官盖住前面的）。VTK 用真正的 z-buffer，遮挡关系正确。

用法
----
    # 参考标注
    python scripts/render_3d.py --label data/cache/amos22/amos_0004.lbl.npy \
        --out outputs/demo/3d_gt.png --view oblique

    # 模型预测
    python scripts/render_3d.py --label outputs/web_cases/<id>/vol.lbl.npy \
        --out outputs/demo/3d_pred.png --view oblique

    # 带半透明体表（更好看，能看到器官在身体里的位置）
    ... --body
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import vtk
from PIL import Image, ImageDraw, ImageFont
from vtk.util.numpy_support import numpy_to_vtk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from render.slice_renderer import ORGAN_NAMES_CN, ORGAN_ORDER, _get_font, _load_palette  # noqa: E402

SPACING = (1.5, 1.5, 1.5)     # 缓存体的各向同性间距（x, y, z）


def label_to_vtk_image(vol: np.ndarray) -> vtk.vtkImageData:
    """numpy [Z,Y,X] → vtkImageData（间距 1.5mm）。"""
    nz, ny, nx = vol.shape
    img = vtk.vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetSpacing(*SPACING)
    img.SetOrigin(0, 0, 0)
    # 关键：VTK 要求 x 变化最快。numpy 数组是 [Z,Y,X]，C 序 ravel 时最后一维(X)最快，
    # 正好符合。若先 transpose 成 [X,Y,Z] 再 ravel，最快轴会变成 Z，数据完全错位
    # （表现为器官被渲染成横跨整个视野的薄片）。
    flat = np.ascontiguousarray(vol.ravel(order="C")).astype(np.float32)
    arr = numpy_to_vtk(flat, deep=True)
    arr.SetName("mask")
    img.GetPointData().SetScalars(arr)
    return img


def extract_surface(binary_vol: np.ndarray, smooth_iters: int = 70, decimate: float = 0.55,
                    blur_sigma: float = 1.8):
    """二值体 → 平滑后的三角网格。

    blur_sigma: 提取等值面前先做高斯模糊。标签是从 5mm 层厚最近邻重采样到 1.5mm 的，
    每层变成 3~4 层相同值，直接 marching cubes 会切出「梯田」状阶梯。先模糊再
    取 0.5 等值面，等价于在亚体素精度上插值，阶梯消失。
    """
    vol = binary_vol.astype(np.float32)
    if blur_sigma > 0:
        from scipy.ndimage import gaussian_filter
        vol = gaussian_filter(vol, sigma=blur_sigma, mode="nearest")
    img = label_to_vtk_image(vol)

    fe = vtk.vtkFlyingEdges3D()          # 比 vtkMarchingCubes 快很多
    fe.SetInputData(img)
    fe.SetValue(0, 0.5)
    fe.ComputeNormalsOn()
    fe.ComputeGradientsOff()
    fe.Update()
    poly = fe.GetOutput()

    if poly.GetNumberOfPoints() == 0:
        return None

    sm = vtk.vtkWindowedSincPolyDataFilter()
    sm.SetInputData(poly)
    sm.SetNumberOfIterations(smooth_iters)
    sm.BoundarySmoothingOff()
    sm.FeatureEdgeSmoothingOff()
    sm.SetPassBand(0.1)
    sm.NormalizeCoordinatesOn()
    sm.Update()
    poly = sm.GetOutput()

    if decimate < 1.0 and poly.GetNumberOfPolys() > 500:
        de = vtk.vtkQuadricDecimation()
        de.SetInputData(poly)
        de.SetTargetReduction(decimate)
        de.Update()
        poly = de.GetOutput()

    return poly


def make_actor(poly, color: tuple[int, int, int], opacity: float = 1.0):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*[c / 255.0 for c in color])
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetAmbient(0.25)
    actor.GetProperty().SetDiffuse(0.85)
    actor.GetProperty().SetSpecular(0.35)
    actor.GetProperty().SetSpecularPower(28)
    actor.GetProperty().SetInterpolationToPhong()
    return actor


def body_surface(ct_vol: np.ndarray | None, hu: float = -300.0):
    """从 CT 提取体表（皮肤）等值面，用于半透明外壳。"""
    if ct_vol is None:
        return None
    img = label_to_vtk_image(np.clip(ct_vol, -1000, 2000).astype(np.float32))
    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputData(img)
    fe.SetValue(0, hu)
    fe.ComputeNormalsOn()
    fe.Update()
    poly = fe.GetOutput()
    if poly.GetNumberOfPoints() == 0:
        return None
    sm = vtk.vtkWindowedSincPolyDataFilter()
    sm.SetInputData(poly)
    sm.SetNumberOfIterations(15)
    sm.SetPassBand(0.1)
    sm.NormalizeCoordinatesOn()
    sm.Update()
    de = vtk.vtkQuadricDecimation()
    de.SetInputData(sm.GetOutput())
    de.SetTargetReduction(0.85)
    de.Update()
    return de.GetOutput()


VIEWS = {
    # 相机方向（相对模型的单位向量），仰角
    "anterior": (0.0, -1.0, 0.15),
    "oblique": (-0.55, -0.75, 0.35),
    "left": (-1.0, -0.15, 0.2),
    "lateral": (0.9, -0.3, 0.25),
}


def render(labels: list, out_png: Path, view: str = "oblique",
           size: tuple[int, int] = (1400, 1100), bg=(255, 255, 255),
           body_poly=None, title: str = "", legend=None) -> Path:
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(*[c / 255.0 for c in bg])
    renderer.SetUseDepthPeeling(1)       # 半透明图层正确混合
    renderer.SetMaximumNumberOfPeels(32)
    renderer.SetOcclusionRatio(0.05)

    if body_poly is not None:
        a = make_actor(body_poly, (235, 235, 235), opacity=0.18)
        a.GetProperty().SetSpecular(0.1)
        renderer.AddActor(a)

    for actor in labels:
        renderer.AddActor(actor)

    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.AddRenderer(renderer)
    win.SetSize(*size)
    win.SetMultiSamples(8)               # 抗锯齿

    # 相机
    renderer.ResetCamera()
    cam = renderer.GetActiveCamera()
    d = np.array(VIEWS.get(view, VIEWS["oblique"]), dtype=float)
    d /= np.linalg.norm(d)

    bounds = renderer.ComputeVisiblePropBounds()
    cx, cy, cz = [(bounds[2 * i] + bounds[2 * i + 1]) / 2 for i in range(3)]
    diag = np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])

    cam.SetFocalPoint(cx, cy, cz)
    cam.SetPosition(cx + d[0] * diag * 1.6, cy + d[1] * diag * 1.6, cz + d[2] * diag * 1.6)
    cam.SetViewUp(0, 0, 1)
    cam.ParallelProjectionOn()
    cam.SetParallelScale(diag * 0.42)
    renderer.ResetCameraClippingRange()

    # 主光 + 补光
    k = vtk.vtkLight()
    k.SetLightTypeToHeadlight()
    renderer.AddLight(k)
    f = vtk.vtkLight()
    f.SetPosition(1, 1, 1)
    f.SetIntensity(0.5)
    renderer.AddLight(f)

    win.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(win)
    w2i.SetInputBufferTypeToRGB()
    w2i.ReadFrontBufferOff()
    w2i.Update()

    writer = vtk.vtkPNGWriter()
    tmp = out_png.with_suffix(".vtk.png")
    writer.SetFileName(str(tmp))
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()

    win.Finalize()

    # 用 PIL 加标题与图例（中文需要 TrueType 字体，VTK 自带字体不支持中文）
    img = Image.open(tmp).convert("RGB")
    tmp.unlink(missing_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if title or legend:
        img = _decorate(img, title, legend)
    img.save(out_png)
    return out_png


def _decorate(img: Image.Image, title: str, legend) -> Image.Image:
    """在渲染图右侧加图例面板，顶部加标题（中文需 TrueType 字体）。"""
    pad_top = 64 if title else 0
    panel_w = 210 if legend else 0
    canvas = Image.new("RGB", (img.width + panel_w, img.height + pad_top), (255, 255, 255))
    canvas.paste(img, (0, pad_top))

    d = ImageDraw.Draw(canvas)
    if title:
        d.text((28, 18), title, fill=(24, 28, 34), font=_get_font(30))
        d.line([28, 58, canvas.width - 28, 58], fill=(220, 224, 230), width=2)

    if legend:
        x0 = img.width + 26
        y = pad_top + 24
        d.text((x0, y), "器官", fill=(110, 118, 130), font=_get_font(16))
        y += 34
        for c, vol in legend:
            color = tuple(int(v) for v in _load_palette()[c])
            d.rectangle([x0, y + 3, x0 + 18, y + 21], fill=color, outline=(200, 205, 212))
            d.text((x0 + 28, y), ORGAN_NAMES_CN.get(ORGAN_ORDER[c], str(c)),
                   fill=(28, 32, 38), font=_get_font(17))
            d.text((x0 + 28, y + 20), f"{vol:.0f} mL", fill=(130, 138, 150), font=_get_font(13))
            y += 46
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="标签体 .npy（[Z,Y,X] uint8）")
    ap.add_argument("--ct", default="", help="可选 CT .npy，用于半透明体表")
    ap.add_argument("--out", required=True)
    ap.add_argument("--view", default="oblique", choices=list(VIEWS))
    ap.add_argument("--organs", type=int, nargs="*", default=None, help="要渲染的类别 id")
    ap.add_argument("--max-organs", type=int, default=8)
    ap.add_argument("--min-ml", type=float, default=8.0, help="小于该体积(ml)的器官不渲")
    ap.add_argument("--body", action="store_true", help="叠加半透明体表")
    ap.add_argument("--title", default="")
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=1100)
    args = ap.parse_args()

    lbl = np.load(args.label)
    print(f"标签体 {lbl.shape}")

    counts = np.bincount(lbl.ravel(), minlength=16)
    vox_ml = 1.5 ** 3 / 1000.0
    vols = {c: counts[c] * vox_ml for c in range(1, 16)}
    for c, v in sorted(vols.items(), key=lambda t: -t[1]):
        if v > 0.5:
            print(f"  {ORGAN_NAMES_CN.get(ORGAN_ORDER[c], c):<10} {v:8.1f} ml")

    # 选择要渲染的器官
    if args.organs:
        chosen = args.organs
    else:
        cand = [(c, v) for c, v in vols.items() if v >= args.min_ml]
        cand.sort(key=lambda t: -t[1])
        chosen = [c for c, _ in cand[: args.max_organs]]
    print(f"渲染器官: {[ORGAN_NAMES_CN.get(ORGAN_ORDER[c], c) for c in chosen]}")

    palette = _load_palette()
    actors = []
    for c in chosen:
        mask = (lbl == c).astype(np.float32)
        if mask.sum() < 500:
            continue
        poly = extract_surface(mask)
        if poly is None:
            continue
        actors.append(make_actor(poly, tuple(int(v) for v in palette[c])))
        print(f"  {ORGAN_NAMES_CN.get(ORGAN_ORDER[c], c)}: {poly.GetNumberOfPolys()} 三角面")

    body = None
    if args.body and args.ct:
        ct = np.load(args.ct)
        body = body_surface(ct)
        print(f"体表网格: {body.GetNumberOfPolys() if body else 0} 三角面")

    out = Path(args.out)
    legend = [(c, vols[c]) for c in chosen if vols[c] > 0.5]
    render(actors, out, view=args.view, size=(args.width, args.height),
           body_poly=body, title=args.title, legend=legend)
    print(f"✅ 已保存 {out}  ({out.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
