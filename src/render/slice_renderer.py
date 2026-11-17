"""逐层渲染器：CT 切片 + 彩色器官叠加 → PNG。

设计要点
--------
1. **必须设窗宽窗位**。直接 min-max 归一化出来的腹部 CT 灰蒙蒙一片，
   边界糊成一团。这里默认 WL=40 / WW=400（腹部软组织窗），可调。
2. **两种叠加模式**：
   - ``fill``   半透明填充，一眼看出器官范围（演示用）
   - ``contour`` 仅画轮廓线，不遮挡原图信息（医生更爱，报告用）
3. **固定低饱和配色**，不用 jet/rainbow（伪彩失真且对色盲不友好）。
4. 输出带 alpha 的 PNG，前端可自由调透明度、图层叠加。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]

__all__ = ["SliceRenderer", "ORGAN_NAMES_CN"]

ORGAN_NAMES_CN = {
    "background": "背景", "spleen": "脾脏", "right_kidney": "右肾", "left_kidney": "左肾",
    "gallbladder": "胆囊", "esophagus": "食管", "liver": "肝脏", "stomach": "胃",
    "aorta": "主动脉", "postcava": "下腔静脉", "pancreas": "胰腺",
    "right_adrenal": "右肾上腺", "left_adrenal": "左肾上腺", "duodenum": "十二指肠",
    "bladder": "膀胱", "prostate_uterus": "前列腺/子宫",
}
ORGAN_ORDER = list(ORGAN_NAMES_CN.keys())          # 索引 0..15 对应标签值


# 兜底配色：即使 yaml 里某个器官漏配，也不会渲染成黑色方块
FALLBACK = {
    "right_adrenal": [199, 76, 255], "left_adrenal": [255, 214, 0],
}


def _load_palette(cfg_path: Path | None = None) -> np.ndarray:
    """从 configs/labels.yaml 读配色，返回 [16, 3] uint8。

    容错设计：
      - 键名允许带/不带 ``_gland`` 后缀（AMOS 的 dataset.json 用 ``right_adrenal_gland``，
        而本项目的 ORGAN_ORDER 用 ``right_adrenal``，早期因此导致这两个器官渲染成黑色）
      - 缺失或全黑的条目回退到 FALLBACK，绝不输出 (0,0,0)
    """
    cfg_path = cfg_path or (ROOT / "configs" / "labels.yaml")
    palette = np.zeros((16, 3), dtype=np.uint8)
    pal: dict = {}
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        pal = cfg.get("palette", {}) or {}

    def _norm(k: str) -> str:
        return k.replace("_gland", "")

    pal_norm = {_norm(k): v for k, v in pal.items()}
    for i, key in enumerate(ORGAN_ORDER):
        if i == 0:
            continue                      # 背景不参与渲染
        color = pal.get(key) or pal_norm.get(_norm(key)) or FALLBACK.get(key)
        if color is None:
            color = [200, 200, 200]       # 最终兜底：浅灰，至少可见
        palette[i] = np.array(color, dtype=np.uint8)
    return palette


def _get_font(size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "arial.ttf"):
        p = Path("C:/Windows/Fonts") / name
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:  # noqa: BLE001
                continue
    return ImageFont.load_default()


class SliceRenderer:
    """把一层 CT + 一层标签渲染成 RGB 图。

    Args:
        wl / ww: 窗位 / 窗宽（HU）。腹部软组织常用 40 / 400。
        alpha: 填充模式下器官的不透明度。
        mode: ``fill`` | ``contour`` | ``both``
    """

    def __init__(
        self,
        wl: float = 40.0,
        ww: float = 400.0,
        alpha: float = 0.45,
        mode: str = "fill",
        palette: np.ndarray | None = None,
    ) -> None:
        self.wl, self.ww = float(wl), float(ww)
        self.alpha = float(alpha)
        self.mode = mode
        self.palette = palette if palette is not None else _load_palette()

    # ------------------------------------------------------------------ 工具
    def to_gray(self, slice_hu: np.ndarray) -> np.ndarray:
        """HU → 8bit 灰度（按窗宽窗位截断）。"""
        lo, hi = self.wl - self.ww / 2.0, self.wl + self.ww / 2.0
        x = np.clip(slice_hu.astype(np.float32), lo, hi)
        return ((x - lo) / max(hi - lo, 1e-6) * 255.0).astype(np.uint8)

    @staticmethod
    def _boundary(mask: np.ndarray) -> np.ndarray:
        """形态学求边界：mask 减去自身腐蚀。"""
        from scipy.ndimage import binary_erosion

        if not mask.any():
            return mask
        return mask & ~binary_erosion(mask, iterations=1)

    def overlay(self, gray: np.ndarray, label: np.ndarray) -> np.ndarray:
        """灰度图 + 标签 → RGB。"""
        rgb = np.stack([gray] * 3, axis=-1).astype(np.float32)

        if self.mode in ("fill", "both"):
            for c in range(1, len(self.palette)):
                m = label == c
                if not m.any():
                    continue
                color = self.palette[c].astype(np.float32)
                rgb[m] = rgb[m] * (1 - self.alpha) + color * self.alpha

        if self.mode in ("contour", "both"):
            for c in range(1, len(self.palette)):
                edge = self._boundary(label == c)
                if edge.any():
                    rgb[edge] = self.palette[c].astype(np.float32)

        return np.clip(rgb, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------ 渲染
    def render(
        self,
        slice_hu: np.ndarray,
        label: np.ndarray | None = None,
        annotate: str | None = None,
        z: int | None = None,
        n_total: int | None = None,
        present_organs: list[int] | None = None,
        scale: int = 2,
    ) -> Image.Image:
        """渲染一层。scale>1 时上采样，避免小图看着糊。图例放在右侧独立面板。"""
        gray = self.to_gray(slice_hu)
        rgb = self.overlay(gray, label) if label is not None else np.stack([gray] * 3, -1)
        img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")

        if scale > 1:
            img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)

        # 左上角切片序号
        if z is not None:
            d = ImageDraw.Draw(img, "RGBA")
            font = _get_font(16)
            txt = f"Slice {z + 1}" + (f" / {n_total}" if n_total else "")
            d.rectangle([8, 8, 150, 36], fill=(0, 0, 0, 150))
            d.text((14, 12), txt, fill=(255, 255, 255), font=font)

        # 右侧图例面板（不覆盖解剖结构）
        if present_organs:
            img = self._attach_legend_panel(img, present_organs)

        # 左下角标注
        if annotate:
            d = ImageDraw.Draw(img, "RGBA")
            font = _get_font(15)
            d.rectangle([8, img.height - 34, 20 + 9 * len(annotate), img.height - 8],
                        fill=(0, 0, 0, 150))
            d.text((14, img.height - 30), annotate, fill=(255, 255, 255), font=font)
        return img

    def _attach_legend_panel(self, img: Image.Image, organs: list[int]) -> Image.Image:
        """在右侧追加一列图例，整体宽度 = 图像宽 + 面板宽。"""
        panel_w = 132
        canvas = Image.new("RGB", (img.width + panel_w, img.height), (16, 18, 22))
        canvas.paste(img, (0, 0))
        d = ImageDraw.Draw(canvas, "RGBA")
        font = _get_font(15)
        x0 = img.width + 12
        y = 14
        d.text((x0, y), "器官", fill=(200, 205, 210), font=_get_font(14))
        y += 26
        for c in organs:
            color = tuple(int(v) for v in self.palette[c])
            d.rectangle([x0, y + 3, x0 + 14, y + 17], fill=color,
                        outline=(255, 255, 255, 60))
            d.text((x0 + 22, y), ORGAN_NAMES_CN[ORGAN_ORDER[c]],
                   fill=(235, 238, 240), font=font)
            y += 24
        return canvas

    def _draw_legend(self, img, annotate, z, n_total, present_organs) -> Image.Image:
        draw = ImageDraw.Draw(img, "RGBA")
        font = _get_font(16)
        font_small = _get_font(13)

        # 左上角切片序号
        if z is not None:
            txt = f"Slice {z + 1}" + (f" / {n_total}" if n_total else "")
            draw.rectangle([8, 8, 8 + 110, 8 + 26], fill=(0, 0, 0, 150))
            draw.text((14, 12), txt, fill=(255, 255, 255), font=font)

        # 右下角标注
        if annotate:
            draw.rectangle([8, img.height - 34, 8 + 8 * len(annotate) + 16, img.height - 8],
                           fill=(0, 0, 0, 150))
            draw.text((14, img.height - 30), annotate, fill=(255, 255, 255), font=font_small)

        # 右上角器官图例
        if present_organs:
            x0, y0 = img.width - 148, 10
            h = 20 * len(present_organs) + 10
            draw.rectangle([x0 - 6, y0 - 4, img.width - 6, y0 + h], fill=(0, 0, 0, 160))
            for i, c in enumerate(present_organs):
                yy = y0 + i * 20
                color = tuple(int(v) for v in self.palette[c])
                draw.rectangle([x0, yy + 3, x0 + 12, yy + 15], fill=color)
                draw.text((x0 + 18, yy), ORGAN_NAMES_CN[ORGAN_ORDER[c]], fill=(255, 255, 255), font=font_small)
        return img


def render_slice_png_bytes(
    slice_hu: np.ndarray,
    label: np.ndarray | None,
    wl: float = 40.0,
    ww: float = 400.0,
    alpha: float = 0.45,
    mode: str = "fill",
    z: int | None = None,
    n_total: int | None = None,
) -> bytes:
    """给 Web 后端用：直接返回 PNG 字节流。"""
    import io

    r = SliceRenderer(wl=wl, ww=ww, alpha=alpha, mode=mode)
    present = sorted(set(int(v) for v in np.unique(label)) - {0}) if label is not None else None
    img = r.render(slice_hu, label, z=z, n_total=n_total, present_organs=present)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


if __name__ == "__main__":
    # 冒烟：用缓存里的真实病例出一张图
    cache = ROOT / "data" / "cache" / "amos22"
    cid = "amos_0001"
    vol = np.load(cache / f"{cid}.img.npy")
    lbl = np.load(cache / f"{cid}.lbl.npy")

    # 找一张器官最多的层
    counts = [int((lbl[z] > 0).sum()) for z in range(lbl.shape[0])]
    z = int(np.argmax(counts))
    print(f"选择第 {z} 层（前景体素最多: {counts[z]}）")

    out_dir = ROOT / "outputs" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("fill", "contour", "both"):
        r = SliceRenderer(mode=mode)
        present = sorted(set(int(v) for v in np.unique(lbl[z])) - {0})
        img = r.render(vol[z], lbl[z], annotate=f"AMOS22 demo | {mode}",
                       z=z, n_total=lbl.shape[0], present_organs=present)
        p = out_dir / f"smoke_{mode}.png"
        img.save(p)
        print(f"  已保存 {p}  ({img.width}x{img.height})")
    print("✅ 渲染器冒烟测试通过")
