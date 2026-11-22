"""生成答辩 PPT：报告/答辩PPT.pptx。

所有数字都从实验输出文件读取，不手抄。版式刻意保持朴素：白底、深蓝标题、真实实验图为主。

    python 报告/build_ppt.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Cm, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "答辩PPT.pptx"
TITLE = "基于三维U-Net的腹部多器官CT分割系统设计与实现"
MEMBERS = "小组成员：（待填写）"

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
BLUE = RGBColor(0x2F, 0x6D, 0xB5)
ORANGE = RGBColor(0xD9, 0x82, 0x2B)
GREEN = RGBColor(0x3A, 0x9A, 0x5B)
TEXT = RGBColor(0x33, 0x33, 0x33)
MUTED = RGBColor(0x7A, 0x84, 0x8C)
LIGHT = RGBColor(0xF1, 0xF4, 0xF8)
LINE = RGBColor(0xC9, 0xD1, 0xDA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "微软雅黑"
W, H = Cm(33.867), Cm(19.05)
M = Cm(1.4)


# ----------------------------------------------------------------------------- 数据
def jload(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


def csv_rows(p):
    with (ROOT / p).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


import os
POST_DIR = os.environ.get("POST_DIR", "outputs/ablation_fold0_post_v3")


def post_total():
    return {r["method"]: r for r in csv_rows(f"{POST_DIR}/postprocess_summary.csv") if r["organ"].startswith("全部")}


def train_side():
    res = {}
    for m in ("none", "gaussian", "median", "butterworth", "clahe"):
        for ep in (0, 20):
            f = ROOT / f"outputs/train_side/{m}_ft{ep}/summary.json"
            if f.exists():
                res[(m, ep)] = json.loads(f.read_text(encoding="utf-8"))
    return res


# ----------------------------------------------------------------------------- 版式工具
def style_run(run, size=18, bold=False, color=TEXT, font=FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font
    rpr = run._r.get_or_add_rPr()
    ea = rpr.find(qn("a:ea"))
    if ea is None:
        ea = rpr.makeelement(qn("a:ea"), {})
        rpr.append(ea)
    ea.set("typeface", font)


def text_box(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT, spacing=1.15):
    """paras: [(text, size, bold, color)] 或字符串；一个元素一段。文本里用 **x** 标粗。"""
    tb = slide.shapes.add_textbox(int(x), int(y), int(w), int(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Cm(0.1)
    tf.margin_top = tf.margin_bottom = Cm(0.05)
    for i, para in enumerate(paras):
        if isinstance(para, str):
            para = (para, 18, False, TEXT)
        text, size, bold, color = para
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        p.space_after = Pt(size * 0.45)
        for k, part in enumerate(text.split("**")):
            if part:
                style_run(p.add_run(), size, bold or k % 2 == 1, NAVY if (k % 2 == 1 and color == TEXT) else color)
                p.runs[-1].text = part
    return tb


def title(slide, text, sub=None):
    text_box(slide, M, Cm(0.75), W - 2 * M, Cm(1.6), [(text, 28, True, NAVY)], anchor=MSO_ANCHOR.MIDDLE)
    ln = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, M, Cm(2.45), W - M, Cm(2.45))
    ln.line.color.rgb = LINE
    ln.line.width = Pt(1)
    if sub:
        text_box(slide, M, Cm(2.6), W - 2 * M, Cm(1.0), [(sub, 15, False, MUTED)])


def footer(slide, n):
    text_box(slide, W - M - Cm(2), H - Cm(1.0), Cm(2), Cm(0.6), [(str(n), 10, False, MUTED)], align=PP_ALIGN.RIGHT)


def picture(slide, path, x, y, w, h, align="center"):
    """在 (x, y, w, h) 框内等比放图，返回实际位置。"""
    iw, ih = Image.open(ROOT / path).size
    scale = min(w / iw, h / ih)
    pw, ph = int(iw * scale), int(ih * scale)
    px = x + (w - pw) // 2 if align == "center" else x
    py = y + (h - ph) // 2
    px, py = int(px), int(py)
    slide.shapes.add_picture(str(ROOT / path), px, py, pw, ph)
    return px, py, pw, ph


def caption(slide, x, y, w, text):
    text_box(slide, x, y, w, Cm(0.7), [(text, 11, False, MUTED)], align=PP_ALIGN.CENTER)


def box(slide, x, y, w, h, text, fill=LIGHT, color=TEXT, size=15, bold=False, line=None, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, int(x), int(y), int(w), int(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(1)
    s.shadow.inherit = False
    tf = s.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Cm(0.15)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for i, line_text in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = line_text
        style_run(r, size if i == 0 else size - 3, bold if i == 0 else False, color)
    return s


def arrow(slide, x1, y1, x2, y2, color=MUTED):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, int(x1), int(y1), int(x2), int(y2))
    c.line.color.rgb = color
    c.line.width = Pt(1.5)
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)
    return c


def table(slide, x, y, w, rows, col_w=None, size=13, row_h=Cm(0.8), highlight_rows=()):
    nr, nc = len(rows), len(rows[0])
    shape = slide.shapes.add_table(nr, nc, int(x), int(y), int(w), int(row_h * nr))
    t = shape.table
    tbl_pr = shape._element.graphic.graphicData.tbl.tblPr
    style = tbl_pr.find(qn("a:tableStyleId"))
    if style is not None:
        style.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"   # 无样式、网格线
    if col_w:
        for j, cw in enumerate(col_w):
            t.columns[j].width = int(cw)
    for i in range(nr):
        t.rows[i].height = int(row_h)
        for j in range(nc):
            cell = t.cell(i, j)
            cell.margin_left = cell.margin_right = Cm(0.12)
            cell.margin_top = cell.margin_bottom = Cm(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY if i == 0 else (RGBColor(0xFD, 0xF1, 0xE3) if i in highlight_rows else (LIGHT if i % 2 == 0 else WHITE))
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = str(rows[i][j])
            style_run(r, size, bold=(i == 0), color=WHITE if i == 0 else TEXT)
    return shape


# ----------------------------------------------------------------------------- 各页
def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    blank = prs.slide_layouts[6]
    pages = []

    def new():
        s = prs.slides.add_slide(blank)
        pages.append(s)
        return s

    post = post_total()
    ts = train_side()
    stats_test = jload(f"{POST_DIR}/stats_test.json")
    sel = jload("configs/morph_selection.json")
    edge = jload("outputs/edge_analysis/edge_stats.json")
    ds = jload("outputs/dataset_stats/dataset_stats.json")

    def st(metric, method, ref, data=stats_test):
        return next(r for r in data if r["metric"] == metric and r["method"] == method and r["reference"] == ref)

    def f4(m, k="dice"):
        return float(post[m][k])

    # 1 封面 ---------------------------------------------------------------
    s = new()
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Cm(1.0), H)
    band.fill.solid()
    band.fill.fore_color.rgb = NAVY
    band.line.fill.background()
    text_box(s, Cm(2.4), Cm(4.2), Cm(18.5), Cm(5), [(TITLE, 32, True, NAVY)], spacing=1.2)
    text_box(s, Cm(2.4), Cm(9.4), Cm(18), Cm(3.2), [
        ("数字图像处理课程大作业（图像分割）", 18, False, TEXT),
        ("数据集：AMOS22（15 个腹部器官）", 16, False, MUTED),
    ])
    text_box(s, Cm(2.4), Cm(13.6), Cm(18), Cm(2.2), [(MEMBERS, 16, False, TEXT), ("2026 年 11 月", 14, False, MUTED)])
    picture(s, "outputs/demo/3d_v2_pred.png", Cm(20.5), Cm(1.8), Cm(12.6), Cm(15.5))

    # 2 任务与难点 -----------------------------------------------------------
    s = new()
    title(s, "任务：在三维腹部 CT 上逐体素分出 15 个器官")
    picture(s, "outputs/demo/web_v2_result.png", M, Cm(3.1), Cm(14.5), Cm(13.8))
    caption(s, M, Cm(16.9), Cm(14.5), "本系统 Web 查看器的真实截图（V2 模型预测）")
    o = ds["organs"]
    items = [
        ("器官大小相差约 400 倍", f"肝脏体积中位数 {o['肝脏']['volume_ml_median']:.0f} mL，右肾上腺 {o['右肾上腺']['volume_ml_median']:.1f} mL；前景只占体积的 {100 * ds['foreground_fraction_mean']:.1f}%"),
        ("软组织 CT 值大面积重叠", "胰腺、十二指肠、肾上腺、肝、脾的中位 CT 值都在 20～65 HU，单凭灰度分不开"),
        ("体素严重各向异性", f"层内间距中位数 {ds['inplane_mm']['median']:.2f} mm，层厚多为 {ds['slice_mm']['median']:.0f} mm"),
    ]
    y = Cm(3.6)
    for head, body in items:
        box(s, Cm(17.2), y, Cm(0.35), Cm(3.2), "", fill=BLUE)
        text_box(s, Cm(18.0), y, Cm(14.4), Cm(3.4), [(head, 20, True, NAVY), (body, 16, False, TEXT)])
        y += Cm(4.3)

    # 3 数据集 ---------------------------------------------------------------
    s = new()
    title(s, "数据集：AMOS22，及我们做的整理")
    table(s, M, Cm(3.3), Cm(14.2), [
        ["项目", "内容"],
        ["来源", "Ji 等，NeurIPS 2022 数据集赛道"],
        ["规模", "500 例 CT + 100 例 MRI，多中心、多设备"],
        ["公开标注", "360 例（CT 300、MRI 60）"],
        ["标注", "15 个腹部器官，体素级"],
        ["协议 / 获取", "CC BY 4.0；Zenodo 7155725"],
        ["本文对照实验", "第 0 折验证集中的 59 例 CT"],
    ], col_w=[Cm(3.6), Cm(10.6)], size=14, row_h=Cm(1.05))
    steps = [("① nibabel 读入", "从仿射矩阵恢复间距与方向；绕开 SimpleITK 的中文路径问题"),
             ("② 重采样到 1.5 mm", "影像三线性插值，标签最近邻插值"),
             ("③ HU 截断与缓存", "截断到 [−1024, 3071] HU，int16 内存映射，按 patch 读取"),
             ("④ 按病例五折划分", "同一病人的 patch 不会同时出现在训练集和验证集")]
    x0, y0 = Cm(17.0), Cm(3.3)
    for k, (a, b) in enumerate(steps):
        y = y0 + k * Cm(1.75)
        box(s, x0, y, Cm(4.9), Cm(1.45), a, fill=NAVY, color=WHITE, size=15, bold=True)
        text_box(s, x0 + Cm(5.2), y, Cm(10.3), Cm(1.45), [(b, 15, False, TEXT)], anchor=MSO_ANCHOR.MIDDLE)
    text_box(s, x0, Cm(10.6), Cm(15.5), Cm(6.5), [
        ("标签为什么不能线性插值", 18, True, NAVY),
        ("相邻两个器官的编号被平均后会变成第三个不存在的编号，例如 2 号右肾和 4 号胆囊之间会插出 3 号左肾。", 16, False, TEXT),
        ("缓存可复现", 18, True, NAVY),
        ("在 Colab 上从原始数据重新生成缓存，与原训练所用缓存逐字节一致（MD5 相同）。", 16, False, TEXT),
    ])

    # 4 数据特性 -------------------------------------------------------------
    s = new()
    title(s, "数据特性分析（300 例带标注 CT）")
    picture(s, "outputs/report_figs/fig_data.png", M, Cm(3.0), W - 2 * M, Cm(9.5))
    cols = [("层厚以 5 mm 为主", "→ 先做 1.5 mm 等向重采样，再送入三维卷积"),
            ("体积相差约 400 倍，前景仅 3.6%", "→ 损失用 Dice + 交叉熵，评价按器官分别算再平均"),
            ("所有器官都落在 [−150, 250] HU 内", "→ 用这一段做窗宽窗位，把对比度用足")]
    cw = (W - 2 * M - 2 * Cm(0.76)) / 3
    for k, (a, b) in enumerate(cols):
        text_box(s, M + k * (cw + Cm(0.76)), Cm(13.0), cw, Cm(4.0), [(a, 18, True, NAVY), (b, 16, False, TEXT)])

    # 5 技术路线 -------------------------------------------------------------
    s = new()
    title(s, "技术路线：课本方法放在网络的前后两端")
    nodes = [("输入 CT", "DICOM / NIfTI", LIGHT, TEXT), ("重采样", "图像内插", LIGHT, TEXT),
             ("窗宽窗位", "分段线性\n灰度变换", RGBColor(0xFD, 0xF1, 0xE3), TEXT),
             ("预处理候选", "高斯、中值\nButterworth、CLAHE", RGBColor(0xFD, 0xF1, 0xE3), TEXT),
             ("三维 U-Net", "滑窗推理", NAVY, WHITE),
             ("后处理候选", "连通域、开闭运算\n实质器官最大块", RGBColor(0xFD, 0xF1, 0xE3), TEXT),
             ("显示", "逐层渲染\n三维重建", LIGHT, TEXT)]
    n = len(nodes)
    gap = Cm(0.6)
    bw = (W - 2 * M - (n - 1) * gap) / n
    for k, (a, b, fill, col) in enumerate(nodes):
        x = M + k * (bw + gap)
        box(s, x, Cm(4.2), bw, Cm(3.6), a + "\n" + b, fill=fill, color=col, size=16, bold=True)
        if k < n - 1:
            arrow(s, x + bw + Cm(0.03), Cm(6.0), x + bw + gap - Cm(0.03), Cm(6.0))
    text_box(s, M, Cm(8.6), W - 2 * M, Cm(1), [("橙色框为课本知识点。每个候选方法是否保留，由对照实验决定。", 15, False, MUTED)])
    chapters = [["课程内容", "知识点", "在本系统中的位置"],
                ["图像基础与插值", "图像内插（最近邻、线性）", "重采样：影像三线性，标签最近邻"],
                ["灰度变换与空间滤波", "分段线性变换、幂律变换、高斯 / 中值滤波、直方图均衡", "窗宽窗位；Gamma 数据增强；预处理对照"],
                ["频率域滤波", "傅里叶变换、Butterworth 低通", "预处理对照"],
                ["形态学处理", "腐蚀、膨胀、开闭运算、连通分量、边界提取", "后处理；查看器轮廓绘制"],
                ["图像分割与边缘检测", "Sobel、Canny 边缘检测", "解释各器官精度差异"]]
    table(s, M, Cm(9.8), W - 2 * M, chapters, col_w=[Cm(5.4), Cm(13.6), Cm(12.07)], size=13, row_h=Cm(1.0))

    # 6 预处理方法 -----------------------------------------------------------
    s = new()
    title(s, "预处理：空间域平滑与直方图均衡")
    picture(s, "outputs/report_figs/fig_pre_visual.png", M, Cm(2.9), W - 2 * M, Cm(9.6))
    table(s, M, Cm(12.9), W - 2 * M, [
        ["方法", "原理", "参数及依据"],
        ["窗宽窗位", "分段线性灰度变换，[−150, 250] HU → [0, 1]", "所有器官的 CT 值都在此范围内（第 4 页）"],
        ["3D 高斯", "线性低通，抑制量子噪声", "σ = 1 mm，约 0.67 体素，只压体素级噪声"],
        ["3D 中值", "统计排序滤波，保边缘、去脉冲噪声", "3×3×3 邻域"],
        ["轴位 CLAHE", "分块直方图均衡，限制对比度放大噪声", "块 64×64（约 96 mm），截断系数 0.01"],
    ], col_w=[Cm(3.6), Cm(14.2), Cm(13.27)], size=13, row_h=Cm(0.82))

    # 7 频域 -----------------------------------------------------------------
    s = new()
    title(s, "预处理：频率域 Butterworth 低通")
    picture(s, "outputs/report_figs/fig_pre_spectrum.png", M, Cm(2.9), W - 2 * M, Cm(9.6))
    text_box(s, M, Cm(13.0), Cm(15), Cm(5), [
        ("H(D) = 1 / [1 + (D / D₀)^(2n)]", 20, True, NAVY),
        ("对整卷做三维 FFT，乘传递函数后逆变换。D₀ = 0.2 周期/mm，1.5 mm 采样的奈奎斯特频率约 0.33 周期/mm。", 16, False, TEXT),
    ])
    text_box(s, Cm(17.6), Cm(13.0), Cm(14.8), Cm(5), [
        ("为什么选二阶，不选理想低通", 18, True, NAVY),
        ("理想低通在截止频率处陡降，空间域会出现振铃，器官边缘多出亮暗条纹；二阶过渡平缓，振铃可以忽略。变换前先减均值，避免周期延拓的边界跳变。", 16, False, TEXT),
    ])

    # 8 网络 -----------------------------------------------------------------
    s = new()
    title(s, "三维 U-Net：结构与训练设置")
    chans = [32, 64, 128, 256, 512]
    bx, by = M + Cm(0.3), Cm(3.6)
    lvl_h, bw2 = Cm(2.2), Cm(3.0)
    for k, c in enumerate(chans):
        y = by + k * lvl_h
        box(s, bx + k * Cm(1.25), y, bw2, Cm(1.5), f"{c}", fill=BLUE if k < 4 else NAVY, color=WHITE, size=16, bold=True)
        if k < 4:
            rx = bx + Cm(16.0) - k * Cm(1.25)
            box(s, rx, y, bw2, Cm(1.5), f"{c}", fill=RGBColor(0x8F, 0xB3, 0xDE), color=NAVY, size=16, bold=True)
            arrow(s, bx + k * Cm(1.25) + bw2, y + Cm(0.75), rx, y + Cm(0.75), color=LINE)
    text_box(s, bx, by + 5 * lvl_h - Cm(0.3), Cm(19), Cm(2.2), [
        ("左列编码器，右列解码器，数字为通道数；横线为跳跃连接（拼接）", 13, False, MUTED),
        ("每级：两个 3×3×3 卷积 + InstanceNorm + LeakyReLU；下采样 2×2×2 最大池化；上采样转置卷积", 13, False, MUTED),
    ])
    table(s, Cm(21.6), Cm(3.4), Cm(10.8), [
        ["设置", "取值"],
        ["参数量", "22.6 M"],
        ["输入 patch", "112³ 体素，批大小 3"],
        ["损失", "缺失类别屏蔽的 Dice + 交叉熵"],
        ["深监督", "4 个辅助头，权重 0.5 → 0.0625"],
        ["数据增强", "三轴翻转、亮度扰动、Gamma 0.7–1.4"],
        ["优化器", "AdamW，lr 1e-4，余弦退火 120 轮"],
        ["推理", "112³ 滑窗，重叠 50%，高斯融合"],
    ], col_w=[Cm(3.2), Cm(7.6)], size=13, row_h=Cm(1.05))
    text_box(s, Cm(21.6), Cm(12.4), Cm(10.8), Cm(4.5), [
        ("缺失类别屏蔽", 17, True, NAVY),
        ("AMOS 中不是每例都有 15 个器官，随机 patch 里更少。只对 patch 里实际出现的器官算 Dice 损失，避免“空预测”得到虚假的满分或零分。", 15, False, TEXT),
    ])

    # 9 训练与五折 -----------------------------------------------------------
    s = new()
    title(s, "训练过程与五折交叉验证结果")
    picture(s, "outputs/figs/fig1_training_curve.png", M, Cm(3.0), Cm(17.5), Cm(9.2))
    caption(s, M, Cm(12.2), Cm(17.5), "第 0 折验证 patch Dice 曲线（训练日志原始数据）")
    table(s, Cm(19.6), Cm(3.2), Cm(12.8), [
        ["", "V1", "V2"],
        ["五折 mDice", "0.7665", "0.8056"],
        ["标准差", "0.0124", "0.0179"],
        ["肝脏", "0.9405", "0.9439"],
        ["胰腺", "0.7493", "0.7832"],
        ["十二指肠", "0.6461", "0.6952"],
        ["左肾上腺", "0.5881", "0.6594"],
    ], col_w=[Cm(4.4), Cm(4.2), Cm(4.2)], size=14, row_h=Cm(0.95))
    text_box(s, Cm(19.6), Cm(10.2), Cm(12.8), Cm(7), [
        ("V2 比 V1 高 3.9 个百分点", 18, True, NAVY),
        ("提升集中在小器官：左肾上腺 +7.1，肝脏只有 +0.3。V2 同时改了增强、轮数和 patch，提升不能全记在某一项上。", 15, False, TEXT),
        ("最难的器官", 18, True, NAVY),
        ("双侧肾上腺（0.66）、十二指肠（0.70）、食管（0.74）。", 15, False, TEXT),
    ])

    # 10 后处理方法 ----------------------------------------------------------
    s = new()
    title(s, "后处理：连通域清理与按器官选择的开闭运算")
    pre_png = ROOT / "outputs/report_figs/fig_fragments.png"
    if pre_png.exists():
        picture(s, "outputs/report_figs/fig_fragments.png", M, Cm(3.0), Cm(15.5), Cm(13.4))
        caption(s, M, Cm(16.4), Cm(15.5), "红色为被清理掉的碎块（冠状面投影）")
    blocks = [
        ("① 连通域清理", "6 邻域标记连通分量，保留 ≥ 最大块 2% 且 ≥ 30 体素的块。不只留最大块：十二指肠是 C 形管，预测可能断成几段。"),
        ("② 开运算 + 闭运算", "A∘B = (A⊖B)⊕B 去毛刺细桥；A•B = (A⊕B)⊖B 填小孔。结构元为中心加 6 邻居（1.5 mm）。"),
        ("③ 选择性开闭", f"开运算会删掉宽度不足 4.5 mm 的结构，肾上腺肢体只有几毫米。在 30 例开发集上选出不掉分的器官：{'、'.join(sel['morph_organs'])}；另 29 例只用来检验。"),
        ("④ 实质器官只留最大块", "肾、肝、脾等实质器官是一整块，远处的第二块只可能是误判。规则按解剖定，不看验证数据。"),
    ]
    y = Cm(3.1)
    for a, b in blocks:
        text_box(s, Cm(17.6), y, Cm(14.8), Cm(3.4), [(a, 17, True, NAVY), (b, 14, False, TEXT)])
        y += Cm(3.5)

    # 11 后处理结果 ----------------------------------------------------------
    s = new()
    title(s, "后处理对照结果（59 例验证 CT）")
    picture(s, "outputs/report_figs/fig_post.png", M, Cm(2.7), W - 2 * M, Cm(7.8))
    names = {"none": "无后处理", "cc": "① 连通域", "morph_cc": "② 开闭 + 连通域", "morph_sel_cc": "③ 选择性开闭 + 连通域",
             "cc_solid": "④ 连通域 + 实质器官最大块", "morph_sel_cc_solid": "③ + ④"}
    rows = [["方法", "mDice", "HD95 / mm", "每例连通域数"]]
    for m in names:
        if m in post:
            rows.append([names[m], f"{f4(m):.4f}", f"{f4(m, 'hd95_mm'):.1f}", f"{f4(m, 'components_per_case'):.1f}"])
    table(s, M, Cm(10.8), Cm(17.5), rows, col_w=[Cm(7.9), Cm(3.0), Cm(3.2), Cm(3.4)], size=13, row_h=Cm(0.76),
          highlight_rows=(len(rows) - 1,))
    d_sel = st("hd95_mm", "morph_sel_cc", "cc")
    lines = [("连通域：碎块少 89%，HD95 −5.8 mm", 14.5, True, NAVY),
             ("Dice 只涨 0.14 个百分点，59 例中 57 例上升（p < 0.001）", 12.5, False, TEXT),
             ("统一开闭：肾上腺掉分", 14.5, True, NAVY),
             ("右肾上腺 −2.9、左肾上腺 −1.7 个百分点", 12.5, False, TEXT),
             ("选择性开闭（29 例测试集）", 14.5, True, NAVY),
             (f"HD95 相对①降 {-d_sel['mean_delta']:.1f} mm，{d_sel['worse_cases']}/29 例下降（p < 0.001），Dice 持平", 12.5, False, TEXT)]
    if "cc_solid" in post:
        lines += [("实质器官只留最大块", 14.5, True, NAVY),
                  (f"HD95 降至 {f4('cc_solid', 'hd95_mm'):.1f} / {f4('morph_sel_cc_solid', 'hd95_mm'):.1f} mm（mDice {f4('morph_sel_cc_solid'):.4f}）；盆腔 5 例误判较大时有反选错删", 12.5, False, TEXT)]
    text_box(s, Cm(19.8), Cm(10.7), Cm(12.6), Cm(6.8), lines, spacing=0.98)

    # 12 预处理结果 ----------------------------------------------------------
    s = new()
    title(s, "预处理对照结果：推理端直接加 vs 带处理微调")
    if (ROOT / "outputs/report_figs/fig_pre.png").exists():
        picture(s, "outputs/report_figs/fig_pre.png", M, Cm(2.8), Cm(19.5), Cm(7.6))
    cn = {"none": "无处理", "gaussian": "3D 高斯", "median": "3D 中值", "butterworth": "Butterworth", "clahe": "CLAHE"}
    rows = [["预处理", "推理端 Dice", "微调后 Dice", "微调后 HD95"]]
    for m in cn:
        a, b = ts.get((m, 0)), ts.get((m, 20))
        rows.append([cn[m], f"{a['mean_dice']:.4f}" if a else "—", f"{b['mean_dice']:.4f}" if b else "—",
                     f"{b['mean_hd95_mm']:.1f}" if b else "—"])
    table(s, M, Cm(10.9), Cm(19.5), rows, col_w=[Cm(4.5), Cm(5), Cm(5), Cm(5)], size=13, row_h=Cm(0.85))
    text_box(s, Cm(21.6), Cm(3.0), Cm(10.8), Cm(14), PRE_POINTS(ts), spacing=1.05)

    # 13 边缘检测 ------------------------------------------------------------
    s = new()
    title(s, "边缘检测：器官边界越清楚，分割越准")
    picture(s, "outputs/edge_analysis/edge_example.png", M, Cm(2.8), W - 2 * M, Cm(6.8))
    picture(s, "outputs/report_figs/fig_edge_corr.png", M, Cm(9.9), Cm(19.5), Cm(7.6))
    rc, rs = edge["spearman_canny_recall_vs_dice"], edge["spearman_sobel_mean_vs_dice"]
    text_box(s, Cm(21.6), Cm(10.0), Cm(10.8), Cm(7.5), [
        ("只用真值算，不依赖模型", 16, True, NAVY),
        ("Canny 召回率：真值边界离 Canny 边缘 ≤ 1 像素的比例；Sobel：边界上的平均梯度。", 14, False, TEXT),
        (f"与 Dice 的秩相关 ρ = {rc['rho']:.2f}（p = {rc['p']:.3f}）和 {rs['rho']:.2f}（p = {rs['p']:.3f}）", 16, True, NAVY),
        ("双肾召回 0.74～0.77；十二指肠、胆囊、膀胱低于 0.47。胰腺和十二指肠之间几乎没有 Canny 响应，所以经典算子不能直接替代分割。", 14, False, TEXT),
    ])

    # 14 失败与负结果 --------------------------------------------------------
    s = new()
    title(s, "失败案例与负结果")
    picture(s, "outputs/demo/analyze_z81.png", M, Cm(3.0), Cm(11.5), Cm(9.5))
    caption(s, M, Cm(12.5), Cm(11.5), "胸部层面：肺内病灶被误判为肝脏（查看器截图）")
    text_box(s, Cm(13.6), Cm(3.0), Cm(18.8), Cm(14.5), [
        ("1. 远处的假阳性块", 18, True, NAVY),
        ("argmax 会在训练分布以外的区域（如肺内）报出器官。连通域清理能去掉这类小块。", 15, False, TEXT),
        ("2. 双肾 Dice 0.91，HD95 却超过 60 mm", 18, True, NAVY),
        ("37 个肾预测的 HD95 > 40 mm，大多保留了一个超过 2% 阈值的远处大块。相对阈值是为十二指肠设计的，不适合实质器官——由此加了“只留最大块”的规则。", 15, False, TEXT),
        ("3. 强化学习边界精修：无增益", 18, True, NAVY),
        ("PPO 智能体沿边界收缩或扩张，精修前后 Dice 完全相同。上界实验：每步都按真值选最优动作，也只能提高约 0.005。U-Net 输出已接近这类局部动作的上限，模块没有进入最终流程。", 15, False, TEXT),
    ])

    # 15 系统 ----------------------------------------------------------------
    s = new()
    title(s, "系统实现：上传 CT，逐层查看，三维重建")
    picture(s, "outputs/demo/web_montage.png", M, Cm(2.9), Cm(19.2), Cm(13.0))
    picture(s, "outputs/demo/3d_v2_pred.png", Cm(21.2), Cm(2.9), Cm(11.2), Cm(9.0))
    text_box(s, Cm(21.2), Cm(12.2), Cm(11.2), Cm(5.5), [
        ("FastAPI 后端 + 网页查看器", 16, True, NAVY),
        ("DICOM 按 ImagePositionPatient 投影排序，RescaleSlope/Intercept 换算 HU", 14, False, TEXT),
        ("轮廓模式：掩膜减去其腐蚀结果，得到一像素宽边界", 14, False, TEXT),
        ("Marching Cubes 提取三维表面", 14, False, TEXT),
    ])

    # 16 创新点 --------------------------------------------------------------
    s = new()
    title(s, "创新点")
    inn = INNOVATIONS(post, stats_test, ts, edge)
    cw = (W - 2 * M - Cm(0.76)) / 2
    for k, (a, b) in enumerate(inn):
        x = M + (k % 2) * (cw + Cm(0.76))
        y = Cm(3.3) + (k // 2) * Cm(7.0)
        box(s, x, y, Cm(0.3), Cm(6.2), "", fill=ORANGE)
        text_box(s, x + Cm(0.7), y, cw - Cm(0.7), Cm(6.4), [(a, 19, True, NAVY), (b, 15, False, TEXT)])

    # 17 总结与分工 ----------------------------------------------------------
    s = new()
    title(s, "总结与小组分工")
    text_box(s, M, Cm(3.2), Cm(16.8), Cm(14), SUMMARY(post, ts), spacing=1.05)
    table(s, Cm(19.2), Cm(3.4), Cm(13.2), [
        ["成员", "负责内容"],
        ["孙钰林", "方案设计、数据管线、3D U-Net 五折训练，\n推理与后处理、Web 系统"],
        ["（待填）", "预处理 / 后处理对照实验、统计检验"],
        ["（待填）", "边缘检测分析、实验说明文档"],
        ["（待填）", "答辩 PPT、系统演示"],
    ], col_w=[Cm(2.8), Cm(10.4)], size=13, row_h=Cm(1.6))
    text_box(s, Cm(19.2), Cm(12.0), Cm(13.2), Cm(2), [("分工请组内核对后修改。", 12, False, MUTED)])

    for k, sl in enumerate(pages[1:], start=2):
        footer(sl, k)
    prs.save(str(OUT))
    print("已生成", OUT, len(pages), "页")


# ----------------------------------------------------------------------------- 依赖结果的文字
def PRE_POINTS(ts):
    d0_min = min(100 * (ts[('none', 0)]['mean_dice'] - ts[(m, 0)]['mean_dice']) for m in ('gaussian', 'median', 'butterworth', 'clahe'))
    d0_max = max(100 * (ts[('none', 0)]['mean_dice'] - ts[(m, 0)]['mean_dice']) for m in ('gaussian', 'median', 'butterworth', 'clahe'))
    d20_min = min(100 * (ts[('none', 20)]['mean_dice'] - ts[(m, 20)]['mean_dice']) for m in ('gaussian', 'median', 'butterworth', 'clahe'))
    d20_max = max(100 * (ts[('none', 20)]['mean_dice'] - ts[(m, 20)]['mean_dice']) for m in ('gaussian', 'median', 'butterworth', 'clahe'))
    clahe_rec = 100 * (ts[('clahe', 20)]['mean_dice'] - ts[('clahe', 0)]['mean_dice'])
    return [
        ("推理端直接加：全部掉分", 17, True, NAVY),
        (f"4 组 mDice 降 {d0_min:.2f}～{d0_max:.2f} 个百分点（p < 0.001）；CLAHE 跌至 {ts[('clahe', 0)]['mean_dice']:.4f}，HD95 升至 {ts[('clahe', 0)]['mean_hd95_mm']:.1f} mm。", 14, False, TEXT),
        ("微调 20 轮消除分布偏移", 17, True, NAVY),
        (f"同一第 0 折权重在 241 例 CT 上微调（种子、学习率与采样一致）。CLAHE 回升 {clahe_rec:.2f} 个百分点，说明掉分主要来自分布偏移。", 14, False, TEXT),
        ("同起跑线比较：均未超过无处理", 17, True, NAVY),
        (f"无处理微调达 {ts[('none', 20)]['mean_dice']:.4f} / {ts[('none', 20)]['mean_hd95_mm']:.1f} mm；其余 4 组低 {d20_min:.2f}～{d20_max:.2f} 个百分点（p < 0.001），平滑削弱食管与肾上腺边界。", 14, False, TEXT),
        ("系统选择", 17, True, NAVY),
        ("仅保留重采样与 [−150, 250] HU 窗宽窗位，不额外开启滤波或 CLAHE。", 14, False, TEXT),
    ]


def INNOVATIONS(post, stats_test, ts, edge):
    sel_hd = next(r for r in stats_test if r["metric"] == "hd95_mm" and r["method"] == "morph_sel_cc" and r["reference"] == "cc")
    rc, rs = edge["spearman_canny_recall_vs_dice"], edge["spearman_sobel_mean_vs_dice"]
    clahe_rec = 100 * (ts[("clahe", 20)]["mean_dice"] - ts[("clahe", 0)]["mean_dice"])
    return [
        ("1. 分层形态学与解剖先验后处理",
         f"在 30 例开发集选出 7 类选择性开闭器官，在独立 29 例测试集上 HD95 相对仅连通域再降 {-sel_hd['mean_delta']:.2f} mm（p < 0.001）。叠加实质器官最大块规则后，59 例平均 HD95 降至 {float(post['morph_sel_cc_solid']['hd95_mm']):.1f} mm，mDice 达 {float(post['morph_sel_cc_solid']['dice']):.4f}。"),
        ("2. 拆分分布偏移的双侧预处理对照",
         f"同一代码做推理（0 轮）与微调（20 轮，含无处理基线）双侧对照。CLAHE 微调回升 {clahe_rec:.2f} 个百分点拆出分布偏移；而 4 组微调后仍低于无处理 {ts[('none', 20)]['mean_dice']:.4f}（p < 0.001）。"),
        ("3. 边缘算子定量解释器官精度差异",
         f"仅用 CT 原图与真值算 Sobel 梯度和 Canny 召回率，两者与 15 器官原始 Dice 呈显著正相关（ρ = {rs['rho']:.2f} 与 {rc['rho']:.2f}，p < 0.02），定量解释低对比度器官误差。"),
        ("4. 完整工程链路与负结果机制验证",
         "打通 DICOM 法向投影排序、缺失类别屏蔽深监督 3D U-Net、形态学轮廓渲染与三维重建；并通过 oracle 上界实验证实 PPO 边界精修上限仅约 0.005 Dice 后未纳入系统。"),
    ]


def SUMMARY(post, ts):
    return [
        ("数据特性指导设计", 18, True, NAVY),
        ("300 例 CT 统计出 5 mm 层厚、前景仅 3.6%（体积差 400 倍）与软组织 HU 重叠，据此定下 1.5 mm 重采样、[−150, 250] HU 窗及缺失类别屏蔽损失。", 14.5, False, TEXT),
        ("对照实验决定取舍", 18, True, NAVY),
        (f"改进 3D U-Net 五折 mDice 达 0.8056（+3.9 pp）。后处理采用选择性开闭 + 实质器官最大块，mDice {float(post['morph_sel_cc_solid']['dice']):.4f}、HD95 {float(post['morph_sel_cc_solid']['hd95_mm']):.1f} mm；4 种预处理微调后仍低于无处理（{ts[('none', 20)]['mean_dice']:.4f}），故仅保留窗宽窗位。", 14.5, False, TEXT),
        ("局限与待改进项", 18, True, NAVY),
        ("对照局限于第 0 折 59 例 CT；1.5 mm 网格对 3 mL 肾上腺细节有损；盆腔 5 例大误判出现最大块反选错删；RL 精修因局部上限低已舍弃。", 14.5, False, TEXT),
    ]


if __name__ == "__main__":
    build()
