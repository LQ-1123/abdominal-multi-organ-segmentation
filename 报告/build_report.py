"""把 report_*.md 排进淮安大学课程论文模板，生成 报告/实验说明文档.docx。

源文件的简单标记：
    # 一级标题 / ## 二级标题 / ### 三级标题
    空行分隔的普通段落（首行缩进两字）
    !fig 图片路径 | 图题
    !eq 公式文本
    !tabcap 表题        （紧跟一个 | 分隔的表格，第一行为表头）
    !abstract 摘要正文   !keywords 关键词
    !ref [n] 参考文献条目
格式按模板“说明”页：宋体；题目小二加粗；标题行小四加粗；正文小四；2 倍行距；
页边距左 3 cm、右 2 cm、上下 2.5 cm；参考文献五号。

    python 报告/build_report.py
"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEMPLATE = HERE / "_ref" / "template.docx"
OUT = HERE / "实验说明文档.docx"
TITLE = "基于三维U-Net的腹部多器官CT分割系统设计与实现"
MEMBERS = "（组员姓名与学号待填写）"
COURSE = "数字图像处理"

XIAO_ER, SI_HAO, XIAO_SI, WU_HAO = Pt(18), Pt(14), Pt(12), Pt(10.5)


def set_font(run, size=XIAO_SI, bold=False, italic=False, east="宋体", latin="Times New Roman"):
    run.font.size = size
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = latin
    run.font.color.rgb = RGBColor(0, 0, 0)
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    for key, val in (("w:ascii", latin), ("w:hAnsi", latin), ("w:eastAsia", east), ("w:cs", latin)):
        fonts.set(qn(key), val)


def para_format(p, spacing=2.0, indent=True, align=None, before=0, after=0, keep_next=False):
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = spacing
    pf.first_line_indent = Pt(24) if indent else Pt(0)
    pf.left_indent = Pt(0)
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.keep_with_next = keep_next
    pf.widow_control = True
    if align is not None:
        p.alignment = align


INLINE = re.compile(r"(\*\*[^*]+\*\*|\^[^^]+\^|~[^~]+~)")


def add_runs(p, text, size=XIAO_SI, bold=False):
    """**粗体**、^上标^、~下标~ 三种行内标记。"""
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**"):
            set_font(p.add_run(part[2:-2]), size, bold=True)
        elif part.startswith("^"):
            r = p.add_run(part[1:-1])
            set_font(r, size, bold)
            r.font.superscript = True
        elif part.startswith("~"):
            r = p.add_run(part[1:-1])
            set_font(r, size, bold)
            r.font.subscript = True
        else:
            set_font(p.add_run(part), size, bold)


def set_cell_border(cell, **edges):
    tcpr = cell._element.get_or_add_tcPr()
    borders = tcpr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcpr.append(borders)
    for edge, sz in edges.items():
        el = OxmlElement(f"w:{edge}")
        if sz:
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), str(sz))
            el.set(qn("w:color"), "000000")
        else:
            el.set(qn("w:val"), "nil")
        borders.append(el)


class Builder:
    def __init__(self, doc: Document, anchor):
        self.doc = doc
        self.anchor = anchor          # 新内容依次插在 anchor 之前
        self.fig_no = 0

    def _new_par(self):
        p = self.doc.add_paragraph()
        self.anchor.addprevious(p._p)
        return p

    def heading(self, text, level):
        p = self._new_par()
        para_format(p, indent=False, keep_next=True, before=6 if level == 1 else 0)
        add_runs(p, text, XIAO_SI, bold=True)

    def body(self, text):
        p = self._new_par()
        para_format(p)
        add_runs(p, text)

    def equation(self, text):
        p = self._new_par()
        para_format(p, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER)
        for part in INLINE.split(text):
            if not part:
                continue
            r = p.add_run(part.strip("^~") if part[:1] in "^~" else part)
            set_font(r, XIAO_SI, east="宋体", latin="Cambria Math")
            r.font.superscript = part.startswith("^")
            r.font.subscript = part.startswith("~")

    def figure(self, path, caption):
        p = self._new_par()
        para_format(p, spacing=1.0, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, before=6, keep_next=True)
        if (ROOT / path).exists():
            p.add_run().add_picture(str(ROOT / path), width=Cm(15.5))
        else:
            print("警告：缺少图片", path, file=sys.stderr)
            set_font(p.add_run(f"【缺图：{path}】"), XIAO_SI)
        c = self._new_par()
        para_format(c, spacing=1.5, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, after=6)
        add_runs(c, caption, WU_HAO)

    def table(self, caption, rows):
        c = self._new_par()
        para_format(c, spacing=1.5, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, before=6, keep_next=True)
        add_runs(c, caption, WU_HAO)
        t = self.doc.add_table(rows=len(rows), cols=len(rows[0]))
        self.anchor.addprevious(t._tbl)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        ncols = len(rows[0])
        weights = [
            max(11, max(sum(2 if ord(ch) > 127 else 1 for ch in re.sub(r"\*\*|\^|~", "", rows[i][j])) for i in range(len(rows))))
            for j in range(ncols)
        ]
        col_widths = [Cm(15.8 * w / sum(weights)) for w in weights]
        for j, cw in enumerate(col_widths):
            t.columns[j].width = cw
        keep_whole = len(rows) <= 10
        for i, row in enumerate(rows):
            tr_pr = t.rows[i]._tr.get_or_add_trPr()
            tr_pr.append(OxmlElement("w:cantSplit"))
            if i == 0:
                tr_pr.append(OxmlElement("w:tblHeader"))
            keep_row = (i < len(rows) - 1) if keep_whole else (i == 0)
            for j, val in enumerate(row):
                cell = t.cell(i, j)
                cell.width = col_widths[j]
                p = cell.paragraphs[0]
                para_format(p, spacing=1.15, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, keep_next=keep_row)
                add_runs(p, val, WU_HAO, bold=(i == 0))
                edges = {"left": 0, "right": 0, "insideH": 0, "insideV": 0,
                         "top": 12 if i == 0 else 0, "bottom": 12 if i == len(rows) - 1 else (6 if i == 0 else 0)}
                set_cell_border(cell, **edges)
        after = self._new_par()
        para_format(after, spacing=1.0, indent=False)

    def label_par(self, label, text, size=XIAO_SI):
        p = self._new_par()
        para_format(p, indent=False)
        set_font(p.add_run(label), size, bold=True)
        add_runs(p, text, size)

    def ref(self, text):
        p = self._new_par()
        para_format(p, spacing=1.5, indent=False)
        p.paragraph_format.left_indent = Pt(21)
        p.paragraph_format.first_line_indent = Pt(-21)
        add_runs(p, text, WU_HAO)


def parse(builder: Builder, text: str) -> None:
    lines = text.splitlines()
    i, buf = 0, []

    def flush():
        if buf:
            builder.body("".join(s.strip() for s in buf))
            buf.clear()

    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            flush()
        elif line.startswith("#"):
            flush()
            level = len(line) - len(line.lstrip("#"))
            builder.heading(line[level:].strip(), level)
        elif line.startswith("!fig "):
            flush()
            path, cap = (s.strip() for s in line[5:].split("|", 1))
            builder.figure(path, cap)
        elif line.startswith("!eq "):
            flush()
            builder.equation(line[4:].strip())
        elif line.startswith("!tabcap "):
            flush()
            cap, rows = line[8:].strip(), []
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            builder.table(cap, rows)
            continue
        elif line.startswith("!abstract "):
            flush()
            builder.label_par("摘要：", line[10:].strip())
        elif line.startswith("!keywords "):
            flush()
            builder.label_par("关键词：", line[10:].strip())
        elif line.startswith("!refhead"):
            flush()
            builder.heading("参考文献", 1)
        elif line.startswith("!ref "):
            flush()
            builder.ref(line[5:].strip())
        else:
            buf.append(line)
        i += 1
    flush()


def fill_cover(doc: Document) -> None:
    paras = doc.paragraphs
    # 封面和“说明”页末尾的空段落会被挤到下一页，形成空白页；保留带分节符的段落
    for i in (8, 16, 17):
        p = paras[i]
        if not p.text.strip() and "w:sectPr" not in p._p.xml:
            p._p.getparent().remove(p._p)
    paras = doc.paragraphs
    for p in paras[:10]:
        if p.text.replace(" ", "").startswith("题目"):
            r = p.add_run(TITLE)
            set_font(r, Pt(15), bold=True)
        elif p.text.startswith("小组成员"):
            r = p.add_run(MEMBERS)
            set_font(r, Pt(14), bold=False)
    table = doc.tables[0]
    for row in table.rows:
        cells = row.cells
        for k, cell in enumerate(cells):
            if cell.text.strip() == "课程名称" and k + 1 < len(cells):
                tgt = cells[k + 1].paragraphs[0]
                if not tgt.text.strip():
                    for r in list(tgt.runs):
                        r._element.getparent().remove(r._element)
                    ppr = tgt._p.get_or_add_pPr()
                    ind = ppr.find(qn("w:ind"))
                    if ind is not None:
                        ppr.remove(ind)
                    tgt.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    tgt.paragraph_format.first_line_indent = Pt(0)
                    tgt.paragraph_format.left_indent = Pt(0)
                    tgt.paragraph_format.right_indent = Pt(0)
                    set_font(tgt.add_run(COURSE), Pt(12))
    # 删除模板中的蓝色说明文字
    for p in doc.paragraphs:
        for r in p.runs:
            if r.font.color is not None and r.font.color.type is not None and str(r.font.color.rgb) == "0000FF":
                r._element.getparent().remove(r._element)
    for cell in table._cells:
        for p in cell.paragraphs:
            for r in p.runs:
                if r.font.color is not None and r.font.color.type is not None and str(r.font.color.rgb) == "0000FF":
                    r._element.getparent().remove(r._element)


def main() -> None:
    doc = Document(str(TEMPLATE))
    for s in doc.sections:
        s.left_margin, s.right_margin = Cm(3), Cm(2)
        s.top_margin = s.bottom_margin = Cm(2.5)
    paras = doc.paragraphs
    # 模板第 3 页的示例段落（“课程论文”、题目、姓名、摘要……参考文献示例）全部替换
    start = next(i for i, p in enumerate(paras) if p.text.strip() == "课程论文")
    end = next(i for i, p in enumerate(paras) if "正式上交课程论文时" in p.text)
    if end + 1 >= len(paras):
        raise SystemExit("模板结构与预期不符")
    anchor = paras[end + 1]._p
    victims = paras[start:end + 1]
    fill_cover(doc)
    builder = Builder(doc, anchor)

    p = builder._new_par()
    para_format(p, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, after=6)
    set_font(p.add_run(TITLE), XIAO_ER, bold=True)
    p = builder._new_par()
    para_format(p, indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, after=6)
    set_font(p.add_run(MEMBERS), SI_HAO, bold=True)

    sources = sorted(HERE.glob("report_part*.md"))
    text = "\n\n".join(s.read_text(encoding="utf-8") for s in sources)
    leftover = re.findall(r"\{\{[A-Z0-9_]+\}\}", text)
    if leftover:
        print("警告：还有未填的占位符", sorted(set(leftover)), file=sys.stderr)
    parse(builder, text)

    for p in victims:
        p._p.getparent().remove(p._p)
    # 去掉正文后面模板残留的空段落，避免多出空白页
    body = doc.element.body
    for p in list(doc.paragraphs)[::-1]:
        if p.text.strip() or p._p.xpath(".//w:drawing") or p._p.xpath(".//w:sectPr"):
            break
        body.remove(p._p)
    doc.save(str(OUT))
    print("已生成", OUT)


if __name__ == "__main__":
    main()
