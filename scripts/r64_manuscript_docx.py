# -*- coding: utf-8 -*-
"""r64 -- render docs/manuscript.md into manuscript.docx.

Contract set by the author's brief:
  * body: Times New Roman 11 pt, left aligned, first line indented two characters;
  * bold is reserved for the title, subheadings, the Abstract heading, the Keywords
    label, and the Figure and Table caption labels; everything else is regular;
  * every table is a three line table, number and title above it, note below it;
  * every figure is inserted once, at its first citation, numbered in order;
  * tables and figures numbered 1..N in order of appearance, no S labels;
  * supplementary material is folded into the main text.
"""
import io
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(ROOT, "docs", "manuscript.md")
ASSETS = os.path.join(ROOT, "docs", "figure_assets.json")
OUT = os.path.join(ROOT, "manuscript.docx")
BODY_FONT = "Times New Roman"

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


def read(path):
    with io.open(path, "r", encoding="utf-8") as f:
        return f.read()


# ------------------------------------------------------------------ docx bits
def set_indent_chars(p, chars=200, pts=22.0):
    p.paragraph_format.first_line_indent = Pt(pts)
    pPr = p._p.get_or_add_pPr()
    ind = pPr.find(qn("w:ind"))
    if ind is None:
        ind = OxmlElement("w:ind")
        pPr.append(ind)
    ind.set(qn("w:firstLineChars"), str(chars))


INLINE = re.compile(r"(\*\*[^*\n]+\*\*|`[^`\n]+`|(?<![\w*])\*[^*\n]+\*(?![\w*]))")


def add_runs(p, text, size=None, bold=False, italic=False, font=BODY_FONT,
             inline_bold=False):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            r = p.add_run(part[2:-2])
            if inline_bold:
                r.bold = True
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            r = p.add_run(part[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt((size or 10) - 1.5)
        elif len(part) > 2 and part[0] == "*" and part[-1] == "*":
            r = p.add_run(part[1:-1])
            r.italic = True
        else:
            r = p.add_run(part)
        if bold:
            r.bold = True
        if italic:
            r.italic = True
        if size is not None:
            r.font.size = Pt(size)
        r.font.name = font
        rpr = r._element.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is not None:
            rf.set(qn("w:eastAsia"), font)
    return p


def para(doc, text, indent=True, size=11, bold=False, italic=False,
         spacing=2.0, before=0, after=6, align=None, inline_bold=False):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = spacing
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    if align is not None:
        p.alignment = align
    if indent:
        set_indent_chars(p)
    add_runs(p, text, size=size, bold=bold, italic=italic, inline_bold=inline_bold)
    return p


def three_line(table):
    tbl = table._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge, val, sz in (("top", "single", "8"), ("bottom", "single", "8"),
                          ("left", "none", "0"), ("right", "none", "0"),
                          ("insideH", "none", "0"), ("insideV", "none", "0")):
        el = OxmlElement("w:" + edge)
        el.set(qn("w:val"), val)
        el.set(qn("w:sz"), sz)
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "auto" if val == "none" else "000000")
        borders.append(el)
    old = tblPr.find(qn("w:tblBorders"))
    if old is not None:
        tblPr.remove(old)
    tblPr.append(borders)
    for cell in table.rows[0].cells:
        tcPr = cell._tc.get_or_add_tcPr()
        oldb = tcPr.find(qn("w:tcBorders"))
        if oldb is not None:
            tcPr.remove(oldb)
        b = OxmlElement("w:tcBorders")
        bt = OxmlElement("w:bottom")
        bt.set(qn("w:val"), "single")
        bt.set(qn("w:sz"), "8")
        bt.set(qn("w:space"), "0")
        bt.set(qn("w:color"), "000000")
        b.append(bt)
        tcPr.append(b)


def cell_text(cell, text, size, bold=False, align=None):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(1)
    if align is not None:
        p.alignment = align
    add_runs(p, text, size=size, bold=bold)


def emit_table(doc, caption, rows, notes):
    if caption:
        para(doc, caption, indent=False, size=11, spacing=1.5, before=10, after=4,
             inline_bold=True)
    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]
    size = 8 if ncols <= 8 else 7
    t = doc.add_table(rows=len(rows), cols=ncols)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell_text(t.cell(i, j), val, size, bold=(i == 0),
                      align=WD_ALIGN_PARAGRAPH.CENTER if i == 0 else None)
    three_line(t)
    if notes:
        para(doc, notes, indent=False, size=9, spacing=1.0, before=4, after=10)


def emit_figure(doc, num, caption, imgpath):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    if imgpath and os.path.exists(imgpath):
        p.add_run().add_picture(imgpath, width=Inches(6.2))
    else:
        p.add_run("[image not found: %s]" % imgpath)
    para(doc, caption, indent=False, size=11, spacing=1.5, before=0, after=10,
         inline_bold=True)


# ------------------------------------------------------------------- pipeline
def main():
    lines = read(MD).split("\n")
    assets = {}
    if os.path.exists(ASSETS):
        assets = json.loads(read(ASSETS))

    def image_for(num):
        a = assets.get(str(num))
        if not a:
            return None
        for rel in (a.get("png"), a.get("legacy")):
            if rel:
                p = os.path.join(ROOT, rel.replace("/", os.sep))
                if os.path.exists(p):
                    return p
        return None

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = BODY_FONT
    st.font.size = Pt(11)
    st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
    st.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Inches(1.0)
        s.left_margin = s.right_margin = Inches(1.0)

    i = 0
    pend_table = False
    fig_n, tab_n = 0, 0
    fig_order, tab_order = [], []
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if re.fullmatch(r"-{3,}", s):
            i += 1
            continue
        m = re.match(r"^(#{1,4}) (.*)$", s)
        if m:
            lvl, txt = len(m.group(1)), m.group(2).strip()
            if lvl == 1:
                para(doc, txt, indent=False, size=16, bold=True, spacing=1.5,
                     before=0, after=12, align=WD_ALIGN_PARAGRAPH.CENTER)
            else:
                para(doc, txt, indent=False, size=14 if lvl == 2 else 12,
                     bold=True, spacing=1.5, before=12, after=6)
            i += 1
            continue
        if s.startswith("|"):
            j = i
            rows = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                cells = [c.strip() for c in lines[j].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c or "---") for c in cells):
                    rows.append(cells)
                j += 1
            notes = ""
            k = j
            while k < len(lines) and not lines[k].strip():
                k += 1
            if k < len(lines) and lines[k].strip().startswith("Notes:"):
                notes = lines[k].strip()
                j = k + 1
            emit_table(doc, pending_caption, rows, notes)
            tab_n += 1
            tab_order.append(pending_caption)
            pending_caption = None
            pend_table = False
            i = j
            continue
        m = re.match(r"^\*\*(Figure|Table) (\d+) \|(.*)\*\*(.*)$", s)
        if m and m.group(1) == "Figure":
            num = int(m.group(2))
            caption = "**Figure %d |**%s%s" % (num, m.group(3), m.group(4))
            fig_n += 1
            fig_order.append(num)
            emit_figure(doc, num, caption, image_for(num))
            i += 1
            continue
        if m and m.group(1) == "Table":
            pending_caption = "**Table %d |**%s" % (int(m.group(2)), m.group(3))
            pend_table = True
            i += 1
            continue
        if s.startswith("- "):
            para(doc, s[2:], indent=False, size=11, spacing=2.0, after=2)
            i += 1
            continue
        if re.match(r"^\[\d+\] ", s):
            para(doc, s, indent=False, size=11, spacing=1.0, after=2)
            i += 1
            continue
        j = i
        buf = []
        while j < len(lines):
            t = lines[j].strip()
            if (not t) or t.startswith("|") or re.match(r"^#{1,4} ", t) \
                    or re.fullmatch(r"-{3,}", t) or t.startswith("Notes:") \
                    or re.match(r"^\[\d+\] ", t) \
                    or re.match(r"^\*\*(Figure|Table) \d+ \|", t):
                break
            buf.append(t)
            j += 1
        if j == i:
            raise RuntimeError("parser stuck at line %d: %r" % (i + 1, s[:80]))
        _body = " ".join(buf)
        _lead = _body.split()[0] if _body.split() else ""
        _inline = _lead in ("**Keywords:**", "**Background:**", "**Methods:**",
                            "**Results:**", "**Conclusion:**")
        para(doc, _body, inline_bold=_inline)
        i = j

    doc.save(OUT)
    print("wrote", OUT)
    print("figures:", fig_n, fig_order)
    print("tables:", tab_n)
    print("figure order monotone:", fig_order == sorted(fig_order))
    print("table order monotone:", tab_order == sorted(tab_order, key=lambda c: int(re.match(r"\*\*Table (\d+)", c).group(1))))
    print("table caption sequence:", [int(re.match(r"\*\*Table (\d+)", c).group(1)) for c in tab_order])


main()
