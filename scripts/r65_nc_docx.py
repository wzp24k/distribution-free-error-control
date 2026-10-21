# -*- coding: utf-8 -*-
"""r65 -- render the Nature Communications build.

  docs/manuscript_nc.md      -> 1015manuscript_NC.docx    (6 figures, 4 tables)
  docs/supplementary_nc.md   -> 1015supplementary_NC.docx (11 figures, 20 tables)

Style contract (author brief + Nature Communications article type):
  * body Times New Roman 11 pt, left aligned, first line indented two characters;
  * bold is reserved for the title, section headings, the Abstract heading, the
    Keywords label, and the Figure and Table caption labels plus panel letters;
  * every table is a three line table, its label and title above it, its note below;
  * every figure is inserted once, at its first citation, numbered in order, its
    legend below it, styled as the Nature Communications articles in the reference
    issue do: the label, the title and the panel letters are bold, the panel
    letters are uppercase;
  * main text and supplementary information use separate numbering series.
"""
import io
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "docs", "figure_assets.json")
BODY_FONT = "Times New Roman"

# Nature Communications figure number -> asset key in docs/figure_assets.json
NC_ASSETS = {1: "1", 2: "2", 3: "3", 4: "5", 5: "10", 6: "17"}

JOBS = [
    (os.path.join(ROOT, "docs", "manuscript_nc.md"),
     os.path.join(ROOT, "1015manuscript_NC.docx"), NC_ASSETS),
    (os.path.join(ROOT, "docs", "supplementary_nc.md"),
     os.path.join(ROOT, "1015supplementary_NC.docx"), None),
]

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


def read(path):
    with io.open(path, "r", encoding="utf-8") as f:
        return f.read()


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
         spacing=2.0, before=0, after=6, align=None, inline_bold=False,
         hanging=False):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = spacing
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    if align is not None:
        p.alignment = align
    if hanging:
        p.paragraph_format.left_indent = Pt(24)
        p.paragraph_format.first_line_indent = Pt(-24)
    elif indent:
        set_indent_chars(p)
    add_runs(p, text, size=size, bold=bold, italic=italic, inline_bold=inline_bold)
    return p


PANEL = re.compile(r"(?:^|(?<=[\.\?] ))([A-Z]) (?=[A-Z])")


def caption_para(doc, label, rest):
    """Legend: bold label, title and panel letters, uppercase, regular body."""
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(label)
    r.bold = True
    r.font.size = Pt(11)
    r.font.name = BODY_FONT
    pos = 0
    for m in PANEL.finditer(rest):
        if m.start() > pos:
            add_runs(p, rest[pos:m.start()], size=11)
        rr = p.add_run(m.group(1))
        rr.bold = True
        rr.font.size = Pt(11)
        rr.font.name = BODY_FONT
        pos = m.end() - 1
    if pos < len(rest):
        add_runs(p, rest[pos:], size=11)
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


def emit_image(doc, imgpath):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    if imgpath and os.path.exists(imgpath):
        p.add_run().add_picture(imgpath, width=Inches(6.2))
    else:
        p.add_run("[image not found: %s]" % imgpath)
    return p


FIGCAP = re.compile(r"^\*\*((?:Supplementary )?Fig\.) (\d+) \| (.*?)\*\*\s*(.*)$")
TABCAP = re.compile(r"^\*\*(Table|Supplementary Table) (\d+) \|\s*(.*)$")
IMG = re.compile(r"^%%IMG:\s*(\S+)\s*$")


def render(md_path, out_path, nc_assets):
    lines = read(md_path).split("\n")
    assets = {}
    if os.path.exists(ASSETS):
        assets = json.loads(read(ASSETS))

    def image_for(nc_num):
        key = nc_assets.get(nc_num)
        if not key:
            return None
        a = assets.get(str(key))
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
    pending_table = None
    pending_fig = None
    fig_order, tab_order = [], []
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        if re.fullmatch(r"-{3,}", s):
            i += 1
            continue
        m = IMG.match(s)
        if m:
            rel = m.group(1).replace("/", os.sep)
            emit_image(doc, os.path.join(ROOT, rel))
            if pending_fig:
                caption_para(doc, pending_fig[0], pending_fig[1])
                pending_fig = None
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
        m = FIGCAP.match(s)
        if m:
            label = "%s %s | %s " % (m.group(1), m.group(2), m.group(3))
            rest = m.group(4)
            n = int(m.group(2))
            fig_order.append((m.group(1), n))
            if nc_assets is not None:
                emit_image(doc, image_for(n))
                caption_para(doc, label, rest)
            else:
                pending_fig = (label, rest)
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
            emit_table(doc, pending_table, rows, notes)
            tab_order.append(pending_table)
            pending_table = None
            i = j
            continue
        m = TABCAP.match(s)
        if m:
            title = m.group(3).strip().lstrip("*").strip().rstrip("*").strip()
            pending_table = "**%s %s | %s**" % (m.group(1), m.group(2), title)
            i += 1
            continue
        if s.startswith("- "):
            para(doc, s[2:], indent=False, size=11, spacing=2.0, after=2)
            i += 1
            continue
        if re.match(r"^\[\d+\] ", s):
            para(doc, s, indent=False, size=11, spacing=1.0, after=2, hanging=True)
            i += 1
            continue
        j = i
        buf = []
        while j < len(lines):
            t = lines[j].strip()
            if (not t) or t.startswith("|") or re.match(r"^#{1,4} ", t) \
                    or re.fullmatch(r"-{3,}", t) or t.startswith("Notes:") \
                    or re.match(r"^\[\d+\] ", t) or IMG.match(t) \
                    or FIGCAP.match(t) or TABCAP.match(t):
                break
            buf.append(t)
            j += 1
        if j == i:
            raise RuntimeError("parser stuck at line %d: %r" % (i + 1, s[:80]))
        body = " ".join(buf)
        lead = body.split()[0] if body.split() else ""
        inline = lead in ("**Keywords:**",)
        para(doc, body, inline_bold=inline)
        i = j

    doc.save(out_path)
    figs = [n for (_, n) in fig_order]
    tabs = [int(re.match(r"\*\*(?:Supplementary )?Table (\d+)", c).group(1))
            for c in tab_order if c]
    print("wrote %s" % out_path)
    print("    figures: %d %s" % (len(figs), figs))
    print("    tables : %d %s" % (len(tabs), tabs))
    print("    figure order monotone:", figs == sorted(figs))
    print("    table  order monotone:", tabs == sorted(tabs))
    return len(figs), len(tabs), figs, tabs


def main():
    for md, out, amap in JOBS:
        render(md, out, amap)


main()
