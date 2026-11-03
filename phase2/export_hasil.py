#!/usr/bin/env python
"""Ekspor keluaran mentah Unlimited OCR 2 ke Excel, Word, PDF yang bisa dicari, dan JSON blok.

Dipanggil otomatis oleh ocr.py, atau manual:
    python phase2/export_hasil.py outputs/paper            # folder hasil ocr.py
    python phase2/export_hasil.py outputs/paper --dpi 200

Masukan: <folder>/hasil_raw.txt dan gambar halaman (page_*.png di folder, atau daftar
"gambar_halaman" di ringkasan.json).
Keluaran di folder yang sama:
    hasil.xlsx            sheet "Blok" (semua blok: halaman, label, bbox, isi) + satu sheet per tabel
    hasil.docx            heading, paragraf, tabel, potongan gambar
    hasil_searchable.pdf  gambar halaman asli + lapisan teks tak terlihat sesuai posisi bbox
    hasil_blok.json       daftar blok terstruktur per halaman
"""
import os, sys, re, json, html
from html.parser import HTMLParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from process_unlimitedOCR2 import _parse_blocks, _base_type, _fix_span_attrs, STRAY_TOKEN_RE

TEXT_LABELS = {"text", "title", "header", "footer", "page_number", "image_caption", "image_footnote",
               "page_footnote", "equation", "ref_text", "aside_text", "algorithm", "other"}
IMAGE_LABELS = {"image", "figure", "picture", "chart", "img", "seal"}


# ---------- parsing ----------
class _TableParser(HTMLParser):
    """HTML <table> -> list baris (list sel), colspan/rowspan diisi sel kosong."""
    def __init__(self):
        super().__init__()
        self.rows, self._row, self._cell, self._in_cell = [], None, None, False
        self._pending_rowspan = {}  # col -> (sisa_baris, teks kosong)
        self._span = (1, 1)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._in_cell = True
            self._cell = []
            self._span = (int(a.get("colspan", "1") or 1), int(a.get("rowspan", "1") or 1))
        elif tag == "br" and self._in_cell:
            self._cell.append("\n")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._in_cell:
            while len(self._row) in self._pending_rowspan:  # sel yang tertutup rowspan baris sebelumnya
                self._row.append("")
                col = len(self._row) - 1
                n = self._pending_rowspan[col] - 1
                if n > 0: self._pending_rowspan[col] = n
                else: del self._pending_rowspan[col]
            text = re.sub(r"[ \t]+", " ", "".join(self._cell)).strip()
            cs, rs = self._span
            start = len(self._row)
            self._row.append(text)
            for k in range(1, cs):
                self._row.append("")
            if rs > 1:
                for c in range(start, start + cs):
                    self._pending_rowspan[c] = rs - 1
            self._in_cell = False
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._in_cell:
            self._cell.append(data)


def html_tables(content):
    """Semua <table> dalam sebuah string -> list of rows."""
    out = []
    for m in re.finditer(r"<table\b.*?</table>", content, re.I | re.S):
        p = _TableParser()
        p.feed(_fix_span_attrs(m.group(0)))
        if p.rows:
            w = max(len(r) for r in p.rows)
            out.append([r + [""] * (w - len(r)) for r in p.rows])
    return out


def clean_text(s):
    s = STRAY_TOKEN_RE.sub("", s or "")
    s = re.sub(r"<[^>]{1,40}>", " ", s)
    return html.unescape(re.sub(r"[ \t]+", " ", s)).strip()


def parse_raw(raw):
    """raw -> list blok {halaman, label, jenis, level, bbox, isi, tabel}"""
    blocks = []
    for b in _parse_blocks(raw or "", page_base=1):
        label = b["label"]
        base = _base_type(label) if label != "__raw__" else "text"
        tables = html_tables(b["content"]) if base == "table" else []
        blocks.append({"halaman": b["page"], "label": label, "jenis": base, "level": b["level"],
                       "bbox": list(b["bbox"]) if b["bbox"] else None,
                       "isi": clean_text(b["content"]) if base != "table" else
                       "\n".join(" | ".join(r) for t in tables for r in t),
                       "tabel": tables})
    return blocks


def bbox_px(bbox, w, h):
    x1, y1, x2, y2 = bbox
    return (int(x1 / 999 * w), int(y1 / 999 * h), int(x2 / 999 * w), int(y2 / 999 * h))


# ---------- ekspor ----------
def export_json(blocks, path):
    data = [{k: v for k, v in b.items() if k != "tabel"} | ({"tabel": b["tabel"]} if b["tabel"] else {}) for b in blocks]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def export_xlsx(blocks, path):
    from openpyxl import Workbook
    from openpyxl.utils import get_column_letter
    wb = Workbook()
    ws = wb.active
    ws.title = "Blok"
    ws.append(["halaman", "label", "level", "x1", "y1", "x2", "y2", "isi"])
    for b in blocks:
        bb = b["bbox"] or ["", "", "", ""]
        ws.append([b["halaman"], b["label"], b["level"] or "", *bb, b["isi"][:32000]])
    ws.column_dimensions["H"].width = 100
    n = 0
    for b in blocks:
        for t in b["tabel"]:
            n += 1
            name = f"Tabel{n}_hal{b['halaman']}"[:31]
            s = wb.create_sheet(name)
            for r in t:
                s.append([c[:32000] for c in r])
            for i in range(1, (max(len(r) for r in t) if t else 0) + 1):
                s.column_dimensions[get_column_letter(i)].width = 24
    wb.save(path)
    return path, n


def export_docx(blocks, page_images, path):
    from docx import Document
    from docx.shared import Inches
    doc = Document()
    cur_page = None
    for b in blocks:
        if b["halaman"] != cur_page:
            if cur_page is not None:
                doc.add_page_break()
            cur_page = b["halaman"]
        base = b["jenis"]
        if base == "title":
            doc.add_heading(b["isi"] or " ", level=min(max(b["level"] or 1, 1), 4))
        elif base == "table" and b["tabel"]:
            for t in b["tabel"]:
                cols = max(len(r) for r in t)
                tb = doc.add_table(rows=len(t), cols=cols)
                tb.style = "Table Grid"
                for i, r in enumerate(t):
                    for j, c in enumerate(r):
                        tb.cell(i, j).text = c
                doc.add_paragraph()
        elif base in IMAGE_LABELS:
            img = _crop(page_images, b)
            if img:
                doc.add_picture(img, width=Inches(5))
            if b["isi"]:
                doc.add_paragraph(b["isi"])
        elif base == "page_number":
            continue
        elif b["isi"]:
            p = doc.add_paragraph(b["isi"])
            if base in ("header", "footer", "page_footnote", "image_caption", "image_footnote"):
                for run in p.runs:
                    run.italic = True
    doc.save(path)
    return path


def _crop(page_images, b):
    """Potong area bbox dari gambar halaman -> path PNG sementara (None bila tidak ada gambar)."""
    if not page_images or not b["bbox"] or b["halaman"] > len(page_images):
        return None
    try:
        from PIL import Image
        src = page_images[b["halaman"] - 1]
        im = Image.open(src)
        x1, y1, x2, y2 = bbox_px(b["bbox"], *im.size)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        out = os.path.join(os.path.dirname(os.path.abspath(src)), f"_crop_p{b['halaman']}_{x1}_{y1}_{x2}_{y2}.png")
        im.crop((x1, y1, x2, y2)).save(out)
        return out
    except Exception:
        return None


def export_searchable_pdf(blocks, page_images, path, dpi=200):
    import pymupdf
    doc = pymupdf.open()
    by_page = {}
    for b in blocks:
        by_page.setdefault(b["halaman"], []).append(b)
    for i, img_path in enumerate(page_images, 1):
        pix = pymupdf.Pixmap(img_path)
        w_pt, h_pt = pix.width * 72 / dpi, pix.height * 72 / dpi
        page = doc.new_page(width=w_pt, height=h_pt)
        page.insert_image(page.rect, pixmap=pix)
        for b in by_page.get(i, []):
            if not b["bbox"] or b["jenis"] in IMAGE_LABELS:
                continue
            text = b["isi"]
            if not text:
                continue
            x1, y1, x2, y2 = b["bbox"]
            rect = pymupdf.Rect(x1 / 999 * w_pt, y1 / 999 * h_pt, x2 / 999 * w_pt, y2 / 999 * h_pt)
            if rect.is_empty or rect.width < 2 or rect.height < 2:
                continue
            fs = max(4.0, min(rect.height * 0.85, 14.0))
            # render_mode=3 -> teks tak terlihat tetapi bisa dicari/di-copy
            for _ in range(8):
                rc = page.insert_textbox(rect, text, fontsize=fs, fontname="helv", render_mode=3)
                if rc >= 0 or fs <= 4.0:
                    break
                fs = max(4.0, fs * 0.8)
    doc.save(path, garbage=3, deflate=True)
    return path


def export_all(folder, raw=None, page_images=None, dpi=200, verbose=True):
    """Jalankan semua ekspor untuk sebuah folder hasil ocr.py. Mengembalikan dict path hasil."""
    folder = os.path.abspath(folder)
    if raw is None:
        with open(os.path.join(folder, "hasil_raw.txt"), encoding="utf-8") as f:
            raw = f.read()
    if page_images is None:
        page_images = sorted(os.path.join(folder, f) for f in os.listdir(folder) if re.match(r"page_\d+\.png$", f))
        if not page_images:
            try:
                page_images = json.load(open(os.path.join(folder, "ringkasan.json"), encoding="utf-8")).get("gambar_halaman") or []
            except Exception:
                page_images = []
    blocks = parse_raw(raw)
    out = {}
    out["json"] = export_json(blocks, os.path.join(folder, "hasil_blok.json"))
    try:
        p, n = export_xlsx(blocks, os.path.join(folder, "hasil.xlsx"))
        out["xlsx"] = p
        out["jumlah_tabel"] = n
    except Exception as e:
        out["xlsx_error"] = repr(e)
    try:
        out["docx"] = export_docx(blocks, page_images, os.path.join(folder, "hasil.docx"))
    except Exception as e:
        out["docx_error"] = repr(e)
    if page_images:
        try:
            out["pdf"] = export_searchable_pdf(blocks, page_images, os.path.join(folder, "hasil_searchable.pdf"), dpi=dpi)
        except Exception as e:
            out["pdf_error"] = repr(e)
    else:
        out["pdf"] = None
    for f in os.listdir(folder):  # bersihkan potongan gambar sementara
        if f.startswith("_crop_"):
            try: os.remove(os.path.join(folder, f))
            except OSError: pass
    if verbose:
        print("[export] blok:", len(blocks), "| tabel:", out.get("jumlah_tabel", 0))
        for k, v in out.items():
            if v and k in ("json", "xlsx", "docx", "pdf"):
                print(f"[export] {k:4s} -> {v}")
            elif k.endswith("_error"):
                print(f"[export] {k}: {v}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="folder hasil ocr.py (berisi hasil_raw.txt)")
    ap.add_argument("--dpi", type=int, default=200, help="dpi gambar halaman (untuk ukuran halaman PDF)")
    a = ap.parse_args()
    export_all(a.folder, dpi=a.dpi)
