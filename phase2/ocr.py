#!/usr/bin/env python
"""OCR satu file (gambar atau PDF) di CPU dengan Unlimited OCR 2.

Contoh:
    python phase2/ocr.py dokumen.pdf                 # semua halaman, 3 halaman per request (base)
    python phase2/ocr.py dokumen.pdf --pages 1-5     # halaman 1 sampai 5
    python phase2/ocr.py foto.jpg                    # satu gambar, mode gundam (detail)
    python phase2/ocr.py dokumen.pdf --per-page      # tiap halaman mode gundam (lebih detail, lebih lambat)

Hasil di outputs/<nama_file>/ :
    hasil.md              Markdown terstruktur (heading, tabel, gambar)
    hasil.txt             teks polos tanpa token layout
    hasil.xlsx            Excel: sheet "Blok" + satu sheet per tabel
    hasil.docx            Word: heading, paragraf, tabel, potongan gambar
    hasil_searchable.pdf  PDF gambar asli + lapisan teks tak terlihat (bisa dicari/di-copy)
    hasil_blok.json       blok terstruktur per halaman (label, bbox, isi)
    hasil_raw.txt         keluaran mentah model (token <PAGE><BLOCK>...)
    ringkasan.json        waktu, token, RAM, CPU
Gunakan --no-export untuk melewati ekspor Excel/Word/PDF.
"""
import argparse, os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uocr_common import load_model, Monitor, save_json, strip_layout

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("file", help="gambar (png/jpg) atau PDF")
ap.add_argument("--pages", default="", help="rentang halaman PDF, mis. 1-5 atau 3 (default semua)")
ap.add_argument("--dpi", type=int, default=200)
ap.add_argument("--group", type=int, default=3, help="halaman per request untuk PDF (default 3)")
ap.add_argument("--per-page", action="store_true", help="tiap halaman sendiri, mode gundam")
ap.add_argument("--threads", type=int, default=12)
ap.add_argument("--max-new-tokens-per-page", type=int, default=4000)
ap.add_argument("--out", default=None, help="folder output (default outputs/<nama_file>)")
ap.add_argument("--stream", action="store_true", help="tampilkan token saat dihasilkan")
ap.add_argument("--no-export", action="store_true", help="jangan buat xlsx/docx/pdf")
a = ap.parse_args()

name = os.path.splitext(os.path.basename(a.file))[0]
out = a.out or os.path.join("outputs", name)
os.makedirs(out, exist_ok=True)

# 1. siapkan daftar gambar halaman
t0 = time.time()
if a.file.lower().endswith(".pdf"):
    import pymupdf
    doc = pymupdf.open(a.file)
    lo, hi = 1, len(doc)
    if a.pages:
        p = a.pages.split("-"); lo = int(p[0]); hi = int(p[-1])
    lo, hi = max(1, lo), min(len(doc), hi)
    mat = pymupdf.Matrix(a.dpi / 72, a.dpi / 72)
    pages = []
    for i in range(lo - 1, hi):
        path = os.path.join(out, f"page_{i + 1:04d}.png")
        doc[i].get_pixmap(matrix=mat).save(path)
        pages.append(path)
    print(f"[pdf] {len(doc)} halaman, diproses {lo}-{hi} ({len(pages)} halaman) @ {a.dpi} dpi, render {time.time() - t0:.1f}s", flush=True)
else:
    pages = [a.file]

# 2. muat model
tok, model, load_rep = load_model(a.threads)

# 3. OCR per kelompok
if a.per_page or len(pages) == 1:
    groups = [[p] for p in pages]
    crop = True
    window = 128
else:
    groups = [pages[i:i + a.group] for i in range(0, len(pages), a.group)]
    crop = False
    window = 1024

raws, stats = [], []
with Monitor() as m:
    for gi, g in enumerate(groups, 1):
        print(f"[ocr] kelompok {gi}/{len(groups)}: {len(g)} halaman, mode={'gundam' if crop else 'base'}", flush=True)
        res = model.infer(tok, image_file=g if len(g) > 1 else g[0], output_path=os.path.join(out, f"grup_{gi}"),
                          save_results=True, crop_mode=crop, eval_mode=not a.stream,
                          max_new_tokens=a.max_new_tokens_per_page * len(g),
                          no_repeat_ngram_size=35, ngram_window=window, return_dict=True)
        raws.append(res["text"])
        st = {"kelompok": gi, "halaman": len(g), "token_prompt": len(res["prompt_token_ids"]),
              "token_keluar": len(res["token_ids"]), "detik": round(res["time"], 1),
              "tok_per_s": round(len(res["token_ids"]) / max(res["time"] - (res.get("prefill_time") or 0), 1e-6), 1),
              "selesai": res["finish_reason"]}
        print("[ocr]", json.dumps(st, ensure_ascii=False), flush=True)
        stats.append(st)
mon = m.report()

# 4. tulis hasil
raw_all = "".join(raws)
text = strip_layout(raw_all)
with open(os.path.join(out, "hasil_raw.txt"), "w", encoding="utf-8") as f:
    f.write(raw_all)
with open(os.path.join(out, "hasil.txt"), "w", encoding="utf-8") as f:
    f.write(text)
try:
    from process_unlimitedOCR2 import convert
    with open(os.path.join(out, "hasil.md"), "w", encoding="utf-8") as f:
        f.write(convert(raw_all))
except Exception as e:
    print("[post] konversi Markdown gagal:", e)
total = time.time() - t0
save_json(os.path.join(out, "ringkasan.json"),
          {"file": a.file, "halaman": len(pages), "kelompok": stats, "total_detik": round(total, 1),
           "detik_per_halaman": round(total / max(len(pages), 1), 1), "ram_puncak_gib": mon["rss_peak_gib"],
           "cpu_persen_semua_core": mon["cpu_pct_of_all_cores_avg"], "threads": a.threads,
           "gambar_halaman": [os.path.abspath(p) for p in pages], "dpi": a.dpi})
if not a.no_export:
    try:
        from export_hasil import export_all
        export_all(out, raw_all, page_images=pages, dpi=a.dpi if a.file.lower().endswith(".pdf") else 150)
    except Exception as e:
        print("[export] gagal:", repr(e))
print("\n========== TEKS HASIL (awal) ==========")
print(text[:2500])
print("=======================================")
print(f"selesai: {len(pages)} halaman dalam {total:.0f} detik ({total / max(len(pages), 1):.0f} s/halaman), "
      f"RAM puncak {mon['rss_peak_gib']} GiB")
print(f"hasil: {out}/hasil.md  |  {out}/hasil.txt  |  {out}/ringkasan.json")
