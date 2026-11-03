#!/usr/bin/env python
"""Test 4: PDF -> gambar (PyMuPDF) -> OCR.
Default: semua halaman dalam SATU request (base mode, seperti infer_multi).
--per-page: tiap halaman sendiri dengan mode gundam (lebih detail, lebih lambat)."""
import argparse, os, json, time
try:
    import pymupdf as fitz
except ImportError:
    import fitz
from uocr_common import load_model, Monitor, sys_mem, save_json, strip_layout

ap = argparse.ArgumentParser()
ap.add_argument("pdf")
ap.add_argument("--dpi", type=int, default=200)
ap.add_argument("--max-pages", type=int, default=3, help="0 = semua halaman")
ap.add_argument("--per-page", action="store_true")
ap.add_argument("--threads", type=int)
ap.add_argument("--max-new-tokens-per-page", type=int, default=4096)
ap.add_argument("--out", default="outputs/test4")
ap.add_argument("--report", default="reports/test4_pdf.json")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
doc = fitz.open(a.pdf)
n = min(len(doc), a.max_pages) if a.max_pages > 0 else len(doc)
mat = fitz.Matrix(a.dpi / 72, a.dpi / 72)
pages = []
t0 = time.time()
for i in range(n):
    p = os.path.join(a.out, f"page_{i + 1:04d}.png")
    doc[i].get_pixmap(matrix=mat).save(p)
    pages.append(p)
print(f"[pdf] {len(doc)} halaman total, {n} dirender @ {a.dpi} dpi dalam {time.time() - t0:.1f}s", flush=True)

tok, model, load_rep = load_model(a.threads)
results = []
if a.per_page:
    raws = []
    with Monitor() as m:
        for i, p in enumerate(pages):
            res = model.infer(tok, image_file=p, output_path=os.path.join(a.out, f"p{i + 1}"), save_results=True,
                              eval_mode=True, max_new_tokens=a.max_new_tokens_per_page,
                              no_repeat_ngram_size=35, ngram_window=128, return_dict=True)
            raws.append(res["text"])
            results.append({"page": i + 1, "prompt_tokens": len(res["prompt_token_ids"]),
                            "output_tokens": len(res["token_ids"]), "total_s": round(res["time"], 2),
                            "prefill_s": round(res.get("prefill_time") or 0, 2), "finish": res["finish_reason"]})
            print("[page]", results[-1], flush=True)
    raw_all = "".join(raws)
else:
    with Monitor() as m:
        res = model.infer(tok, image_file=pages, output_path=a.out, save_results=True, crop_mode=False,
                          eval_mode=True, max_new_tokens=a.max_new_tokens_per_page * n,
                          no_repeat_ngram_size=35, ngram_window=1024, return_dict=True)
    results.append({"pages": n, "prompt_tokens": len(res["prompt_token_ids"]),
                    "output_tokens": len(res["token_ids"]), "total_s": round(res["time"], 2),
                    "prefill_s": round(res.get("prefill_time") or 0, 2), "finish": res["finish_reason"],
                    "pages_out": res["num_pages_out"]})
    raw_all = res["text"]
mon = m.report()
text = strip_layout(raw_all)
with open(os.path.join(a.out, "text_only.txt"), "w", encoding="utf-8") as f:
    f.write(text)
try:
    from process_unlimitedOCR2 import convert
    with open(os.path.join(a.out, "result_eval.md"), "w", encoding="utf-8") as f:
        f.write(convert(raw_all))
except Exception as e:
    print("[post] process_unlimitedOCR2 gagal:", e)
rep = {"test": "pdf", "pdf": a.pdf, "pages": n, "dpi": a.dpi, "per_page_mode": a.per_page, "results": results,
       "monitor": mon, "load": load_rep, "sys_after": sys_mem()}
print("\n===== TEKS HASIL (awal) =====\n" + text[:2000] + "\n=============================")
print(json.dumps(rep, indent=2, ensure_ascii=False))
save_json(a.report, rep)
print("TEST 4 SELESAI")
