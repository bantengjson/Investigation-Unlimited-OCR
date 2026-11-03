#!/usr/bin/env python
"""Test 3: OCR satu gambar di CPU. Mengukur prefill/decode, token/detik, RAM, CPU, kemiripan dengan ground truth."""
import argparse, os, json
from uocr_common import load_model, Monitor, sys_mem, save_json, similarity, strip_layout

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--gt", help="file teks ground truth (opsional)")
ap.add_argument("--threads", type=int)
ap.add_argument("--dtype", default="bf16")
ap.add_argument("--mode", choices=["auto", "gundam", "base"], default="auto", help="auto = gundam untuk 1 gambar")
ap.add_argument("--max-new-tokens", type=int, default=4096, help="batas aman untuk CPU")
ap.add_argument("--stream", action="store_true", help="cetak token saat dihasilkan")
ap.add_argument("--out", default="outputs/test3")
ap.add_argument("--report", default="reports/test3_single.json")
a = ap.parse_args()

tok, model, load_rep = load_model(a.threads, a.dtype)
crop = {"auto": None, "gundam": True, "base": False}[a.mode]
print(f"[infer] {a.image} mode={a.mode} max_new_tokens={a.max_new_tokens}", flush=True)
with Monitor() as m:
    res = model.infer(tok, image_file=a.image, output_path=a.out, save_results=True, crop_mode=crop,
                      eval_mode=not a.stream, max_new_tokens=a.max_new_tokens,
                      no_repeat_ngram_size=35, ngram_window=128, return_dict=True)
mon = m.report()
ntok = len(res["token_ids"])
nprompt = len(res["prompt_token_ids"])
pre = res.get("prefill_time") or 0.0
dec = res["time"] - pre
rep = {"test": "single_image", "image": a.image, "mode": a.mode, "threads": load_rep.get("threads"),
       "prompt_tokens": nprompt, "output_tokens": ntok, "finish_reason": res["finish_reason"],
       "pages_out": res["num_pages_out"], "prefill_s": round(pre, 2), "decode_s": round(dec, 2),
       "total_s": round(res["time"], 2),
       "decode_tok_per_s": round(ntok / dec, 2) if dec > 0 and ntok else None,
       "monitor": mon, "load": load_rep, "sys_after": sys_mem()}
text = strip_layout(res["text"])
if a.gt and os.path.exists(a.gt):
    rep["similarity_to_gt"] = similarity(text, open(a.gt, encoding="utf-8").read())
os.makedirs(a.out, exist_ok=True)
with open(os.path.join(a.out, "text_only.txt"), "w", encoding="utf-8") as f:
    f.write(text)
try:
    from process_unlimitedOCR2 import convert
    with open(os.path.join(a.out, "result_eval.md"), "w", encoding="utf-8") as f:
        f.write(convert(res["text"]))
except Exception as e:
    print("[post] process_unlimitedOCR2 gagal:", e)
print("\n===== TEKS HASIL (tanpa token layout) =====\n" + text[:3000] + "\n===========================================")
print(json.dumps(rep, indent=2, ensure_ascii=False))
save_json(a.report, rep)
print("TEST 3 SELESAI")
