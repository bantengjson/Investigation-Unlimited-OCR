#!/usr/bin/env python
"""Test 5: beban bertahap. Memproses gambar dari sebuah direktori dalam tahap 1 -> 3 -> 10 halaman
(satu request per tahap), mencatat RAM/CPU/swap, dan BERHENTI bila swap terpakai melebihi batas."""
import argparse, os, json, glob
from uocr_common import load_model, Monitor, sys_mem, save_json

ap = argparse.ArgumentParser()
ap.add_argument("image_dir")
ap.add_argument("--stages", default="1,3,10")
ap.add_argument("--threads", type=int)
ap.add_argument("--mode", choices=["gundam", "base"], default="base")
ap.add_argument("--max-new-tokens-per-page", type=int, default=3000)
ap.add_argument("--swap-limit-gib", type=float, default=1.0)
ap.add_argument("--out", default="outputs/test5")
ap.add_argument("--report", default="reports/test5_stress.json")
a = ap.parse_args()

imgs = sorted(glob.glob(os.path.join(a.image_dir, "*.png")) + glob.glob(os.path.join(a.image_dir, "*.jpg")))
assert imgs, "tidak ada gambar di " + a.image_dir
tok, model, load_rep = load_model(a.threads)
stages = []
for k in [int(s) for s in a.stages.split(",")]:
    batch = (imgs * ((k // len(imgs)) + 1))[:k]
    print(f"\n[stage] {k} halaman, mode={a.mode}", flush=True)
    with Monitor() as m:
        res = model.infer(tok, image_file=batch, output_path=os.path.join(a.out, f"stage_{k}"), save_results=True,
                          crop_mode=(a.mode == "gundam"), eval_mode=True,
                          max_new_tokens=a.max_new_tokens_per_page * k,
                          no_repeat_ngram_size=35, ngram_window=1024 if k > 1 else 128, return_dict=True)
    mon = m.report()
    sm = sys_mem()
    st = {"pages": k, "prompt_tokens": len(res["prompt_token_ids"]), "output_tokens": len(res["token_ids"]),
          "total_s": round(res["time"], 2), "s_per_page": round(res["time"] / k, 2), "finish": res["finish_reason"],
          "pages_out": res["num_pages_out"], "monitor": mon, "sys": sm}
    print("[stage]", json.dumps(st), flush=True)
    stages.append(st)
    save_json(a.report, {"test": "stress", "mode": a.mode, "stages": stages, "load": load_rep})
    if sm["swap_used_gib"] > a.swap_limit_gib:
        print(f"[stop] swap {sm['swap_used_gib']} GiB > batas {a.swap_limit_gib} GiB, berhenti.")
        break
print("TEST 5 SELESAI")
