#!/usr/bin/env python
"""Langkah 3: unduh baidu/Unlimited-OCR-2 (~6.7 GB) ke direktori model terpisah."""
import os, time
from huggingface_hub import snapshot_download
MODELS = os.environ.get("UOCR_MODELS", "/mnt/data/ai-models/Unlimited-OCR")
dst = os.path.join(MODELS, "Unlimited-OCR-2")
t0 = time.time()
path = snapshot_download("baidu/Unlimited-OCR-2", local_dir=dst,
                         ignore_patterns=["assets/*", "wheel/*", "*.pdf", "*.gif"])
dt = time.time() - t0
total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)
print(f"selesai dalam {dt/60:.1f} menit -> {path}  ({total/2**30:.2f} GiB)")
for f in sorted(os.listdir(path)): print("  ", f)
