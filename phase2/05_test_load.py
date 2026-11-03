#!/usr/bin/env python
"""Test 2: muat model ke RAM, ukur waktu dan memori. Tidak ada inference."""
import argparse
from uocr_common import load_model, sys_mem, save_json

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int)
ap.add_argument("--dtype", default="bf16")
ap.add_argument("--report", default="reports/test2_load.json")
a = ap.parse_args()

before = sys_mem()
print("[sys] sebelum:", before)
tok, model, rep = load_model(a.threads, a.dtype)
after = sys_mem()
print("[sys] sesudah:", after)
import torch
print("[check] cuda available:", torch.cuda.is_available(), "| devices:", rep["devices"], "| dtypes:", rep["dtypes"])
assert rep["devices"] == ["cpu"], "ada parameter bukan di CPU!"
save_json(a.report, {"test": "load", "load": rep, "sys_before": before, "sys_after": after})
print("TEST 2 SELESAI")
