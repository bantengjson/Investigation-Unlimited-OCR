#!/usr/bin/env python
"""Langkah 4: patch minimal agar modeling_unlimitedocr.py device-agnostic.

Hanya dua baris diubah: `.cuda()` pada mask masked_scatter_ -> `.to(<tensor>.device)`.
Tidak menyentuh bobot, arsitektur, maupun file lain. Idempoten: aman dijalankan ulang.
File asli disimpan sebagai modeling_unlimitedocr.py.orig di direktori yang sama.
Pakai --revert untuk mengembalikan file asli.

Pemakaian:
    python 04_apply_cpu_patch.py                 # target /mnt/data/ai-models/Unlimited-OCR/Unlimited-OCR-2
    python 04_apply_cpu_patch.py /path/ke/model  # target lain
    python 04_apply_cpu_patch.py --revert
"""
import os, sys, shutil, hashlib, datetime

MODELS = os.environ.get("UOCR_MODELS", "/mnt/data/ai-models/Unlimited-OCR")
_args = [x for x in sys.argv[1:] if not x.startswith("--")]
MODEL_DIR = _args[0] if _args else os.path.join(MODELS, "Unlimited-OCR-2")
TARGET = os.path.join(MODEL_DIR, "modeling_unlimitedocr.py")
ORIG = TARGET + ".orig"

REPLACEMENTS = [
    ("inputs_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1).cuda(), images_in_this_batch)",
     "inputs_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1).to(images_in_this_batch.device), images_in_this_batch)"),
    ("inputs_embeds[0].masked_scatter_(images_seq_mask[0].unsqueeze(-1).cuda(), _packed_feats)",
     "inputs_embeds[0].masked_scatter_(images_seq_mask[0].unsqueeze(-1).to(_packed_feats.device), _packed_feats)"),
]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


if "--revert" in sys.argv:
    if not os.path.exists(ORIG):
        sys.exit("tidak ada file .orig, tidak ada yang dikembalikan")
    shutil.copy2(ORIG, TARGET)
    print("dikembalikan dari", ORIG)
    sys.exit(0)

src = open(TARGET, encoding="utf-8").read()
if all(new in src for _, new in REPLACEMENTS):
    print("sudah dipatch, tidak ada perubahan. sha256:", sha(TARGET))
    sys.exit(0)
missing = [old for old, _ in REPLACEMENTS if src.count(old) != 1]
if missing:
    sys.exit("PATCH DIBATALKAN: baris target tidak ditemukan tepat satu kali:\n  " + "\n  ".join(missing))
if not os.path.exists(ORIG):
    shutil.copy2(TARGET, ORIG)
before = sha(TARGET)
for old, new in REPLACEMENTS:
    src = src.replace(old, new)
with open(TARGET, "w", encoding="utf-8", newline="\n") as f:
    f.write(src)
log = os.path.join(MODEL_DIR, "CPU_PATCH_CHANGES.md")
with open(log, "a", encoding="utf-8") as f:
    f.write(f"# CPU patch {datetime.datetime.now():%Y-%m-%d %H:%M}\n\n")
    f.write(f"File: modeling_unlimitedocr.py (asli: .orig, sha {before} -> {sha(TARGET)})\n\n")
    for old, new in REPLACEMENTS:
        f.write(f"- `{old.strip()}`\n  -> `{new.strip()}`\n")
    f.write("\nAlasan: `.cuda()` hard-coded gagal pada PyTorch CPU build. "
            "Pengganti memindahkan mask ke device tensor fitur, identik di GPU maupun CPU.\n\n")
print(f"patch diterapkan: {TARGET}\n  backup: {ORIG}\n  log: {log}\n  sha256 {before} -> {sha(TARGET)}")
