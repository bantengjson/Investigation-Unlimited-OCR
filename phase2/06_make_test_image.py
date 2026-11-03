#!/usr/bin/env python
"""Membuat gambar uji A4 (150 dpi) berisi teks yang diketahui, plus file ground-truth .txt."""
import os, sys
from PIL import Image, ImageDraw, ImageFont

out = sys.argv[1] if len(sys.argv) > 1 else "samples/test_page.png"
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
TEXT = """LAPORAN PENGUJIAN OCR PADA CPU

Dokumen ini dibuat untuk menguji kemampuan Baidu Unlimited OCR 2 membaca teks
tanpa GPU. Server menggunakan prosesor AMD Ryzen 9 9900X dengan 12 core dan
RAM 64 GB. Sistem operasi Ubuntu Server 24.04 LTS, Python 3.12.3.

1. Tujuan: memastikan model berjalan sepenuhnya di CPU.
2. Metode: satu halaman gambar diproses dengan model.infer().
3. Metrik: waktu, RAM, CPU, dan kemiripan teks hasil ekstraksi.

Tabel contoh:
Nomor   Komponen        Nilai
1       CPU             Ryzen 9 9900X
2       RAM             64 GB DDR5 6000 MHz
3       Storage         Samsung 9100 PRO 2 TB

The quick brown fox jumps over the lazy dog. 0123456789
Alamat: Jl. Merdeka No. 17, Jakarta 10110. Telepon (021) 555-0199.
Total biaya: Rp 1.250.000,00 (satu juta dua ratus lima puluh ribu rupiah)."""
W, H = 1240, 1754  # A4 @150dpi
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    mono = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 26)
except Exception:
    font = mono = ImageFont.load_default()
y = 100
for line in TEXT.split("\n"):
    f = mono if line[:1].isdigit() or line.startswith("Nomor") else font
    d.text((100, y), line, fill="black", font=f)
    y += 44
img.save(out)
gt = os.path.splitext(out)[0] + ".gt.txt"
with open(gt, "w", encoding="utf-8") as f:
    f.write(TEXT)
print("gambar uji:", out, img.size, "| ground truth:", gt)
