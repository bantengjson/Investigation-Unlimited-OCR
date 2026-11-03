# Tahap 2 — Panduan eksekusi di server (CPU-only)

Semua perintah dijalankan sebagai user biasa di server Ubuntu. Tidak ada `sudo`.
Direktori yang disentuh hanya:

- `/mnt/data/ai-projects/Unlimited-OCR` (proyek, venv, skrip, output, laporan)
- `/mnt/data/ai-models/Unlimited-OCR` (bobot model + patch)

Kirim output setiap langkah sebelum lanjut ke langkah berikutnya.

## Langkah 0 — salin skrip dan periksa server (tanpa perubahan)

Di PC Windows (ganti `USER@SERVER`):

```bash
scp -r F:/work/Investigation-Unlimited-OCR/phase2 USER@SERVER:/tmp/uocr-phase2
```

Di server:

```bash
mkdir -p /mnt/data/ai-projects/Unlimited-OCR && mv /tmp/uocr-phase2 /mnt/data/ai-projects/Unlimited-OCR/phase2
```

```bash
bash /mnt/data/ai-projects/Unlimited-OCR/phase2/00_check_server.sh
```

Fungsi: menampilkan versi Python, apakah `venv`/`ensurepip` tersedia, disk, RAM, flag CPU (cari `avx512_bf16`), dan memastikan `nvidia-smi` tidak ada.
Diharapkan: `ensurepip OK`, flag `avx512_bf16` muncul, `/mnt/data` punya ruang > 20 GB.
Jika `ensurepip TIDAK ada`: jangan `sudo apt`; pasang `uv` tanpa sudo:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
```

## Langkah 1 — buat venv dan pasang dependency CPU

```bash
bash /mnt/data/ai-projects/Unlimited-OCR/phase2/01_setup_env.sh
```

Fungsi: membuat `.venv`, memasang `torch==2.13.0+cpu` dan `torchvision==0.28.0+cpu` dari index CPU PyTorch (bukan PyPI, agar tidak menarik paket NVIDIA/Triton), lalu `transformers==4.52.1` dan dependensi lain yang diuji Baidu.
Diharapkan: daftar paket di akhir **tidak** memuat `nvidia-*` maupun `triton`; ukuran venv ≈ 1–2 GB; unduhan ≈ 300–400 MB.

## Test 1 — environment

```bash
cd /mnt/data/ai-projects/Unlimited-OCR && source .venv/bin/activate && python phase2/02_test_env.py
```

Diharapkan: `cuda available: False`, `cpu capability: AVX512`, `mkldnn/oneDNN: True`, `threads: 24`, angka TFLOPS untuk matmul fp32 dan bf16 (bf16 seharusnya lebih cepat), `SDPA bf16 + mask: OK`, `TEST 1 SELESAI`.

## Langkah 3 — unduh model (± 6,7 GB)

```bash
cd /mnt/data/ai-projects/Unlimited-OCR && source .venv/bin/activate && python phase2/03_download_model.py
```

Diharapkan: folder `/mnt/data/ai-models/Unlimited-OCR/Unlimited-OCR-2` berisi 2 file `.safetensors`, `config.json`, `tokenizer.json`, dan 10 file `.py`; total ≈ 6,25 GiB.

## Langkah 4 — patch CPU (2 baris)

```bash
cd /mnt/data/ai-projects/Unlimited-OCR && source .venv/bin/activate && python phase2/04_apply_cpu_patch.py
```

Fungsi: mengganti dua `.cuda()` di `modeling_unlimitedocr.py` dengan `.to(<tensor>.device)`. Menyimpan file asli sebagai `.orig` dan menulis `CPU_PATCH_CHANGES.md`. Diff lengkapnya ada di `phase2/cpu.patch`. Revert: tambahkan `--revert`.
Diharapkan: `patch diterapkan: ...` dengan dua sha256 berbeda.

## Test 2 — model loading

```bash
cd /mnt/data/ai-projects/Unlimited-OCR && source .venv/bin/activate && python phase2/05_test_load.py
```

Diharapkan: baris `[load] {...}` dengan `elapsed_s` (perkiraan 10–60 detik dari NVMe), `rss_peak_gib` ≈ 7–9, `params_B` ≈ 3.34, `devices: ["cpu"]`, `dtypes: ["torch.bfloat16"]`, `attn_impl: eager`, lalu `TEST 2 SELESAI`.
Jika gagal dengan `Torch not compiled with CUDA enabled`: patch belum diterapkan.

## Test 3 — satu gambar

Buat gambar uji dengan teks yang diketahui (atau pakai gambar Anda sendiri):

```bash
cd /mnt/data/ai-projects/Unlimited-OCR && source .venv/bin/activate && python phase2/06_make_test_image.py samples/test_page.png
```

```bash
cd /mnt/data/ai-projects/Unlimited-OCR && source .venv/bin/activate && python phase2/07_test_single_image.py samples/test_page.png --gt samples/test_page.gt.txt --threads 12
```

Fungsi: OCR satu halaman mode gundam (default untuk 1 gambar), batas 4.096 token output, pengukuran prefill/decode, token/detik, RSS puncak, CPU%, dan kemiripan teks dengan ground truth. Output di `outputs/test3/` (`result.md`, `result_raw.md`, `text_only.txt`, `result_eval.md`, gambar dengan kotak) dan laporan JSON di `reports/test3_single.json`.
Ini pengujian yang menentukan. Perkiraan durasi: beberapa menit. Ulangi dengan `--threads 24` untuk membandingkan.
Tambahkan `--stream` jika ingin melihat token keluar secara langsung.

Pemantauan di terminal kedua selama Test 3:

```bash
vmstat 5
```

## Test 4 — PDF

```bash
cd /mnt/data/ai-projects/Unlimited-OCR && source .venv/bin/activate && python phase2/08_test_pdf.py /path/ke/dokumen.pdf --max-pages 3 --dpi 200
```

Fungsi: render 3 halaman pertama ke PNG dengan PyMuPDF, lalu satu request multi-halaman mode base (cara yang dipakai Baidu untuk dokumen). `--per-page` memproses tiap halaman mode gundam.

## Test 5 — beban bertahap (hanya jika Test 3–4 lulus)

```bash
cd /mnt/data/ai-projects/Unlimited-OCR && source .venv/bin/activate && python phase2/09_stress.py outputs/test4 --stages 1,3,10
```

Fungsi: memproses 1, lalu 3, lalu 10 halaman dalam satu request per tahap; berhenti otomatis bila swap terpakai > 1 GiB.

## Mengembalikan semuanya

Tidak ada yang berubah di luar dua direktori di atas. Hapus `/mnt/data/ai-projects/Unlimited-OCR` dan `/mnt/data/ai-models/Unlimited-OCR` bila ingin bersih total.
