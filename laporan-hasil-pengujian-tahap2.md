# Laporan Hasil Pengujian CPU-Only — Baidu Unlimited OCR 2

Tanggal: 2026-11-03
Server: `banteng-data` (AMD Ryzen 9 9900X 12C/24T, 60 GiB RAM terlihat OS, swap 8 GiB, Ubuntu Server 24.04.5, Python 3.12.3, tanpa GPU)
Semua angka di bagian 2–7 adalah **hasil pengukuran** di server tersebut. Perkiraan diberi label [ESTIMASI].
Laporan JSON mentah: `results/reports/*.json`. Output OCR: `results/test*/`.

---

## 0. Kesimpulan

**Unlimited OCR 2 TERBUKTI berjalan sepenuhnya di CPU** pada server Anda, tanpa driver NVIDIA, CUDA, Triton, maupun SGLang. Patch yang diperlukan hanya 2 baris. Hasil OCR akurat (kemiripan 0,993 pada halaman uji sintetis; abstrak paper ditranskripsi kata demi kata).

Kinerja terukur dengan 12 thread:

| Skenario | Waktu | Decode |
|---|---|---|
| 1 halaman teks (gundam) | 24,5 s | 20,6 tok/s |
| 3 halaman paper (base, 1 request) | 94 s (31 s/hal) | ≈18 tok/s |
| 3 halaman paper (gundam, per halaman) | 123 s (41 s/hal) | ≈18 tok/s |
| 10 halaman paper (base, 1 request) | 1.195 s (119 s/hal) | 6,8 tok/s rata-rata |

RAM puncak proses 7,0–8,6 GiB. Swap tidak bertambah. Keterbatasan utama CPU: **throughput decode turun seiring panjang konteks** (bagian 6), sehingga dokumen puluhan halaman dalam satu request menjadi lambat. Untuk pemakaian operasional sebelum GPU dipasang: layak untuk volume kecil (puluhan halaman per jam) dengan memproses per halaman atau per kelompok kecil halaman.

---

## 1. Yang dilakukan di server

| Langkah | Hasil |
|---|---|
| Folder | `/mnt/data/ai-projects/Unlimited-OCR` (venv, skrip, output) dan `/mnt/data/ai-models/Unlimited-OCR` (model). Tidak ada folder lain yang disentuh; aplikasi lain (`BantengJson`, `Chimera`, pm2, docker, proses user `ojan`) tidak diganggu. |
| venv | `python3 -m venv` tidak bisa dipakai karena paket `python3.12-venv` tidak ada dan tidak boleh `sudo`. Solusi tanpa sudo: `uv 0.13.7` dipasang ke `~/.local/bin` (tanpa mengubah `.bashrc`), venv dibuat dari interpreter sistem 3.12.3. |
| Dependency | `torch 2.13.0+cpu`, `torchvision 0.28.0+cpu` (index CPU PyTorch), `transformers 4.52.1`, `tokenizers 0.21.4`, `huggingface-hub 0.36.0`, `accelerate`, `pymupdf 2.1`, `psutil`. Tidak ada paket `nvidia-*`, `triton`, `flash-attn`, `xformers`, `sglang`. venv 1,3 GB. |
| Model | `baidu/Unlimited-OCR-2` diunduh dalam 2,9 menit, 6,24 GiB. |
| Patch | 2 baris di `modeling_unlimitedocr.py` (`.cuda()` → `.to(<tensor>.device)`). Asli disimpan sebagai `.orig`, log di `CPU_PATCH_CHANGES.md`, diff di `phase2/cpu.patch`. Revert: `python phase2/04_apply_cpu_patch.py --revert`. |

## 2. Test 1 — Environment

| Item | Hasil |
|---|---|
| `torch.cuda.is_available()` | False |
| CPU capability PyTorch | AVX512 |
| oneDNN (mkldnn) | tersedia |
| matmul 2048³ fp32 | 10,8 ms ≈ 1,60 TFLOPS |
| matmul 2048³ bf16 | 5,6 ms ≈ 3,09 TFLOPS |
| SDPA bf16 dengan mask | berjalan |

bf16 sekitar 1,9× lebih cepat dari fp32 berkat AVX-512 BF16, membenarkan pilihan dtype bf16.

## 3. Test 2 — Model loading

| Item | Hasil |
|---|---|
| Waktu load | 1,4 s |
| RSS setelah load | 0,47 GiB |
| Parameter | 3,344 miliar, semua di `cpu`, dtype `bfloat16` |
| Attention impl | `eager` (bukan flash/sdpa) |
| Error CUDA | tidak ada |

Catatan: load 1,4 s dan RSS kecil karena safetensors di-memory-map; bobot baru masuk RAM saat inference pertama (terlihat sebagai RSS 5 → 7 GiB di Test 3). Ini sekaligus berarti model bisa dimuat ulang hampir gratis selama file masih di page cache.

## 4. Test 3 — Satu gambar (A4 150 dpi, teks sintetis dengan ground truth)

| Parameter | 12 thread | 24 thread |
|---|---|---|
| Mode | gundam (2×3 tile + global) | sama |
| Prompt token | 909 | 909 |
| Output token | 454 | 454 |
| Prefill | 2,49 s | 2,92 s |
| Decode | 21,99 s | 25,80 s |
| **Decode tok/s** | **20,6** | 17,6 |
| Total | 24,5 s | 28,7 s |
| RSS puncak | 7,05 GiB | 7,07 GiB |
| CPU | 1.195 % (≈12 core penuh) | 2.386 % (24 thread penuh) |
| Kemiripan dengan ground truth | 0,9932 | 0,9932 |

Kesalahan OCR yang ditemukan pada 806 karakter: kata "teks" hilang di satu pemenggalan baris, dan "dan" terbaca "of". Tabel sederhana terbaca utuh, angka, alamat, dan nominal rupiah benar. **12 thread (core fisik) lebih cepat 15 % dari 24 thread**; SMT tidak membantu beban bf16 yang sudah memenuhi unit vektor.

## 5. Test 4 — PDF (paper Unlimited-OCR-2, 15 halaman, dirender 200 dpi)

| Parameter | Base, 3 hal dalam 1 request | Gundam, per halaman |
|---|---|---|
| Prompt token | 827 (273/hal) | 909/hal |
| Output token | 1.685 | 511 + 819 + 898 = 2.228 |
| Prefill | 2,75 s | 2,4–2,5 s/hal |
| Total | 94,2 s | 122,9 s |
| Detik per halaman | 31 | 41 |
| RSS puncak | 7,41 GiB | 7,51 GiB |
| Kemiripan dengan teks PDF (3 hal) | 0,883 | 0,869 |

Kemiripan terhadap teks ekstraksi PDF lebih rendah dari Test 3 karena urutan baca dua kolom, caption, dan footnote berbeda antara OCR dan ekstraktor PDF; pemeriksaan manual menunjukkan abstrak, daftar isi, dan sitasi `[5, 6, 8, ...]` terbaca benar. Post-processing resmi (`process_unlimitedOCR2.py`) menghasilkan Markdown dengan heading yang benar. PDF harus dikonversi ke gambar dulu (model tidak membaca PDF langsung); PyMuPDF melakukannya dalam < 1 s per halaman.

## 6. Test 5 — Beban bertahap (mode base, 1 request per tahap)

| Halaman | Prompt token | Output token | Total | Detik/hal | Tok/s rata-rata | RSS puncak | Swap |
|---|---|---|---|---|---|---|---|
| 1 | 279 | 495 | 23,8 s | 23,8 | 20,8 | 6,86 GiB | tetap 0,25 GiB |
| 3 | 827 | 1.685 | 96,4 s | 32,1 | 17,5 | 7,33 GiB | tetap |
| 10 | 2.745 | 8.148 | 1.194,6 s | 119,5 | 6,8 | 8,61 GiB | tetap |

Semua tahap selesai dengan `finish=stop` dan jumlah `<PAGE>` keluar sama dengan jumlah halaman masuk. Kemiripan 10 halaman terhadap teks PDF 0,836 (25.429 vs 27.983 karakter).

**Temuan utama:** throughput decode turun dari ≈20 tok/s menjadi 6,8 tok/s rata-rata saat satu request mencakup 10 halaman (konteks ≈ 11 ribu token). Penyebabnya di jalur CPU: attention decode memakai fallback dense fp32 (`_decode_attend`) yang membaca seluruh KV cache tiap token, dan router DASH menghitung skor untuk semua blok; di GPU bagian ini ditangani kernel Triton block-sparse yang biayanya hampir konstan. Kurva terukur ada di bagian 6.1.

### 6.1 Kurva tok/s terhadap panjang konteks (5 halaman, pelaporan tiap 100 token)

Dokumen 5 halaman paper, mode base, prompt 1.375 token, output 3.244 token, total 233,9 s (prefill 4,5 s). Tok/s interval dihitung dari laporan kumulatif `[TPS]` tiap 100 token.

| Token keluar ke- | Panjang konteks (prompt + keluar) | Tok/s rata-rata kumulatif | Tok/s interval terakhir |
|---|---|---|---|
| 100 | ≈1.500 | 19,6 | 19,6 |
| 500 | ≈1.900 | 18,6 | 17,8 |
| 1.000 | ≈2.400 | 17,7 | 16,8 |
| 1.500 | ≈2.900 | 16,9 | 15,6 |
| 2.000 | ≈3.400 | 16,1 | 13,1 |
| 2.500 | ≈3.900 | 15,3 | 13,2 |
| 3.000 | ≈4.400 | 14,5 | 12,1 |
| 3.200 | ≈4.600 | 14,2 | 9,9 |

Throughput decode turun kira-kira linier terhadap panjang konteks: dari ≈20 tok/s pada 1,5 ribu token menjadi ≈10–12 tok/s pada 4,6 ribu token, dan 6,8 tok/s rata-rata pada ≈11 ribu token (Test 5). Ini konsisten dengan biaya per token yang bertambah sebanding dengan jumlah KV yang dibaca di jalur fallback CPU. Implikasi praktis: pada CPU, batasi satu request ke ≈3 halaman (≈3–4 ribu token konteks) agar tetap di 15–18 tok/s.

## 7. Jawaban untuk 10 pertanyaan output akhir

| # | Pertanyaan | Jawaban (terukur) |
|---|---|---|
| 1 | Bisa berjalan tanpa GPU? | **Ya, terbukti.** Semua 5 test lulus di CPU. |
| 2 | Ryzen 9 9900X mampu? | Ya. 12 core bf16 memberi 20 tok/s pada dokumen pendek. |
| 3 | PyTorch CPU cukup? | Ya. Tidak perlu ONNX atau backend lain. |
| 4 | Perlu modifikasi source? | Ya, 2 baris, tersimpan sebagai patch terpisah dan bisa di-revert. |
| 5 | Perlu backend alternatif? | Tidak. SGLang tetap GPU-only. |
| 6 | RAM yang diperlukan | 7–9 GiB per proses (terukur 6,9–8,6 GiB). 64 GB jauh dari cukup. |
| 7 | Performa OCR di CPU | 24 s/halaman teks; 31–41 s/halaman paper; turun ke 120 s/halaman bila 10 halaman dalam satu request. |
| 8 | Keterbatasan vs GPU | Decode 20 tok/s vs ratusan di GPU; penurunan tajam pada konteks panjang; tidak ada serving paralel; dokumen ratusan halaman dalam satu request tidak praktis. |
| 9 | Operasional sebelum GPU? | Ya untuk volume kecil: proses per halaman atau per 3 halaman, ≈ 100–120 halaman/jam [ESTIMASI dari angka terukur]. Bukan untuk produksi throughput tinggi. |
| 10 | Yang berubah saat RTX 5070 Ti 16 GB dipasang | Pasang driver NVIDIA + torch CUDA (cu13), `model.cuda()`; patch tidak perlu dicabut (`.to(device)` benar di GPU). Kernel Triton aktif otomatis, penurunan tok/s pada konteks panjang hilang. Bobot 6,2 GB muat di 16 GB VRAM. Untuk SGLang pakai wheel di repo dengan `--mem-fraction-static 0.7` dan `--max-running-requests` lebih kecil dari contoh A800. Dukungan sm_120 (Blackwell) di flashinfer/flash-attn-4 perlu dicek saat itu. |

## 8. Rekomendasi konfigurasi CPU

- `torch.set_num_threads(12)` (core fisik), bukan 24.
- dtype bf16 (wajib untuk kode pra-pemrosesan; terbukti 1,9× lebih cepat dari fp32).
- Proses dokumen **per halaman (gundam) atau per 3 halaman (base)**, bukan puluhan halaman dalam satu request, untuk menjaga ≈18–20 tok/s. Konteks lintas halaman (penggabungan tabel antar halaman) akan hilang di batas kelompok.
- Satu proses dengan 12 thread. Dua proses paralel akan berbagi 12 core yang sama dan tidak menambah throughput total [ESTIMASI, belum diukur].
- `max_new_tokens` dibatasi (mis. 4.096 per halaman) dan `no_repeat_ngram_size=35` seperti setelan deployment Baidu.
- Simpan model di NVMe bila ingin load pertama lebih cepat; di HDD `/mnt/data` load pertama tetap < 10 s karena page cache.

## 9. Yang belum diuji

- Dokumen fotokopi/scan nyata dengan noise (hanya gambar sintetis dan PDF render bersih).
- Bahasa selain Indonesia dan Inggris; tulisan tangan.
- Dua proses OCR bersamaan.
- Temperatur CPU (sensor tidak diperiksa).
- Perbandingan hasil bit-per-bit dengan GPU (belum ada GPU).
