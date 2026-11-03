# Rangkuman Hasil Pengujian — Unlimited OCR 2 di CPU

Tanggal pengujian: 3 November 2026
Server: `banteng-data`, AMD Ryzen 9 9900X (12C/24T), 60 GiB RAM terlihat OS, swap 8 GiB, Ubuntu Server 24.04.5, Python 3.12.3, **tanpa GPU**
Software: torch 2.13.0+cpu, torchvision 0.28.0+cpu, transformers 4.52.1, bf16, patch 2 baris
Semua angka adalah hasil pengukuran langsung. Data mentah: `results/reports/*.json`, `results/reports/tps_curve.log`.

## Ringkasan satu baris

**Lulus semua 5 pengujian. Unlimited OCR 2 berjalan sepenuhnya di CPU: 1 halaman ≈ 24 detik (20,6 tok/s), RAM 7–9 GiB, akurasi 0,993 pada halaman uji; throughput turun pada dokumen panjang dalam satu request.**

## Tabel hasil per pengujian

| Test | Skenario | Hasil utama | Status |
|---|---|---|---|
| 1 Environment | import torch/transformers, benchmark | `cuda.is_available()=False`, AVX512, oneDNN aktif; matmul fp32 1,60 TFLOPS, bf16 3,09 TFLOPS; SDPA bf16 OK | Lulus |
| 2 Model loading | muat 3,34 M parameter bf16 | 1,4 s (memory-map), semua parameter di `cpu`, attention `eager`, tidak ada error CUDA | Lulus |
| 3 Satu gambar | A4 150 dpi teks sintetis, gundam | 24,5 s total; decode 20,6 tok/s; RSS 7,05 GiB; kemiripan 0,9932 | Lulus |
| 4 PDF | paper 15 hal, 3 hal pertama, 200 dpi | base 1 request: 94 s (31 s/hal); gundam per hal: 123 s (41 s/hal); RSS ≤ 7,5 GiB | Lulus |
| 5 Beban bertahap | 1 → 3 → 10 halaman, base | 23,8 s / 96,4 s / 1.194,6 s; RSS puncak 8,61 GiB; swap tidak bertambah; semua `finish=stop` | Lulus |

## Tabel laporan sesuai format yang diminta

| Parameter | Hasil |
|---|---|
| Model loading time | 1,4 s (memory-map; bobot masuk RAM saat inference pertama) |
| Inference time | 24,5 s per halaman teks (1 hal, gundam); 31 s/hal (3 hal base); 41 s/hal (gundam per hal); 119 s/hal (10 hal dalam 1 request) |
| CPU utilization | ≈1.195 % dari satu core = 12 core penuh (setelan 12 thread) |
| RAM usage | 5,1 GiB sebelum inference → 7,0–7,5 GiB saat inference 1–3 halaman |
| Peak RAM usage | 8,61 GiB (10 halaman dalam 1 request) |
| OCR accuracy | 0,9932 kemiripan karakter vs ground truth sintetis (2 kesalahan kecil / 806 karakter); 0,88 vs teks ekstraksi PDF paper 3 hal (perbedaan terutama urutan baca 2 kolom, bukan salah baca) |
| Number of pages | 27 halaman diproses di seluruh pengujian (per request: 1, 1, 3, 3×1, 1, 3, 10, 5); maksimum 10 halaman dalam satu request |
| Error | tidak ada |

## Detail Test 3 — 12 vs 24 thread

| Parameter | 12 thread | 24 thread |
|---|---|---|
| Prompt / output token | 909 / 454 | 909 / 454 |
| Prefill | 2,49 s | 2,92 s |
| Decode | 21,99 s | 25,80 s |
| Decode tok/s | **20,6** | 17,6 |
| Total | **24,5 s** | 28,7 s |
| RSS puncak | 7,05 GiB | 7,07 GiB |
| Kemiripan GT | 0,9932 | 0,9932 |

Kesimpulan: pakai 12 thread (core fisik). SMT tidak membantu.

## Detail Test 5 dan kurva throughput

| Halaman dalam 1 request | Prompt token | Output token | Total | Detik/halaman | Tok/s rata-rata | RSS puncak |
|---|---|---|---|---|---|---|
| 1 | 279 | 495 | 23,8 s | 23,8 | 20,8 | 6,86 GiB |
| 3 | 827 | 1.685 | 96,4 s | 32,1 | 17,5 | 7,33 GiB |
| 5 (kurva) | 1.375 | 3.244 | 233,9 s | 46,8 | 13,9 | — |
| 10 | 2.745 | 8.148 | 1.194,6 s | 119,5 | 6,8 | 8,61 GiB |

Kurva tok/s sesaat terhadap panjang konteks (dokumen 5 halaman):

| Konteks (prompt + keluar) | ≈1.500 | ≈2.400 | ≈3.400 | ≈4.400 | ≈4.600 |
|---|---|---|---|---|---|
| Tok/s interval | 19,6 | 16,8 | 13,1 | 12,1 | 9,9 |

Penurunan kira-kira linier. Penyebab: di CPU attention memakai fallback dense yang membaca seluruh KV cache tiap token; di GPU kernel Triton block-sparse membuat biaya hampir konstan.

## Kualitas OCR yang diamati

- Teks sintetis Indonesia: hampir sempurna; kata "teks" hilang di satu pemenggalan baris, "dan" terbaca "of". Tabel, angka, alamat, nominal rupiah benar.
- Paper (Inggris, 2 kolom, rumus, tabel): abstrak kata demi kata benar, daftar isi dan nomor halaman benar, sitasi `[5, 6, 8, ...]` benar, heading Markdown benar setelah post-processing resmi.
- Format keluaran: token layout `<PAGE><BLOCK>label[bbox]<FORK>teks`, jumlah `<PAGE>` keluar selalu sama dengan jumlah halaman masuk.

## Kesimpulan dan rekomendasi

1. **CPU-only terbukti**: tidak perlu GPU untuk menjalankan model; patch 2 baris, tanpa backend alternatif.
2. **Kapasitas praktis di CPU**: sekitar 100–120 halaman per jam bila diproses per halaman atau per 3 halaman (perkiraan dari 24–32 s/halaman terukur). Cukup untuk uji akurasi dan volume kecil, bukan untuk produksi throughput tinggi.
3. **Setelan terbaik**: 12 thread, bf16, ≤3 halaman per request, `max_new_tokens` ≈4.000/halaman, `no_repeat_ngram_size=35`.
4. **Keterbatasan vs GPU**: decode ≈20 tok/s versus ratusan tok/s di GPU; throughput turun pada konteks panjang; tidak ada serving paralel.
5. **Saat RTX 5070 Ti 16 GB dipasang**: driver + torch CUDA, `model.cuda()`, patch tidak perlu dicabut; bobot 6,2 GB muat di VRAM.

## Belum diuji

Dokumen scan nyata dengan noise, bahasa selain Indonesia/Inggris, tulisan tangan, dua proses bersamaan, suhu CPU, perbandingan bit-per-bit dengan GPU.
