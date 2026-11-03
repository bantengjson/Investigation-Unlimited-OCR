# Baidu Unlimited OCR 2 di CPU — Dokumentasi Sistem

Proyek ini menjalankan model OCR **Baidu Unlimited OCR 2** (`baidu/Unlimited-OCR-2`, 3,34 miliar parameter) di server Ubuntu **tanpa GPU**, memakai PyTorch CPU build, dan mengukur kinerjanya. Tujuannya menilai kelayakan pemakaian sebelum GPU NVIDIA dibeli.

Status: **selesai dan terbukti berjalan** (3 November 2026). Rangkuman angka ada di [HASIL-PENGUJIAN.md](HASIL-PENGUJIAN.md); laporan rinci di [laporan-kelayakan-cpu-tahap1.md](laporan-kelayakan-cpu-tahap1.md) (investigasi kode) dan [laporan-hasil-pengujian-tahap2.md](laporan-hasil-pengujian-tahap2.md) (pengukuran).

---

## 1. Gambaran umum

| Hal | Nilai |
|---|---|
| Model | Unlimited-OCR-2: encoder visi SAM ViT-B + CLIP-L, decoder DeepSeek-V2 MoE 12 layer (64 expert, 6 aktif), sparse attention DASH |
| Bobot | 6,24 GiB, bfloat16 |
| Jalur inference yang dipakai | Transformers (`trust_remote_code`), **bukan** SGLang (SGLang hanya GPU) |
| Patch | 2 baris di `modeling_unlimitedocr.py` (`.cuda()` → `.to(<tensor>.device>)`) |
| Server | AMD Ryzen 9 9900X 12C/24T, 64 GB DDR5, Ubuntu Server 24.04.5, Python 3.12.3, tanpa GPU |
| Kinerja terukur | 1 halaman ≈ 24 s (20 tok/s); 3 halaman ≈ 31 s/halaman; RAM 7–9 GiB |
| Yang tidak dipasang | driver NVIDIA, CUDA, Triton, flash-attn, xformers, SGLang, Docker |

## 2. Struktur folder

### 2.1 Di PC (repositori ini)

```
Investigation-Unlimited-OCR/
├── README.md                           ← dokumen ini
├── HASIL-PENGUJIAN.md                  ← rangkuman hasil pengujian
├── laporan-kelayakan-cpu-tahap1.md     ← investigasi kode & kompatibilitas CPU
├── laporan-hasil-pengujian-tahap2.md   ← laporan pengukuran lengkap
├── phase2/                             ← paket skrip yang dijalankan di server (lihat §4)
├── results/                            ← salinan laporan JSON dan output OCR dari server
│   ├── reports/*.json, tps_curve.log
│   ├── test3_t12/, test3_t24/, test4_base/, test4_gundam/, test5/
├── Unlimited-OCR/                      ← clone repo GitHub baidu/Unlimited-OCR (referensi)
├── hf-model-code/v2/, v1/              ← salinan kode model dari Hugging Face (referensi, tanpa bobot)
└── sglang-wheel/                       ← isi wheel SGLang yang dibongkar (referensi)
```

### 2.2 Di server

```
/mnt/data/ai-projects/Unlimited-OCR/        ← proyek
├── .venv/                                  ← virtualenv (uv, Python 3.12.3 sistem), 1,3 GB
├── phase2/                                 ← skrip (sama dengan folder phase2 di PC)
├── samples/                                ← gambar/PDF uji + ground truth
│   ├── test_page.png, test_page.gt.txt
│   ├── paper.pdf, paper_gt_3p.txt, paper_gt_10p.txt, paper_pages/page_01..10.png
├── outputs/<nama_test>/                    ← hasil OCR per pengujian
├── reports/*.json                          ← metrik per pengujian
└── *.log                                   ← log setiap langkah

/mnt/data/ai-models/Unlimited-OCR/Unlimited-OCR-2/   ← bobot + kode model (dipatch)
├── model-0000{1,2}-of-00002.safetensors
├── modeling_unlimitedocr.py                ← sudah dipatch
├── modeling_unlimitedocr.py.orig           ← asli
└── CPU_PATCH_CHANGES.md                    ← log patch

~/.local/bin/uv                              ← satu-satunya tambahan di luar dua folder di atas
~/.cache/huggingface/modules/transformers_modules/Unlimited-OCR-2/   ← salinan kode model oleh transformers
```

## 3. Cara kerja sistem

### 3.1 Alur pemrosesan satu dokumen

```
PDF ──PyMuPDF 200 dpi──► PNG per halaman
                             │
                             ▼
              _uocr_build_inputs()  (modeling_unlimitedocr.py)
              ├─ gundam (1 halaman): pecah jadi ≤16 tile 640×640 + 1 global 1024×1024
              └─ base (multi-halaman): hanya global 1024×1024 per halaman (273 token/hal)
                             │  tensor gambar bf16 + input_ids ("<PAGE><image>...Multi page parsing.")
                             ▼
              Encoder visi (deepencoder.py)
              SAM ViT-B ──► CLIP-L ──► concat ──► projector linear 2048→1280
                             │  embedding visual disisipkan ke posisi token <image>  ← PATCH di sini
                             ▼
              Decoder DeepSeek-V2 MoE 12 layer (_sparse_base_impl.py + dash_attention.py)
              ├─ prefill: attention DASH block-sparse → fallback dense SDPA di CPU
              └─ decode token demi token (loop Python di _uocr_generate), KV cache DynamicCache
                 ├─ router DASH memilih 12 blok remote + jendela lokal 128 token
                 ├─ MoE: 6 dari 64 expert + 2 shared per token
                 └─ guard: no-repeat-ngram(35), page-count guard, max_new_tokens
                             │  teks dengan token layout
                             ▼
              "<PAGE><BLOCK>title<LEVEL1>[x1,y1,x2,y2]<FORK>Judul<BLOCK>text[...]<FORK>isi ..."
                             │
                ┌────────────┴─────────────┐
                ▼                          ▼
   _uocr_save_results()           process_unlimitedOCR2.py (resmi, stdlib)
   result.md, result_raw.md,      result_eval.md  (Markdown: heading, tabel HTML,
   gambar dengan kotak bbox        gambar sebagai ![](pageN_x1_y1_x2_y2.jpg))
                                  + strip_layout() di uocr_common → text_only.txt
```

### 3.2 Mengapa bisa jalan di CPU

Kode model ditulis device-agnostic kecuali dua baris. Semua kernel GPU punya gerbang `q.is_cuda` dan fallback PyTorch murni:

| Komponen GPU | Gerbang | Yang dipakai di CPU |
|---|---|---|
| Kernel Triton DASH (`sparse_gather_e2e_triton.py`) | `dash_attention.py:1342`, `modeling_deepseekv2.py:447` | `_forward_dense()`: SDPA dengan mask blok, chunk 2.048 query |
| Kernel Triton gather indexer (`sparse_gather_triton.py`) | `_sparse_base_impl.py:1748` | block-mask eager (matmul + softmax fp32) |
| Attention decode | selalu `_decode_attend()` | matmul fp32 (sudah PyTorch murni) |
| FlashAttention-2 | hanya jika paket ada dan diminta | eager (`SlidingWindowLlamaAttention`) |
| xformers / flex_attention | `use_xformers=false`, `use_flex_attention=false` di config | nonaktif |
| `torch.cuda.synchronize()` | hanya jika env `TIME_DIAG=1` | tidak dipanggil |
| `.cuda()` di `masked_scatter_` (2 baris) | **tidak ada gerbang** | **dipatch** menjadi `.to(<tensor>.device)` |

Konsekuensi jalur fallback: biaya decode per token bertambah sebanding panjang konteks (seluruh KV cache dibaca tiap token), sehingga throughput turun dari 20 tok/s (1,5 ribu token) ke 6,8 tok/s (11 ribu token). Di GPU, kernel Triton membuat biaya ini hampir konstan.

### 3.3 Presisi dan thread

- **bf16 wajib**: kode pra-pemrosesan meng-hardcode `.to(torch.bfloat16)` untuk tensor gambar. Ryzen 9 9900X punya AVX-512 BF16; terukur matmul bf16 3,09 TFLOPS vs fp32 1,60 TFLOPS.
- **12 thread (core fisik)** lebih cepat 15 % daripada 24 thread. Skrip memakai `torch.set_num_threads(12)` secara default (`psutil.cpu_count(logical=False)`), bisa diubah lewat `--threads` atau env `UOCR_THREADS`.
- Router DASH dan attention decode sudah dihitung fp32 oleh kode asli; tidak ada penurunan presisi tambahan.

### 3.4 Patch

```diff
--- a/modeling_unlimitedocr.py
+++ b/modeling_unlimitedocr.py
@@ -785 +785 @@
-  inputs_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1).cuda(), images_in_this_batch)
+  inputs_embeds[idx].masked_scatter_(images_seq_mask[idx].unsqueeze(-1).to(images_in_this_batch.device), images_in_this_batch)
@@ -789 +789 @@
-  inputs_embeds[0].masked_scatter_(images_seq_mask[0].unsqueeze(-1).cuda(), _packed_feats)
+  inputs_embeds[0].masked_scatter_(images_seq_mask[0].unsqueeze(-1).to(_packed_feats.device), _packed_feats)
```

Tidak mengubah bobot, arsitektur, maupun hasil; di GPU patch ini tetap benar sehingga tidak perlu dicabut. Diterapkan oleh `phase2/04_apply_cpu_patch.py` (idempoten, backup `.orig`, `--revert` untuk mengembalikan). Penting: patch diterapkan **sebelum** model dimuat pertama kali, karena transformers menyalin kode ke `~/.cache/huggingface/modules/`.

## 4. Skrip (`phase2/`)

| Skrip | Fungsi | Output |
|---|---|---|
| `00_check_server.sh` | Inventaris server tanpa perubahan: Python, venv, disk, RAM, flag CPU, GPU, folder yang ada, proses lain | stdout |
| `01_setup_env.sh` | Buat `.venv` (venv standar, atau uv bila `ensurepip` tidak ada) dan pasang dependency CPU dari index `download.pytorch.org/whl/cpu` | `.venv/` |
| `02_test_env.py` | Test 1: import, `cuda.is_available()==False`, oneDNN, benchmark matmul fp32/bf16, SDPA | stdout |
| `03_download_model.py` | Unduh `baidu/Unlimited-OCR-2` ke `/mnt/data/ai-models/Unlimited-OCR/Unlimited-OCR-2` (tanpa assets/wheel/pdf) | folder model |
| `04_apply_cpu_patch.py` | Terapkan/revert patch 2 baris | `.orig`, `CPU_PATCH_CHANGES.md` |
| `05_test_load.py` | Test 2: muat model, ukur waktu, RSS, device, dtype | `reports/test2_load.json` |
| `06_make_test_image.py` | Buat gambar uji A4 150 dpi dengan teks diketahui + `.gt.txt` | `samples/` |
| `07_test_single_image.py` | Test 3: OCR satu gambar; `--mode auto/gundam/base`, `--threads`, `--max-new-tokens`, `--gt`, `--stream` | `outputs/<out>/`, `reports/test3_*.json` |
| `08_test_pdf.py` | Test 4: PDF → PNG (PyMuPDF) → OCR; default base 1 request, `--per-page` gundam per halaman, `--max-pages`, `--dpi` | `outputs/<out>/`, `reports/test4_*.json` |
| `09_stress.py` | Test 5: tahap 1→3→10 halaman, berhenti bila swap > batas | `reports/test5_stress.json` |
| `ocr.py` | **Perintah pemakaian sehari-hari**: OCR satu gambar atau PDF; PDF dipecah 3 halaman per request (base) atau `--per-page` gundam; `--pages 1-5`, `--dpi`, `--threads`, `--stream` | `outputs/<nama_file>/hasil.md`, `hasil.txt`, `hasil_raw.txt`, `ringkasan.json` |
| `export_hasil.py` | Ekspor keluaran mentah ke `hasil.xlsx`, `hasil.docx`, `hasil_searchable.pdf`, `hasil_blok.json`; dipanggil otomatis oleh `ocr.py` atau manual pada folder hasil | file-file tersebut |
| `uocr_common.py` | `load_model()`, `Monitor` (RSS puncak, CPU %, swap), `similarity()`, `strip_layout()` | — |
| `process_unlimitedOCR2.py` | Salinan post-processing resmi Baidu (raw → Markdown) | — |
| `cpu.patch` | Diff patch untuk referensi | — |
| `README-LANGKAH.md` | Urutan perintah langkah demi langkah | — |

Semua laporan JSON berisi: token prompt/keluar, `prefill_s`, `decode_s`, `decode_tok_per_s`, `rss_peak_gib`, `cpu_pct_of_all_cores_avg`, `swap_used_peak_gib`, dan `similarity_to_gt` bila ground truth diberikan.

## 5. Alur kerja (flow) dari nol sampai hasil

```
[0] 00_check_server.sh      → pastikan ensurepip/uv, disk, RAM, tidak ada GPU
[1] 01_setup_env.sh         → .venv + torch cpu + transformers        (≈3 menit, 400 MB unduhan)
[2] 02_test_env.py          → Test 1 lulus                            (≈30 detik)
[3] 03_download_model.py    → bobot 6,24 GiB                          (≈3 menit)
[4] 04_apply_cpu_patch.py   → patch 2 baris, .orig tersimpan          (instan)
[5] 05_test_load.py         → Test 2: model di RAM, device=cpu        (≈2 detik)
[6] 06_make_test_image.py   → gambar uji + ground truth               (instan)
[7] 07_test_single_image.py → Test 3: 1 halaman, metrik lengkap       (≈25 detik)
[8] 08_test_pdf.py          → Test 4: PDF 3 halaman                   (≈1,5–2 menit)
[9] 09_stress.py            → Test 5: 1/3/10 halaman                  (≈22 menit)
```

Perintah lengkap tiap langkah ada di [phase2/README-LANGKAH.md](phase2/README-LANGKAH.md). Semua dijalankan dari `/mnt/data/ai-projects/Unlimited-OCR` dengan `source .venv/bin/activate`.

## 6. Pemakaian sehari-hari di CPU

Cara paling sederhana: `phase2/ocr.py`, menerima gambar atau PDF dan langsung menghasilkan Markdown.

```bash
cd /mnt/data/ai-projects/Unlimited-OCR && source .venv/bin/activate && python phase2/ocr.py /path/dokumen.pdf
```

Varian: `--pages 1-5` (rentang halaman), `--per-page` (tiap halaman mode gundam, lebih detail), `--stream` (lihat teks keluar langsung), `--out folder` (lokasi hasil), `--no-export` (lewati Excel/Word/PDF). Terukur: 2 halaman paper dalam 44 s.

File hasil di `outputs/<nama_file>/`:

| File | Isi | Dibuat oleh |
|---|---|---|
| `hasil.md` | Markdown terstruktur: heading, tabel HTML, penanda gambar | post-processing resmi Baidu |
| `hasil.txt` | teks polos tanpa token layout | `uocr_common.strip_layout` |
| `hasil.xlsx` | Excel: sheet `Blok` (halaman, label, bbox, isi setiap blok) + satu sheet per tabel yang terdeteksi | `export_hasil.py` (openpyxl) |
| `hasil.docx` | Word: heading sesuai level, paragraf, tabel, potongan gambar dari bbox | `export_hasil.py` (python-docx) |
| `hasil_searchable.pdf` | PDF berisi gambar halaman asli ditambah lapisan teks tak terlihat pada posisi bbox, sehingga bisa dicari dan di-copy | `export_hasil.py` (PyMuPDF, `render_mode=3`) |
| `hasil_blok.json` | daftar blok per halaman: label, level, bbox (skala 0–999), isi, tabel sebagai array | `export_hasil.py` |
| `hasil_raw.txt` | keluaran mentah model | — |
| `ringkasan.json` | waktu, token, tok/s, RAM, CPU, daftar gambar halaman | — |
| `grup_N/result_with_boxes_*.jpg` | halaman dengan kotak deteksi | kode model |

Ekspor bisa dijalankan ulang sendiri pada folder hasil mana pun: `python phase2/export_hasil.py outputs/<nama_file>`. Koordinat bbox model berskala 0–999 dan dipetakan ke piksel dengan `x/999 × lebar`, sama seperti kode penggambar kotak milik Baidu. Catatan: lapisan teks PDF memakai font Helvetica, cukup untuk Latin; aksara CJK butuh font lain.

Skrip pengujian di bawah ini tetap bisa dipakai bila ingin metrik lebih rinci.

Contoh OCR satu halaman gambar:

```bash
cd /mnt/data/ai-projects/Unlimited-OCR && source .venv/bin/activate && python phase2/07_test_single_image.py /path/gambar.png --threads 12 --out outputs/dokumen_a
```

Contoh OCR PDF, 3 halaman per request (setelan yang disarankan di CPU):

```bash
cd /mnt/data/ai-projects/Unlimited-OCR && source .venv/bin/activate && python phase2/08_test_pdf.py /path/dokumen.pdf --max-pages 3 --dpi 200 --threads 12 --out outputs/dokumen_b
```

Hasil yang bisa dipakai: `outputs/<out>/result_eval.md` (Markdown terstruktur), `text_only.txt` (teks polos), `result_with_boxes_*.jpg` (visualisasi bbox).

Pemakaian dari Python sendiri:

```python
import sys; sys.path.insert(0, "phase2")
from uocr_common import load_model
tok, model, _ = load_model(threads=12)
text = model.infer(tok, image_file=["p1.png", "p2.png", "p3.png"], crop_mode=False,
                   eval_mode=True, max_new_tokens=12000, no_repeat_ngram_size=35, ngram_window=1024)
```

Setelan yang disarankan di CPU:

| Setelan | Nilai | Alasan |
|---|---|---|
| threads | 12 | core fisik; 24 lebih lambat |
| dtype | bf16 | wajib oleh kode, dan tercepat |
| halaman per request | 1 (gundam) atau ≤3 (base) | menjaga 15–20 tok/s; konteks > 4 ribu token menurunkan throughput |
| `max_new_tokens` | ≈4.000 per halaman | pengaman bila model tidak berhenti |
| `no_repeat_ngram_size` / `ngram_window` | 35 / 128 (1 hal) atau 1024 (multi) | setelan deployment Baidu |
| proses paralel | 1 | semua proses berbagi 12 core yang sama |

## 6a. Format masukan yang didukung

Model hanya menerima **gambar raster** (piksel) yang dibaca lewat Pillow. Ia tidak membaca teks di dalam PDF, struktur Word, atau sel Excel; ia melihat tampilannya saja. Apa pun formatnya harus menjadi gambar dulu, dan `ocr.py` melakukan itu untuk PDF.

| Format | Status di `ocr.py` | Cara |
|---|---|---|
| PNG, JPG/JPEG, BMP, TIFF satu halaman, WebP | langsung | Pillow membaca, dikirim ke model |
| PDF (scan maupun PDF teks) | langsung | PyMuPDF merender tiap halaman ke PNG (`--dpi`, default 200), lalu per halaman ke model |
| TIFF multi-halaman | hanya halaman pertama | pecah per halaman dulu |
| GIF | hanya frame pertama | sama |
| HEIC/HEIF (foto iPhone) | tidak | butuh paket `pillow-heif` |
| Word (.docx), Excel (.xlsx), PowerPoint (.pptx) | tidak langsung | konversi ke PDF dulu, mis. `soffice --headless --convert-to pdf file.docx` |
| HTML, teks, Markdown | tidak relevan | sudah berupa teks, tidak perlu OCR |

Yang sudah diuji: PNG (gambar sintetis) dan PDF (paper 15 halaman). JPG, TIFF, dan WebP ditangani Pillow dengan cara yang sama tetapi belum dijalankan.

Hal yang mempengaruhi hasil, apa pun formatnya:

- **Resolusi**: model bekerja pada 1024 piksel (global) dan tile 640 piksel. A4 pada 150–300 dpi sudah cukup; foto beresolusi sangat tinggi tidak menambah akurasi, hanya waktu resize.
- **Orientasi**: pembaca gambar memakai `exif_transpose`, jadi foto HP yang tersimpan terputar dibetulkan. Miring beberapa derajat biasanya masih terbaca; terbalik 180° tidak dikoreksi otomatis.
- **Warna**: semua gambar dikonversi ke RGB; hitam putih, grayscale, dan berwarna diperlakukan sama.
- **Ukuran kertas**: tidak ada batasan; tiling gundam menyesuaikan rasio aspek sampai 16 tile.

## 7. Pemantauan

Di terminal kedua saat inference berjalan:

```bash
vmstat 5
```

```bash
free -h
```

Skrip sendiri sudah mencatat RSS puncak dan CPU % ke JSON, jadi pemantauan manual hanya perlu untuk melihat aplikasi lain di server.

## 8. Troubleshooting

| Gejala | Penyebab | Solusi |
|---|---|---|
| `Torch not compiled with CUDA enabled` | patch belum diterapkan, atau model dimuat sebelum patch sehingga cache transformers berisi kode lama | jalankan `04_apply_cpu_patch.py`, lalu hapus `~/.cache/huggingface/modules/transformers_modules/Unlimited-OCR-2/` |
| `No module named 'ensurepip'` saat `python3 -m venv` | paket `python3.12-venv` tidak ada | `01_setup_env.sh` otomatis memakai uv; pasang uv tanpa sudo: `curl -LsSf https://astral.sh/uv/install.sh \| env UV_NO_MODIFY_PATH=1 sh` |
| pip menarik `nvidia-*` dan `triton` | torch dipasang dari PyPI | pasang dengan `--index-url https://download.pytorch.org/whl/cpu` |
| dtype mismatch bf16/fp32 | model dimuat fp32 | muat dengan `torch_dtype=torch.bfloat16` |
| Throughput turun di dokumen panjang | fallback attention CPU membaca seluruh KV tiap token | pecah dokumen ≤3 halaman per request |
| Output tidak berhenti / sangat panjang | halaman sangat padat atau loop | pakai `max_new_tokens`, `no_repeat_ngram_size=35`; page-count guard sudah aktif |
| `import fitz` gagal | PyMuPDF 2.x | skrip sudah memakai `import pymupdf as fitz` |

## 9. Mengembalikan dan membersihkan

- Cabut patch: `python phase2/04_apply_cpu_patch.py --revert`
- Bersih total (hanya dua folder ini yang dibuat): hapus `/mnt/data/ai-projects/Unlimited-OCR` dan `/mnt/data/ai-models/Unlimited-OCR`; opsional `~/.local/bin/uv`, `~/.local/bin/uvx`, dan `~/.cache/huggingface/modules/transformers_modules/Unlimited-OCR-2`.

## 10. Migrasi ke GPU (RTX 5070 Ti 16 GB) nanti

1. Pasang driver NVIDIA dan torch build CUDA (cu13 untuk Blackwell) di venv terpisah.
2. Patch tidak perlu dicabut; `.to(device)` benar di GPU.
3. `model = AutoModel.from_pretrained(..., torch_dtype=torch.bfloat16).eval().cuda()`; kernel Triton aktif otomatis karena `q.is_cuda` menjadi True, sehingga penurunan tok/s pada konteks panjang hilang.
4. Bobot 6,2 GB muat di 16 GB VRAM. Untuk SGLang pakai wheel di repo dengan `--mem-fraction-static 0.7` dan `--max-running-requests` lebih kecil dari contoh A800-80GB; periksa dukungan sm_120 pada flashinfer/flash-attn-4 saat itu.

## 11. Penjelasan library dan komponen

Bagian ini menjelaskan setiap library dan istilah yang muncul di proyek ini, dikelompokkan menurut perannya, supaya keputusan "mana yang dipakai di CPU dan mana yang hanya untuk GPU" bisa dipahami.

### 11.1 Framework dasar (dipakai di CPU)

| Library | Peran | Catatan |
|---|---|---|
| **PyTorch (torch)** | Library komputasi tensor tempat model ditulis: perkalian matriks, konvolusi, softmax, autograd. Ada build CUDA (GPU NVIDIA) dan build CPU. | Terpasang `torch 2.13.0+cpu`. Di CPU PyTorch memakai **oneDNN** (dulu MKL-DNN), library Intel yang memanfaatkan AVX-512 dan AVX-512 BF16; sebabnya bf16 terukur 2× lebih cepat dari fp32. |
| **torchvision** | Pelengkap PyTorch untuk gambar. | Model hanya memakai transformasi resize dan normalisasi. |
| **Transformers (Hugging Face)** | Memuat model dari Hugging Face Hub: membaca `config.json`, mengunduh bobot, menjalankan kode model. | `trust_remote_code=True` berarti kode model (10 file Python milik Baidu) ikut diunduh dari repo HF. Ini jalur yang dipakai di CPU. |
| **tokenizers** | Memecah teks menjadi token; ditulis dalam Rust agar cepat. | Dipakai `LlamaTokenizerFast`. |
| **huggingface-hub** | Mengunduh file dari Hub. | Dipakai `03_download_model.py`. |
| **safetensors** | Format file bobot yang aman dan bisa di-memory-map. | Sebabnya model "dimuat" dalam 1,4 s; bobot masuk RAM saat inference pertama. |
| **accelerate** | Dipakai Transformers untuk memuat model tanpa alokasi memori ganda (`low_cpu_mem_usage`). | Opsional; skrip memakainya bila ada. |
| **einops, addict, easydict** | Library kecil untuk manipulasi bentuk tensor dan konfigurasi bergaya dictionary. | Dipakai kode Baidu. |
| **pillow** | Membaca dan menggambar gambar. | Pra-pemrosesan dan visualisasi bbox. |
| **PyMuPDF (pymupdf / fitz)** | Merender halaman PDF menjadi gambar. | Model tidak membaca PDF langsung. |
| **psutil** | Mengukur RAM, CPU, swap. | Ditambahkan untuk pengukuran, bukan dependency model. |

### 11.2 Mesin serving: SGLang (hanya GPU)

**SGLang** adalah server inference untuk model bahasa besar, sekelas dengan vLLM. Tugasnya menerima banyak permintaan HTTP sekaligus, menjadwalkan batch, mengelola KV cache di VRAM, dan menyajikan API kompatibel OpenAI. Baidu menyediakan wheel SGLang kustom berisi implementasi khusus Unlimited OCR 2: backend attention `unlimited_ocr_v2`, pemecahan dokumen per jendela halaman, dan logit processor anti-pengulangan.

Mengapa tidak bisa di CPU: wheel itu mewajibkan `cuda-python`, `flashinfer`, `flash-attn-4`, `sgl-kernel`, dan paket NVIDIA lain; backend attention untuk model ini seluruhnya ditulis dalam Triton; dan backend CPU generik yang ada di SGLang (`intel_amx_backend`) ditolak oleh override konfigurasi model. SGLang baru relevan setelah GPU terpasang, terutama bila dibutuhkan banyak permintaan paralel.

### 11.3 Kernel GPU khusus (akselerasi, bukan syarat kebenaran)

"Kernel" di sini berarti program kecil yang berjalan langsung di ribuan core GPU. Semua komponen berikut adalah cara mempercepat attention di GPU; kode Baidu menyediakan jalur alternatif PyTorch murni untuk masing-masing.

| Komponen | Apa itu | Di CPU |
|---|---|---|
| **Triton** | Bahasa dan compiler dari OpenAI untuk menulis kernel GPU dengan sintaks mirip Python, tanpa CUDA C++. Baidu menulis kernel attention block-sparse DASH dalam Triton (`sparse_gather_triton.py`, `sparse_gather_e2e_triton.py`); kernel ini hanya membaca blok KV yang dipilih router sehingga biaya per token hampir konstan. | Tidak bisa jalan; kode otomatis beralih ke fallback PyTorch yang membaca seluruh KV. Inilah sumber penurunan dari 20 ke 6,8 tok/s pada dokumen panjang. |
| **FlashAttention (flash-attn)** | Algoritma attention yang menghindari penyimpanan matriks attention berukuran kuadrat dengan menghitungnya per blok di SRAM GPU; lebih hemat memori dan lebih cepat. Hanya untuk GPU NVIDIA. | Tidak terpasang; jalur `eager` (implementasi standar PyTorch) dipakai. |
| **xformers** | Library Meta berisi kernel attention hemat memori untuk GPU. | `use_xformers=false` di config; tidak dipakai bahkan di GPU. |
| **flex_attention** | Fitur PyTorch ≥ 2.5 yang mengompilasi pola attention kustom (mis. mask block-sparse) menjadi kernel efisien. | `use_flex_attention=false` di config; tidak dipakai. |
| **SDPA (scaled dot product attention)** | Fungsi attention bawaan PyTorch yang memilih implementasi terbaik yang tersedia; di GPU bisa memakai FlashAttention internal, di CPU memakai kernel CPU PyTorch. | Dipakai oleh encoder SAM/CLIP dan fallback DASH; terbukti berjalan. |

### 11.4 Istilah NVIDIA

| Istilah | Arti | Relevansi |
|---|---|---|
| **CUDA** | Platform komputasi GPU NVIDIA; build CUDA PyTorch butuh driver NVIDIA dan runtime CUDA. | Tidak dipasang sekarang; wajib setelah GPU ada. |
| **cuDNN** | Library NVIDIA untuk konvolusi dan operasi jaringan saraf. | Ikut terpasang bersama torch CUDA nanti. |
| **cuBLAS** | Library NVIDIA untuk perkalian matriks. | Sama. |
| **TensorRT** | Compiler inference NVIDIA yang mengoptimalkan model untuk GPU tertentu. | Tidak dipakai model ini sama sekali. |
| **CUDA graph** | Teknik merekam urutan kernel sekali lalu memutarnya ulang untuk mengurangi overhead peluncuran (muncul di flag SGLang). | Hanya GPU. |
| **CUDA core** | Unit komputasi di dalam GPU NVIDIA. | Bukan fitur CPU; tidak bisa "dialihkan" ke CPU dengan konfigurasi. |

### 11.5 Komponen di dalam model

| Komponen | Penjelasan |
|---|---|
| **SAM ViT-B** | Encoder gambar dari Segment Anything (Meta), varian ViT-Base. Mengekstrak fitur detail dari gambar 1024 piksel. |
| **CLIP-L** | Encoder gambar OpenAI yang menangkap makna semantik. Fiturnya digabung dengan SAM lalu diproyeksikan ke dimensi decoder. |
| **DeepSeek-V2 MoE** | Arsitektur decoder bahasa. **MoE (Mixture of Experts)**: lapisan feed-forward dipecah menjadi 64 expert, hanya 6 aktif per token. Total 3,34 miliar parameter, tetapi hanya ≈0,6 miliar dihitung per token; sebabnya CPU masih mencapai 20 tok/s. |
| **DASH / SR-SWA** | Inovasi Unlimited OCR 2: router memilih 12 blok KV relevan ditambah jendela lokal 128 token, sehingga perhatian model tetap terbatas berapa pun panjang dokumen. Di GPU diwujudkan kernel Triton; di CPU oleh fallback dense yang benar hasilnya tetapi lebih lambat. |
| **KV cache** | Penyimpanan key dan value attention dari semua token sebelumnya agar tidak dihitung ulang tiap langkah decode. ≈60 KiB per token untuk model ini. |
| **Prefill / decode** | Prefill: memproses seluruh prompt (gambar + instruksi) sekali. Decode: menghasilkan token keluaran satu per satu; bagian yang mendominasi waktu di CPU. |
| **Gundam / base** | Dua mode encoding halaman. Gundam: ≤16 tile 640×640 ditambah global 1024×1024 (≈900–1.900 token/halaman, lebih detail). Base: hanya global (273 token/halaman, dipakai untuk multi-halaman). |
| **bf16 (bfloat16)** | Format angka 16-bit dengan rentang eksponen sama seperti fp32 tetapi presisi lebih rendah. Hemat memori setengahnya dan 2× lebih cepat di CPU ber-AVX-512 BF16. Model dilatih dan disimpan dalam bf16. |

### 11.6 Ringkasan: mana yang dipakai di mana

| Komponen | Peran | Di CPU sekarang | Setelah GPU |
|---|---|---|---|
| PyTorch CPU + oneDNN | mesin komputasi | dipakai | diganti build CUDA |
| Transformers | memuat dan menjalankan model | dipakai | tetap dipakai |
| SGLang | server inference paralel | tidak bisa | opsional, untuk throughput |
| Triton | kernel DASH block-sparse | fallback dense | aktif otomatis |
| FlashAttention / xformers / flex | attention cepat GPU | tidak dipakai | opsional |
| CUDA / cuDNN / driver | runtime GPU | tidak dipasang | wajib |
| TensorRT | compiler GPU | tidak relevan | tidak relevan |

Intinya: semua yang berupa kernel GPU adalah akselerasi, bukan syarat kebenaran. Kode Baidu memberi jalan alternatif untuk tiap komponen tersebut, dan patch 2 baris menutup satu-satunya celah yang tidak punya jalan alternatif.

## 12. Referensi

- Repo: https://github.com/baidu/Unlimited-OCR
- Model: https://huggingface.co/baidu/Unlimited-OCR-2
- Paper: https://arxiv.org/abs/2610.39308 (salinan di `Unlimited-OCR/Unlimited-OCR-2.pdf`, dipakai sebagai dokumen uji)
