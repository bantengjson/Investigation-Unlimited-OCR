# Laporan Kelayakan CPU-Only — Baidu Unlimited OCR 2

Tanggal: 2026-11-03
Tahap: 1 (investigasi, belum ada instalasi)
Sumber yang dianalisis:
- GitHub `baidu/Unlimited-OCR` (commit `ea42e6d`)
- Hugging Face `baidu/Unlimited-OCR-2` (kode model + config, tanpa bobot)
- Hugging Face `baidu/Unlimited-OCR` (v1, pembanding)
- Wheel `sglang-0.5.20.dev13+g75b9b06b02` yang dibundel di repo

Keterangan label:
- **[TERUKUR]** = dibaca langsung dari kode, config, metadata, atau index PyPI.
- **[ESTIMASI]** = dihitung dari arsitektur; harus diverifikasi pada Test 3.

---

## 0. Kesimpulan

**STATUS: PARTIALLY SUPPORTED**

Unlimited OCR 2 dapat dijalankan sepenuhnya di CPU dengan PyTorch CPU build, **tanpa CUDA, tanpa driver NVIDIA, tanpa Triton**, dengan syarat:

1. Jalur yang dipakai adalah jalur **Transformers (HF)**, bukan SGLang. Jalur SGLang murni GPU.
2. Diperlukan **patch minimal 2 baris** pada `modeling_unlimitedocr.py` (dua pemanggilan `.cuda()` yang hard-coded). Semua kernel GPU lainnya sudah punya fallback PyTorch murni yang aktif otomatis saat tensor bukan CUDA.
3. Model harus dimuat dalam **bfloat16** (kode pra-pemrosesan gambar meng-hardcode bf16). Ryzen 9 9900X mendukung AVX-512 BF16, jadi ini jalur yang tepat.
4. Performa akan jauh di bawah GPU. Perkiraan kasar 5–15 token/detik saat decode; satu halaman padat bisa memakan beberapa menit. Angka pasti baru diketahui setelah Test 3.

---

## 1. Struktur repository dan di mana kode model sebenarnya berada

Repo GitHub **tidak berisi kode model**. Isinya hanya:

| File | Fungsi | Relevan CPU? |
|---|---|---|
| `README.md` | Dokumentasi dua jalur inference: Transformers dan SGLang | Ya |
| `infer.py` | Client HTTP batch untuk benchmark; butuh server SGLang dan `import sglang` | Tidak |
| `utils/process_unlimitedOCR2.py` | Post-processing output mentah ke Markdown; stdlib murni | Ya (bisa dipakai) |
| `wheel/sglang-*.whl` | SGLang kustom 15 MB | Tidak (GPU) |
| `Unlimited-OCR-2.pdf`, `assets/` | Paper dan gambar | Referensi |

Kode model dimuat lewat `trust_remote_code=True` dari Hugging Face. Totalnya 10.225 baris Python dalam 10 file [TERUKUR]:

| File HF | Baris | Isi |
|---|---|---|
| `_sparse_base_impl.py` | 4.471 | Decoder DeepSeek-V2 MoE + sparse attention (indexer) |
| `dash_attention.py` | 1.349 | Router DASH (entmax top-k block) + fallback dense |
| `modeling_unlimitedocr.py` | 1.228 | Wrapper VLM, pra-pemrosesan gambar, `infer()` / `infer_multi()` |
| `deepencoder.py` | 1.055 | Encoder visi: SAM ViT-B + CLIP-L + projector |
| `modeling_deepseekv2.py` | 531 | Overlay jalur inference (prefill/decode DASH) |
| `sparse_gather_triton.py` | 524 | Kernel Triton (GPU) — opsional |
| `sparse_gather_e2e_triton.py` | 407 | Kernel Triton (GPU) — opsional |
| `conversation.py`, `configuration_deepseek_v2.py`, `packing_patch.py` | 660 | Util |

## 2. Arsitektur dan ukuran model [TERUKUR dari config.json dan model.safetensors.index.json]

| Parameter | Nilai |
|---|---|
| Arsitektur | `UnlimitedOCRForCausalLM` = SAM ViT-B + CLIP-L → projector linear (2048→1280) → decoder DeepSeek-V2 MoE |
| Decoder | 12 layer, hidden 1280, 10 head × 128 dim (MHA, bukan MLA) |
| MoE | 64 expert routed, top-6 aktif + 2 shared expert, intermediate 896; layer 0 dense (intermediate 6848) |
| Vocab | 129.280 |
| Konteks maksimum | 131.072 (YaRN, factor 4 dari 32.768) |
| Sparse attention | DASH: block 32, top-12 block remote + jendela lokal; indexer + router per layer |
| Ukuran bobot | 6.687.958.536 byte = **6,23 GiB** dalam bf16 |
| Jumlah parameter | **≈ 3,34 miliar** (6,69 GB / 2 byte) |
| Parameter aktif per token | ≈ 0,55–0,6 miliar [ESTIMASI dari config: 1 dense MLP + 11 × (6+2 expert) + attention + lm_head] |
| Token visi per halaman | Base mode: 273. Gundam mode: 273 + (10·w+1)·(10·h) dengan w×h ≤ 16 tile 640 px; untuk 4×4 tile = 1.913 token |

## 3. Jawaban pertanyaan teknis bagian A (kompatibilitas CPU)

| # | Pertanyaan | Jawaban |
|---|---|---|
| 1 | Mendukung CPU inference? | Tidak secara resmi (README hanya menguji A800-80GB), tetapi kodenya **device-agnostic** kecuali 2 baris. |
| 2 | Bisa dimuat dengan PyTorch CPU? | Ya. `AutoModel.from_pretrained(..., torch_dtype=torch.bfloat16)` tanpa `.cuda()`. |
| 3 | Ketergantungan CUDA? | Jalur HF: hanya 2 baris `.cuda()`. Jalur SGLang: total (lihat bagian 5). |
| 4 | Ada `.cuda()` / `.to("cuda")` eksplisit? | Ya: `modeling_unlimitedocr.py:785` dan `:789` (mask untuk `masked_scatter_`). Hanya itu. |
| 5 | CUDA custom / Triton / FlashAttention? | Ada Triton (2 file) dan FlashAttention2 opsional, **semuanya di-gate** oleh `q.is_cuda` atau ketersediaan paket; di CPU otomatis ke fallback PyTorch. |
| 6 | Semua op bisa di CPU? | Ya: Linear, Conv2d, LayerNorm/RMSNorm, SDPA, einsum, softmax, argsort/scatter (MoE), interpolate bicubic (di-cast ke fp32 oleh kode). Tidak ada op kustom. |
| 7 | Backend alternatif? | Tidak diperlukan. ONNX **tidak realistis**: routing MoE dinamis, router entmax, state cache kustom (`_dash_idx_k`, `_uocr_pg`), dan loop decode manual di Python. |
| 8 | Butuh BF16/FP16/FP32? | Checkpoint bf16. Kode meng-hardcode `.to(torch.bfloat16)` untuk tensor gambar (`modeling_unlimitedocr.py:561,566,579`). Router dan attention decode sudah dihitung fp32 oleh kode. |
| 9 | 9900X mendukung format itu? | Ya. Zen 5 punya AVX-512 BF16, dipakai oneDNN untuk matmul/conv bf16. |
| 10 | Python 3.12.3 kompatibel? | Ya, semua paket ada wheel-nya (bagian 6). |

## 4. Pertanyaan bagian B (kebutuhan resource)

| Item | Nilai | Dasar |
|---|---|---|
| Storage bobot | 6,23 GiB + tokenizer 10 MB | [TERUKUR] |
| Storage venv | ≈ 1,5–2 GB (torch CPU + deps) | [ESTIMASI] |
| RAM saat model dimuat | ≈ 6,3 GiB bobot + 1–2 GiB overhead runtime ≈ 8–9 GiB | [ESTIMASI]; gunakan `low_cpu_mem_usage=True` agar tidak ada puncak 2× saat load |
| KV cache | 12 layer × 2 × 10 head × 128 × 2 byte = **60 KiB/token**. 5k token ≈ 0,3 GiB; 131k token ≈ 7,5 GiB | [TERUKUR dari config] |
| Memori transien attention prefill (fallback dense) | Chunk 2.048 query × S key × 10 head: S=20k ≈ 1,6 GB; S=131k ≈ 10,7 GB | [ESTIMASI dari `_forward_dense`] |
| Total RAM realistis 1–10 halaman | 10–14 GiB | [ESTIMASI] |
| Swap | Tidak diharapkan terpakai (57 GiB tersedia) | — |
| CPU | PyTorch akan memakai semua thread; diuji 12 vs 24 thread | — |
| Waktu per halaman | Lihat bagian 7 | [ESTIMASI] |
| Beberapa proses paralel | Secara memori muat 3–4 proses, tetapi semuanya berebut 12 core yang sama; tidak menambah throughput. Satu proses dengan thread penuh lebih masuk akal. | [ESTIMASI] |

## 5. Pertanyaan bagian C (pemetaan ketergantungan GPU)

### 5.1 Wajib GPU — tidak bisa dipakai di CPU

| Komponen | Bukti |
|---|---|
| Seluruh jalur **SGLang** | Wheel mewajibkan `cuda-python>=13`, `flashinfer_python[cu13]`, `flash-attn-4`, `sglang-kernel`, `nvidia-cutlass-dsl[cu13]`, `sgl-deep-gemm`, `tilelang`, dll. Backend `unlimited_ocr_v2_backend.py` seluruhnya Triton (`attend_triton.py`, `route_triton.py`, `decode_meta_triton.py`). `model_overrides/unlimited_ocr_v2.py` menolak backend attention selain `unlimited_ocr_v2`, sehingga backend `intel_amx_backend.py` yang ada di wheel pun tidak bisa dipakai untuk model ini. |
| `infer.py` di repo | Hanya client HTTP ke server SGLang; `import sglang` di baris 15. |

### 5.2 Bisa diganti CPU — butuh patch minimal

| Lokasi | Kode asli | Pengganti | Dampak |
|---|---|---|---|
| `modeling_unlimitedocr.py:785` | `images_seq_mask[idx].unsqueeze(-1).cuda()` | `.to(inputs_embeds.device)` | Nol; hanya memindahkan mask ke device yang sama dengan embedding |
| `modeling_unlimitedocr.py:789` | `images_seq_mask[0].unsqueeze(-1).cuda()` | `.to(inputs_embeds.device)` | Nol |
| README (kode pengguna) | `.eval().cuda()` | `.eval()` | Bukan patch repo |

Tidak ada perubahan bobot maupun arsitektur. Hasil OCR tetap dapat dibandingkan 1:1 dengan GPU karena jalur fallback dense dirancang penulis sebagai "validation reference" untuk kernel Triton (komentar di `dash_attention.py:1229`).

### 5.3 Opsional — fallback otomatis, tanpa patch

| Komponen | Gate | Fallback CPU |
|---|---|---|
| `sparse_gather_e2e_triton` (attention DASH prefill) | `dash_attention.py:1342`, `modeling_deepseekv2.py:447`: `self.use_triton and q.is_cuda` | `_forward_dense()`: SDPA dengan mask block, chunk 2.048 query |
| `sparse_gather_triton` (gather attention indexer) | `_sparse_base_impl.py:1748`: `q.is_cuda` | Block-mask eager (matmul + softmax fp32) |
| `triton_block_max_score` | `_sparse_base_impl.py:3322`: `idx_q.is_cuda` | Versi eager |
| Attention decode | selalu `_decode_attend()` matmul fp32 | Sudah PyTorch murni |
| FlashAttention-2 | `is_flash_attn_2_available()` + `attn_implementation="flash_attention_2"`; default `eager` → `SlidingWindowLlamaAttention` (`_sparse_base_impl.py:3627`) | Eager |
| xformers / flex_attention | `config.sparse_attention_config.use_xformers=false`, `use_flex_attention=false` | Nonaktif |
| `torch.cuda.synchronize()` | `_sparse_base_impl.py:88–112`, hanya jika env `TIME_DIAG=1` (default 0) | Tidak dipanggil |
| `torch.compile` | hanya di `build_sam_fast_vit_b` yang tidak dipakai | — |
| Import Triton | `modeling_unlimitedocr.py:12–36` dibungkus `try/except`; impor lain lazy di dalam gate `is_cuda` | Paket `triton` tidak perlu dipasang |

### 5.4 Tidak relevan untuk CPU-only

`torch.distributed` / `ep_size` (bernilai 1), logging training (`DASH_LOG_*`, KL loss), `PYTORCH_CUDA_ALLOC_CONF`, semua flag CUDA-graph SGLang, cuDNN, TensorRT (tidak dipakai sama sekali).

### 5.5 Catatan tentang v1 (Unlimited-OCR 1)

v1 justru **lebih sulit** di CPU: 14 pemanggilan `.cuda()` dan `torch.autocast("cuda")` di `modeling_unlimitedocr.py` (baris 581, 998–1065, 1233–1254). Jadi v2 adalah target yang tepat.

## 6. Kompatibilitas dependency dengan Python 3.12.3 [TERUKUR dari PyPI dan index PyTorch]

| Paket (versi yang diuji Baidu) | Wheel untuk cp312 Linux x86_64 | Catatan |
|---|---|---|
| `torch==2.13.0` | `torch-2.13.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl` di `download.pytorch.org/whl/cpu` | **Wajib pakai index CPU.** Versi PyPI menarik `nvidia-cudnn-cu13`, `nvidia-nccl-cu13`, `triton==3.7.1` (ratusan MB, tidak berguna). |
| `torchvision` | `0.28.0+cpu` (pasangan resmi torch 2.13.0; `torchvision 0.28.0` requires `torch==2.13.0`) | Index CPU |
| `transformers==4.52.1` | pure Python, `>=3.9` | Kombinasi dengan torch 2.13 adalah yang diuji Baidu |
| `tokenizers==0.21.4` | `cp39-abi3-manylinux_2_17_x86_64` | Dipakai oleh `LlamaTokenizerFast` |
| `huggingface-hub==0.36.0` | pure Python | |
| `einops==0.8.2`, `addict==2.4.0`, `easydict==1.13`, `pillow` | pure Python / wheel tersedia | |
| `requests`, `tqdm`, `numpy` | ditarik transformers | diimpor langsung oleh kode model |
| `pymupdf` | opsional, untuk PDF → gambar (Test 4) | Model tidak membaca PDF langsung |
| `triton`, `flash-attn`, `xformers`, `sglang` | **tidak dipasang** | |

## 7. Perkiraan performa [ESTIMASI — wajib diukur di Test 3]

Dasar perhitungan:

- **Decode** (token demi token) bersifat memory-bound dan overhead-bound. Parameter aktif ≈ 0,58 miliar → ≈ 1,2 GB bobot dibaca per token. DDR5-6000 dual-channel ≈ 96 GB/s teoretis → batas atas keras ≈ 80 token/detik. Dalam praktik, PyTorch eager menjalankan ±1.500 op kecil per token (12 layer × attention + router + loop 8 expert), dan overhead dispatch di CPU biasanya mendominasi. Perkiraan realistis: **5–15 token/detik**.
- Output per halaman dalam format layout (`<BLOCK>label[bbox]<FORK>teks`) biasanya 1.000–3.000 token untuk halaman padat → **1,5–10 menit per halaman** di CPU.
- **Encoder visi**: SAM ViT-B pada 1024² ≈ 1 TFLOP; per tile 640² ≈ 0,3 TFLOP; gundam 16 tile ≈ 5–6 TFLOP. Dengan throughput bf16 realistis 1–2 TFLOPS di 12 core → **beberapa detik hingga belasan detik per halaman**. Base mode (multi-halaman) jauh lebih ringan, ≈ 1–2 detik per halaman.
- **Prefill** decoder ~2.000 token ≈ 2 TFLOP → 1–3 detik.

Artinya waktu didominasi decode, bukan encoder. Mode base (273 token/halaman) menurunkan beban visi tetapi tidak menurunkan panjang output.

Pembanding GPU (dari README, bukan pengukuran saya): Baidu menguji di A800; jalur SGLang memakai kernel Triton khusus dan CUDA graph. Di RTX 5070 Ti, jalur HF diperkirakan 10–30× lebih cepat dari CPU untuk decode; SGLang lebih cepat lagi dan mendukung banyak request paralel.

## 8. Jawaban awal untuk 10 pertanyaan output akhir

| # | Pertanyaan | Jawaban tahap 1 |
|---|---|---|
| 1 | Bisa berjalan tanpa GPU? | Ya, jalur Transformers, dengan patch 2 baris. Belum dibuktikan dengan eksekusi. |
| 2 | 9900X mampu? | Secara kompatibilitas ya (AVX-512 BF16). Secara kecepatan: lambat tetapi fungsional. |
| 3 | PyTorch CPU cukup? | Ya. Tidak perlu backend lain. |
| 4 | Perlu modifikasi source? | Ya, minimal: 2 baris `.cuda()` → `.to(device)`. Disimpan sebagai patch terpisah pada salinan di `/mnt/data/ai-models`. |
| 5 | Perlu backend alternatif? | Tidak. ONNX tidak layak (MoE dinamis + state kustom). SGLang tidak bisa di CPU. |
| 6 | Estimasi RAM? | 8–9 GiB saat idle setelah load; 10–14 GiB saat memproses 1–10 halaman. |
| 7 | Performa OCR di CPU? | Perkiraan 5–15 tok/s decode, menit per halaman. Diukur di Test 3. |
| 8 | Keterbatasan vs GPU? | Kecepatan 10–30× lebih lambat (HF) hingga >50× (SGLang); tidak ada serving paralel; dokumen ratusan halaman tidak praktis di CPU. |
| 9 | Operasional sebelum GPU? | Untuk volume kecil (beberapa halaman per jam) dan uji akurasi: ya. Untuk produksi: tidak. |
| 10 | Yang berubah saat RTX 5070 Ti 16 GB dipasang? | Pasang driver NVIDIA + torch CUDA (cu13 untuk Blackwell), hapus patch (atau biarkan; `.to(device)` tetap benar di GPU), `model.cuda()` → kernel Triton aktif otomatis. Bobot 6,2 GB muat di 16 GB VRAM. Untuk SGLang: pakai wheel di repo dengan `--mem-fraction-static 0.7` dan turunkan `--max-running-requests` karena VRAM 16 GB, bukan 80 GB. Dukungan sm_120 di flashinfer/flash-attn-4 perlu dicek saat itu. |

## 9. Rencana tahap 2 (menunggu persetujuan)

Semua langkah berikut hanya menyentuh `/mnt/data/ai-projects/Unlimited-OCR` dan `/mnt/data/ai-models/Unlimited-OCR`. Tidak ada `sudo`, tidak ada paket sistem, tidak ada driver.

1. **Environment**: `python3 -m venv /mnt/data/ai-projects/Unlimited-OCR/.venv`; install torch 2.13.0+cpu dan torchvision 0.28.0+cpu dari index CPU; lalu transformers 4.52.1, tokenizers 0.21.4, huggingface-hub 0.36.0, einops, addict, easydict, pillow, pymupdf.
2. **Test 1**: import torch, cek `torch.cuda.is_available() == False`, `torch.backends.mkldnn.is_available()`, jumlah thread, dukungan bf16 (`torch.backends.cpu.get_cpu_capability()`).
3. **Unduh model** (6,7 GB) ke `/mnt/data/ai-models/Unlimited-OCR/Unlimited-OCR-2` dengan `huggingface_hub.snapshot_download`.
4. **Patch**: salin `modeling_unlimitedocr.py` → `.orig`, terapkan patch 2 baris, simpan `cpu.patch` dan catatan perubahan di folder proyek.
5. **Test 2**: load model bf16 dengan `low_cpu_mem_usage=True`, ukur waktu dan RSS (`/proc/self/status`, `free -h`).
6. **Test 3**: satu gambar teks, `model.infer(...)`, catat waktu prefill/decode (fungsi `infer` sudah mengembalikan `time` dan `prefill_time` jika `return_dict=True`), token/detik, RSS puncak, utilisasi CPU; bandingkan 12 vs 24 thread.
7. **Test 4**: satu PDF → gambar via PyMuPDF 300 dpi → `infer_multi` (base mode) dan `infer` per halaman (gundam); post-processing dengan `utils/process_unlimitedOCR2.py`.
8. **Test 5**: bertahap 1 → 3 → 10 halaman sambil memantau `free -h`, `vmstat 1`, `sensors` jika tersedia.

Yang **tidak** akan dilakukan: instalasi SGLang, Triton, flash-attn, driver NVIDIA, Docker, atau perubahan apa pun di luar dua direktori di atas.
