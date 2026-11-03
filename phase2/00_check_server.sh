#!/usr/bin/env bash
# Langkah 0: pemeriksaan server sebelum instalasi. Tidak mengubah apa pun.
set -u
echo "== Python =="; python3 --version; which python3
echo "== venv/ensurepip tersedia? =="
python3 -c "import ensurepip, venv; print('ensurepip OK')" 2>&1 || echo "ensurepip TIDAK ada -> butuh paket python3.12-venv (sudo apt) atau pakai uv (tanpa sudo)"
echo "== uv tersedia? =="; command -v uv && uv --version || echo "uv tidak ada (opsional)"
echo "== Disk =="; df -h / /mnt/data 2>/dev/null
echo "== RAM/Swap =="; free -h
echo "== CPU =="; lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket" ; echo -n "Flags bf16/avx512: "; grep -o -E "avx512_bf16|avx512f|avx512_vnni|avx2" /proc/cpuinfo | sort -u | tr '\n' ' '; echo
echo "== GPU (diharapkan tidak ada) =="; command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "nvidia-smi tidak ada (OK)"
echo "== Inventaris folder yang SUDAH ADA (tidak membuat apa pun) =="
echo "-- /mnt/data:"; ls -la /mnt/data 2>/dev/null || echo "/mnt/data tidak ada"
echo "-- /mnt/data/ai-projects:"; ls -la /mnt/data/ai-projects 2>/dev/null || echo "belum ada (akan dibuat di langkah 1)"
echo "-- /mnt/data/ai-models:"; ls -la /mnt/data/ai-models 2>/dev/null || echo "belum ada (akan dibuat di langkah 1)"
echo "-- Home ($HOME):"; ls -la "$HOME" 2>/dev/null | head -40
echo "-- Cache HuggingFace lama?"; du -sh "$HOME/.cache/huggingface" 2>/dev/null || echo "tidak ada"
echo "-- venv/conda lain yang terlihat:"; ls -d "$HOME"/*/.venv "$HOME"/.venv "$HOME"/miniconda3 "$HOME"/anaconda3 /opt/conda 2>/dev/null || echo "tidak ada di lokasi umum"
echo "-- Proses Python/AI yang sedang berjalan (aplikasi lain, jangan diganggu):"; ps -eo pid,user,%cpu,%mem,rss,comm,args --sort=-%mem 2>/dev/null | grep -iE "python|torch|ollama|docker|node" | grep -v grep | head -15
echo "== Font untuk gambar uji =="; ls /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf 2>/dev/null || echo "DejaVuSans tidak ada (skrip akan pakai font bawaan PIL)"
