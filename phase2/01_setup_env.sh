#!/usr/bin/env bash
# Langkah 1: buat venv terpisah dan pasang dependency CPU-only.
# Tidak memakai sudo, tidak menyentuh Python sistem.
set -euo pipefail
PROJ=/mnt/data/ai-projects/Unlimited-OCR
MODELS=/mnt/data/ai-models/Unlimited-OCR
mkdir -p "$PROJ" "$MODELS"
cd "$PROJ"

export PATH="$HOME/.local/bin:$PATH"
if [ ! -x .venv/bin/python ]; then
  if python3 -c "import ensurepip" 2>/dev/null; then
    python3 -m venv .venv
  elif command -v uv >/dev/null; then
    # venv dari interpreter sistem 3.12.3 (tanpa unduh Python baru), lalu pasang pip ke dalamnya
    uv venv --python /usr/bin/python3 .venv
    uv pip install --python .venv/bin/python pip
  else
    echo "Tidak bisa membuat venv: ensurepip tidak ada dan uv tidak ada."
    echo "Opsi tanpa sudo: curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh   (lalu jalankan skrip ini lagi)"
    exit 1
  fi
fi
source .venv/bin/activate
python -m pip install --upgrade pip 2>/dev/null || true

# PyTorch CPU build dari index resmi CPU (BUKAN PyPI, agar tidak menarik nvidia-*/triton)
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu

# Versi yang diuji Baidu untuk jalur Transformers
pip install transformers==4.52.1 huggingface-hub==0.36.0 tokenizers==0.21.4 \
  addict==2.4.0 einops==0.8.2 easydict==1.13 pillow accelerate pymupdf psutil openpyxl python-docx

echo "== Paket terpasang (torch/transformers/nvidia/triton) =="
pip list 2>/dev/null | grep -iE "^(torch|torchvision|transformers|tokenizers|huggingface|accelerate|triton|nvidia|pymupdf|psutil)" || true
echo "== Ukuran venv =="; du -sh .venv
