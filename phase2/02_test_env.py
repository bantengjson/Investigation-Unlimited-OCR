#!/usr/bin/env python
"""Test 1: environment. Harus dijalankan di dalam venv."""
import os, sys, time, platform
print("python", sys.version.split()[0], platform.machine())
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(), "(diharapkan False)")
print("cpu capability:", torch.backends.cpu.get_cpu_capability())
print("mkldnn/oneDNN:", torch.backends.mkldnn.is_available(), "| threads:", torch.get_num_threads(), "| interop:", torch.get_num_interop_threads())
import transformers, tokenizers, torchvision, huggingface_hub, PIL
print("transformers", transformers.__version__, "| tokenizers", tokenizers.__version__,
      "| torchvision", torchvision.__version__, "| hf-hub", huggingface_hub.__version__, "| pillow", PIL.__version__)
for mod in ("triton", "flash_attn", "xformers", "sglang"):
    try:
        __import__(mod); print(f"{mod}: TERPASANG (tidak diperlukan, tidak masalah)")
    except Exception:
        print(f"{mod}: tidak ada (OK)")

# bf16 matmul kecil: memastikan jalur bf16 jalan dan memberi gambaran throughput GEMM
for dt in (torch.float32, torch.bfloat16):
    a = torch.randn(2048, 2048, dtype=dt); b = torch.randn(2048, 2048, dtype=dt)
    for _ in range(2): (a @ b)
    t0 = time.perf_counter(); n = 5
    for _ in range(n): (a @ b)
    dt_s = (time.perf_counter() - t0) / n
    print(f"matmul 2048^3 {str(dt).replace('torch.','')}: {dt_s*1e3:.1f} ms  ~{2*2048**3/dt_s/1e12:.2f} TFLOPS")
# SDPA bf16 dengan mask (jalur fallback attention DASH)
q = torch.randn(1, 10, 512, 128, dtype=torch.bfloat16); m = torch.zeros(1, 1, 512, 512, dtype=torch.bfloat16)
o = torch.nn.functional.scaled_dot_product_attention(q, q, q, attn_mask=m)
print("SDPA bf16 + mask: OK", tuple(o.shape))
print("TEST 1 SELESAI")
