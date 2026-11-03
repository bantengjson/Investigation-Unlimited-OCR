"""Util bersama: muat model CPU, pantau RAM/CPU, laporan JSON."""
import os, sys, time, json, threading, re, difflib
import psutil

MODELS = os.environ.get("UOCR_MODELS", "/mnt/data/ai-models/Unlimited-OCR")
MODEL_DIR = os.environ.get("UOCR_MODEL_DIR", os.path.join(MODELS, "Unlimited-OCR-2"))


def setup_threads(n=None):
    import torch
    phys = psutil.cpu_count(logical=False) or 12
    n = int(n or os.environ.get("UOCR_THREADS") or phys)
    torch.set_num_threads(n)
    return n


def sys_mem():
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return {"ram_used_gib": round(vm.used / 2**30, 2), "ram_avail_gib": round(vm.available / 2**30, 2),
            "swap_used_gib": round(sw.used / 2**30, 2)}


class Monitor:
    """Sampel RSS proses, CPU% proses, dan swap sistem tiap `interval` detik di thread terpisah."""

    def __init__(self, interval=0.5):
        self.p = psutil.Process()
        self.interval = interval
        self.peak_rss = 0
        self.cpu_samples = []
        self.swap_peak = 0
        self._stop = threading.Event()
        self.p.cpu_percent(None)

    def _run(self):
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, self.p.memory_info().rss)
            self.cpu_samples.append(self.p.cpu_percent(None))
            self.swap_peak = max(self.swap_peak, psutil.swap_memory().used)
            self._stop.wait(self.interval)

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.rss0 = self.p.memory_info().rss
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self.th.join()
        self.elapsed = time.perf_counter() - self.t0

    def report(self):
        ncpu = psutil.cpu_count()
        cpu = self.cpu_samples[1:] or [0]
        return {"elapsed_s": round(self.elapsed, 2),
                "rss_start_gib": round(self.rss0 / 2**30, 2),
                "rss_peak_gib": round(self.peak_rss / 2**30, 2),
                "cpu_pct_of_one_core_avg": round(sum(cpu) / len(cpu), 1),
                "cpu_pct_of_all_cores_avg": round(sum(cpu) / len(cpu) / ncpu, 1),
                "swap_used_peak_gib": round(self.swap_peak / 2**30, 2)}


def load_model(threads=None, dtype="bf16"):
    import torch
    from transformers import AutoModel, AutoTokenizer
    n = setup_threads(threads)
    dt = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    print(f"[load] {MODEL_DIR} dtype={dtype} threads={n}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    kw = dict(trust_remote_code=True, use_safetensors=True, torch_dtype=dt)
    try:
        import accelerate  # noqa: F401
        kw["low_cpu_mem_usage"] = True
    except ImportError:
        pass
    with Monitor() as m:
        model = AutoModel.from_pretrained(MODEL_DIR, **kw).eval()
    rep = m.report()
    devs = {str(p.device) for p in model.parameters()}
    dts = {str(p.dtype) for p in model.parameters()}
    nparam = sum(p.numel() for p in model.parameters())
    rep.update({"threads": n, "params_B": round(nparam / 1e9, 3), "devices": sorted(devs), "dtypes": sorted(dts),
                "attn_impl": getattr(model.config, "_attn_implementation", None)})
    print("[load]", json.dumps(rep), flush=True)
    return tok, model, rep


def save_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print("[report] ->", path)


def similarity(a, b):
    """Rasio kemiripan karakter 0..1 (difflib) setelah normalisasi spasi; indikator kasar akurasi."""
    na = re.sub(r"\s+", " ", a).strip()
    nb = re.sub(r"\s+", " ", b).strip()
    return round(difflib.SequenceMatcher(None, na, nb).ratio(), 4)


def strip_layout(raw):
    """Buang token layout (<PAGE>, <BLOCK>label[bbox]<FORK>) agar tersisa teks saja."""
    t = re.sub(r"<\s*/?\s*(PAGE|FORK|MERGE)\s*\d*\s*>", "\n", raw)
    t = re.sub(r"<\s*BLOCK\s*>\s*[a-zA-Z_]+(\s*<\s*LEVEL\s*\d*\s*>)?(\s*\[[^\]]*\])?", "\n", t)
    t = re.sub(r"<\|[^<>]*?\|>", "", t)
    t = re.sub(r"<[^>]{1,40}>", " ", t)
    return re.sub(r"\n{2,}", "\n", t).strip()
