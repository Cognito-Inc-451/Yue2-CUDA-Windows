# YuE2-3B Music Generator

[m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B)

Local, patched build of Gradio app by [mrfakename](https://huggingface.co/spaces/mrfakename/yue2-3b)

**Hardware tested:** RTX 4090 24 GB, Windows, CUDA 13.1 driver (WDDM)

**Result:** ~90–110s per song (down from ~715s before the fixes below)

---

### One-click setup

```powershell
git clone https://github.com/Cognito-Inc-451/Yue2-CUDA-Windows
cd Yue2-CUDA-Windows
.\setup.ps1
.venv\Scripts\python app.py
```

`setup.ps1` creates the venv, installs CUDA torch first, installs the
requirements, and applies the patches.

## Manual Setup

```powershell
# 1. Create the venv (Python 3.11)
python -m venv .venv
.venv\Scripts\Activate

# 2. Install CUDA torch FIRST (the default PyPI build is CPU-only on Windows)
.venv\Scripts\python -m pip install torch==2.10.0 torchaudio==2.10.0 `
    --index-url https://download.pytorch.org/whl/cu128

# 3. Install the rest
.venv\Scripts\python -m pip install -r pre-requirements.txt
.venv\Scripts\python -m pip install -r requirements.txt

# 4. Apply the Windows patches (idempotent — safe to re-run)
.venv\Scripts\python apply_patches.py

# 5. Run
.venv\Scripts\python app.py
# → http://127.0.0.1:7860
```

> `ffmpeg` must be on PATH (used to encode MP3 output).

---

## Fixes

The upstream app runs on Linux. On a Windows torch build things silently
degrade performance. All are fixed by `apply_patches.py`.

### 1. NAR synthesis used the slow "math" attention kernel  ← the big one

The acoustic (NAR) phase runs a 32-step ODE solver = **64 full forward passes**
through the 3B model, each with attention over ~13k tokens. This is 87% of total
runtime.

On the Windows torch 2.10.0 build, PyTorch's SDPA auto-selection is broken:
the flash and memory-efficient kernels aren't compiled in, so it falls back to
the **math kernel** — measured **73× slower** than cuDNN:

| Backend            | Per attention (13k tokens) |
|--------------------|----------------------------|
| math (auto fallback) | 680 ms                  |
| **cuDNN**            | **9.2 ms**              |

**Fix:** force the cuDNN backend in the NAR path.
- `yue2/nar.py` — `attention()` now accepts a `"cudnn"` backend
  (`SDPBackend.CUDNN_ATTENTION`).
- `yue2/pipeline.py` — `synthesize()` passes `attention="cudnn"`.

Effect: NAR phase ~623 s → ~10–20 s.

### 2. AR decode mis-detected flash attention

`yue2/cuda_graph.py` auto-detects the attention backend by checking whether the
`_flash_attention_forward` op exists with the right schema. On Windows the op
*exists* but the kernel isn't compiled in, so it picked "flash" and crashed with
`USE_FLASH_ATTENTION was not enabled for build`.

**Fix:** added `torch.backends.cuda.is_flash_attention_available()` to the
detection condition, so `auto` correctly resolves to `cudnn`.

### 3. Long-song semantic hang (72 separate cudaMalloc calls)

`GraphAR.__init__` allocated the KV cache as 72 separate `torch.zeros` calls
(one per layer × K/V). On Windows WDDM each `cudaMalloc` is a kernel round-trip.
After the ABC phase fragments the allocator pool, each ~22 MB allocation took
~12 s → 72 × 12 s ≈ 872 s stall.

**Fix:** single contiguous tensor, sliced into per-layer views
(`cuda_graph.py` constructor). Also added `gc.collect()` +
`torch.cuda.empty_cache()` in the `finally` block of `sampling.py` to free
memory between phases.

### 4. Long-song semantic prefill (60 s for 4486-token prefix)

`GraphAR.prefill()` calls `self.model(...)` which uses `sdpa()` in
`modeling_yue2.py` with the **auto** SDPA backend → math kernel on Windows
(O(n²), ~73× slower). For a 4486-token prefix this took 60.8 s.

**Fix:** wrapped the prefill model call in
`sdpa_kernel(SDPBackend.CUDNN_ATTENTION)` in `cuda_graph.py`.

### 5. Long-song NAR slowness / prefill hang

After the semantic phase frees its ~1.5 GB KV cache, the CUDA allocator pool
is fragmented. The NAR phase then allocates ~2 GB of its own KV cache. On
Windows WDDM, fragmented allocations are extremely slow — for a long song the
NAR **prefill** (full forward over ~14k AR tokens) can hang for 1000+ s at
"0 steps".

**Fix (two parts):**
- `pipeline.py` `synthesize()` — `gc.collect()` + `torch.cuda.empty_cache()`
  before the NAR phase.
- `nar.py` `CachedNAR` — the per-layer KV cache (72 separate `cudaMalloc`
  calls) is now pre-allocated as **one contiguous tensor** and filled with
  `.copy_()`, mirroring the `GraphAR` fix.

### 6. Missing dependencies in `requirements.txt`

`gradio` and `spaces` were not listed (the HF Space image provides them).

### 7. CPU-only torch by default

`pip install torch` on Windows pulls the CPU build. Install from the
`cu128` index first (see Manual Setup).

### 8. Auxiliary models starve the GPU (the long-song killer)

`app.py` loads **four** models onto the GPU at startup: the YuE2 pipeline
(~8 GB) **plus** three auxiliaries used only by the "cover" and "write lyrics"
buttons — SheetSage2 (transcriber), Qwen3-4B (lyric writer), and Qwen3-ASR-1.7B.
Those three hold **~15 GB** resident the whole time.

During generation the card therefore has 15 GB (idle aux) + 8 GB (YuE2) + the
working set. A short song peaks just under 24 GB and runs fine; a **long** song
(semantic > ~8k tokens) pushes the card to **near-OOM**, and on WDDM every
allocation becomes a slow `cudaMalloc` round-trip:

| Song   | semantic | semantic prefill | NAR        |
|--------|----------|------------------|------------|
| short  | 3062 tok | 0.6 s            | 5.1 s      |
| long   | 8494 tok | **114.8 s**      | **~300 s/step** |

**Fix:** `app.py` now offloads the three auxiliary models to CPU before
generation (`_offload_aux()`) and reloads them after (`_load_aux()`), giving the
YuE2 pipeline the full 24 GB. The cover / lyric features reload them on demand.

---

## App changes (`app.py`)

- `progress=True` (was `False`) — live per-phase progress in the terminal.
- A `[Timing]` line prints after each generation with the per-phase breakdown
  (abc / semantic / nar / vae / e2e).
- Auxiliary models (transcriber / lyric writer / ASR) are offloaded to CPU
  during generation and reloaded after, so the YuE2 pipeline gets the full GPU
  (see fix #8).
- **Auto-save:** every generated song is written to `output/<timestamp>/` as
  `song.flac`, `song.mp3`, `score.abc`, `lyrics.txt` and `style.txt` — all
  sharing the folder's timestamp. Change `OUTPUT_DIR` at the top of `app.py`
  to save elsewhere.
- **Seed 0 = random:** the Seed field defaults to `0`, which draws a fresh
  random seed each generation. Enter a specific number to reproduce a song.
- **Memory cleanup:** `gc.collect()` + `torch.cuda.empty_cache()` run after
  each generation so the GPU footprint stays flat across turns.
- **Theme:** a custom dark Gradio theme (violet→cyan accents)

---

## Generation phases (typical, post-fix)

| Phase                  | Time   | Notes                              |
|------------------------|--------|------------------------------------|
| ABC planning           | ~9 s   | symbolic score (AR)                |
| Semantic tokens        | ~76 s  | AR decode, CUDA graphs             |
| NAR synthesis          | ~10–20 s | 32 ODE steps, cuDNN attention  |
| VAE decode             | ~1 s   | latents → audio                    |

The **semantic (AR)** phase is now the largest cost. The `Render quality`
setting controls NAR ODE steps (16 = Fast, 32 = Best) — it does **not** affect
the AR phase.

---

## Re-applying patches

The fixes live in `.venv\Lib\site-packages\yue2\`. If you reinstall `yue2`,
re-run:

```powershell
.venv\Scripts\python apply_patches.py
```

It is idempotent — it detects already-applied patches and skips them.

---

## Repository

The repo only needs the source + patch script + requirements; 
anyone can rebuild the environment in a minute.

| File / folder      | Purpose                                            |
|--------------------|----------------------------------------------------|
| `app.py`           | the Gradio app (with all the polish)               |
| `apply_patches.py` | idempotent Windows patches for the `yue2` package  |
| `requirements.txt` | CUDA torch + model wheel + deps                    |
| `pre-requirements.txt` | `qwen-asr` (installed before the main set)     |
| `setup.ps1`        | one-click Windows setup                            |
| `README.md`        | this file                                          |
| `.gitignore`       | excludes `.venv/`, `output/`, `__pycache__/`, etc. |

`.gitignore` excludes `.venv/`, `output/`, `__pycache__/`, and common logs.

### Notes for contributors

- **Python 3.11** is the tested version.
- **`ffmpeg` must be on PATH** (MP3 encoding).
- The `yue2` package is installed from a Hugging Face wheel (see
  `requirements.txt`); the patches in `apply_patches.py` target that exact
  version (`yue2-infer 0.1.5`). If the wheel version changes, re-verify the
  patches.
- The `[DIAG-MEM]` / `[DIAG-AR]` / `[DIAG-NAR]` prints in the patched
  `yue2` files are optional diagnostics — they show per-phase memory and
  attention timing.
