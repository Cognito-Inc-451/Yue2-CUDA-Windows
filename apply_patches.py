"""Apply Windows performance/correctness patches to the installed yue2 package.

Idempotent: safe to re-run. Detects already-applied patches and skips them.

Run:  .venv\\Scripts\\python apply_patches.py
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import yue2
except ImportError:
    sys.exit("yue2 is not installed. Run the pip install steps first (see README.md).")

PKG = Path(yue2.__file__).resolve().parent
print(f"yue2 package: {PKG}\n")


def patch(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"  [skip] {label} (already applied)")
        return
    if old not in text:
        print(f"  [warn] {label}: anchor not found — check manually")
        return
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"  [ok]   {label}")


# ---------------------------------------------------------------------------
# 1. cuda_graph.py — don't pick flash unless the kernel is actually compiled in
# ---------------------------------------------------------------------------
print("cuda_graph.py:")
patch(
    PKG / "cuda_graph.py",
    old=(
        '            "seqused_k" in str(torch.ops.aten._flash_attention_forward.default._schema))\n'
    ),
    new=(
        '            "seqused_k" in str(torch.ops.aten._flash_attention_forward.default._schema)) and (\n'
        '            torch.backends.cuda.is_flash_attention_available())\n'
    ),
    label="flash-attention availability check",
)

# ---------------------------------------------------------------------------
# 2. nar.py — add a "cudnn" attention backend
# ---------------------------------------------------------------------------
print("nar.py:")
patch(
    PKG / "nar.py",
    old='    if backend not in {"sdpa", "math", "flash"}:\n        raise ValueError("attention must be sdpa, math, or flash")',
    new='    if backend not in {"sdpa", "math", "flash", "cudnn"}:\n        raise ValueError("attention must be sdpa, math, flash, or cudnn")',
    label="allow cudnn backend",
)
patch(
    PKG / "nar.py",
    old=(
        "    context = nullcontext()\n"
        '    if backend != "sdpa":\n'
        "        from torch.nn.attention import SDPBackend, sdpa_kernel\n"
        "        context = sdpa_kernel(SDPBackend.MATH if backend == \"math\" else SDPBackend.FLASH_ATTENTION)\n"
    ),
    new=(
        "    context = nullcontext()\n"
        '    if backend != "sdpa":\n'
        "        from torch.nn.attention import SDPBackend, sdpa_kernel\n"
        '        if backend == "math":\n'
        "            context = sdpa_kernel(SDPBackend.MATH)\n"
        '        elif backend == "cudnn":\n'
        "            context = sdpa_kernel(SDPBackend.CUDNN_ATTENTION)\n"
        "        else:\n"
        "            context = sdpa_kernel(SDPBackend.FLASH_ATTENTION)\n"
    ),
    label="cudnn SDPA branch",
)

# ---------------------------------------------------------------------------
# 3. pipeline.py — force the NAR phase onto the cuDNN backend
# ---------------------------------------------------------------------------
print("pipeline.py:")
patch(
    PKG / "pipeline.py",
    old=(
        "                                context=self.generation_config.context, offload_ar=self.offload_ar,\n"
        "                                cancelled=cancelled, on_progress=report)"
    ),
    new=(
        "                                context=self.generation_config.context, offload_ar=self.offload_ar,\n"
        '                                attention="cudnn", cancelled=cancelled, on_progress=report)'
    ),
    label="NAR uses cudnn attention",
)

# ---------------------------------------------------------------------------
# 4. cuda_graph.py — single KV cache allocation (fixes Windows WDDM stall)
# ---------------------------------------------------------------------------
print("cuda_graph.py (KV cache):")
patch(
    PKG / "cuda_graph.py",
    old=(
        "        shape = (self.branches, self.capacity, config.num_key_value_heads, config.head_dim)\n"
        "        self.keys = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in model.model.layers]\n"
        "        self.values = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in model.model.layers]\n"
    ),
    new=(
        "        shape = (self.branches, self.capacity, config.num_key_value_heads, config.head_dim)\n"
        "        # Single allocation avoids 2*N separate cudaMalloc calls, which are\n"
        "        # extremely slow on Windows WDDM after the allocator pool is fragmented.\n"
        "        _n = len(model.model.layers)\n"
        "        _kv = torch.zeros(2, _n, *shape, device=self.device, dtype=self.dtype)\n"
        "        self.keys = [_kv[0, i] for i in range(_n)]\n"
        "        self.values = [_kv[1, i] for i in range(_n)]\n"
    ),
    label="single KV cache allocation",
)

# ---------------------------------------------------------------------------
# 5. sampling.py — free CUDA memory between phases
# ---------------------------------------------------------------------------
print("sampling.py (memory cleanup):")
patch(
    PKG / "sampling.py",
    old=(
        "        return history, timing, not eos\n"
        "    finally:\n"
        "        if graph is not None:\n"
        "            graph.close()\n"
        "        positive_cache = negative_cache = None\n"
    ),
    new=(
        "        return history, timing, not eos\n"
        "    finally:\n"
        "        if graph is not None:\n"
        "            graph.close()\n"
        "            del graph\n"
        "        positive_cache = negative_cache = None\n"
        "        import gc\n"
        "        gc.collect()\n"
        "        torch.cuda.empty_cache()\n"
    ),
    label="free CUDA memory between phases",
)

# ---------------------------------------------------------------------------
# 6. cuda_graph.py — force cuDNN in the prefill forward (fixes 60s prefill)
# ---------------------------------------------------------------------------
print("cuda_graph.py (prefill cuDNN):")
patch(
    PKG / "cuda_graph.py",
    old=(
        "import torch\n"
        "import torch.nn.functional as F\n"
    ),
    new=(
        "import torch\n"
        "import torch.nn.functional as F\n"
        "from torch.nn.attention import SDPBackend, sdpa_kernel\n"
    ),
    label="import sdpa_kernel for prefill",
)
patch(
    PKG / "cuda_graph.py",
    old=(
        "            cache = _PrefixCache(self.keys, self.values, branch)\n"
        "            result = self.model(torch.tensor([prefix], dtype=torch.long, device=self.device),\n"
        "                                past_key_values=cache, use_cache=True, logits_to_keep=1)\n"
    ),
    new=(
        "            cache = _PrefixCache(self.keys, self.values, branch)\n"
        "            # Force cuDNN for the prefill forward — on Windows the auto\n"
        "            # SDPA backend falls back to the math kernel (O(n^2), ~73x slower).\n"
        "            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):\n"
        "                result = self.model(torch.tensor([prefix], dtype=torch.long, device=self.device),\n"
        "                                    past_key_values=cache, use_cache=True, logits_to_keep=1)\n"
    ),
    label="prefill uses cuDNN attention",
)

# ---------------------------------------------------------------------------
# 7. pipeline.py — free CUDA memory before NAR (fixes 344s NAR on long songs)
# ---------------------------------------------------------------------------
print("pipeline.py (NAR memory cleanup):")
patch(
    PKG / "pipeline.py",
    old=(
        "        model = self._load_model(for_nar=True)\n"
        "        with self._status(\"Synthesizing audio\", unit=\"steps\") as status:\n"
    ),
    new=(
        "        model = self._load_model(for_nar=True)\n"
        "        # Free fragmented CUDA memory from the AR phase (GraphAR KV cache)\n"
        "        # before the NAR allocates its own tensors. On Windows WDDM,\n"
        "        # fragmented allocations are extremely slow.\n"
        "        import gc\n"
        "        gc.collect()\n"
        "        torch.cuda.empty_cache()\n"
        "        with self._status(\"Synthesizing audio\", unit=\"steps\") as status:\n"
    ),
    label="free CUDA memory before NAR",
)

# ---------------------------------------------------------------------------
# 8. nar.py — single KV cache allocation (fixes 1451s NAR prefill on long songs)
# ---------------------------------------------------------------------------
print("nar.py (KV cache):")
patch(
    PKG / "nar.py",
    old=(
        "        self.pos_emb = model.latent_pos_embed(local)[None]\n"
        "        self.cache = []\n"
        "        self._prefill()\n"
    ),
    new=(
        "        self.pos_emb = model.latent_pos_embed(local)[None]\n"
        "        # Pre-allocate KV cache as a single contiguous tensor. On Windows WDDM,\n"
        "        # 72 separate cudaMalloc calls (one per layer x K/V) are extremely slow\n"
        "        # after the allocator pool is fragmented by the AR phase.\n"
        "        config = model.config\n"
        "        _n = len(model.model.layers)\n"
        "        self._kv_cache = torch.zeros(2, _n, self.visible_length, config.num_key_value_heads, config.head_dim,\n"
        "                                     device=self.device, dtype=self.dtype)\n"
        "        self.cache = []\n"
        "        self._prefill()\n"
    ),
    label="single NAR KV cache allocation",
)
patch(
    PKG / "nar.py",
    old=(
        "        x = backbone.embed_tokens(ids)\n"
        "        for layer in backbone.layers:\n"
        "            q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos, sin)\n"
        "            # Clone only for restricted visibility; a slice would retain the\n"
        "            # storage of invisible codec tokens for every layer.\n"
        "            cached = (k[0, :self.visible_length], v[0, :self.visible_length])\n"
        "            if self.visible_length != self.ar_length:\n"
        "                cached = tuple(t.clone() for t in cached)\n"
        "            self.cache.append(cached)\n"
        "            h = self._attention(q[0], k[0], v[0], causal=True)\n"
    ),
    new=(
        "        x = backbone.embed_tokens(ids)\n"
        "        for _li, layer in enumerate(backbone.layers):\n"
        "            q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos, sin)\n"
        "            # Copy into the pre-allocated contiguous KV cache (avoids 72\n"
        "            # separate cudaMalloc calls on Windows WDDM).\n"
        "            self._kv_cache[0, _li, :self.visible_length].copy_(k[0, :self.visible_length])\n"
        "            self._kv_cache[1, _li, :self.visible_length].copy_(v[0, :self.visible_length])\n"
        "            self.cache.append((self._kv_cache[0, _li, :self.visible_length],\n"
        "                               self._kv_cache[1, _li, :self.visible_length]))\n"
        "            h = self._attention(q[0], k[0], v[0], causal=True)\n"
    ),
    label="NAR prefill copies into contiguous KV cache",
)

print("\nDone. Re-run any time — already-applied patches are skipped.")
