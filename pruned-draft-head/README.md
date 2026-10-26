# Pruned FP8 draft head for MTP speculative decoding in vLLM

With MTP speculative decoding at batch size 1, every draft step multiplies the draft hidden state by the whole output head: 248,320 vocabulary rows for Qwen3.8. This patch has the drafter score a fixed-size slice of the head instead, 65,536 rows stored in FP8, which is 336 MB per draft step for Qwen3.8-27B (hidden size 5,120).

The slice holds:

- a static keep list, [`keep-32768-16384.pt`](keep-32768-16384.pt): 37,656 token ids, the union of the 32,768 most frequent tokens on a generic code and text corpus, the 16,384 lowest BPE ranks and every added or special token;
- every token id that appears in the target's inputs (prompt chunks and verified tokens), appended on the GPU without a host sync until the slots are full. When a large (prefill) batch arrives and the slice is more than 95% full, it is reset to the static list.

Rows are quantized to FP8 e4m3 with one scale per row, and a Triton GEMV computes the drafter's argmax over them. Draft buffers keep fixed addresses, so the captured CUDA graphs stay valid.

**Verification is untouched.** The target still scores the full vocabulary and accepts or rejects each proposed token, so with greedy decoding the output is the target's own greedy output. The patch changes only which tokens are proposed: a token outside the slice cannot be drafted, and FP8 rounding can change the drafter's argmax. Both cost acceptance, not correctness. Output can still differ from a run without the patch through floating-point effects of how tokens are batched for verification, as with any change in acceptance.

On [Percivio/Qwen3.8-27B-NVFP4-all](https://huggingface.co/Percivio/Qwen3.8-27B-NVFP4-all), one RTX PRO 6000 Blackwell Server Edition, single stream, three benches each on one host ([details](../qwen3.8-27b-nvfp4-all/)):

| | 64K | 128K | 256K |
|---|---:|---:|---:|
| Unmodified vLLM: decode tok/s | 152.5 | 135.6 | 125.7 |
| With this patch: decode tok/s | 163.8 | 142.8 | 132.8 |
| Verification rounds per second, unmodified → patched | 45.8 → 48.6 | 43.2 → 45.8 | 38.7 → 40.7 |

That is 5–6% more rounds per second at every length; with vLLM's Marlin kernels forced the gain is the same (47.5 → 50.4, 44.8 → 47.5, 39.9 → 42.0). Draft acceptance stayed within run-to-run variation, and GSM8K-200 and needle scores were unchanged within noise (187/200 and 18/18 patched, 188/200 and 18/18 unmodified).

## Files

| File | |
|---|---|
| [`speculator.py`](speculator.py) | Drop-in replacement for `vllm/v1/worker/gpu/spec_decode/mtp/speculator.py` |
| [`speculator.diff`](speculator.diff) | The same change as a diff against vLLM commit [`c50aca1`](https://github.com/vllm-project/vllm/commit/c50aca14416ef29f6ed0ed256302f30646726dcd); `git apply speculator.diff` in a vLLM checkout |
| [`keep-32768-16384.pt`](keep-32768-16384.pt) | Static keep list: a 1-D int64 tensor of Qwen3.8 token ids (`torch.load` with the default `weights_only=True` reads it) |
| [`build_keep.py`](build_keep.py) | Builds the keep list: `build_keep.py TOKENIZER_DIR keep-32768-16384.pt 32768 16384`, run inside the vLLM image below. Its corpus is the image's Python 3.12 standard library and installed packages (`*.py`, `*.md`, `*.rst`, `*.txt`), so no benchmark prompt is in it |

Tested only on vLLM nightly `0.32.1rc1.dev435+gc50aca144` (`vllm/vllm-openai@sha256:3379c8e773c619972cc81e9f1db648c38ab5029369d71c7a24149753666c6480`), on one GPU. `speculator.py` is unchanged on vLLM main as of 26 October 2026, but the classes it builds on may have changed.

## Settings

The patch is off unless `VLLM_DRAFT_VOCAB_KEEP` is set.

| Environment variable | Default | Meaning |
|---|---|---|
| `VLLM_DRAFT_VOCAB_KEEP` | unset | Path of the static keep list; unset leaves vLLM's drafter unchanged |
| `VLLM_DRAFT_VOCAB_MAX` | `65536` | Slots in the slice |
| `VLLM_DRAFT_VOCAB_FP8` | `0` | `1` stores the rows in FP8 with per-row scales; `0` keeps them in BF16 |
| `VLLM_DRAFT_LM_HEAD_FILE` | unset | A safetensors file holding a dense `lm_head.weight`. Needed when the model's own head is quantized (for example NVFP4), since rows are copied from a BF16 or FP16 head |
| `VLLM_DRAFT_VOCAB_DYNAMIC` | `1` | `0` keeps only the static list |
| `VLLM_DRAFT_VOCAB_TRITON` | `1` | `0` appends seen tokens with PyTorch ops instead of the Triton kernel |
| `VLLM_DRAFT_VOCAB_GEMV` | `64,256,4,3` | Triton GEMV `BLOCK_N,BLOCK_K,num_warps,num_stages` |

It needs plain greedy drafting, vLLM's default for MTP; with another draft sampling method it logs a warning and stays off. With tensor parallelism each rank scores its own vocabulary shard and the ranks exchange their best (logit, token) pairs, but only a single GPU was measured. All measurements used greedy decoding (temperature 0).

When it is active, the server log shows a line such as:

```
Pruned draft head: 37656 static of 65536 slots (shard 0-248320 of 248320, tp=1, dynamic=True, triton=True, fp8=True, src=/fp8/outside.safetensors)
```

[`../qwen3.8-27b-nvfp4-all/serve.sh`](../qwen3.8-27b-nvfp4-all/serve.sh) `draft-head` shows how to mount it into the vLLM image.
