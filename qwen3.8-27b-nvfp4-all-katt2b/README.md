# Qwen3.8-27B-NVFP4-all with katt2b prefill attention on one RTX PRO 6000 Blackwell

[Percivio/Qwen3.8-27B-NVFP4-all](https://huggingface.co/Percivio/Qwen3.8-27B-NVFP4-all) served as in [`../qwen3.8-27b-nvfp4-all/`](../qwen3.8-27b-nvfp4-all/)'s `draft-head` configuration with vLLM's Marlin kernels forced, plus **katt2b**, a prefill path for the FP8 KV cache. Against the same configuration without katt2b, on one host, its time to first token was 1.11–1.19× shorter and its full 2,048-token request time 1.04–1.16× shorter at 64K–256K, at the same decode steps per second ([Results](#results)). [`serve.sh`](serve.sh) serves it with the vLLM build and settings of the measurements.

```bash
hf download Percivio/Qwen3.8-27B-NVFP4-all --revision f089aefe4d5ef33cedf64ca5fcf644bc0d619419 --local-dir /data/qwen3.8-27b-nvfp4-all
(cd /data/qwen3.8-27b-nvfp4-all && grep -v -e '\.complete$' -e quant_summary.txt -e README.md "$OLDPWD/Qwen3.8-27B-NVFP4-all.sha256" | sha256sum -c)
hf download Qwen/Qwen3.8-27B-FP8 outside.safetensors --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --local-dir /data/qwen3.8-27b-fp8
katt/build.sh                                                     # builds katt/libkatt.so in the vLLM image
MODEL_DIR=/data/qwen3.8-27b-nvfp4-all FP8_DIR=/data/qwen3.8-27b-fp8 ./serve.sh            # benchmark settings
AGENT=1 VLLM_API_KEY_FILE=~/.config/vllm-key MODEL_DIR=/data/qwen3.8-27b-nvfp4-all FP8_DIR=/data/qwen3.8-27b-fp8 ./serve.sh
```

The server listens on 127.0.0.1:8000 (`HOST`, `PORT`) as `qwen3.8-27b`, on GPU 0 (`GPUS`). The measurements pinned the checkpoint by the 21-file manifest [`Qwen3.8-27B-NVFP4-all.sha256`](Qwen3.8-27B-NVFP4-all.sha256) (also published in the checkpoint's `reproduce/`); at revision `f089aefe` all 18 of its weight, config and tokenizer files on the Hub match it. The other three entries are a download marker and a quantization summary, which are not on the Hub, and the model card, which has changed since; the check above skips them.

## What the stack changes

Everything is mounted into the unchanged vLLM image `vllm/vllm-openai@sha256:3379c8e773c619972cc81e9f1db648c38ab5029369d71c7a24149753666c6480` (nightly `0.32.1rc1.dev435+gc50aca144`); only the katt2b library is built.

- **The `draft-head` configuration of [`../qwen3.8-27b-nvfp4-all/`](../qwen3.8-27b-nvfp4-all/)**: MTP with 5 draft tokens and the [pruned FP8 draft head](../pruned-draft-head/) (65,536 slots), the FP8 KV cache and FlashInfer attention, with vLLM's Marlin kernels forced for the NVFP4 and FP8 linears (`--kernel-config`). That directory explains why its own `serve.sh` leaves the default kernels instead: Marlin made prefill 1.4–1.9× slower there. katt2b was developed and adopted on the Marlin configuration.
- **katt2b** ([`katt/`](katt/), [`vllm/flashinfer.py`](vllm/flashinfer.py), [`vllm/gpu_worker.py`](vllm/gpu_worker.py)). With an FP8 KV cache, FlashInfer's prefill reads the cache in FP8. For Qwen3.8-27B's attention shape (24 query heads, 4 KV heads, head dimension 256) on SM120, katt2b instead gathers each prefill's KV pages, widens them exactly from FP8 (e4m3) to BF16 into a staging pool reserved at model load (the `katt_gather` kernel in [`katt/katt.cu`](katt/katt.cu)), and runs FlashInfer's BF16 FA2 prefill on the staged pages. Pages of 16, 32 and 64 tokens are supported: the benchmark settings use 16, the agent settings 64. Decode is unchanged.
  - `vllm/flashinfer.py` ([diff](vllm/flashinfer.diff)) builds the staged wrapper ([`katt/staged_prefill.py`](katt/staged_prefill.py)) when `VLLM_SM120_FP8KV_PREFILL=1` and the shape, dtypes and page size qualify, and logs `KATT2_DISABLED reason=...` otherwise.
  - `vllm/gpu_worker.py` ([diff](vllm/gpu_worker.diff)) keeps at least `VLLM_SM120_KATT_PROFILE_FLOOR_GIB=3.4` GiB of activation memory out of the KV-cache budget, so the staging does not run out of memory.
  - [`katt/katt.py`](katt/katt.py) loads `libkatt.so` with ctypes; [`katt/build.sh`](katt/build.sh) builds it from `katt.cu` with nvcc 13.0 in the image (`-gencode arch=compute_120a,code=sm_120a`). `katt.cu` also holds a standalone attention kernel that the served path does not call.
- The [strict-tool grammar fix](../strict-tool-grammar/), mounted in both settings.

**Check that it is active.** At load the log shows `KATT2_READY ... page_size=...` and `Pruned draft head: 37656 static of 65536 slots ...`, and `Using MarlinNvFp4LinearKernel`; at the first long prefill, `KATT2_ACTIVE` and `KATT2_LONG_PREFILL_ACTIVE`.

## Results

All on AWS g7e hosts with one RTX PRO 6000 Blackwell Server Edition, 26 October 2026, the image pinned in `serve.sh`.

**Benchmark** (benchmark settings; each bench pools the review, tests and explain prompts; greedy, thinking on, one request at a time):

| Run | Decode tok/s, 64K / 128K / 256K | Tokens per step | Steps/s | Time to first token, s |
|---|---|---|---|---|
| katt2b, g7e.4xlarge: three benches, two on one instance and one on its replacement | 151.1 / 140.0–140.1 / 131.8–131.9 | 3.00 / 2.96 / 3.14 | 50.4 / 47.4 / 42.0 | 16.45 / 31.55 / 71.46 |
| katt2b, g7e.2xlarge: mean of three benches | 151.3 / 140.3 / 132.3 | 3.00 / 2.96 / 3.15 | 50.4 / 47.4 / 42.0 | 16.39 / 31.37 / 71.01 |
| without katt2b, g7e.4xlarge: mean of three benches | 155.2 / 155.8 / 129.9 | | 50.6 / 47.5 / 42.0 | 18.19 / 35.94 / 85.35 |

katt2b's time to first token is 1.11 / 1.14 / 1.19× shorter than the control's, and its full request time (prefill plus 2,048 output tokens) 1.038 / 1.074 / 1.157× shorter. Decode steps per second are within 0.4%. Decode tokens per second differ through acceptance: katt2b changes floating-point results (the widening is exact, but the attention kernel differs), so greedy outputs and the tokens accepted per step change with it. On 31 new prompts its acceptance differed from the control's by −0.015 ± 0.042 tokens per step. katt2b repeated its outputs on all 9 prompts across restarts, so each host's three benches count as one draw.

**Quality** (replacement g7e.4xlarge): GSM8K, the first 200 test problems, 188/200, with needles (18 passphrase lookups in 64K–200K-token prompts) 18/18; GSM8K, all 1,319 test problems, 1,233 correct.

**Agentic tasks** (the six tasks of [`../qwen3.8-27b-nvfp4-all/`](../qwen3.8-27b-nvfp4-all/), `AGENT=1` settings with the grammar fix, one run per row):

| Run | Tasks passed | Reasoning loops | Summed task time, client clock | Completion tokens |
|---|---:|---:|---:|---:|
| replacement g7e.4xlarge | 4/6 | 0 | 842.2 s | 76,034 |
| same, second run | 4/6 | 0 | 850.6 s | 77,967 |
| g7e.2xlarge | 4/6 | 0 | 838.8 s | 75,023 |
| same, second run | 5/6 | 0 | 874.0 s | 80,839 |

All four code tasks passed in every run; the failures are the two question-answering tasks. The configuration without katt2b scored 4/6 in 896 s on the g7e.4xlarge. On the g7e.2xlarge, katt2b on vLLM's default NVFP4 kernels (`MARLIN=0`) took 813.1 s on average over two runs against 856.4 s for this stack, passing 4/6 in both. That variant was never adopted, and these two runs are its only record here, so `serve.sh` keeps the adopted Marlin configuration as its default.

Pass counts are under our first grading of the question-answering tasks (adj1, a regrade that clears known grader errors); a later revision of those checks (w01-qa/v2) grades the code tasks the same. Under w01-qa/v2 the four runs above score the same (4/6, 4/6, 4/6 and 5/6), and the two `MARLIN=0` runs score 6/6 each. [artus.ai/frankenstein](https://artus.ai/frankenstein/) shows no pass counts for this stack.

The benches ran [`../bench/bench.py`](../bench/bench.py) (76.7K, 126.1K and 223.7K-token Django 5.1 prompts, 2,048 output tokens); see [`../qwen3.8-27b-nvfp4-all/`](../qwen3.8-27b-nvfp4-all/#benchmark) for the method and for the official FP8 release's numbers, which were measured on a different host and date.

## Limits

- **Specific to this model's attention shape** (24 query heads, 4 KV heads, head dimension 256), an FP8 KV cache, BF16 activations and SM120; anything else falls back to FlashInfer's own prefill with a `KATT2_DISABLED` line.
- **Memory.** The staging pool is reserved at model load for `--max-num-seqs` requests of full model length, plus the 3.4 GiB activation floor, so the KV cache is smaller than without katt2b, and more so in the benchmark settings (4 sequences) than in the agent settings (1).
- **Not in this recipe:** `--prefix-match-unit 64`, which `AGENT=1` of [`../qwen3.8-27b-nvfp4-all/serve.sh`](../qwen3.8-27b-nvfp4-all/serve.sh) adds. It was adopted after these runs, and no katt2b run has used it.
- **Determinism.** A later test made this stack restart-deterministic with `TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK=0`; the measured runs did not set it, and neither does `serve.sh`.
- **Measured on Server Edition cards and one vLLM build only.** The mounted vLLM files replace whole modules of that build.

## Files

| File | |
|---|---|
| [`serve.sh`](serve.sh) | Serves the stack with the measured settings |
| [`katt/katt.cu`](katt/katt.cu), [`katt/build.sh`](katt/build.sh) | The katt2b CUDA source and its build (`libkatt.so`) |
| [`katt/katt.py`](katt/katt.py), [`katt/staged_prefill.py`](katt/staged_prefill.py) | ctypes launcher and the staged FlashInfer prefill wrapper, loaded from `VLLM_SM120_KATT_DIR` |
| [`vllm/flashinfer.py`](vllm/flashinfer.py), [`.diff`](vllm/flashinfer.diff) | Replaces `vllm/v1/attention/backends/flashinfer.py` |
| [`vllm/gpu_worker.py`](vllm/gpu_worker.py), [`.diff`](vllm/gpu_worker.diff) | Replaces `vllm/v1/worker/gpu_worker.py` |
| [`Qwen3.8-27B-NVFP4-all.sha256`](Qwen3.8-27B-NVFP4-all.sha256) | sha256 of the measured checkpoint's 21 files |

Also used: [`../pruned-draft-head/`](../pruned-draft-head/) (`speculator.py`, `keep-32768-16384.pt`) and [`../strict-tool-grammar/structural_tag_registry.py`](../strict-tool-grammar/structural_tag_registry.py). The `.diff` files are against vLLM commit [`c50aca1`](https://github.com/vllm-project/vllm/commit/c50aca14416ef29f6ed0ed256302f30646726dcd). `serve.sh`'s settings differ from the measured runs only in the served model name, the optional API key and the container paths of the checkpoint and draft-head file.

## Credits

katt2b and the vLLM changes were written by Percivio's agents. katt2b runs [FlashInfer](https://github.com/flashinfer-ai/flashinfer)'s BF16 FA2 prefill (`BatchPrefillWithPagedKVCacheWrapper`) on the staged pages; the modified files of [vLLM](https://github.com/vllm-project/vllm) keep vLLM's Apache-2.0 header. The base model is [Qwen](https://huggingface.co/Qwen)'s Qwen3.8-27B; the checkpoint is derived from NVIDIA's NVFP4 release and Qwen's FP8 and BF16 releases, as its model card describes.
