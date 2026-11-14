# Qwen3.8-Flash-Next NVFP4 on two RTX PRO 6000 Blackwell

[`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4), NVIDIA's NVFP4 release, unmodified, served with tensor parallelism over two cards and Percivio's tuned stack: **ph4 + kmoe1 + K-FDEC v8**, plus the [strict-tool grammar fix](../strict-tool-grammar/). On one host it decoded 1.76–1.87× faster than the official FP8 release with MTP, one request at a time ([Results](#results)). [`serve.sh`](serve.sh) serves it, or the FP8 release for comparison, with the vLLM build and settings of the measurements.

```bash
hf download nvidia/Qwen3.8-Flash-Next-NVFP4 --revision fc694b54fb0174e0913e6adf86691ef85a4ead47 \
  --local-dir /data/qwen3.8-flash-next-nvfp4
(cd /data/qwen3.8-flash-next-nvfp4 && sha256sum -c "$OLDPWD/../qwen3.8-flash-next-common/nvidia-Qwen3.8-Flash-Next-NVFP4-fc694b54.sha256")
../qwen3.8-flash-next-common/k-fdec/build.sh                     # builds kfdec_moe_v8.so in the vLLM image
MODEL_DIR=/data/qwen3.8-flash-next-nvfp4 ./serve.sh              # the tuned stack, benchmark settings
AGENT=1 VLLM_API_KEY_FILE=~/.config/vllm-key MODEL_DIR=/data/qwen3.8-flash-next-nvfp4 ./serve.sh   # for agents
FP8_MODEL_DIR=/data/qwen3.8-flash-next-fp8 ./serve.sh baseline   # Qwen/Qwen3.8-Flash-Next-FP8 @236dfdf2, MTP3
```

The server listens on 127.0.0.1:8000 (`HOST`, `PORT`) as `qwen3.8-flash-next`, on GPUs 0 and 1 (`GPUS`). The first start compiles and autotunes into `CACHE_DIR`. The NVIDIA checkpoint is under the [NVIDIA Open Model License](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4); this repository ships no weights.

## What the stack changes

Everything is mounted into the unchanged vLLM image `vllm/vllm-openai@sha256:3379c8e773c619972cc81e9f1db648c38ab5029369d71c7a24149753666c6480` (nightly `0.32.1rc1.dev435+gc50aca144`); nothing is rebuilt except the K-FDEC library. The routed experts run on vLLM's Marlin W4A16 kernels (`--moe-backend marlin --quantization modelopt`), and MTP drafts 3 tokens per step.

**ph4**, the base stack (named after its last step, the INT4 pruned draft head):

- *W8A16 decode GEMMs.* The dense BF16 linears keep FP8 (e4m3) copies with one scale per output row; decode batches (M ≤ 16) run a Triton skinny GEMM over them, reading half the bytes. Prefill keeps BF16. [`vllm/low_latency_gemm.py`](vllm/low_latency_gemm.py) ([diff](vllm/low_latency_gemm.diff)); `QWEN4_EXP_W8A16` lists the tile plans per shape, and [`sm120_plans2.json`](sm120_plans2.json) holds skinny-GEMM plans measured on the RTX PRO 6000 for the TP2 shapes (vLLM ships none for SM120).
- *HC4.* The hyper-connection boundaries in two fused kernels ([`../qwen3.8-flash-next-common/hc_fused.py`](../qwen3.8-flash-next-common/hc_fused.py)).
- *PH4, the INT4 pruned draft head.* As [`../pruned-draft-head`](../pruned-draft-head/) for the 27B, the MTP drafter scores a slice of the output head instead of all 248,320 rows, here in INT4 with one scale per 128 values. Each tensor-parallel rank keeps 45,056 rows from its own half of the vocabulary. Verification still uses the full head. [`vllm/speculator.py`](vllm/speculator.py) ([diff](vllm/speculator.diff)), the `VLLM_DRAFT_VOCAB_*` variables in `serve.sh`.
- vLLM's FlashInfer PCIe IPC all-reduce (`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`, an option of the image).

**kmoe1** ([`kmoe1/`](kmoe1/)), a vLLM general plugin for prefill, on with `VLLM_SM120_FLASH_PREFILL=moe,dense8`:

- `moe`: each MoE layer also keeps a FlashInfer-layout copy of its experts, and calls with at least 2,560 tokens run FlashInfer's TensorRT-LLM CUTLASS fused MoE, W4A4 with the checkpoint's static input scales. Smaller calls (decode, MTP verification, short chunks) stay on Marlin. The copy costs 708 MB per layer per GPU, 31.6 GiB per GPU in all.
- `dense8`: four dense prefill shapes run as per-token FP8 activations times the FP8 weight rows through CUTLASS scaled-mm, instead of dequantizing to BF16 first.

**K-FDEC v8**: decode steps of up to 4 tokens (one request's MTP verification) run the routed experts through two native sm_120 kernels instead of Marlin's chain of launches. Described, with its build, in [`../qwen3.8-flash-next-common/`](../qwen3.8-flash-next-common/#k-fdec-v8).

The other mounted vLLM changes are in [`../qwen3.8-flash-next-common/`](../qwen3.8-flash-next-common/), and the grammar fix in [`../strict-tool-grammar/`](../strict-tool-grammar/).

**Left out.** The measured runs also mounted four modules that their settings never switched on: a fused MoE router, a two-launch shared-expert tail, a side-stream L2 prefetch, and a QSA top-k kernel with its prebuilt library. Their switches (`QWEN4_EXP_MOE_ROUTE`, `QWEN4_EXP_SE_TAIL`, `QWEN4_EXP_L2PF`, `QWEN4_EXP_QSA_TOPK`) were unset, and the measured server log shows none of their start-up lines, so they are not part of this release. Setting one of those switches with this `low_latency_gemm.py` fails at import.

## Results

### Against the official FP8 release, one host

One AWS g7e.12xlarge (two RTX PRO 6000 Blackwell Server Edition), 28 October 2026. Three arms interleaved, every bench on a fresh server start, three benches per arm, greedy, thinking on, one request at a time; the image pinned in `serve.sh`, with the grammar fix mounted in every arm. The reference is `serve.sh baseline`. Decode tokens per second (ratio to the reference):

| | 64K | 128K | 256K |
|---|---:|---:|---:|
| `Qwen/Qwen3.8-Flash-Next-FP8`, no speculation | 118.2 (0.63×) | 117.4 (0.62×) | 117.4 (0.62×) |
| `baseline`: the same with MTP, 3 draft tokens | 188.2 | 190.5 | 188.7 |
| `tuned`: this stack | **331.6 (1.76×)** | **338.8 (1.78×)** | **353.3 (1.87×)** |

| Time to first token, s (client) | 64K | 128K | 256K |
|---|---:|---:|---:|
| no speculation | 5.21 | 8.08 | 14.76 |
| `baseline` | 5.43 | 8.45 | 15.44 |
| `tuned` | 5.02 | 7.81 | 14.30 |

Both speculative arms accepted about the same number of tokens per verification step (reference 2.57–2.62, tuned 2.50–2.65); the tuned stack ran 132.2–133.7 verification steps per second against 72.8–73.3 (means per context). One reference bench repeated itself in one 128K request; without that request the reference's 128K mean is 189.2 and the ratio 1.79×.

The six agentic tasks ([below](#benchmark-and-agentic-tasks)), on the same host and day, each run on a fresh server start with `AGENT=1` settings, two rounds per arm. Summed time of the six tasks, client clock, and tasks passed:

| | Round 1 | Round 2 |
|---|---:|---:|
| no speculation | 735.6 s, 4/6 | 669.6 s, 4/6 |
| `baseline` | 627.4 s, 4/6 (one task looped and was cut off at 65,536 tokens) | 468.0 s, 5/6 |
| `tuned` | **292.8 s, 4/6** | **355.8 s, 5/6** |

Each run takes its own path through the tasks. In round 2 the tuned stack wrote more tokens than the reference (77,852 against 66,826), so its time there is 1.32× faster where its decoding is 1.8× faster. No run of the tuned stack looped. Pass counts are under our first grading of the question-answering tasks (adj1, a regrade that clears known grader errors); a later revision of those checks (w01-qa/v2) grades the code tasks the same. Under w01-qa/v2, which this comparison's own report uses, the same runs score 4/6 and 5/6 (no speculation), 4/6 and 5/6 (`baseline`) and 5/6 and 6/6 (`tuned`). The figures for this stack on [artus.ai/frankenstein](https://artus.ai/frankenstein/) come from three earlier runs on another host on 26 October: 331.3, 287.9 and 279.3 s, with 4/6, 6/6 and 5/6 under adj1.

### Earlier measurement of the stack

On 26 October 2026, on another g7e.12xlarge: three benches each, decode 332.4 / 336.7 / 348.1 tok/s at 64K / 128K / 256K, against 314.7 / 316.7 / 325.1 for ph4 + kmoe1 without K-FDEC on the same host (+5.6 / +6.3 / +7.1%). The stack ran 132.9 / 132.0 / 133.3 verification steps per second and accepted 2.50 / 2.55 / 2.61 tokens per step. Three agentic-task runs on the same host (`AGENT=1` settings, grammar fix mounted) passed 4, 6 and 5 of 6 tasks, with no reasoning loop, in 331.3, 287.9 and 279.3 s summed task time (client clock).

Quality, measured with the same stack on a different host: GSM8K, all 1,319 test problems, 1,272 correct (the official FP8 release: 1,273); needles (18 passphrase lookups in 64K–200K-token prompts) 18/18.

### Benchmark and agentic tasks

The benches ran [`../bench/bench.py`](../bench/bench.py) (the copy used on 28 October differs from the published one only in a docstring and the default `--long-src` path): three tasks (review, tests, explain) over prompts cut from the Django 5.1 source, 76.7K, 126.1K and 223.7K prompt tokens, 2,048 output tokens each, greedy; decode tok/s is output tokens over the decode window, pooled over the three tasks.

The agentic tasks are the small internal battery described in [`../qwen3.8-27b-nvfp4-all/`](../qwen3.8-27b-nvfp4-all/): two question-answering, two bug-repair and two proof-of-concept tasks over 128K–200K-token code contexts, graded by fixture tests and fact and citation checks.

## Limits

- **Measured on Server Edition cards only** (AWS g7e). The tile plans, the K-FDEC schedule, the HC4 tiles, the draft-head GEMV tiles and the kmoe1 thresholds were chosen on those hosts and were not re-tuned for Workstation Edition cards, whose clocks, power limits and memory bandwidth differ. The all-reduce's crossover points depend on the PCIe topology.
- **One request at a time.** Every number above is single-stream. K-FDEC serves decode steps of up to 4 tokens, one request's verification group; with two or more requests in a step, those steps fall back to Marlin (logged once as `first Marlin fallback`), which is correct but loses K-FDEC's gain.
- **Numerics.** kmoe1's W4A4 prefill experts and FP8 dense prefill, and K-FDEC's accumulation order, change floating-point results compared with Marlin, so greedy outputs differ from an unpatched run. FlashInfer's fused MoE finalizes with atomics, so runs are not bit-for-bit repeatable.
- **Memory.** kmoe1's expert copy takes 31.6 GiB per GPU. The measuring host reported a KV cache of 717,081 tokens at `--gpu-memory-utilization 0.90` (2.74 × 262,144). The n-gram (engram) tables stay in pinned host memory (`cpu_offload`).
- **Tested only on the pinned image.** The mounted vLLM files replace whole modules of that build.

## Files

| File | |
|---|---|
| [`serve.sh`](serve.sh) | Serves the tuned stack or the FP8 reference with the measured settings |
| [`vllm/low_latency_gemm.py`](vllm/low_latency_gemm.py), [`.diff`](vllm/low_latency_gemm.diff) | Replaces `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py`: W8A16 decode GEMMs, SM120 plans, the HC4 switch |
| [`vllm/speculator.py`](vllm/speculator.py), [`.diff`](vllm/speculator.diff) | Replaces `vllm/v1/worker/gpu/spec_decode/mtp/speculator.py`: pruned FP8 / INT4 draft head, sharded under tensor parallelism |
| [`sm120_plans2.json`](sm120_plans2.json) | Skinny-GEMM plans for the TP2 shapes, read through `QWEN4_EXP_SM120_PLANS` |
| [`kmoe1/`](kmoe1/) | The kmoe1 prefill plugin and its `dist-info` entry point |

Shared with the one-GPU stack: [`../qwen3.8-flash-next-common/`](../qwen3.8-flash-next-common/). Also used: [`../pruned-draft-head/keep-32768-16384.pt`](../pruned-draft-head/keep-32768-16384.pt) (the draft head's static keep list) and [`../strict-tool-grammar/structural_tag_registry.py`](../strict-tool-grammar/structural_tag_registry.py).

The `.diff` files are against vLLM commit [`c50aca1`](https://github.com/vllm-project/vllm/commit/c50aca14416ef29f6ed0ed256302f30646726dcd). `serve.sh`'s settings differ from the measured runs only in the served model name, the optional API key, the container paths of the mounted files and the four left-out modules.

## Credits

ph4, kmoe1, K-FDEC and the vLLM changes were written by Percivio's agents. They build on [vLLM](https://github.com/vllm-project/vllm) (the modified files keep vLLM's Apache-2.0 header), NVIDIA's NVFP4 checkpoint and ModelOpt format, [FlashInfer](https://github.com/flashinfer-ai/flashinfer)'s fused MoE (NVIDIA [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) CUTLASS kernels) and PCIe all-reduce, NVIDIA [CUTLASS](https://github.com/NVIDIA/cutlass) scaled-mm in vLLM, the [Marlin](https://github.com/IST-DASLab/marlin) kernels and [Triton](https://github.com/triton-lang/triton), all of which ship in the vLLM image. The model is [Qwen](https://huggingface.co/Qwen)'s Qwen3.8-Flash-Next.
