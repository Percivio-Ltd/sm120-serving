# Qwen3.8-Flash-Next NVFP4 on one RTX PRO 6000 Blackwell

Qwen3.8-Flash-Next on a single card, from NVIDIA's NVFP4 release [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) with Percivio's tuned stack **f1-dn4g-i4s16 + K-FDEC v8**. On one host it decoded 1.76–1.84× faster than the NVIDIA release with MTP, one request at a time ([Results](#results)). [`serve.sh`](serve.sh) serves it, or the release for comparison, with the vLLM build and settings of the measurements.

```bash
hf download nvidia/Qwen3.8-Flash-Next-NVFP4 --revision fc694b54fb0174e0913e6adf86691ef85a4ead47 \
  --local-dir /data/qwen3.8-flash-next-nvfp4
(cd /data/qwen3.8-flash-next-nvfp4 && sha256sum -c "$OLDPWD/../qwen3.8-flash-next-common/nvidia-Qwen3.8-Flash-Next-NVFP4-fc694b54.sha256")
NVFP4_DIR=/data/qwen3.8-flash-next-nvfp4 DN4G_DIR=/data/qwen3.8-flash-next-nvfp4-dn4g dn4g/build_dn4g.sh   # one GPU
../qwen3.8-flash-next-common/k-fdec/build.sh                    # builds kfdec_moe_v8.so in the vLLM image
DN4G_DIR=/data/qwen3.8-flash-next-nvfp4-dn4g ./serve.sh         # the tuned stack, benchmark settings
AGENT=1 VLLM_API_KEY_FILE=~/.config/vllm-key DN4G_DIR=/data/qwen3.8-flash-next-nvfp4-dn4g ./serve.sh   # for agents
NVFP4_DIR=/data/qwen3.8-flash-next-nvfp4 ./serve.sh stock       # the release with MTP3: the reference
```

The server listens on 127.0.0.1:8000 (`HOST`, `PORT`) as `qwen3.8-flash-next`, on GPU 0 (`GPUS`). The first start compiles and autotunes into `CACHE_DIR`. The NVIDIA checkpoint, and so the dn4g checkpoint built from it, is under the [NVIDIA Open Model License](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4); this repository ships no weights.

## What the stack changes

Everything is mounted into the unchanged vLLM image `vllm/vllm-openai@sha256:3379c8e773c619972cc81e9f1db648c38ab5029369d71c7a24149753666c6480` (nightly `0.32.1rc1.dev435+gc50aca144`); only the K-FDEC library is built. The name lists the steps:

- **dn4g, the checkpoint.** NVIDIA's release keeps every dense linear in BF16. [`dn4g/build_dn4g.sh`](dn4g/build_dn4g.sh) re-quantizes the 108 Gated DeltaNet projections (`in_proj_qkv`, `in_proj_z` and `out_proj` of the 36 GDN layers) to NVFP4 with an MSE block-scale search ([`dn4g/flash_nvfp4_convert.py`](dn4g/flash_nvfp4_convert.py) and its helper [`dn4g/nvfp4_convert.py`](dn4g/nvfp4_convert.py)), rewrites shards 1–8 and three JSON files and hard-links the rest. It runs on one GPU in the vLLM image and needs about 66 GB of new disk space. The measured checkpoint was built seven times on Server Edition cards and once on a Workstation Edition card, always with identical bytes; the script checks the result against those md5s ([`dn4g/dn4g-shards.md5`](dn4g/dn4g-shards.md5), [`dn4g/dn4g-json.md5`](dn4g/dn4g-json.md5)). These projections run on Marlin W4A16 (`--kernel-config '{"linear_backend_per_quant":{"nvfp4_w4a4":"marlin"}}'`), as do the routed experts (`--moe-backend marlin`).
- **i4: INT4 output head.** The target's 248,320 × 2,560 output head is read in INT4 with one FP16 scale per 128 values, half the bytes of an FP8 head (`QWEN4_EXP_LMHEAD_INT4=32/4/2/4`, in [`vllm/low_latency_gemm.py`](vllm/low_latency_gemm.py)). Unlike the draft head below, this changes the target's logits.
- **s16: Gated DeltaNet state in BF16** (`--mamba-ssm-cache-dtype bfloat16`; the reference runs it in FP32).
- **MTP with 3 draft tokens and the pruned FP8 draft head** of [`../pruned-draft-head`](../pruned-draft-head/), with 45,056 slots (`VLLM_DRAFT_VOCAB_MAX=45056`). The served file is that directory's `speculator.py`.
- **HC4**, the fused hyper-connection kernels, and **W8A16 decode GEMMs** for the remaining BF16 linears, with skinny-GEMM plans for the one-GPU shapes ([`sm120_plans_f1.json`](sm120_plans_f1.json) and the `QWEN4_EXP_W8A16` tile list). See [`../qwen3.8-flash-next-common/`](../qwen3.8-flash-next-common/) and the two-GPU stack's [README](../qwen3.8-flash-next-nvfp4-2gpu/#what-the-stack-changes).
- **K-FDEC v8**, the native decode kernel for the routed experts ([`../qwen3.8-flash-next-common/`](../qwen3.8-flash-next-common/#k-fdec-v8)), the same build as the two-GPU stack's; it takes the one-GPU expert width at run time. `KFDEC=0` leaves it out.
- The [strict-tool grammar fix](../strict-tool-grammar/), mounted in every mode.

kmoe1, the two-GPU stack's prefill plugin, is not used: its FlashInfer copy of the experts (about 63 GiB) does not fit beside the model on one card.

**Left out.** The measured runs also mounted a fused MoE router module that their settings never switched on (`QWEN4_EXP_MOE_ROUTE` unset); it is not part of this release.

## Results

One AWS g7e.4xlarge (one RTX PRO 6000 Blackwell Server Edition, 124 GiB RAM), 28 October 2026. Three arms interleaved, every bench on a fresh server start, three benches per arm, greedy, thinking on, one request at a time, the grammar fix mounted in every arm; the image pinned in `serve.sh`. The reference is `serve.sh stock`. Decode tokens per second (ratio to the reference):

| | 64K | 128K | 256K |
|---|---:|---:|---:|
| `nvidia/Qwen3.8-Flash-Next-NVFP4`, no speculation | 91.5 | 91.6 | 91.4 |
| `stock`: the same with MTP, 3 draft tokens | 145.0 | 145.1 | 152.1 |
| `tuned`: this stack | **258.1 (1.78×)** | **267.1 (1.84×)** | **267.5 (1.76×)** |

The tuned stack ran 104.7 / 104.5 / 104.7 verification steps per second and accepted 2.46 / 2.55 / 2.56 tokens per step. Its time to first token was 6.53 / 10.34 / 18.80 s, 1.18–1.19× the reference's. Every bench of the tuned stack was free of reasoning loops.

What K-FDEC adds, on another g7e host on 26 October 2026 (three benches each, same benchmark): 255.6 / 265.0 / 264.0 tok/s with it against 251.7 / 266.1 / 251.6 without it (`KFDEC=0`), at 104.5–104.9 against 100.3–100.8 verification steps per second. At 128K it decoded 0.4% fewer tokens per second at 4.1% more steps per second, a difference in acceptance, not speed.

The six agentic tasks (two question-answering, two bug-repair and two proof-of-concept tasks over 128K–200K-token code contexts, graded by fixture tests and fact and citation checks; see [`../qwen3.8-27b-nvfp4-all/`](../qwen3.8-27b-nvfp4-all/)), one run with `AGENT=1` settings on the same host and day: 5/6 tasks passed, no reasoning loop, 343.5 s summed task time (client clock), 54,923 completion tokens. The failed task is a citation check on a 200K-token question, which the reference and the no-speculation arm also failed.

Quality of this stack, measured on 25–26 October 2026 on other hosts: GSM8K, the first 200 test problems, 191/200, with needles (18 passphrase lookups in 64K–200K-token prompts) 18/18; GSM8K, all 1,319 test problems, 1,264 correct, against 1,263 without K-FDEC and 1,273 for the official FP8 release.

The benches ran [`../bench/bench.py`](../bench/bench.py) (the copy used on 28 October differs from the published one only in a docstring and the default `--long-src` path): three tasks (review, tests, explain) over prompts cut from the Django 5.1 source, 76.7K, 126.1K and 223.7K prompt tokens, 2,048 output tokens each; decode tok/s is output tokens over the decode window, pooled over the three tasks.

## Limits

- **Measured on Server Edition cards only** (AWS g7e). The tile plans, the INT4 head tiles, the draft-head GEMV tiles, the HC4 tiles and the K-FDEC schedule were chosen there and not re-tuned for Workstation Edition cards.
- **One request at a time.** Every number above is single-stream. K-FDEC serves decode steps of up to 4 tokens; with two or more requests in a step those steps fall back to Marlin (logged once), which is correct but loses its gain.
- **Numerics.** dn4g and the INT4 output head change the model's weights, and K-FDEC's accumulation order changes floating-point results, so outputs differ from the NVIDIA release; the quality checks above are the evidence that they stay close.
- **Memory.** At `--gpu-memory-utilization 0.95` the measuring host loaded 74.98 GiB of model and reported a KV cache of 463,556 tokens (1.77 × 262,144). The n-gram (engram) tables stay in pinned host memory (`cpu_offload`); the measuring host had 124 GiB of RAM.
- **Tested only on the pinned image.** The mounted vLLM files replace whole modules of that build.

## Files

| File | |
|---|---|
| [`serve.sh`](serve.sh) | Serves the tuned stack or the NVIDIA release with the measured settings |
| [`vllm/low_latency_gemm.py`](vllm/low_latency_gemm.py), [`.diff`](vllm/low_latency_gemm.diff) | Replaces `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py`: W8A16 decode GEMMs, SM120 plans, the INT4 output head, the HC4 switch |
| [`sm120_plans_f1.json`](sm120_plans_f1.json) | Skinny-GEMM plans for the one-GPU shapes, read through `QWEN4_EXP_SM120_PLANS` |
| [`dn4g/`](dn4g/) | The dn4g build script, converter and md5 lists |

Shared with the two-GPU stack: [`../qwen3.8-flash-next-common/`](../qwen3.8-flash-next-common/). Also used: [`../pruned-draft-head/speculator.py`](../pruned-draft-head/speculator.py) with its keep list, and [`../strict-tool-grammar/structural_tag_registry.py`](../strict-tool-grammar/structural_tag_registry.py).

`serve.sh`'s settings differ from the measured runs only in the served model name, the optional API key, the container paths of the mounted files and the left-out router module.

## Credits

dn4g, the INT4 head, K-FDEC and the vLLM changes were written by Percivio's agents. The converter writes NVIDIA's ModelOpt NVFP4 layout and models its block-scale search on ModelOpt's MSE calibrator. The stack builds on [vLLM](https://github.com/vllm-project/vllm) (the modified files keep vLLM's Apache-2.0 header), the [Marlin](https://github.com/IST-DASLab/marlin) kernels and [Triton](https://github.com/triton-lang/triton), all in the vLLM image. The model is [Qwen](https://huggingface.co/Qwen)'s Qwen3.8-Flash-Next.
