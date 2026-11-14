# Shared files for the Qwen3.8-Flash-Next stacks

Files that both Flash Next stacks mount unchanged: [`../qwen3.8-flash-next-nvfp4-2gpu/`](../qwen3.8-flash-next-nvfp4-2gpu/) (two cards, tensor parallel) and [`../qwen3.8-flash-next-nvfp4-1gpu/`](../qwen3.8-flash-next-nvfp4-1gpu/) (one card). Their `serve.sh` scripts mount them from here; nothing in this directory is run on its own except the K-FDEC build.

| File | Mounted over | What it does |
|---|---|---|
| [`k-fdec/`](k-fdec/) | `dist-packages/sm120_flash_decode.py`, its `dist-info` and `kfdec_moe_v8.so` | K-FDEC v8, a native decode kernel for the routed experts (below) |
| [`hc_fused.py`](hc_fused.py) | `vllm/models/qwen4_exp/nvidia/hc_fused.py` (new file) | HC4: Qwen4Exp's hyper-connection boundary at decode (M ≤ 16) in two Triton kernels instead of four launches, with FP8 per-row weights and programmatic dependent launch. On with `QWEN4_EXP_HC_FUSED=1`; tile shapes from `QWEN4_EXP_HC_K1` / `QWEN4_EXP_HC_K2` (both stacks use `64/256/4/1` and `8/4/1`, chosen by a sweep on the measuring hosts) |
| [`vllm/model.py`](vllm/model.py), [`model.diff`](vllm/model.diff) | `vllm/models/qwen4_exp/nvidia/model.py` | Adds EAGLE-3 style auxiliary hidden-state capture. It is inactive with MTP drafting; the file is here because every measured run mounted it |
| [`vllm/qsa_cache.py`](vllm/qsa_cache.py), [`qsa_cache.diff`](vllm/qsa_cache.diff) | `vllm/models/qwen4_exp/common/qsa_cache.py` | Widens the QSA compressor ring, when needed, to the next whole-group size that divides the attention block size, so that the speculative tokens fit |
| [`nvidia-Qwen3.8-Flash-Next-NVFP4-fc694b54.sha256`](nvidia-Qwen3.8-Flash-Next-NVFP4-fc694b54.sha256) | – | sha256 of the 13 large files of `nvidia/Qwen3.8-Flash-Next-NVFP4` at revision `fc694b54fb0174e0913e6adf86691ef85a4ead47`; run `sha256sum -c` on it inside the downloaded directory |

The `.diff` files are against vLLM commit [`c50aca1`](https://github.com/vllm-project/vllm/commit/c50aca14416ef29f6ed0ed256302f30646726dcd), the build in the pinned image; `git apply` them in a vLLM checkout. Each modified vLLM file keeps vLLM's license header and states the change in a header of its own.

## K-FDEC v8

With MTP and 3 draft tokens, one decode step of a single request verifies 4 tokens (M = 4). vLLM serves the NVFP4 routed experts of these stacks with Marlin W4A16 (`--moe-backend marlin`), whose MoE path at that size is a chain of launches (align, sort, gate/up GEMM, activation, down GEMM, sum). K-FDEC replaces it for calls with 1 ≤ M ≤ 4 tokens by two sm_120 kernels chained with programmatic dependent launch:

- kernel A computes the gate/up GEMM and SiLU × up for each routed slot;
- kernel B computes the down GEMM and the top-k weighted sum, reducing across a thread-block cluster.

Both read the Marlin tensors already in memory, so there is no second copy of the experts. Routing ids outside [0, E), which vLLM uses for padded token slots, contribute nothing (the change from v7 to v8). Larger M and any layer it cannot serve stay on Marlin, which the log reports once. Accumulation is FP32 in a different order from Marlin's, so greedy outputs can differ from a run without it.

It is a vLLM general plugin: the `sm120_flash_decode-0.1.dist-info` entry point loads `sm120_flash_decode.py` in every vLLM process, and it does nothing unless `VLLM_SM120_FLASH_DECODE=1`. `VLLM_SM120_FLASH_DECODE_CFG=4:1:4:5:0:0` is the kernel schedule both stacks were measured with (4 expert groups, PDL on, kernel A and kernel B row-batch presets 4 and 5, B's L2 prefetch off, no register cap); it is also the plugin's default.

**Build** (no GPU needed):

```bash
k-fdec/build.sh        # writes k-fdec/build/kfdec_moe_v8.so
```

[`k-fdec/build.sh`](k-fdec/build.sh) compiles [`k-fdec/csrc/kfdec_moe_v8.cu`](k-fdec/csrc/kfdec_moe_v8.cu) in the pinned vLLM image with torch's `cpp_extension` (`TORCH_CUDA_ARCH_LIST=12.0`, `-O3 -lineinfo`), the flags of the measured build. The release ships no binary. A rebuilt `.so` will not have the measured binary's hash, because nvcc output is not byte-reproducible.

**Check that it is active.** The server log shows, per process:

```
SM120_FLASH_DECODE enabled moe max_m=4 cfg={'groups': 4, 'pdl': True, 'rba': 4, 'rbb': 5, 'pfb': False, 'regs': 0} lib=...
```

then, after the first requests, `SM120_FLASH_DECODE moe: first K-FDEC call, M=...` and a rising `K-FDEC calls N` counter. If the library is missing, the plugin logs `disabled: K-FDEC extension ... not found` and the server runs on Marlin; the `serve.sh` scripts refuse to start without it.

Tested only on vLLM `0.32.1rc1.dev435+gc50aca144` (`vllm/vllm-openai@sha256:3379c8e773c619972cc81e9f1db648c38ab5029369d71c7a24149753666c6480`), on RTX PRO 6000 Blackwell Server Edition cards, with Qwen3.8-Flash-Next's expert shapes (512 experts, top-10, hidden 2,560, expert width 640 at one GPU and 320 per GPU at two). The kernel takes the hidden size and expert width at run time and requires multiples of 64, at most 512 experts, M ≤ 16 and M × top-k ≤ 256.

## Credits

K-FDEC, HC4 and the vLLM changes here were written by Percivio's agents. K-FDEC reads the weight layout of the [Marlin](https://github.com/IST-DASLab/marlin) kernels as vLLM ships them, and the plugin hooks vLLM's ModelOpt NVFP4 MoE and Marlin expert classes. The HC4 kernels are written in [Triton](https://github.com/triton-lang/triton). `vllm/model.py` and `vllm/qsa_cache.py` are modified files of [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0, copyright the vLLM contributors).
