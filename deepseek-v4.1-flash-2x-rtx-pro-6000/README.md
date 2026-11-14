# DeepSeek-V4.1-Flash-1017 on 2x RTX PRO 6000: "the Frankenstein"

This directory holds the exact serving configuration we run at home: DeepSeek-V4.1-Flash-1017 (763B parameters) on two RTX PRO 6000 Blackwell Workstation Edition cards (SM120, 96 GB each), a 64-core Threadripper PRO 9985WX and 512 GB of RAM.

It builds on [diffbot's EXL3 3-bit pack and serving recipe](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000), at revision `451f7dcb`. Everything diffbot ships stays as they ship it. This directory adds our parts and settings on top, as patches and files.

The full story and every measurement are at [artus.ai/frankenstein](https://artus.ai/frankenstein/), with one note per part.

## What it does

All figures are on our own 6-task agent test: two questions about 128K- and 200K-token documents, and four coding tasks with tools. Grading is mechanical. These are not live users.

- **One request at a time:** 34.8 s of model time per task, on the server's clock (54 passes of the test, 5 November). NVIDIA's full-quality FP4 version on 4 RTX PRO 6000 Server Edition cards needed 36.4 s on the same test. diffbot's recipe as published, on our box, needed about 40.5 s (3 passes, on the client's clock).
- **8 agent sessions at once:** with the KV tier in host RAM, 208 graded tasks per hour against 87 without it, and 440 output tokens per second (12 to 13 November).
- **Quality:** our 3-bit version has no public benchmark score of its own. It passed 83% of our test's tasks.

The figures above were measured before the last change, the capture sizes of 14 November (below).

## Our parts

| Part | What it is | Where | Measured when it went in |
|---|---|---|---|
| CPU expert schedule | The recipe's CPU expert pool runs with schedule 2, which regroups the fp32 partial sums of decode-sized jobs. It was picked from the pool's settings after profiling our cores. | `serve.sh`: `EXL3_CPU_EXPERTS_SCHED=2` | 2.07% less time per verify step, lower in 5 of 5 randomized blocks ([Part 1](https://artus.ai/2026/11/02/frankenstein-part-1.html)) |
| Prompt region cache | Agent conversations resend the same 100K+ token regions with every request. Regions seen before reuse their token IDs instead of being tokenized again; prompt IDs are unchanged. | `patches/sitecustomize.diff` (`DS41_REGION_CACHE=1`) | 5.57 s less server time per pass of the test, outside queueing and prefill, p = 1/70 (Part 1) |
| vLLM-Moet's mHC overlap | kacper-daftcode's SM120 port takes the mHC projection and Sinkhorn off the decode critical path. It is not our code: we serve it. | `serve.sh` (`MHC_OVERLAP=1`); see "Build" | 4.88% less time per verify step, lower in 5 of 5 blocks ([Part 2](https://artus.ai/2026/11/03/frankenstein-part-2.html)) |
| Host-expert plan | Which 5,296 experts live in host RAM, fitted to the expert routing of our own agent tasks instead of the recipe's plan. | `plans/plan-rt-5296.json` | 2.42% less time per verify step, lower in 5 of 5 blocks ([Part 3](https://artus.ai/2026/11/04/frankenstein-part-3.html)) |
| `ixrs`, an exact multi-CTA indexer top-k | It replaces the recipe's one-CTA-per-row decode top-k for the sparse attention indexer, so a 6-row verify step uses thread-block clusters across the GPU instead of 6 SMs. It has the same op schema and gives byte-identical output. | `kernels/ixrs/`, `patches/sitecustomize.diff` (`DS41_ROWTOPK_IMPL=ixrs`) | 0.314 ms less per verify step on served traffic, 95% interval 0.217 to 0.410 ([Part 5](https://artus.ai/2026/11/07/frankenstein-part-5.html)) |
| RAM KV tier | vLLM's CPU KV offloading: 37 GiB of host RAM per rank (about 6 million tokens) behind the 1.07 million tokens of KV cache on the cards. Without it, 8 agent sessions overflow the cards and recompute their conversations. | `start.sh` (`KV_OFFLOAD_GB=37`, `ALLOC_CONF=`, `CONTAINER_MEM=279g`) | 208 against 87 graded tasks per hour at 8 sessions, 2.4x ([Part 9](https://artus.ai/2026/11/13/frankenstein-part-9.html)); 4.65x the output at 16 sessions ([Part 8](https://artus.ai/2026/11/12/frankenstein-part-8.html)) |
| KV offload worker fix | On our box, the second worker's `cudaHostRegister` of the shared offload region failed, so it stayed pageable. The workers now register one at a time under a file lock, and a failed chunk is retried. vLLM [#59544](https://github.com/vllm-project/vllm/pull/59544), merged on 15 October after our vLLM base, changes the same code path upstream. | `patches/vllm-kv-offload-gpu-worker.diff` | needed for the tier to start as measured |
| Capture sizes | CUDA graph sizes at multiples of 1 + 5 drafted tokens (6, 12, ... 48) instead of the recipe's 1, 2, 4, ... 48, so drafted batches don't pad up to the next size. | `start.sh` (`CAPTURE_SIZES`) | adopted 14 November; the measurement goes into a coming part |
| Engine key handling | The API key is read from a mode-600 file and reaches vLLM through a read-only mount, never through a command line, `docker inspect` or the container's environment. | `serve.sh` (`API_KEY_FILE`) | a safety change, not a speed change |

The drafter is the recipe's: DeepSeek's DSpark guessing 5 tokens ahead, with sampled guesses. With 8 sessions it adds 34% output, and alone it cuts model time per task 2.3x ([Part 10](https://artus.ai/2026/11/14/frankenstein-part-10.html)).

## Build

These steps are written from the records of our own box. We have not run them end to end on a fresh machine. If something is missing, please open an issue.

1. **The pack and recipe.** Download [diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000) at revision `451f7dcb`, weights and `recipe/`.
   - Newer revisions changed the recipe. diffbot's 9 November update reports a faster CPU runtime that we haven't tested.
   - Build the image with `recipe/docker/build.sh`. It pins `vllm/vllm-openai:nightly-5555a2d8f7116b2b4877f66516f4128176cf722a` and its patch series. Our image ID is `sha256:9653961995d3…`, a local build.
   - Build the kernels with `recipe/kernels/build.sh` and the CPU expert runtime with `recipe/cpuexp/build.sh`, as the recipe's README says.
2. **Our recipe changes.** From the pack directory:
   ```
   patch -p1 < .../patches/serve.sh.diff        # or copy serve.sh to recipe/serve/serve.sh
   patch -p1 < .../patches/sitecustomize.diff
   cp .../plans/plan-rt-5296.json recipe/plans/
   ```
3. **ixrs.** Inside the image, run `IXTK_BUILD_DIR=<empty dir> python3 kernels/ixrs/build_ixrs.py`. This writes `ixrs.so`.
4. **The KV offload worker.** Take `vllm/v1/kv_offload/cpu/gpu_worker.py` from vLLM `5555a2d8` (the image's copy) and apply `patches/vllm-kv-offload-gpu-worker.diff` with `patch -p1`. Our patched file's sha256 is `449d278a5973c085…`.
5. **The mHC overlap.** serve.sh looks for it in `mhc-overlap/` two levels above `recipe/` (`MHC_OVERLAP_DIR`).
   - From [kacper-daftcode/vLLM-Moet](https://github.com/kacper-daftcode/vLLM-Moet) at commit `ec8a1e05e1cb` (Apache-2.0), copy `tools/dsv41_sm120/mhc_overlap/` to `mhc-overlap/moet/mhc_overlap/`.
   - Copy the image's `vllm/models/deepseek_v41/nvidia/ops/mhc.py` to `mhc-overlap/patched/mhc.py`.
   - Run `python3 tools/dsv41_sm120/patch_vllm_mhc_overlap_sm120.py --file mhc-overlap/patched/mhc.py --kernel-dir /opt/moet/mhc_overlap`.
   - The kernel extension compiles on its first import, under `TORCH_EXTENSIONS_DIR`. Build it once inside the image, with `mhc-overlap/ext/` as that directory, for example by running Moet's `test_mhc_post_norm_sm120.py` on one GPU. Then write `mhc-overlap/SHA256SUMS` over `patched/mhc.py`, the files of `moet/mhc_overlap/` and the one `.so` under `ext/`.
   - serve.sh refuses to start with the overlap on unless that directory matches its `SHA256SUMS` with exactly one `.so`. `MHC_OVERLAP=0` serves without it.
   - Checked on 14 November: Moet's patcher at that commit, run on vLLM `5555a2d8`'s `mhc.py`, reproduces our served file except one default path string. `VLLM_MOET_MHC_OVERLAP_DIR` overrides that string at start.
6. **The key.** Put one line in a file with mode 600, and pass it as `API_KEY_FILE`.
7. **Start:**
   ```
   RECIPE=<pack>/recipe IXRS_SO=<dir>/ixrs.so KV_WORKER=<path>/gpu_worker.py API_KEY_FILE=<key file> ./start.sh
   ```
   - The server is OpenAI-compatible on port 8000 as `deepseek-v4.1-flash`. Clients send the key as the Bearer token.
   - serve.sh binds `0.0.0.0`; we bind `127.0.0.1` on our box.
   - The first start compiles kernels for several minutes. `[warmup] done` means it's ready.

**How close this is to what we serve.** We ran `start.sh` against a stand-in `docker` and compared the `docker run` it produces with the container we serve, recorded from `docker inspect` on 14 November.
- The vLLM arguments are identical, apart from the host address.
- The environment it sets is identical, except for one variable. We serve the `ixrs` hook as a second `sitecustomize.py` on `PYTHONPATH`; here it is folded into the recipe's own `sitecustomize.py` (`patches/sitecustomize.diff`), so that variable and one mount fall away.

## Files

| File | What it is |
|---|---|
| `serve.sh` | diffbot's `recipe/serve/serve.sh` (MIT) with our changes. `patches/serve.sh.diff` is the same change as a diff against `451f7dcb`. |
| `start.sh` | The settings we serve on top of serve.sh's defaults. |
| `patches/sitecustomize.diff` | Against the recipe's `patches/vllm/sitecustomize.py` (MIT): the prompt region cache and the `DS41_ROWTOPK_IMPL` switch for `ixrs`. |
| `patches/vllm-kv-offload-gpu-worker.diff` | Against vLLM `5555a2d8`'s `vllm/v1/kv_offload/cpu/gpu_worker.py` (Apache-2.0). |
| `kernels/ixrs/` | `ixrs.cu` and its build script. |
| `plans/plan-rt-5296.json` | The host-expert plan: per MoE layer, the expert indices kept in host RAM. |
| `MANIFEST.md` | Every file's sha256, and how it relates to the bytes we serve. |

## Licenses and credits

- This directory is Apache-2.0, like the rest of this repository.
- `serve.sh` and the two recipe diffs modify diffbot's MIT-licensed recipe. Their notice is in `LICENSE-diffbot-recipe`.
- The KV worker diff modifies a vLLM file (Apache-2.0, copyright the vLLM contributors).
- vLLM-Moet's overlap is fetched from its repository, not copied here.

Credits:
- **DeepSeek** for the model and the DSpark drafter.
- **diffbot** for the 3-bit pack and the recipe everything here builds on.
- **turboderp** for EXL3 and ExLlamaV3.
- **vLLM** for the engine, and **Or Ozeri** and **Yihua Cheng** for its KV offloading.
- **kacper-daftcode** for vLLM-Moet's mHC overlap.

## AI assistance

Percivio's AI agents (Claude and GPT models) designed, built and measured these parts, and an AI agent assembled this directory. Artus set the goal and owns the hardware.
