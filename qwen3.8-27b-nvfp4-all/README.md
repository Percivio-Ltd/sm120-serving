# Qwen3.8-27B-NVFP4-all on one RTX PRO 6000 Blackwell

[Percivio/Qwen3.8-27B-NVFP4-all](https://huggingface.co/Percivio/Qwen3.8-27B-NVFP4-all) is Qwen3.8-27B with the attention, MLP and main Gated DeltaNet projections and the output head in NVFP4, and an FP8 MTP draft layer; its model card describes how it was built. [`serve.sh`](serve.sh) serves it, or the official FP8 release for comparison, with the vLLM build and settings of the measurements below.

```bash
./serve.sh baseline      # Qwen/Qwen3.8-27B-FP8, MTP with 5 draft tokens: the reference
./serve.sh stock         # this checkpoint on unmodified vLLM: default kernels, FP8 KV cache, MTP5
FP8_DIR=/data/qwen3.8-27b-fp8 ./serve.sh draft-head   # the same plus ../pruned-draft-head
AGENT=1 ./serve.sh stock # for agents: tool calling with ../strict-tool-grammar, prefix caching
```

`draft-head` copies the draft head's rows from the dense `lm_head` in the FP8 release's `outside.safetensors` (6.0 GB), because this checkpoint's own head is NVFP4:

```bash
hf download Qwen/Qwen3.8-27B-FP8 outside.safetensors --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --local-dir /data/qwen3.8-27b-fp8
```

`AGENT=1` also sets `--prefix-match-unit 64`. With MTP, vLLM keeps the Gated DeltaNet state for prefix reuse at least one match unit before the end of the cached prompt, so each agent turn recomputes the new tokens plus one to two units. The default unit is the 1,664-token page. On the six agentic tasks below (the Marlin kernels and the draft head, three runs each, one host), the 64-token unit cut the median tokens recomputed per turn from about 3,000 to about 300 and the mean time per six-task run from 934.7 to 834.6 s (−10.7%), with 14/18 tasks passed in both arms.

For a 32 GB card (RTX PRO 4500 or RTX 5090 class), [`32gb/`](32gb/) has a serving script and the measurements on an RTX PRO 4500.

## Results

One RTX PRO 6000 Blackwell Server Edition (AWS g7e.2xlarge), the image pinned in `serve.sh`, means of three benches on fresh server starts (four for the FP8 release), all on one host:

| Decode tok/s (speed-up) | 64K | 128K | 256K |
|---|---:|---:|---:|
| `baseline`: official FP8 release | 77.8 | 72.4 | 68.8 |
| `stock` | 152.5 (1.96×) | 135.6 (1.87×) | 125.7 (1.83×) |
| `draft-head` | 163.8 (2.10×) | 142.8 (1.97×) | 132.8 (1.93×) |
| `stock` with the Marlin kernels forced | 145.9 (1.87×) | 138.4 (1.91×) | 122.5 (1.78×) |
| `draft-head` with the Marlin kernels forced | 160.2 (2.06×) | 144.5 (1.99×) | 132.2 (1.92×) |

| Full request, s (of which prefill) | 64K | 128K | 256K |
|---|---:|---:|---:|
| `baseline` | 37.5 (11.2) | 51.2 (23.0) | 86.5 (56.6) |
| `stock` | 23.3 (9.8) | 37.3 (22.2) | 77.3 (61.0) |
| `draft-head` | 22.3 (9.8) | 36.6 (22.2) | 76.5 (61.0) |
| `stock` with the Marlin kernels forced | 32.2 (18.2) | 50.7 (35.9) | 102.1 (85.4) |
| `draft-head` with the Marlin kernels forced | 31.0 (18.2) | 50.1 (36.0) | 100.9 (85.4) |

Forcing vLLM's Marlin kernels (`--kernel-config '{"linear_backend_per_quant":{"nvfp4_w4a4":"marlin","fp8_w8a8":"marlin"}}'`) gives 3–4% more verification rounds per second but makes prefill 1.4–1.9× slower, so `serve.sh` leaves vLLM's default (`FlashInferCutlassNvFp4LinearKernel`). No bench produced a reasoning loop.

Quality, greedy with thinking on, on the same host (GSM8K: the first 200 test problems, answers capped at 8,192 tokens; needles: 18 passphrase lookups in 64K–200K-token prompts):

| | GSM8K | Needles |
|---|---:|---:|
| `stock` | 188/200 | 18/18 |
| `draft-head` | 187/200 | 18/18 |
| `stock` with the Marlin kernels forced | 190/200 | 18/18 |
| `draft-head` with the Marlin kernels forced | 190/200 | 18/18 |

The official FP8 release scored 186/200 and 18/18 in the same evaluation on another host. On all 1,319 GSM8K test problems, `stock` scored 1,240 against the FP8 release's 1,231 (24 problems lost, 33 gained).

Six long-context agentic tasks (two question-answering, two bug-repair and two proof-of-concept tasks over 128K–200K-token code contexts, graded by fixture tests and fact and citation checks; a small internal battery), three runs each, with tool calling and prefix caching, one request at a time:

| Agentic tasks passed | `baseline` | `draft-head` with the Marlin kernels forced |
|---|---:|---:|
| strict tools, vLLM's unfixed grammar | 16/18 | 18/18 |
| strict tools with [`../strict-tool-grammar`](../strict-tool-grammar) | 14/18 | 12/18 |
| strict tools off | 18/18 | 16/18 |
| all nine runs | 48/54 | 46/54 |

None of these ran on the benchmark host; the second and third rows ran after the first, on another RTX PRO 6000 host, each run on a fresh server start. Repeat runs of one configuration have moved by up to two tasks, so these differences are within run-to-run variation. No run produced a reasoning loop. `stock` was not run on these tasks.

## Benchmark

[`../bench/bench.py`](../bench/bench.py) sends one streaming request at a time to an OpenAI-compatible server: three tasks (review, tests, explain) over prompts cut from the Django 5.1 source at 64K, 128K and 256K (76.7K, 126.1K and 223.7K prompt tokens), 2,048 output tokens each, greedy. Decode tok/s is output tokens over the decode window (first to last streamed chunk), pooled over the three tasks; it also records time to first token, request time and the server's speculative-decoding acceptance, and flags looping output.

```bash
git clone --depth 1 --branch 5.1 https://github.com/django/django
python3 ../bench/bench.py --label stock-1 --model Percivio/Qwen3.8-27B-NVFP4-all --long-src django/django
python3 ../bench/bench_stats.py results base=baseline-1,baseline-2 stock=stock-1,stock-2 --base base
```

Each bench above ran on a fresh server start. `bench_stats.py` prints each entry's per-context mean, spread and ratio over the base entry.
