# Qwen3.8-27B-NVFP4-all on one RTX PRO 6000 Blackwell

[Percivio/Qwen3.8-27B-NVFP4-all](https://huggingface.co/Percivio/Qwen3.8-27B-NVFP4-all) is Qwen3.8-27B with the attention, MLP and main Gated DeltaNet projections and the output head in NVFP4, and an FP8 MTP draft layer; its model card describes how it was built. [`serve.sh`](serve.sh) serves it, or the official FP8 release for comparison, with the vLLM build and settings of the measurements below.

```bash
./serve.sh baseline      # Qwen/Qwen3.8-27B-FP8, MTP with 5 draft tokens: the reference
./serve.sh stock         # this checkpoint on unmodified vLLM: default kernels, FP8 KV cache, MTP5
FP8_DIR=/data/qwen3.8-27b-fp8 ./serve.sh draft-head   # the same plus ../pruned-draft-head
```

`draft-head` copies the draft head's rows from the dense `lm_head` in the FP8 release's `outside.safetensors` (6.0 GB), because this checkpoint's own head is NVFP4:

```bash
hf download Qwen/Qwen3.8-27B-FP8 outside.safetensors --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --local-dir /data/qwen3.8-27b-fp8
```

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

The official FP8 release scored 186/200 and 18/18 in the same evaluation on another host. The model card has the full GSM8K and the agentic tasks.

## Benchmark

[`../bench/bench.py`](../bench/bench.py) sends one streaming request at a time to an OpenAI-compatible server: three tasks (review, tests, explain) over prompts cut from the Django 5.1 source at 64K, 128K and 256K (76.7K, 126.1K and 223.7K prompt tokens), 2,048 output tokens each, greedy. Decode tok/s is output tokens over the decode window (first to last streamed chunk), pooled over the three tasks; it also records time to first token, request time and the server's speculative-decoding acceptance, and flags looping output.

```bash
git clone --depth 1 --branch 5.1 https://github.com/django/django
python3 ../bench/bench.py --label stock-1 --model Percivio/Qwen3.8-27B-NVFP4-all --long-src django/django
python3 ../bench/bench_stats.py results base=baseline-1,baseline-2 stock=stock-1,stock-2 --base base
```

Each bench above ran on a fresh server start. `bench_stats.py` prints each entry's per-context mean, spread and ratio over the base entry.
