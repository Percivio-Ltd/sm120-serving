# Qwen3.8-27B in NVFP4 on one 32 GB Blackwell card

These scripts serve [Qwen3.8-27B-NVFP4-all](https://huggingface.co/Percivio/Qwen3.8-27B-NVFP4-all) on one 32 GB sm_120 card with vLLM. You get single-stream chat with MTP speculative decoding and up to 262,144 tokens of context. Weights take 18 GiB; on a headless RTX PRO 4500 a 256K context fit beside them in 32 GB with an FP8 KV cache.

> **What we tested, and what we did not.** Our work targets NVIDIA's RTX PRO Blackwell series, and the model is tuned and measured mainly on the RTX PRO 6000. This kit is the 32 GB configuration we measured on an **RTX PRO 4500 Blackwell** (AWS g7). We have **not** run it on an RTX 5090. The 5090 has the same 32 GB and the same sm_120 architecture, with about twice the memory bandwidth (1,792 against 896 GB/s) and 170 SMs against 82. Its bandwidth and SM count provide potential headroom; speed and the 256K fit on the 5090 remain unmeasured.

## Requirements

- Linux with an NVIDIA driver for CUDA 13 (580 or newer; we used 595.91.07).
- Docker with the NVIDIA Container Toolkit.
- About 35 GB of disk for the vLLM image and 19 GB for the model.
- For the benchmark: `git`, `curl`, `sha256sum` and Python 3.

## Serve

```bash
bash serve.sh    # the first start downloads the model (18.4 GB), then serves on 127.0.0.1:8000
```

Then talk to it with any OpenAI-compatible client, model name `q27`:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"q27","messages":[{"role":"user","content":"Explain NVFP4 in two sentences."}],"max_tokens":2048}'
```

Thinking is on by default. Pass `"chat_template_kwargs":{"enable_thinking":false}` in a request to turn it off. Tool calls use vLLM's `qwen3_xml` parser, with our [strict-tool grammar fix](../../strict-tool-grammar/) mounted so that strict calls of tools without arguments close properly. Stop the server with `docker rm -f q27`.

Settings are environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `Percivio/Qwen3.8-27B-NVFP4-all` | Hugging Face id, or an absolute path to a local copy |
| `MAX_LEN` | `262144` | longest context in tokens |
| `KV_BYTES` | `10737418240` (10 GiB) | KV cache pool; `0` lets vLLM size it from free memory |
| `PREFIX_CACHE` | `1` | reuse earlier turns of a conversation (much faster multi-turn chats); `0` turns it off |
| `MAX_SEQS` | `1` | concurrent requests; the speed numbers below are single stream |
| `PORT` | `8000` | port on 127.0.0.1 |
| `HF_CACHE` | `~/.cache/huggingface` | where the model is downloaded |
| `MARLIN` | `0` | `1` forces vLLM's Marlin kernels, the setting the results below were measured with |

**If the card also drives your display:** the 256K setting leaves only about 1.3 GiB free on a headless card, and a desktop takes some of that. Start with `MAX_LEN=131072 KV_BYTES=0 bash serve.sh`. If the server reports too little KV cache memory for `MAX_LEN`, lower `MAX_LEN`. We did not test a display-attached card.

## Measure your card

```bash
bash run_all.sh    # about 1.5 hours; results in ./results
```

`run_all.sh` fetches the Django 5.1 source and the GSM8K test split (`fetch_data.sh`), then runs two parts:

- **Speed** (`bench.py`). Prompts are built from the Django source at 64K, 128K and 256K tokens, with three tasks each (review, write tests, explain). Each request is greedy with thinking on and 2,048 output tokens, and prefix caching is off.
- **Quality** (`quality.py`). Needle retrieval plants eight passphrases in 64K, 128K and 200K-token haystacks and asks for one, six probes per size. The first 200 GSM8K test problems run greedy with thinking on.

## Results on an RTX PRO 4500 Blackwell (32 GB, 165 W)

Measured on 26 October 2026 on an AWS g7.4xlarge with one RTX PRO 4500 Blackwell Server Edition (32 GB, driver 595.91.07, headless). `run_all.sh` ran with `MAX_LEN=262144`, `KV_BYTES` 10 GiB, a single stream and the Marlin kernels (`MARLIN=1` today). The model was a local copy whose 21 files match the release's SHA-256 manifest.

**Speed** (`bench.py`). Each row covers three requests, all greedy with thinking on and 2,048 output tokens. Prefix caching was off.

| context | prompt tokens | time to first token | decode tok/s (mean) | MTP mean acceptance length |
|---|---:|---:|---:|---:|
| 64K | 76,742 | 51 s | 80.7 | 3.13 |
| 128K | 126,139 | 101 s | 72.7 | 3.12 |
| 256K | 223,705 | 237 s | 60.1 | 3.07 |

None of the nine requests fell into a repetition loop.

**Quality** (`quality.py`):

- Needle retrieval: 18 of 18 (six probes each at 64K, 128K and 200K tokens).
- GSM8K: 188 of the first 200 test problems.
- These quality results match our RTX PRO 6000 run of the same model (needles 18/18, GSM8K-200 189/200).

The time to first token is the cost of a cold prompt. With prefix caching on (the default), later turns of a conversation reuse the cached prefix and start much sooner.

Since this measurement, `serve.sh` has changed in three ways. It mounts the strict-tool grammar fix, which neither run exercised because they sent no tools. It adds `--prefix-match-unit 64` to prefix caching, which the speed run had off. And it uses vLLM's default NVFP4 kernels instead of forcing Marlin (below). On an RTX PRO 6000, with the Marlin kernels, FP8 KV cache and MTP as here, `--prefix-match-unit 64` cut the time of six long-context agentic tasks by 10.7% at equal scores ([details](../README.md)); it is not measured on 32 GB cards.

**Kernels.** `serve.sh` now leaves vLLM's default NVFP4 kernels in place, as the [main recipe](../) does. On the RTX PRO 6000 they make prefill 1.4–1.9× faster than the Marlin kernels, at 3–4% fewer decode steps per second. They take the same memory there (16.5 GiB of weights and the same room for the KV cache), so the 256K setting should still fit. We expect a much shorter time to first token than in the table above and about the same decode speed, but we have not yet measured them on a 32 GB card. `MARLIN=1 bash run_all.sh` repeats the measured configuration.

## How the model was made

Qwen3.8-27B-NVFP4-all starts from NVIDIA's `nvidia/Qwen3.8-27B-NVFP4`. Its attention and Gated DeltaNet projections are re-quantized from FP8 to NVFP4 from Qwen's BF16 weights, and the MTP draft layer uses Qwen's official block-FP8 weights. The model card has the details, quality results and build scripts.

## Versions

- vLLM nightly `0.32.1rc1.dev435+gc50aca144`, image `vllm/vllm-openai@sha256:3379c8e773c619972cc81e9f1db648c38ab5029369d71c7a24149753666c6480` (pinned in `serve.sh`).
- Django 5.1 (git tag `5.1`), and the GSM8K test split from `openai/grade-school-math` (SHA-256 checked).

## License

Apache-2.0, like the model.
