# sm120-serving

Recipes, vLLM patches, kernels and a benchmark from Percivio's work on fast serving on NVIDIA RTX PRO 6000 Blackwell cards (SM120, 96 GB): Qwen3.8 on one and two cards, and DeepSeek-V4.1-Flash on two. The story behind the DeepSeek server is at [artus.ai/frankenstein](https://artus.ai/frankenstein/).

| Directory | Contents |
|---|---|
| [`deepseek-v4.1-flash-2x-rtx-pro-6000/`](deepseek-v4.1-flash-2x-rtx-pro-6000/) | DeepSeek-V4.1-Flash-1017 (763B) on 2 Workstation Edition cards as we serve it, "the Frankenstein": our patches, kernel, expert plan and RAM KV tier settings on top of diffbot's EXL3 3-bit recipe |
| [`qwen3.8-flash-next-nvfp4-2gpu/`](qwen3.8-flash-next-nvfp4-2gpu/) | Qwen3.8-Flash-Next (NVIDIA's NVFP4 release) on 2 cards: ph4 + kmoe1 + K-FDEC v8, 1.76–1.87× the official FP8 release's decode speed |
| [`qwen3.8-flash-next-nvfp4-1gpu/`](qwen3.8-flash-next-nvfp4-1gpu/) | Qwen3.8-Flash-Next on 1 card: f1-dn4g-i4s16 + K-FDEC v8, 1.76–1.84× NVIDIA's NVFP4 release with MTP |
| [`qwen3.8-flash-next-common/`](qwen3.8-flash-next-common/) | What both Flash Next stacks share: the K-FDEC v8 decode MoE kernel, the HC4 kernels and two vLLM changes |
| [`qwen3.8-27b-nvfp4-all/`](qwen3.8-27b-nvfp4-all/) | Serving script and measurements for [Percivio/Qwen3.8-27B-NVFP4-all](https://huggingface.co/Percivio/Qwen3.8-27B-NVFP4-all), and a [kit for 32 GB cards](qwen3.8-27b-nvfp4-all/32gb/) |
| [`qwen3.8-27b-nvfp4-all-katt2b/`](qwen3.8-27b-nvfp4-all-katt2b/) | The same model with katt2b, a prefill path for the FP8 KV cache: 1.11–1.19× shorter time to first token at 64K–256K |
| [`pruned-draft-head/`](pruned-draft-head/) | vLLM patch: MTP drafting against a pruned FP8 copy of the output head |
| [`strict-tool-grammar/`](strict-tool-grammar/) | vLLM patch: strict Qwen tool calls without arguments close instead of running on in hidden whitespace |
| [`bench/`](bench/) | Single-stream long-context decode benchmark for any OpenAI-compatible server (Python standard library only) |

Each directory's README gives its measurements: the host, the clock, the number of runs and the limits.

## Qwen3.8-27B

On one RTX PRO 6000 Blackwell Server Edition, one request at a time, 2,048 output tokens. Each figure is the mean of three benches on one host (four for the FP8 release):

| | Decode tok/s, 64K / 128K / 256K | Full request, s, 64K / 128K / 256K |
|---|---:|---:|
| `Qwen/Qwen3.8-27B-FP8` (official FP8 release), vLLM with MTP | 77.8 / 72.4 / 68.8 | 37.5 / 51.2 / 86.5 |
| `Percivio/Qwen3.8-27B-NVFP4-all`, unmodified vLLM with MTP | 152.5 / 135.6 / 125.7 | 23.3 / 37.3 / 77.3 |
| the same plus the pruned draft head | 163.8 / 142.8 / 132.8 | 22.3 / 36.6 / 76.5 |

The prompts hold 76.7K, 126.1K and 223.7K tokens. [`qwen3.8-27b-nvfp4-all/`](qwen3.8-27b-nvfp4-all/) has the full results and the commands.

## Builds and hosts

The Qwen directories were tested on a single vLLM build, nightly `0.32.1rc1.dev435+gc50aca144` (`vllm/vllm-openai@sha256:3379c8e773c619972cc81e9f1db648c38ab5029369d71c7a24149753666c6480`). The hosts were AWS g7e.2xlarge, g7e.4xlarge and g7e.12xlarge instances (RTX PRO 6000 Blackwell Server Edition), with greedy decoding.

The DeepSeek directory builds on diffbot's recipe and its pinned vLLM nightly (`5555a2d8`). It runs on our own workstation: 2 RTX PRO 6000 Blackwell Workstation Edition cards, a Threadripper PRO 9985WX and 512 GB of RAM.

## License

Apache-2.0 ([LICENSE](LICENSE)).

Files that modify vLLM (Apache-2.0, copyright the vLLM contributors) keep vLLM's header and state the change:
- `pruned-draft-head/speculator.py` and `speculator.diff`;
- the `vllm/` files of the Qwen directories;
- the KV-offload diff in the DeepSeek directory.

The DeepSeek directory's `serve.sh` and recipe diffs modify diffbot's MIT-licensed recipe; see its `LICENSE-diffbot-recipe`.

No model weights are included. Each model is under its own license, which its README names.
