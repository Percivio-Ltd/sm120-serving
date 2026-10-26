# sm120-serving

Recipes, vLLM patches and the benchmark from Percivio's work on fast single-stream decoding on one NVIDIA RTX PRO 6000 Blackwell (SM120, 96 GB).

| Directory | Contents |
|---|---|
| [`qwen3.8-27b-nvfp4-all/`](qwen3.8-27b-nvfp4-all/) | Serving script and measurements for [Percivio/Qwen3.8-27B-NVFP4-all](https://huggingface.co/Percivio/Qwen3.8-27B-NVFP4-all) |
| [`pruned-draft-head/`](pruned-draft-head/) | vLLM patch: MTP drafting against a pruned FP8 copy of the output head |
| [`bench/`](bench/) | Single-stream long-context decode benchmark for any OpenAI-compatible server (Python standard library only) |

Qwen3.8-27B on one RTX PRO 6000 Blackwell Server Edition, one request at a time, 2,048 output tokens, means of three benches on one host (four for the FP8 release):

| | Decode tok/s, 64K / 128K / 256K | Full request, s, 64K / 128K / 256K |
|---|---:|---:|
| `Qwen/Qwen3.8-27B-FP8` (official FP8 release), vLLM with MTP | 77.8 / 72.4 / 68.8 | 37.5 / 51.2 / 86.5 |
| `Percivio/Qwen3.8-27B-NVFP4-all`, unmodified vLLM with MTP | 152.5 / 135.6 / 125.7 | 23.3 / 37.3 / 77.3 |
| the same plus the pruned draft head | 163.8 / 142.8 / 132.8 | 22.3 / 36.6 / 76.5 |

The prompts hold 76.7K, 126.1K and 223.7K tokens. [`qwen3.8-27b-nvfp4-all/`](qwen3.8-27b-nvfp4-all/) has the full results and the commands.

Everything here was tested on a single vLLM build, nightly `0.32.1rc1.dev435+gc50aca144` (`vllm/vllm-openai@sha256:3379c8e773c619972cc81e9f1db648c38ab5029369d71c7a24149753666c6480`), on AWS g7e.2xlarge instances (one RTX PRO 6000 Blackwell Server Edition), with one request at a time and greedy decoding.

## License

Apache-2.0 ([LICENSE](LICENSE)). `pruned-draft-head/speculator.py` and `speculator.diff` modify a file of [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0, copyright the vLLM contributors); the modified file states the change in its header.
