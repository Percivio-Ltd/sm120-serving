# Manifest

Each file below is listed with how it relates to the bytes we serve. The "served" hashes are the files in our running stack, pinned on 14 November 2026. The edits listed are to comments and messages only, and every edit was checked as described.

| File | sha256 | Relation to the served files |
|---|---|---|
| `serve.sh` | `03726d8abcd4eecd833cea4e0551e5eaa85bbc7036548b279b2a5733c91b7ec8` | The served `recipe/serve/serve.sh` (`49b8c097…`), with internal references removed from our comments and from one error message. Every other line is identical (`diff`, with comment lines set aside). We start a copy with `--host 127.0.0.1`. |
| `patches/serve.sh.diff` | `2580ca4e645e7c60147a600782ebb04eb6f1d53fc0ade0d2f8b16a38c8be65e3` | The recipe's `serve/serve.sh` at `451f7dcb` (`dafe8db1…`) → `serve.sh` above. It applies with `patch -p1` from the pack directory. |
| `patches/sitecustomize.diff` | `b95e0606ee48f66101a80437f563373afa699790cebf68a3e4dc5d6b35482247` | The recipe's `patches/vllm/sitecustomize.py` at `451f7dcb` (`23d1add7…`) → our served hook file (`0759f805…`, the region cache plus the `ixrs` switch), with four comments reworded. It applies cleanly. We serve the result as a second `sitecustomize.py` on `PYTHONPATH`; here it replaces the recipe's own. |
| `patches/vllm-kv-offload-gpu-worker.diff` | `ba2aa61d45f0b4543d83b004927797a1ae4382deeaeb465da03967e0a1aa29c1` | Our served patch (`185249d93f9c6ed1…`) with two comments reworded. The served patch on vLLM `5555a2d8`'s file gives the served file `449d278a5973c085…`. This one gives a file with the same syntax tree apart from those comments. |
| `kernels/ixrs/ixrs.cu` | `d7adeb5ab4b6faa6e15cf47141b0b845795ff8057f1d751dc1881b679cda03cd` | The source of the served `ixrs.so` (`ab90d70c…`), with its header comment rewritten. With comments set aside, the code is identical (331 lines). |
| `kernels/ixrs/build_ixrs.py` | `8da678174ef92e63f7c04e0034dd1d3394b8cf84ffcf215acb0c3c1b30fdabab` | Rewritten from our build script (`854d22c6…`), with the same compiler flags and build variants, and without our internal paths. |
| `plans/plan-rt-5296.json` | `6006bc924b6d2fdd02a24e00a73b4b4bc6408edbdbdd17a2980e12f74b38fa2c` | Byte-identical to the served plan. |
| `start.sh` | `f8d1dd69b633273910261c63dbc6f1d98092027a939b203079d9edbe3e170c8a` | New. With a stand-in `docker`, the `docker run` it produces matches the served container's `docker inspect` of 14 November. The vLLM arguments are identical apart from the host. The environment is identical except `PYTHONPATH`, which the folded hook (above) makes unnecessary. |
| `LICENSE-diffbot-recipe` | `72d0add5aad86c1a452b789df4822a681300694de0c275939a76f3c386d50d2a` | The recipe's `LICENSE` at `451f7dcb`, byte for byte. |

Not in this directory:
- **The model weights and diffbot's recipe:** download them from Hugging Face.
- **vLLM-Moet's overlap:** fetch it from its repository (README, "Build", step 5). Its kernel files at `ec8a1e05e1cb` are byte-identical to the ones we serve.
- **Any built binary.**
