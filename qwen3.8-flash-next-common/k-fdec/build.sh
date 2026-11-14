#!/bin/bash
# build.sh -- build the K-FDEC v8 extension, kfdec_moe_v8.so (sm_120), from csrc/kfdec_moe_v8.cu inside the pinned
# vLLM image, with the compiler flags of the measured build (torch cpp_extension, TORCH_CUDA_ARCH_LIST=12.0,
# -O3 -lineinfo). No GPU is needed. The result goes to build/kfdec_moe_v8.so next to this script (OUT_DIR overrides);
# the files are written by the container's root user. nvcc builds are not byte-reproducible, so the result will not
# match the measured binary's hash; README.md lists what to check in the server log instead.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT_DIR=${OUT_DIR:-$HERE/build}
IMAGE=vllm/vllm-openai@sha256:3379c8e773c619972cc81e9f1db648c38ab5029369d71c7a24149753666c6480  # 0.32.1rc1.dev435+gc50aca144
mkdir -p "$OUT_DIR"
docker run --rm --network none -v "$HERE/csrc:/src:ro" -v "$OUT_DIR:/out" -e TORCH_CUDA_ARCH_LIST=12.0 \
  --entrypoint python3 "$IMAGE" -I -c '
from torch.utils.cpp_extension import load
load(name="kfdec_moe_v8", sources=["/src/kfdec_moe_v8.cu"], build_directory="/out",
     extra_cuda_cflags=["-O3", "-lineinfo", "-Xptxas=-v"], extra_cflags=["-O3"], verbose=True)
print("BUILD_OK")'
ls -l "$OUT_DIR/kfdec_moe_v8.so"
