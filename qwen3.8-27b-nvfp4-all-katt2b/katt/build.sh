#!/bin/bash
# build.sh -- build libkatt.so (sm_120a) from katt.cu inside the pinned vLLM image (nvcc 13.0), with the flags of the
# measured build. No GPU is needed. The library is written next to this script, where katt.py looks for it
# (VLLM_SM120_KATT_LIBRARY overrides that at serve time); the file is written by the container's root user. nvcc
# builds are not byte-reproducible, so the result will not match the measured binary's hash.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=vllm/vllm-openai@sha256:3379c8e773c619972cc81e9f1db648c38ab5029369d71c7a24149753666c6480  # 0.32.1rc1.dev435+gc50aca144
docker run --rm --network none -v "$HERE:/kern/katt" -w /kern/katt --entrypoint nvcc "$IMAGE" \
  -std=c++17 -O3 -lineinfo -shared -Xcompiler=-fPIC -Xptxas=-v \
  -gencode arch=compute_120a,code=sm_120a katt.cu -o libkatt.so
ls -l "$HERE/libkatt.so"
