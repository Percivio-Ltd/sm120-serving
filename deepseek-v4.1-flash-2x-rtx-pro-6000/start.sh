#!/bin/bash
# Start DeepSeek-V4.1-Flash-1017 exactly as we serve it on 2x RTX PRO 6000 Blackwell Workstation Edition (since 14 Nov 2026):
# diffbot's recipe with this directory's patches (README.md, "Build"), plus these settings on top of its defaults.
#
# Usage: RECIPE=/path/to/pack/recipe IXRS_SO=/path/to/ixrs.so KV_WORKER=/path/to/patched/gpu_worker.py \
#        API_KEY_FILE=/path/to/key ./start.sh
# Everything else is serve.sh's defaults (this directory's serve.sh, installed as $RECIPE/serve/serve.sh).
# Needs about 260 GiB of host RAM for the container, plus the 37 GiB KV tier in /dev/shm: we run it on a 512 GB machine.
set -u
: "${RECIPE:?set RECIPE to the pack's recipe directory (patched as README.md says)}"
: "${IXRS_SO:?set IXRS_SO to the ixrs.so you built (kernels/ixrs/build_ixrs.py)}"
: "${KV_WORKER:?set KV_WORKER to vLLM's v1/kv_offload/cpu/gpu_worker.py with patches/vllm-kv-offload-gpu-worker.diff applied}"
for f in "$IXRS_SO" "$KV_WORKER"; do [[ -f $f ]] || { echo "[start] missing $f"; exit 1; }; done
VLLM_PKG=/usr/local/lib/python3.12/dist-packages/vllm

# Host experts on the CPU core pool (the recipe's CPU_EXPERTS=1), with the 2-regroup schedule (serve.sh's default here).
export CPU_EXPERTS=1
# Our exact multi-CTA indexer top-k (kernels/ixrs) in place of the recipe's rtx8::rowtopk; same output bytes.
export DS41_ROWTOPK=1 DS41_ROWTOPK_IMPL=ixrs DS41_IXTK_SO=/opt/ixe-so/ixrs.so
# The KV tier in host RAM: vLLM's CPU KV offloading, 37 GiB per rank (about 6M tokens behind the cards' 1.07M), the
# allocator's expandable segments off (the offload connector needs that), and a higher container memory limit.
export KV_OFFLOAD_GB=37 ALLOC_CONF= CONTAINER_MEM=279g
# CUDA graph capture sizes at multiples of 1 + 5 drafted tokens (6 .. 48) instead of the recipe's 1 2 4 6 8 12 16 24 32 48.
export CAPTURE_SIZES="6 12 18 24 30 36 42 48"
export EXTRA_ARGS="--enable-prompt-tokens-details"
export DOCKER_EXTRA="${DOCKER_EXTRA:-} -v $IXRS_SO:/opt/ixe-so/ixrs.so:ro -v $KV_WORKER:$VLLM_PKG/v1/kv_offload/cpu/gpu_worker.py:ro"
exec bash "$RECIPE/serve/serve.sh"
