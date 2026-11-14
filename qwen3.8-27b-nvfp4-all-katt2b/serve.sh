#!/bin/bash
# serve.sh -- serve Qwen3.8-27B-NVFP4-all on one RTX PRO 6000 Blackwell (SM120) with the pruned FP8 draft head, vLLM's
# Marlin kernels forced, and the katt2b prefill attention: the vLLM build and server settings of the published
# measurements (README.md).
#   MODEL_DIR  local copy of Percivio/Qwen3.8-27B-NVFP4-all at revision f089aefe4d5ef33cedf64ca5fcf644bc0d619419
#   FP8_DIR    directory holding outside.safetensors from Qwen/Qwen3.8-27B-FP8 (revision 017b9c7a), which supplies
#              the draft head's dense lm_head rows
# Build katt2b first: katt/build.sh writes katt/libkatt.so.
# Default: the benchmark settings (--max-num-seqs 4, no prefix caching). AGENT=1 serves agents instead: tool calling
# (qwen3_xml parser), prefix caching with the Gated DeltaNet state kept per block (--mamba-cache-mode align) and
# --max-num-seqs 1, the settings of the agent-task runs. MAX_NUM_SEQS overrides either default.
# MARLIN=0 drops the forced Marlin kernels (vLLM's default NVFP4 kernels with katt2b): measured on two agent-task
# runs only (README.md).
# API key: VLLM_API_KEY_FILE (a file holding the key, preferred) or VLLM_API_KEY in the environment; neither is ever
# put on a command line. Without one the server accepts any request.
# Other variables: GPUS (one GPU index, default 0), HOST (default 127.0.0.1), PORT (default 8000), SERVED_NAME (default
# qwen3.8-27b), CACHE_DIR (compile cache, default ~/.cache/sm120-serving/<name>).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")
IMAGE=vllm/vllm-openai@sha256:3379c8e773c619972cc81e9f1db648c38ab5029369d71c7a24149753666c6480  # 0.32.1rc1.dev435+gc50aca144
V=/usr/local/lib/python3.12/dist-packages/vllm
: "${MODEL_DIR:?set MODEL_DIR to a local copy of Percivio/Qwen3.8-27B-NVFP4-all at f089aefe (README.md)}"
: "${FP8_DIR:?set FP8_DIR to the directory holding outside.safetensors from Qwen/Qwen3.8-27B-FP8}"
[ -f "$HERE/katt/libkatt.so" ] || { echo "katt/libkatt.so not found: run $HERE/katt/build.sh first" >&2; exit 1; }
GPUS=${GPUS:-0}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
SERVED_NAME=${SERVED_NAME:-qwen3.8-27b}
NAME=qwen3.8-27b-katt2b-gpu$GPUS
CACHE_DIR=${CACHE_DIR:-$HOME/.cache/sm120-serving/$NAME}
if [ "${AGENT:-0}" = 1 ]; then
  MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
  WORKLOAD=(--enable-prefix-caching --mamba-cache-mode align --enable-auto-tool-choice --tool-call-parser qwen3_xml)
else
  MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
  WORKLOAD=(--no-enable-prefix-caching)
fi
KERNELS=(--kernel-config '{"linear_backend_per_quant":{"nvfp4_w4a4":"marlin","fp8_w8a8":"marlin"}}')
[ "${MARLIN:-1}" = 0 ] && KERNELS=()
KEY=()
if [ -n "${VLLM_API_KEY_FILE:-}" ]; then
  [ -r "$VLLM_API_KEY_FILE" ] || { echo "VLLM_API_KEY_FILE $VLLM_API_KEY_FILE is not readable" >&2; exit 1; }
  KEY=(-v "$(cd "$(dirname "$VLLM_API_KEY_FILE")" && pwd)/$(basename "$VLLM_API_KEY_FILE"):/run/sm120-serving/api-key:ro")
elif [ -n "${VLLM_API_KEY:-}" ]; then
  KEY=(-e VLLM_API_KEY)  # docker copies the value from this environment; it is not on the command line
else
  echo "note: neither VLLM_API_KEY_FILE nor VLLM_API_KEY is set; the server accepts requests without a key" >&2
fi
DOCKER=(
  # pruned FP8 draft head (../pruned-draft-head)
  -e VLLM_DRAFT_VOCAB_KEEP=/patches/keep-32768-16384.pt -e VLLM_DRAFT_LM_HEAD_FILE=/fp8/outside.safetensors
  -e VLLM_DRAFT_VOCAB_MAX=65536 -e VLLM_DRAFT_VOCAB_FP8=1
  -v "$ROOT/pruned-draft-head/keep-32768-16384.pt:/patches/keep-32768-16384.pt:ro"
  -v "$ROOT/pruned-draft-head/speculator.py:$V/v1/worker/gpu/spec_decode/mtp/speculator.py:ro"
  # strict-tool grammar fix (../strict-tool-grammar), mounted in both modes as in every measured run
  -v "$ROOT/strict-tool-grammar/structural_tag_registry.py:$V/tool_parsers/structural_tag_registry.py:ro"
  # katt2b
  -e VLLM_SM120_FP8KV_PREFILL=1 -e VLLM_SM120_KATT_DIR=/kern/katt -e VLLM_SM120_KATT_PROFILE_FLOOR_GIB=3.4
  -v "$HERE/katt:/kern/katt:ro"
  -v "$HERE/vllm/flashinfer.py:$V/v1/attention/backends/flashinfer.py:ro"
  -v "$HERE/vllm/gpu_worker.py:$V/v1/worker/gpu_worker.py:ro")
mkdir -p "$CACHE_DIR"
exec docker run --rm --name "$NAME" --gpus "\"device=$GPUS\"" --ipc host --network host --ulimit memlock=-1 \
  --ulimit stack=67108864 -v "$CACHE_DIR:/root/.cache" -v "$MODEL_DIR:/model:ro" -v "$FP8_DIR:/fp8:ro" \
  -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 ${KEY[@]+"${KEY[@]}"} "${DOCKER[@]}" \
  --entrypoint /bin/sh "$IMAGE" -c \
  'if [ -r /run/sm120-serving/api-key ]; then VLLM_API_KEY=$(cat /run/sm120-serving/api-key); export VLLM_API_KEY; fi; exec vllm "$@"' \
  sh serve /model --served-model-name "$SERVED_NAME" --host "$HOST" --port "$PORT" \
  --max-model-len 262144 --gpu-memory-utilization 0.90 --seed 42 --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":true}' --generation-config vllm \
  --limit-mm-per-prompt '{"image":0,"video":0}' --mamba-ssm-cache-dtype float32 --max-num-batched-tokens 8192 \
  --attention-config '{"backend":"FLASHINFER"}' --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --kv-cache-dtype fp8 ${KERNELS[@]+"${KERNELS[@]}"} --max-num-seqs "$MAX_NUM_SEQS" "${WORKLOAD[@]}"
