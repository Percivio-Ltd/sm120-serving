#!/bin/bash
# serve.sh [tuned|baseline] -- serve Qwen3.8-Flash-Next on two RTX PRO 6000 Blackwell (SM120), tensor parallel, with the
# vLLM build and server settings of the published measurements (README.md).
#   tuned     nvidia/Qwen3.8-Flash-Next-NVFP4 with ph4 + kmoe1 + K-FDEC v8 + the strict-tool grammar fix. Set MODEL_DIR to
#             a local copy of that repository at revision fc694b54fb0174e0913e6adf86691ef85a4ead47, and build K-FDEC
#             first (../qwen3.8-flash-next-common/k-fdec/build.sh).
#   baseline  Qwen/Qwen3.8-Flash-Next-FP8, the official FP8 release, with MTP (3 draft tokens): the reference. Set
#             FP8_MODEL_DIR to a local copy at revision 236dfdf285828023ca3bcd3f37366c58a3469b13.
# Default: the benchmark settings (--max-num-seqs 4, no prefix caching). AGENT=1 serves agents instead: tool calling
# (qwen3_coder parser), prefix caching with the Gated DeltaNet state kept per block (--mamba-cache-mode align) and
# --max-num-seqs 1, the settings of the agent-task runs. MAX_NUM_SEQS overrides either default.
# API key: VLLM_API_KEY_FILE (a file holding the key, preferred) or VLLM_API_KEY in the environment; neither is ever
# put on a command line. Without one the server accepts any request.
# Other variables: GPUS (default 0,1), HOST (default 127.0.0.1), PORT (default 8000), SERVED_NAME (default
# qwen3.8-flash-next), CACHE_DIR (compile and autotune cache, default ~/.cache/sm120-serving/<name>), KFDEC_SO.
set -euo pipefail
MODE=${1:-tuned}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")
COMMON=$ROOT/qwen3.8-flash-next-common
IMAGE=vllm/vllm-openai@sha256:3379c8e773c619972cc81e9f1db648c38ab5029369d71c7a24149753666c6480  # 0.32.1rc1.dev435+gc50aca144
P=/usr/local/lib/python3.12/dist-packages
V=$P/vllm
D=$V/models/qwen4_exp/nvidia
GPUS=${GPUS:-0,1}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
SERVED_NAME=${SERVED_NAME:-qwen3.8-flash-next}
NAME=qwen3.8-flash-next-2gpu-$MODE
CACHE_DIR=${CACHE_DIR:-$HOME/.cache/sm120-serving/$NAME}
KFDEC_SO=${KFDEC_SO:-$COMMON/k-fdec/build/kfdec_moe_v8.so}
if [ "${AGENT:-0}" = 1 ]; then
  MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
  WORKLOAD=(--enable-prefix-caching --mamba-cache-mode align --enable-auto-tool-choice --tool-call-parser qwen3_coder)
else
  MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
  WORKLOAD=(--no-enable-prefix-caching)
fi
FLAGS=(--max-model-len 262144 --gpu-memory-utilization 0.90 --seed 42 --reasoning-parser qwen3
  --default-chat-template-kwargs '{"enable_thinking":true}' --generation-config vllm
  --limit-mm-per-prompt '{"image":0,"video":0}' --tensor-parallel-size 2
  --kv-cache-dtype bfloat16 --attention-config.indexer_kv_dtype bf16 --mamba-cache-dtype bfloat16
  --mamba-ssm-cache-dtype float32 --engram-config '{"cpu_offload":true,"dp_shared_memory":false}'
  --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens 8192
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}')
# The strict-tool grammar fix is mounted in both modes, as in every measured run.
DOCKER=(-v "$ROOT/strict-tool-grammar/structural_tag_registry.py:$V/tool_parsers/structural_tag_registry.py:ro")
case $MODE in
  tuned)
    : "${MODEL_DIR:?set MODEL_DIR to a local copy of nvidia/Qwen3.8-Flash-Next-NVFP4 at fc694b54 (README.md)}"
    [ -f "$KFDEC_SO" ] || { echo "K-FDEC library $KFDEC_SO not found: run $COMMON/k-fdec/build.sh first" >&2; exit 1; }
    FLAGS+=(--moe-backend marlin --quantization modelopt)
    DOCKER+=(
      # ph4: W8A16 decode GEMMs with SM120 plans, HC4, the INT4 pruned draft head
      -e QWEN4_EXP_SM120_PLANS=/prof/sm120_plans2.json
      -e QWEN4_EXP_W8A16=1,124160x2560:32/256/1/4/3,336x10240:16/128/16/4/4,2560x320:32/64/1/4/4,2560x3072:32/256/2/4/3
      -e QWEN4_EXP_W8_PREFETCH=0 -e QWEN4_EXP_W8_PDL=0
      -e QWEN4_EXP_HC_FUSED=1 -e QWEN4_EXP_HC_K1=64/256/4/1 -e QWEN4_EXP_HC_K2=8/4/1
      -e VLLM_DRAFT_VOCAB_KEEP=/patches/keep-32768-16384.pt -e VLLM_DRAFT_VOCAB_MAX=45056 -e VLLM_DRAFT_VOCAB_FP8=1
      -e VLLM_DRAFT_VOCAB_GEMV=128,128,4,3 -e VLLM_DRAFT_VOCAB_INT4=1 -e VLLM_DRAFT_VOCAB_GEMV4=16,4,4,2
      -e VLLM_DRAFT_VOCAB_GEMV4D=32,4,3,2
      -e VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1
      # kmoe1 (prefill) and K-FDEC v8 (decode)
      -e VLLM_SM120_FLASH_PREFILL=moe,dense8
      -e VLLM_SM120_FLASH_DECODE=1 -e VLLM_SM120_FLASH_DECODE_CFG=4:1:4:5:0:0
      -v "$HERE/sm120_plans2.json:/prof/sm120_plans2.json:ro"
      -v "$ROOT/pruned-draft-head/keep-32768-16384.pt:/patches/keep-32768-16384.pt:ro"
      -v "$HERE/vllm/low_latency_gemm.py:$D/low_latency_gemm.py:ro"
      -v "$HERE/vllm/speculator.py:$V/v1/worker/gpu/spec_decode/mtp/speculator.py:ro"
      -v "$COMMON/vllm/model.py:$D/model.py:ro"
      -v "$COMMON/vllm/qsa_cache.py:$V/models/qwen4_exp/common/qsa_cache.py:ro"
      -v "$COMMON/hc_fused.py:$D/hc_fused.py:ro"
      -v "$HERE/kmoe1/sm120_flash_prefill.py:$P/sm120_flash_prefill.py:ro"
      -v "$HERE/kmoe1/sm120_flash_prefill-0.1.dist-info:$P/sm120_flash_prefill-0.1.dist-info:ro"
      -v "$COMMON/k-fdec/sm120_flash_decode.py:$P/sm120_flash_decode.py:ro"
      -v "$COMMON/k-fdec/sm120_flash_decode-0.1.dist-info:$P/sm120_flash_decode-0.1.dist-info:ro"
      -v "$KFDEC_SO:$P/kfdec_moe_v8.so:ro")
    MODEL=$MODEL_DIR ;;
  baseline)
    : "${FP8_MODEL_DIR:?set FP8_MODEL_DIR to a local copy of Qwen/Qwen3.8-Flash-Next-FP8 at 236dfdf2 (README.md)}"
    FLAGS+=(--enable-expert-parallel --moe-backend triton)
    MODEL=$FP8_MODEL_DIR ;;
  *) echo "usage: [AGENT=1] [MODEL_DIR=...|FP8_MODEL_DIR=...] $0 [tuned|baseline]" >&2; exit 2 ;;
esac
KEY=()
if [ -n "${VLLM_API_KEY_FILE:-}" ]; then
  [ -r "$VLLM_API_KEY_FILE" ] || { echo "VLLM_API_KEY_FILE $VLLM_API_KEY_FILE is not readable" >&2; exit 1; }
  KEY=(-v "$(cd "$(dirname "$VLLM_API_KEY_FILE")" && pwd)/$(basename "$VLLM_API_KEY_FILE"):/run/sm120-serving/api-key:ro")
elif [ -n "${VLLM_API_KEY:-}" ]; then
  KEY=(-e VLLM_API_KEY)  # docker copies the value from this environment; it is not on the command line
else
  echo "note: neither VLLM_API_KEY_FILE nor VLLM_API_KEY is set; the server accepts requests without a key" >&2
fi
mkdir -p "$CACHE_DIR"
exec docker run --rm --name "$NAME" --gpus "\"device=$GPUS\"" --ipc host --network host --ulimit memlock=-1 \
  --ulimit stack=67108864 -v "$CACHE_DIR:/root/.cache" -v "$MODEL:/model:ro" \
  -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 ${KEY[@]+"${KEY[@]}"} "${DOCKER[@]}" \
  --entrypoint /bin/sh "$IMAGE" -c \
  'if [ -r /run/sm120-serving/api-key ]; then VLLM_API_KEY=$(cat /run/sm120-serving/api-key); export VLLM_API_KEY; fi; exec vllm "$@"' \
  sh serve /model --served-model-name "$SERVED_NAME" --host "$HOST" --port "$PORT" "${FLAGS[@]}" "${WORKLOAD[@]}"
