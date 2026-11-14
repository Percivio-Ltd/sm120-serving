#!/bin/bash
# serve.sh [tuned|stock] -- serve Qwen3.8-Flash-Next on ONE RTX PRO 6000 Blackwell (SM120) with the vLLM build and server
# settings of the published measurements (README.md).
#   tuned  the dn4g checkpoint (dn4g/build_dn4g.sh) with the INT4 output head, the BF16 Gated DeltaNet state, MTP3 with
#          the pruned FP8 draft head, HC4, W8A16 decode GEMMs, K-FDEC v8 and the strict-tool grammar fix. Set DN4G_DIR
#          to the built checkpoint, and build K-FDEC first (../qwen3.8-flash-next-common/k-fdec/build.sh). KFDEC=0 serves
#          the same stack without K-FDEC (decode MoE on Marlin).
#   stock  nvidia/Qwen3.8-Flash-Next-NVFP4 as released, with MTP (3 draft tokens): the reference. Set NVFP4_DIR to a
#          local copy at revision fc694b54fb0174e0913e6adf86691ef85a4ead47.
# Default: the benchmark settings (--max-num-seqs 4, no prefix caching). AGENT=1 serves agents instead: tool calling
# (qwen3_coder parser), prefix caching with the Gated DeltaNet state kept per block (--mamba-cache-mode align) and
# --max-num-seqs 1, the settings of the agent-task run. MAX_NUM_SEQS overrides either default.
# API key: VLLM_API_KEY_FILE (a file holding the key, preferred) or VLLM_API_KEY in the environment; neither is ever
# put on a command line. Without one the server accepts any request.
# Other variables: GPUS (one GPU index, default 0), HOST (default 127.0.0.1), PORT (default 8000), SERVED_NAME (default
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
GPUS=${GPUS:-0}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
SERVED_NAME=${SERVED_NAME:-qwen3.8-flash-next}
NAME=qwen3.8-flash-next-1gpu-$MODE-gpu$GPUS
CACHE_DIR=${CACHE_DIR:-$HOME/.cache/sm120-serving/$NAME}
KFDEC=${KFDEC:-1}
KFDEC_SO=${KFDEC_SO:-$COMMON/k-fdec/build/kfdec_moe_v8.so}
if [ "${AGENT:-0}" = 1 ]; then
  MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
  WORKLOAD=(--enable-prefix-caching --mamba-cache-mode align --enable-auto-tool-choice --tool-call-parser qwen3_coder)
else
  MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
  WORKLOAD=(--no-enable-prefix-caching)
fi
FLAGS=(--max-model-len 262144 --gpu-memory-utilization 0.95 --seed 42 --reasoning-parser qwen3
  --default-chat-template-kwargs '{"enable_thinking":true}' --generation-config vllm
  --limit-mm-per-prompt '{"image":0,"video":0}'
  --kv-cache-dtype bfloat16 --attention-config.indexer_kv_dtype bf16 --mamba-cache-dtype bfloat16
  --engram-config '{"cpu_offload":true,"dp_shared_memory":false}'
  --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens 8192 --quantization modelopt
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}')
# The strict-tool grammar fix is mounted in both modes, as in every measured run.
DOCKER=(-e PYTORCH_CUDA_ALLOC_CONF=pinned_max_round_threshold_mb:256
  -v "$ROOT/strict-tool-grammar/structural_tag_registry.py:$V/tool_parsers/structural_tag_registry.py:ro")
case $MODE in
  tuned)
    : "${DN4G_DIR:?set DN4G_DIR to the checkpoint built by dn4g/build_dn4g.sh (README.md)}"
    FLAGS+=(--mamba-ssm-cache-dtype bfloat16 --moe-backend marlin
      --kernel-config '{"linear_backend_per_quant":{"nvfp4_w4a4":"marlin"}}')
    DOCKER+=(
      -e QWEN4_EXP_SM120_PLANS=/prof/sm120_plans_f1.json
      -e QWEN4_EXP_W8A16=2560x640:32/64/1/4/4,2560x6144:32/256/2/4/3,16384x2560:32/128/1/4/4,248320x2560:32/256/1/4/3
      -e QWEN4_EXP_W8_PREFETCH=0 -e QWEN4_EXP_W8_PDL=0
      -e QWEN4_EXP_HC_FUSED=1 -e QWEN4_EXP_HC_K1=64/256/4/1 -e QWEN4_EXP_HC_K2=8/4/1
      -e VLLM_DRAFT_VOCAB_KEEP=/patches/keep-32768-16384.pt -e VLLM_DRAFT_VOCAB_MAX=45056 -e VLLM_DRAFT_VOCAB_FP8=1
      -e VLLM_DRAFT_VOCAB_GEMV=128,128,4,3
      -e QWEN4_EXP_LMHEAD_INT4=32/4/2/4
      -v "$HERE/sm120_plans_f1.json:/prof/sm120_plans_f1.json:ro"
      -v "$ROOT/pruned-draft-head/keep-32768-16384.pt:/patches/keep-32768-16384.pt:ro"
      -v "$ROOT/pruned-draft-head/speculator.py:$V/v1/worker/gpu/spec_decode/mtp/speculator.py:ro"
      -v "$HERE/vllm/low_latency_gemm.py:$D/low_latency_gemm.py:ro"
      -v "$COMMON/vllm/model.py:$D/model.py:ro"
      -v "$COMMON/vllm/qsa_cache.py:$V/models/qwen4_exp/common/qsa_cache.py:ro"
      -v "$COMMON/hc_fused.py:$D/hc_fused.py:ro")
    if [ "$KFDEC" = 1 ]; then
      [ -f "$KFDEC_SO" ] || { echo "K-FDEC library $KFDEC_SO not found: run $COMMON/k-fdec/build.sh (or KFDEC=0)" >&2; exit 1; }
      DOCKER+=(-e VLLM_SM120_FLASH_DECODE=1 -e VLLM_SM120_FLASH_DECODE_CFG=4:1:4:5:0:0
        -v "$COMMON/k-fdec/sm120_flash_decode.py:$P/sm120_flash_decode.py:ro"
        -v "$COMMON/k-fdec/sm120_flash_decode-0.1.dist-info:$P/sm120_flash_decode-0.1.dist-info:ro"
        -v "$KFDEC_SO:$P/kfdec_moe_v8.so:ro")
    fi
    MODEL=$DN4G_DIR ;;
  stock)
    : "${NVFP4_DIR:?set NVFP4_DIR to a local copy of nvidia/Qwen3.8-Flash-Next-NVFP4 at fc694b54 (README.md)}"
    FLAGS+=(--mamba-ssm-cache-dtype float32)
    MODEL=$NVFP4_DIR ;;
  *) echo "usage: [AGENT=1] [KFDEC=0] [DN4G_DIR=...|NVFP4_DIR=...] $0 [tuned|stock]" >&2; exit 2 ;;
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
