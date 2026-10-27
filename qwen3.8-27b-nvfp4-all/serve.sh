#!/bin/bash
# serve.sh [baseline|stock|draft-head] -- serve Qwen3.8-27B on one RTX PRO 6000 Blackwell (SM120) with the vLLM build
# and server settings of the published measurements. The server listens on 127.0.0.1:8000; weights go to the Hugging
# Face cache (HF_HOME, default ~/.cache/huggingface).
#   baseline    Qwen/Qwen3.8-27B-FP8, the official FP8 release, with MTP (5 draft tokens): the reference
#   stock       Percivio/Qwen3.8-27B-NVFP4-all on unmodified vLLM: default NVFP4 kernels, FP8 KV cache, MTP
#   draft-head  the same plus the pruned FP8 draft head (../pruned-draft-head). Set FP8_DIR to a directory holding
#               outside.safetensors from Qwen/Qwen3.8-27B-FP8, which supplies the draft head's dense lm_head rows.
# AGENT=1 serves agents instead of the benchmark: tool calling (qwen3_coder parser) with the strict-tool grammar fix
# (../strict-tool-grammar) mounted, and prefix caching with a 64-token match unit, so that each turn reuses all but
# the last few dozen tokens of the previous one (README.md).
set -euo pipefail
MODE=${1:-stock}
HERE=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=vllm/vllm-openai@sha256:3379c8e773c619972cc81e9f1db648c38ab5029369d71c7a24149753666c6480  # 0.32.1rc1.dev435+gc50aca144
SP=/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/mtp/speculator.py
TR=/usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/structural_tag_registry.py
MODEL=Percivio/Qwen3.8-27B-NVFP4-all
FAST=(--kv-cache-dtype fp8)  # vLLM's default NVFP4 kernels; forcing Marlin (--kernel-config) makes prefill 1.4-1.9x slower
DOCKER=()
WORKLOAD=(--no-enable-prefix-caching)
case $MODE in
  baseline) MODEL=Qwen/Qwen3.8-27B-FP8; FAST=() ;;
  stock) ;;
  draft-head)
    : "${FP8_DIR:?set FP8_DIR to the directory holding outside.safetensors from Qwen/Qwen3.8-27B-FP8}"
    DOCKER=(-v "$HERE/pruned-draft-head/speculator.py:$SP:ro" -v "$HERE/pruned-draft-head:/draft-head:ro"
      -v "$FP8_DIR:/fp8:ro" -e VLLM_DRAFT_VOCAB_KEEP=/draft-head/keep-32768-16384.pt -e VLLM_DRAFT_VOCAB_MAX=65536
      -e VLLM_DRAFT_VOCAB_FP8=1 -e VLLM_DRAFT_LM_HEAD_FILE=/fp8/outside.safetensors) ;;
  *) echo "usage: [AGENT=1] $0 [baseline|stock|draft-head]" >&2; exit 2 ;;
esac
if [ "${AGENT:-0}" = 1 ]; then
  DOCKER+=(-v "$HERE/strict-tool-grammar/structural_tag_registry.py:$TR:ro")
  WORKLOAD=(--enable-prefix-caching --mamba-cache-mode align --prefix-match-unit 64
    --enable-auto-tool-choice --tool-call-parser qwen3_coder)
fi
exec docker run --rm --name sm120-serving --gpus all --ipc host --network host --ulimit memlock=-1 \
  --ulimit stack=67108864 -v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface" \
  ${HF_TOKEN:+-e HF_TOKEN} -e VLLM_NO_USAGE_STATS=1 ${DOCKER[@]+"${DOCKER[@]}"} \
  --entrypoint vllm "$IMAGE" serve "$MODEL" --host 127.0.0.1 --port 8000 \
  --max-model-len 262144 --gpu-memory-utilization 0.90 --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --mamba-ssm-cache-dtype float32 --attention-config '{"backend":"FLASHINFER"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' ${FAST[@]+"${FAST[@]}"} \
  "${WORKLOAD[@]}" --seed 42 --generation-config vllm --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":true}' --limit-mm-per-prompt '{"image":0,"video":0}'
