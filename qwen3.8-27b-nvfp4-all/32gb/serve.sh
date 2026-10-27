#!/usr/bin/env bash
# serve.sh -- serve Qwen3.8-27B-NVFP4-all on one 32 GB Blackwell card (sm_120: RTX PRO 4500, RTX 5090) with vLLM in
# Docker, single stream, MTP speculative decoding, up to 262,144 tokens of context. Waits until the server answers.
#
# Settings (environment variables):
#   MODEL         Hugging Face id or an absolute path to a local copy   (default Percivio/Qwen3.8-27B-NVFP4-all)
#   MAX_LEN       maximum context in tokens                            (default 262144)
#   KV_BYTES      KV cache pool in bytes; 0 lets vLLM size it          (default 10737418240 = 10 GiB, fits 262144)
#   PREFIX_CACHE  1 reuses earlier conversation turns, 0 disables      (default 1)
#   MAX_SEQS      concurrent requests                                  (default 1; speed numbers are single stream)
#   PORT          port on 127.0.0.1                                    (default 8000)
#   HF_CACHE      Hugging Face cache on the host                      (default ~/.cache/huggingface)
#   NAME          container name                                       (default q27)
# Tool calls use the qwen3_xml parser with the strict-tool grammar fix (../../strict-tool-grammar) mounted.
# Stop with: docker rm -f q27
set -euo pipefail
MODEL=${MODEL:-Percivio/Qwen3.8-27B-NVFP4-all}
MAX_LEN=${MAX_LEN:-262144}
KV_BYTES=${KV_BYTES:-10737418240}
PREFIX_CACHE=${PREFIX_CACHE:-1}
MAX_SEQS=${MAX_SEQS:-1}
PORT=${PORT:-8000}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
NAME=${NAME:-q27}
# The vLLM nightly every number in README.md was measured with (0.32.1rc1.dev435+gc50aca144, CUDA 13).
IMAGE=vllm/vllm-openai@sha256:3379c8e773c619972cc81e9f1db648c38ab5029369d71c7a24149753666c6480
FIX=$(cd "$(dirname "$0")/../../strict-tool-grammar" && pwd)/structural_tag_registry.py
TR=/usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/structural_tag_registry.py

mkdir -p "$HF_CACHE"
mounts=(-v "$HF_CACHE":/root/.cache/huggingface -v "$FIX:$TR:ro")
model_arg=$MODEL
if [[ "$MODEL" = /* ]]; then
  [ -f "$MODEL/config.json" ] || { echo "no config.json in $MODEL"; exit 1; }
  mounts+=(-v "$MODEL":/model:ro); model_arg=/model
fi

args=(--served-model-name q27 --host 127.0.0.1 --port "$PORT"
  --max-model-len "$MAX_LEN" --gpu-memory-utilization 0.90 --max-num-seqs "$MAX_SEQS" --max-num-batched-tokens 2048
  --kernel-config '{"linear_backend_per_quant":{"nvfp4_w4a4":"marlin","fp8_w8a8":"marlin"}}'
  --kv-cache-dtype fp8 --mamba-ssm-cache-dtype float32 --attention-config '{"backend":"FLASHINFER"}'
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
  --reasoning-parser qwen3 --default-chat-template-kwargs '{"enable_thinking":true}'
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
  --generation-config vllm --limit-mm-per-prompt '{"image":0,"video":0}')
if [ "$KV_BYTES" != 0 ]; then args+=(--kv-cache-memory-bytes "$KV_BYTES"); fi
if [ "$PREFIX_CACHE" = 1 ]; then
  args+=(--enable-prefix-caching --mamba-cache-mode align --prefix-match-unit 64)
else
  args+=(--no-enable-prefix-caching)
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 -e VLLM_NO_USAGE_STATS=1 "${mounts[@]}" \
  --entrypoint vllm "$IMAGE" serve "$model_arg" "${args[@]}" > /dev/null
echo "starting $NAME: $MODEL, max-model-len $MAX_LEN, prefix cache $PREFIX_CACHE (first start downloads about 18 GB)"
for i in $(seq 1 720); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; then echo "READY after $((i * 5)) s"; exit 0; fi
  if ! docker ps -q -f "name=^${NAME}$" | grep -q .; then echo DIED; docker logs "$NAME" 2>&1 | tail -40; exit 1; fi
  sleep 5
done
echo TIMEOUT; docker logs "$NAME" 2>&1 | tail -20; exit 1
