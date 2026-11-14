#!/usr/bin/env bash
# Serve DeepSeek-V4.1-Flash-1017 (calibrated EXL3 3-bit routed experts, int4 Engram, EXL3 3-bit DSpark drafter experts) with
# vLLM TP2 on 2x RTX PRO 6000 Blackwell (sm_120, 96 GB each). The defaults below are the configuration behind the model card
# numbers, except CPU_EXPERTS (see below).
# Memory layout: the most-used routed experts live in VRAM; the coldest 5,296 target experts (plans/plan-rt-5296.json,
# fitted to the expert routing of our own agent tasks; see plans/README.md) stay in pinned host RAM, and a 512-slot VRAM cache per GPU holds
# copies of the host experts that recent decode steps used most. Host experts run next to the VRAM experts: on a side stream
# that reads them over PCIe (default), or on a CPU core pool (CPU_EXPERTS=1). Prefill runs the VRAM experts through the
# decode-once kernel with int8 tensor-core math (kernels/xmoe_do_i8, EXL3_DO=1 DO_INT8=1). The Engram tables (int4) and the
# embedding table are pinned host RAM too, read over UVA. The container uses ~230 GiB of host RAM; CONTAINER_MEM caps it.
# CPU_EXPERTS=0 (default): GPU-only path, runs on any host with enough RAM.
# CPU_EXPERTS=1: host experts of decode steps and prefill chunks are computed by a per-GPU pool of CPU cores (cpuexp/): needs a
#   big AVX-512 host CPU with high memory bandwidth (tested: Threadripper PRO 9995WX, 96 cores, 8-channel DDR5-4800) and
#   cpuexp/build.sh run on this host. serve.sh runs cpuexp/check_host.sh first and falls back to CPU_EXPERTS=0 if it fails
#   (CPU_EXPERTS_STRICT=1: stop instead). On the tested host: +17% agentic decode in decode-only form, then -6% / -9%
#   agentic wall time (1 / 4 sessions) with prefill on the CPU too.
# Chat prompts are rendered and tokenized like the official DeepSeek API and tool calls are parsed from emitted DSML tokens only
# (patches/vllm/v41-encoder-api.diff, v41-parser-strict-dsml.diff, v41-namespaces-thinking.diff; the VLLM_DS41_* switches below,
# all on).
# Text and image input (VISION=1). Foreground; Ctrl-C removes the container. Requires the engine key in API_KEY_FILE (below; clients send it as the Bearer token).
# PROFILE=2x (default): the 2-GPU configuration above. PROFILE=8x REPLICA=0|1: one of the two replicas of an 8x RTX PRO 6000
#   node (TP4 attention + EP4 experts on GPUs 0-3 or 4-7, every expert in VRAM, NUMA-pinned); serve/profile-8x.env holds its
#   settings and serve/serve-8x.sh starts both replicas. The settings below that the profile also sets are noted there.
# DS41_DRAFTER=released (default) | code: the DSpark head. code = drafter-code-k7/ in the pack (a head fine-tuned for code and
#   agentic tool-use workloads); the default for PROFILE=8x, an opt-in for PROFILE=2x (use with SPEC_K=7; README).
set -u
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# The engine key: read from API_KEY_FILE, never put on a command line. Default: .eb_key two levels above the
# recipe (the work directory, where the clients read it; outside the recipe, which is mounted into the container). It must
# be a regular file (not a symlink) owned by this user, mode 600, holding one line of printable characters without spaces.
# A caller's API_KEY, if set, must equal it (callers that set it from the same file keep working); it is then unset. The
# key reaches vllm only through a read-only mount and the entrypoint below (VLLM_API_KEY, vLLM's fallback for --api-key),
# so no command line (ps), no docker inspect field and no container env set by docker run carries it.
API_KEY_FILE=${API_KEY_FILE:-$(cd "$REPO/../.." && pwd)/.eb_key}
[[ -f $API_KEY_FILE && ! -L $API_KEY_FILE ]] \
  || { echo "[serve] API_KEY_FILE=$API_KEY_FILE is missing or not a regular file (a symlink is refused); not starting"; exit 1; }
[[ -n $(find "$API_KEY_FILE" -maxdepth 0 -type f -user "$(id -u)" -perm 600) ]] \
  || { echo "[serve] API_KEY_FILE=$API_KEY_FILE must be owned by $(id -un) with mode 600; not starting"; exit 1; }
unset SERVE_KEY; SERVE_KEY=$(<"$API_KEY_FILE")   # not exported: no child process inherits it
[[ -n $SERVE_KEY && $SERVE_KEY != *[![:graph:]]* ]] \
  || { echo "[serve] API_KEY_FILE=$API_KEY_FILE must hold one key: one line of printable characters, no spaces; not starting"; exit 1; }
[[ -z ${API_KEY+x} || $API_KEY == "$SERVE_KEY" ]] \
  || { echo "[serve] API_KEY in the environment is not the key in API_KEY_FILE=$API_KEY_FILE; not starting"; exit 1; }
unset API_KEY
PROFILE=${PROFILE:-2x}
case $PROFILE in
  2x) ;;
  8x) source "$REPO/serve/profile-8x.env" || exit 1 ;;   # sets only what the caller left unset
  *) echo "PROFILE=$PROFILE: use 2x or 8x"; exit 1 ;;
esac
# Downloaded from Hugging Face, this script lives in <pack>/recipe/serve, so the pack is the recipe's parent directory.
if [[ -z ${PACK:-} && -f $REPO/../config.json ]]; then PACK=$(cd "$REPO/.." && pwd); fi
MODELS_DIR=${MODELS_DIR:-$HOME/models}
PACK=${PACK:-$MODELS_DIR/DeepSeek-V4.1-Flash-1017-EXL3-3bpw}
[[ $PACK == "$MODELS_DIR"/* ]] || MODELS_DIR=$(dirname "$PACK")   # the pack's parent is mounted at the same path
IMAGE=${IMAGE:-dsv41-flash-1017-exl3-3bpw-sm120}
NAME=${NAME:-dsv41-flash-exl3}
PORT=${PORT:-8000}
SERVED=${SERVED:-deepseek-v4.1-flash}
CACHE_DIR=${CACHE_DIR:-$REPO/.cache}        # TileLang/Triton/FlashInfer kernel caches (later boots mostly load)
CONTAINER_MEM=${CONTAINER_MEM:-260g}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1048576}      # 1M tokens (the model's full window); 524288 + KV_MEM=4294967296 = a 512K setup
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
MAX_BATCHED=${MAX_BATCHED:-16384}           # prefill chunk; 16K beats 8K on fan-out and 4-stream decode
KV_MEM=${KV_MEM:-4563402752}                # 4.25 GiB fp8 KV = 1,071,558 tokens: one full 1M request (5.25 GiB = 1.32M)
NO_CUSTOM_AR=${NO_CUSTOM_AR-1}              # vLLM's own custom all-reduce stays off (no gain here); DS41_CAR_PCIE_IPC below
LINEAR_BACKEND=${LINEAR_BACKEND-marlin}     # small dense MXFP8 layers; the big ones use the one-copy FP8 path below
XMOE_EXT=${XMOE_EXT-xmoe}                   # kernels/build/xmoe (kernels/build.sh); empty = stock ExLlamaV3 exl3_moe
EXL3_DO=${EXL3_DO-1}                        # 1 = decode-once prefill MoE kernel; 0 = the fused kernel
DO_INT8=${DO_INT8:-1}                       # with EXL3_DO=1: 1 = its int8 tensor-core build (kernels/build/xmoe_do_i8); 0 = the
                                            #   fp16 build (kernels/build/xmoe_do, bit-exact with the fused kernel's math)
UTIL=${UTIL:-0.92}
EAGER=${EAGER:-0}                           # 1 = no CUDA graphs
CG_REPLAY=${CG_REPLAY:-false}               # true = vLLM's decoder-replay CUDA graphs (upstream default; measured at parity here)
SPEC=${SPEC:-dspark}                        # none = no speculative decoding
SPEC_K=${SPEC_K:-5}                         # DSpark draft length; CAPTURE_SIZES must reach MAX_NUM_SEQS x (1 + SPEC_K)
DS41_DRAFTER=${DS41_DRAFTER:-released}      # released = the pack's DSpark head; code = drafter-code-k7/ (see the header)
DRAFTER_DIR=${DRAFTER_DIR:-$PACK/drafter-code-k7}
KV_OFFLOAD_GB=${KV_OFFLOAD_GB:-0}           # >0: vLLM's CPU KV offload tier of this many GB (native backend; the 8x profile: 150)
FI_ARCH=${FI_ARCH:-12.0f}
THINKING=${THINKING:-default}               # default = model decides (on, effort 50); off = force thinking off
VISION=${VISION:-1}                         # 1 = load the vision encoder (0.77 GiB bf16; images via the image_url API); 0 = text only
# Host-expert plan: plan-rt-5296.json, fitted to the expert routing of our own agent tasks. Against the recipe's plan it took
# 2.42% less time per verify step on the mean, lower in 5 of 5 paired blocks, one request at a time (not task time).
# PLAN=plan-1017-agent30-c512.json serves the previous plan (also 5,296 host experts, with HOST_CACHE=512).
PLAN=${PLAN:-plan-rt-5296.json}             # host-expert plan in plans/; see plans/README.md
HOST_CACHE=${HOST_CACHE:-512}               # VRAM slots per GPU for copies of host experts; must match the plan (README there)
CPU_EXPERTS=${CPU_EXPERTS:-0}               # 1 = host experts on a CPU core pool (cpuexp/README.md); 0 = GPU-only (PCIe) path
CPU_EXPERTS_PREFILL=${CPU_EXPERTS_PREFILL:-1}   # with CPU_EXPERTS=1: also the host experts of prefill chunks
WARMUP=${WARMUP:-1}                         # compile the remaining TileLang/Triton kernels before serving clients
[[ -f $PACK/config.json ]] || { echo "no config.json in PACK=$PACK"; exit 1; }
[[ -z ${XMOE_EXT:-} || -d $REPO/kernels/build/$XMOE_EXT ]] || { echo "kernels/build/$XMOE_EXT missing: run kernels/build.sh (or XMOE_EXT= for the stock kernel)"; exit 1; }
DO_MOD=${EXL3_DO_MOD:-$([[ $DO_INT8 == 1 ]] && echo xmoe_do_i8 || echo xmoe_do)}
[[ $EXL3_DO != 1 || -d $REPO/kernels/build/$DO_MOD ]] || { echo "kernels/build/$DO_MOD missing: run kernels/build.sh (or DO_INT8=0 / EXL3_DO=0)"; exit 1; }
DS41_FP8AR=${DS41_FP8AR:-1}                 # 1 = 8-bit prefill all-reduce (kernels/fp8ar); 0 = the exact NCCL all-reduce
[[ $DS41_FP8AR != 1 || -d $REPO/kernels/build/fp8ar_ext ]] || { echo "kernels/build/fp8ar_ext missing: run kernels/build.sh (or DS41_FP8AR=0)"; exit 1; }
DS41_ROWTOPK=${DS41_ROWTOPK:-1}             # 1 = length-aware indexer decode top-k (kernels/rowtopk, exact); 0 = vLLM persistent_topk
[[ $DS41_ROWTOPK != 1 || -f $REPO/kernels/build/rowtopk/rowtopk.so ]] || { echo "kernels/build/rowtopk missing: run kernels/build.sh rowtopk (or DS41_ROWTOPK=0)"; exit 1; }
[[ -f $REPO/plans/$PLAN ]] || { echo "plans/$PLAN missing"; exit 1; }
[[ ${DS41_AR8_GRAPH:-0} != 1 || -d $REPO/kernels/build/fp8arg_ext ]] || { echo "kernels/build/fp8arg_ext missing: run kernels/build.sh fp8arg (or kernels/build.sh 8x)"; exit 1; }
case $DS41_DRAFTER in
  released) DRAFT_MODEL= ;;
  code) for i in 44 45 46; do [[ -f $DRAFTER_DIR/model-000$i-of-00048.safetensors ]] || { echo "DS41_DRAFTER=code: $DRAFTER_DIR/model-000$i-of-00048.safetensors missing (download drafter-code-k7/ from the repo)"; exit 1; }; done
        [[ $DRAFTER_DIR == "$MODELS_DIR"/* ]] || { echo "DRAFTER_DIR must be under MODELS_DIR=$MODELS_DIR (mounted into the container)"; exit 1; }
        DRAFT_MODEL=$DRAFTER_DIR ;;
  *) echo "DS41_DRAFTER=$DS41_DRAFTER: use released or code"; exit 1 ;;
esac
if [[ $CPU_EXPERTS == 1 ]]; then
  why=""
  [[ -f $REPO/cpuexp/build/libcpuexp_rt.so && -f $REPO/cpuexp/build/libcpuexp_gpu.so ]] || why="cpuexp/build missing: run cpuexp/build.sh on this host"
  [[ -n $why ]] || bash "$REPO/cpuexp/check_host.sh" || why="cpuexp/check_host.sh failed (see above)"
  if [[ -n $why ]]; then
    [[ ${CPU_EXPERTS_STRICT:-0} == 1 ]] && { echo "CPU_EXPERTS=1: $why"; exit 1; }
    echo "CPU_EXPERTS=1 requested but $why -> serving on the GPU-only path (CPU_EXPERTS=0)"; CPU_EXPERTS=0
  fi
fi
if [[ $CPU_EXPERTS == 1 ]]; then
  export EXL3_CPU_EXPERTS=1 EXL3_CPU_EXPERTS_LIB=${EXL3_CPU_EXPERTS_LIB:-/opt/recipe/cpuexp/build}
  [[ $CPU_EXPERTS_PREFILL == 1 ]] && export EXL3_CPU_EXPERTS_PREFILL=1
  # the pool's schedule: 2 regroups the fp32 K partials of decode-sized jobs (served since 1 Nov: 2.07% less time per verify
  # step in our A/B; its fidelity check was not conclusive, so this is a policy call, easy to revert); 1 = the old schedule
  export EXL3_CPU_EXPERTS_SCHED=${EXL3_CPU_EXPERTS_SCHED:-2}
fi
export EXL3_DO EXL3_DO_MOD=$DO_MOD           # the plugin loads kernels/build/$EXL3_DO_MOD (EXL3_DO_DIR overrides the directory)
mkdir -p "$CACHE_DIR"
# Runtime knobs (all read by patches/vllm/sitecustomize.py or the patched vllm-exl3 plugin). Each can be overridden from the
# caller's environment; recipe/README.md explains them.
export EXL3_HOST_EXPERTS=${EXL3_HOST_EXPERTS:-/opt/recipe/plans/$PLAN}
export EXL3_HOST_CACHE=${EXL3_HOST_CACHE:-$HOST_CACHE}                     # VRAM cache of host experts (slots per GPU)
export EXL3_HOST_CACHE_HOT=${EXL3_HOST_CACHE_HOT:-1}                       # cached experts run in the hot (all-SM) decode launch
export EXL3_HOST_OVERLAP_SMS=${EXL3_HOST_OVERLAP_SMS:-32}                  # SMs for the cold (host) experts during decode
export EXL3_HOST_OVERLAP_SMS_PREFILL=${EXL3_HOST_OVERLAP_SMS_PREFILL:-64}  # ... during prefill
export EXL3_HOST_OVERLAP_GROUP=${EXL3_HOST_OVERLAP_GROUP:-16}
export EXL3_FAT_SLICE=${EXL3_FAT_SLICE:-0}
export EXL3_LEAN_PREP=${EXL3_LEAN_PREP:-1}                                 # lean routing prep on decode-sized calls
export EXL3_FUSED_PREP=${EXL3_FUSED_PREP:-1}                               # decode routing prep in one fused kernel
export EXL3_PREP_IO=${EXL3_PREP_IO:-1}                                     # ... which also writes the kernel's fp16 input
export DS41_MHC_SPLITS=${DS41_MHC_SPLITS:-80}                              # mHC prenorm GEMM split cap (stock 16)
export DS41_DENSE_GEMV=${DS41_DENSE_GEMV:-1}                               # split-K FP8 GEMV for dense layers at decode sizes
export DS41_ROUTER_SPLITK=${DS41_ROUTER_SPLITK:-1}                         # router GEMM split-K + Triton partial-sum reduce
export DS41_DENSE_ONE_COPY=${DS41_DENSE_ONE_COPY:-1}                       # big dense layers keep one FP8 copy (no Marlin repack)
export DS41_DENSE_GEMV_MAX=${DS41_DENSE_GEMV_MAX:-64} DS41_DENSE_MBLOCK=${DS41_DENSE_MBLOCK:-64}
export DS41_DENSE_PY=${DS41_DENSE_PY:-/opt/recipe/kernels/sm120_dense2.py}
export DS41_DENSE_TABLE=${DS41_DENSE_TABLE:-/opt/recipe/kernels/sm120_dense_table_v2.json}
export DS41_WOA_FP8=${DS41_WOA_FP8:-1}                                     # attention wo_a held FP8 (exact requantize)
export DS41_WOA_ACTQ=${DS41_WOA_ACTQ:-32}                                  # attention output rounded through FP8 before wo_a, as
export DS41_WOA_ACTQ_TRITON=${DS41_WOA_ACTQ_TRITON:-1}                     #   the 1017 reference does (one fused Triton pass)
export DS41_EMBED_HOST=${DS41_EMBED_HOST:-1}                               # embedding table in pinned host RAM
export DS41_INDEXER_BUF_MULT=${DS41_INDEXER_BUF_MULT:-8}                   # indexer prefill workspace = max_model_len x 8
export DS41_INDEXER_EXPAND_CAP=${DS41_INDEXER_EXPAND_CAP:-1}               # indexer decode block-table buffer trimmed
export DS41_ROPE_CAP=${DS41_ROPE_CAP:-1}                                   # rope cache sized to max_model_len
export DS41_FP8_LMHEAD=${DS41_FP8_LMHEAD:-1}                               # FP8 lm_head
export DS41_FUSE_ROUTER=${DS41_FUSE_ROUTER:-1}                             # decode fusions (bit-exact): router split-K reduce
export DS41_FUSE_HCPREP=${DS41_FUSE_HCPREP:-1}                             #   in the top-k kernel, cache hotness counts in the
export DS41_FUSE_IDX=${DS41_FUSE_IDX:-1}                                   #   routing prep, sparse-MLA top-k + pad fix, inverse
export DS41_FUSE_OPROJ=${DS41_FUSE_OPROJ:-1}                               #   RoPE + FP8 rounding of the attention output
export DS41_CAR=${DS41_CAR:-1} DS41_CAR_PCIE_IPC=${DS41_CAR_PCIE_IPC:-1}   # FlashInfer PCIe IPC all-reduce in the decode graphs
export DS41_FP8AR=${DS41_FP8AR:-1}                                         # prefill all-reduces (>= 256 rows): int8 + per-32 bf16
                                                                           #   scales pushed over PCIe P2P, bit-identical on both ranks
export DS41_ROWTOPK=${DS41_ROWTOPK:-1}                                     # indexer decode top-k reads only each row's valid prefix
                                                                           #   (vLLM's scales with max_model_len); exact, + bounded mask
export DENSE_HYBRID_M=${DENSE_HYBRID_M:-128}                               # FlashInfer MXFP8 for dense prefill rows > 128
# vLLM patches in the image (patches/vllm/v41-*.diff); each switch is on by default, 0 restores the previous behavior:
export VLLM_DS41_REPLAY_FINAL_WINDOW=${VLLM_DS41_REPLAY_FINAL_WINDOW:-1}   # decoder replay: chunks the prompt's final window
                                                                           #   cannot read keep one placeholder row (exact)
export VLLM_DS41_SEGMENT_ENCODE=${VLLM_DS41_SEGMENT_ENCODE:-1}             # control tokens only from the template; client text
                                                                           #   that spells one stays plain text
export VLLM_DS41_EFFORT_ALIASES=${VLLM_DS41_EFFORT_ALIASES:-1}             # reasoning_effort minimal/medium/ultra accepted
export VLLM_DS41_API_TOOLS=${VLLM_DS41_API_TOOLS:-1}                       # tool schemas as {name, description, parameters}
export VLLM_DS41_API_SYSTEM=${VLLM_DS41_API_SYSTEM:-1}                     # tools + response format in a leading system message
export VLLM_DS41_API_SYSTEM_CLEAN=${VLLM_DS41_API_SYSTEM_CLEAN:-1}         # billing-header prefix stripped, empty system dropped
export VLLM_DS41_DEVELOPER_AS_SYSTEM=${VLLM_DS41_DEVELOPER_AS_SYSTEM:-1}   # a developer message is a system message
export VLLM_DS41_RESPONSE_FORMAT=${VLLM_DS41_RESPONSE_FORMAT:-1}           # response_format json_object / json_schema rendered
export VLLM_DS41_TOOL_CHOICE_NONE=${VLLM_DS41_TOOL_CHOICE_NONE:-1}         # tool_choice none renders no tools
export VLLM_DS41_SYSTEM_TURN=${VLLM_DS41_SYSTEM_TURN:-1}                   # system-only prompt gets the assistant header
export VLLM_DS41_DSML_STRICT=${VLLM_DS41_DSML_STRICT:-1}                   # tool calls only from emitted DSML tokens
export VLLM_DS41_NAMESPACES=${VLLM_DS41_NAMESPACES:-1}                     # tool namespaces render as "ns::name" (1017 reference)
export VLLM_DS41_THINKING_PRECEDENCE=${VLLM_DS41_THINKING_PRECEDENCE:-1}   # thinking=false wins over reasoning_effort
# V4.1 prompt region cache (patches/vllm/sitecustomize.py's rtx8 hook): repeated prompt regions are encoded once and
# reused. Served since 1 Nov (5.57 s less server time per pass of our 6-task agent test outside queueing and prefill,
# prompt ids identical on all eight test starts); 0 encodes every prompt as before. DS41_PROMPT_IDS_LOG (a measurement log) stays off.
export DS41_REGION_CACHE=${DS41_REGION_CACHE:-1}
# Moet's sm_120 mHC overlap (vllm-moet): the patched ops/mhc.py and Moet's post-norm kernel, built for this image, mounted
# from MHC_OVERLAP_DIR; decode batches of at most 64 tokens run the post-mix, collapse and RMSNorm in one kernel with the
# projection and Sinkhorn on a side stream, larger batches the image's path. Served since 2 Nov: 4.88% less
# time per verify step, lower in 5 of 5 paired blocks (replayed steps, not task time), after a fidelity check. MHC_OVERLAP=0 serves the image's own
# ops/mhc.py as before; a host without the directory needs it. The directory must match its SHA256SUMS (README.md,
# "The mHC overlap") with exactly one .so; ext/ stays writable because torch's JIT load takes a lock file in it.
MHC_OVERLAP=${MHC_OVERLAP:-1}
MHC_OVERLAP_DIR=${MHC_OVERLAP_DIR:-$(cd "$REPO/../.." && pwd)/mhc-overlap}
if [[ $MHC_OVERLAP == 1 ]]; then
  [[ -f $MHC_OVERLAP_DIR/patched/mhc.py && -d $MHC_OVERLAP_DIR/moet/mhc_overlap && -d $MHC_OVERLAP_DIR/ext && -f $MHC_OVERLAP_DIR/SHA256SUMS ]] \
    || { echo "[serve] MHC_OVERLAP=1: $MHC_OVERLAP_DIR lacks patched/mhc.py, moet/mhc_overlap, ext or SHA256SUMS (MHC_OVERLAP=0 serves without the overlap)"; exit 1; }
  ( cd "$MHC_OVERLAP_DIR" && sha256sum --quiet -c SHA256SUMS ) && [[ $(find "$MHC_OVERLAP_DIR/ext" -name '*.so' -type f | grep -c .) == 1 ]] \
    || { echo "[serve] MHC_OVERLAP=1: $MHC_OVERLAP_DIR does not match its SHA256SUMS or holds other than one .so (README.md); not starting"; exit 1; }
  DOCKER_EXTRA="${DOCKER_EXTRA:-} -v $MHC_OVERLAP_DIR/patched/mhc.py:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/nvidia/ops/mhc.py:ro -v $MHC_OVERLAP_DIR/moet/mhc_overlap:/opt/moet/mhc_overlap:ro -v $MHC_OVERLAP_DIR/ext:/opt/moet/ext -e VLLM_MOET_MHC_OVERLAP_DIR=/opt/moet/mhc_overlap -e TORCH_EXTENSIONS_DIR=/opt/moet/ext"
fi
extra=()
# default: these sizes up to MAX_NUM_SEQS x (1 + SPEC_K), plus that maximum (8 seqs x (1 + 5 DSpark tokens) = 48)
if [[ -z ${CAPTURE_SIZES:-} ]]; then
  cmax=$(( MAX_NUM_SEQS * (1 + $([[ $SPEC == dspark ]] && echo "$SPEC_K" || echo 0)) )); CAPTURE_SIZES=""
  for c in 1 2 4 6 8 12 16 24 32 48 64 96 128 192 256 384 512; do (( c < cmax )) && CAPTURE_SIZES+="$c "; done; CAPTURE_SIZES+="$cmax"
fi
[[ $EAGER == 1 ]] && extra+=(--enforce-eager) || extra+=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"],"cudagraph_decoder_replay":'"$CG_REPLAY"'}' --cudagraph-capture-sizes $CAPTURE_SIZES)
[[ -n ${NO_CUSTOM_AR:-} ]] && extra+=(--disable-custom-all-reduce)
[[ -n ${LINEAR_BACKEND:-} ]] && extra+=(--linear-backend "$LINEAR_BACKEND")
[[ $THINKING == off ]] && extra+=(--default-chat-template-kwargs '{"thinking":false,"reasoning_effort":"low"}')
[[ $SPEC == dspark ]] && extra+=(--speculative-config '{"method":"dspark",'"${DRAFT_MODEL:+\"model\":\"$DRAFT_MODEL\",}"'"num_speculative_tokens":'"$SPEC_K"',"draft_sample_method":"probabilistic"}')
[[ $KV_OFFLOAD_GB != 0 ]] && extra+=(--kv-offloading-size "$KV_OFFLOAD_GB" --kv-offloading-backend native)
# pass-through into the container: EXL3_*/VLLM_EXL3_*, DS41_*, VLLM_DS41_*, XMOE_*, NCCL_PROTO/ALGO, DENSE_HYBRID_M
PASS_ENV=$(env | grep -E "^(VLLM_EXL3_[A-Z0-9_]+|EXL3_[A-Z0-9_]+|DS41_[A-Z0-9_]+|VLLM_DS41_[A-Z0-9_]+|XMOE_[A-Z_]+|NCCL_PROTO|NCCL_ALGO|DENSE_HYBRID_M)=" | sed "s/^/-e /" | tr "\n" " ")
# an image built before the v41-* vLLM patches would ignore the VLLM_DS41_* switches: say so (docker/build.sh labels its series)
docker image inspect -f '{{index .Config.Labels "ds41.vllm.series"}}' "$IMAGE" 2>/dev/null | grep -q v41-parser-strict-dsml \
  || echo "[serve] WARNING: image $IMAGE predates this recipe's vLLM patches (no ds41.vllm.series label): rebuild it with docker/build.sh"
docker image inspect -f '{{index .Config.Labels "ds41.vllm.series"}}' "$IMAGE" 2>/dev/null | grep -q v41-namespaces-thinking \
  || echo "[serve] NOTE: image $IMAGE predates patches/vllm/v41-namespaces-thinking.diff (tool namespaces, thinking precedence): rebuild it with docker/build.sh"
if [[ ${DS41_IDX_CHUNK:-0} == 1 ]] && ! docker image inspect -f '{{index .Config.Labels "ds41.vllm.series"}}' "$IMAGE" 2>/dev/null | grep -q v41-indexer-chunked-decode; then
  echo "[serve] DS41_IDX_CHUNK=1 needs patches/vllm/v41-indexer-chunked-decode.diff in the image: rebuild it with docker/build.sh (or MAX_MODEL_LEN=524288 DS41_IDX_CHUNK=0)"; exit 1
fi

# The container's entrypoint: sh reads the engine key from the read-only mount into VLLM_API_KEY and execs vllm with the
# arguments below (vllm stays the container's pid 1, as with --entrypoint vllm); an empty or unreadable key stops the start
# (vLLM with no key would serve without authentication).
KEY_ENTRY='VLLM_API_KEY=$(cat /run/secrets/engine-key) && [ -n "$VLLM_API_KEY" ] && export VLLM_API_KEY && exec vllm "$@"; echo "[serve] no engine key in /run/secrets/engine-key inside the container; not starting" >&2; exit 1'
cleanup() { [ -n "${WPID:-}" ] && kill $WPID 2>/dev/null; docker rm -f "$NAME" >/dev/null 2>&1; }
trap 'cleanup; exit 130' INT TERM
echo "[serve] profile $PROFILE${REPLICA:+ replica $REPLICA}, GPUs ${DOCKER_GPUS:-all}, TP ${TP:-2}, DSpark $([[ $SPEC == dspark ]] && echo "k=$SPEC_K head $DS41_DRAFTER" || echo off)"
echo "[serve] pack $PACK, image $IMAGE, plan $PLAN, host cache $EXL3_HOST_CACHE, CPU host experts $([[ $CPU_EXPERTS == 1 ]] && echo "on (prefill $CPU_EXPERTS_PREFILL)" || echo off), 8-bit prefill all-reduce $([[ $DS41_FP8AR == 1 ]] && echo on || echo off), decode-once kernel $([[ $EXL3_DO == 1 ]] && echo $DO_MOD || echo off)"
if [[ $WARMUP == 1 ]]; then
  # the Authorization header reaches curl through a pipe (@<(printf ...), printf a builtin), not its command line
  ( t0=$(date +%s); until curl -sf -o /dev/null -H @<(printf 'Authorization: Bearer %s\n' "$SERVE_KEY") "http://127.0.0.1:$PORT/v1/models"; do sleep 5; [ $(( $(date +%s)-t0 )) -gt 3600 ] && exit 0; done
    echo "[warmup] start"; w0=$(date +%s)
    req() { curl -s -o /dev/null -H @<(printf 'Authorization: Bearer %s\n' "$SERVE_KEY") -H 'Content-Type: application/json' "http://127.0.0.1:$PORT/v1/completions" \
      -d "{\"model\":\"$SERVED\",\"prompt\":\"$(python3 -c "print(('warm up the kernels with some text. ' * $1).strip())")\",\"max_tokens\":$2,\"temperature\":0}"; }
    req 400 24; req 1600 24; for i in 1 2 3 4; do req 300 48 & done; wait; req 12 64
    echo "[warmup] done in $(( $(date +%s)-w0 ))s" ) &
  WPID=$!
fi
docker run --rm --name "$NAME" --gpus "${DOCKER_GPUS:-all}" ${DOCKER_EXTRA:-} --network host --ipc host --shm-size 32g \
  --memory="$CONTAINER_MEM" --memory-swap="$CONTAINER_MEM" --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
  -v "$MODELS_DIR":"$MODELS_DIR":ro -v "$CACHE_DIR":/root/.cache -v "$REPO":/opt/recipe:ro \
  -v "$REPO/patches/vllm/sitecustomize.py":/usr/lib/python3.12/sitecustomize.py:ro \
  -v "$API_KEY_FILE":/run/secrets/engine-key:ro \
  ${XMOE_EXT:+-e XMOE_EXT=$XMOE_EXT} \
  -e TORCH_CUDA_ARCH_LIST=12.0a -e FLASHINFER_CUDA_ARCH_LIST="$FI_ARCH" -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0} \
  -e VLLM_NO_USAGE_STATS=1 -e DO_NOT_TRACK=1 \
  ${NO_CUSTOM_AR:+-e PYTORCH_CUDA_ALLOC_CONF=${ALLOC_CONF-expandable_segments:True}} \
  -e VLLM_PLUGINS=vllm_exl3 -e DSV41_ENGRAM_DISK=0 -e DSV41_ENGRAM_INT4=1 -e VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=0 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_DEBUG=WARN \
  -e TILELANG_CACHE_DIR=/root/.cache/tilelang -e TRITON_CACHE_DIR=/root/.cache/triton \
  $PASS_ENV \
  --entrypoint sh "$IMAGE" -c "$KEY_ENTRY" vllm serve "$PACK" \
  --tensor-parallel-size "${TP:-2}" ${PARALLEL_ARGS:-} --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" --kv-cache-dtype fp8 --kv-cache-memory "$KV_MEM" \
  --gpu-memory-utilization "$UTIL" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED" \
  --kernel-config '{"enable_flashinfer_autotune":false,"enable_jit_warmup":false}' \
  --block-size 64 --quantization exl3 $([[ $VISION == 1 ]] || echo --language-model-only) \
  --tokenizer-mode deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v41 \
  --served-model-name "$SERVED" --trust-remote-code "${extra[@]}" ${EXTRA_ARGS:-}
rc=$?; cleanup; exit $rc
