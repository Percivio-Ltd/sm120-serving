#!/bin/bash
# build_dn4g.sh -- build the dn4g checkpoint from nvidia/Qwen3.8-Flash-Next-NVFP4 (revision fc694b54) on one GPU, inside
# the pinned vLLM image, then check its ten shards and three JSON files against the md5 lists of the measured build.
#   NVFP4_DIR  the NVIDIA release (25 files); DN4G_DIR  the new checkpoint, which must not exist yet and must be in the
#   same directory as NVFP4_DIR (unchanged files are hard-linked, not copied). GPU (default 0).
# The converter re-quantizes the 108 Gated DeltaNet projections (in_proj_qkv, in_proj_z and out_proj of the 36 GDN
# layers) from BF16 to NVFP4, rewrites shards 1-8 and three JSON files, and links everything else. It needs about
# 66 GB of new disk space. The new files are written by the container's root user.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=vllm/vllm-openai@sha256:3379c8e773c619972cc81e9f1db648c38ab5029369d71c7a24149753666c6480  # 0.32.1rc1.dev435+gc50aca144
: "${NVFP4_DIR:?set NVFP4_DIR to the local copy of nvidia/Qwen3.8-Flash-Next-NVFP4 at fc694b54}"
: "${DN4G_DIR:?set DN4G_DIR to the checkpoint to create, next to NVFP4_DIR}"
GPU=${GPU:-0}
SRC=$(cd "$NVFP4_DIR" && pwd)
PARENT=$(dirname "$SRC")
[ -e "$DN4G_DIR" ] && { echo "$DN4G_DIR exists; not overwritten" >&2; exit 1; }
[ "$(cd "$(dirname "$DN4G_DIR")" && pwd)" = "$PARENT" ] || { echo "DN4G_DIR must be in $PARENT (hard links)" >&2; exit 1; }
OUT=$(basename "$DN4G_DIR")
docker run --rm --gpus "\"device=$GPU\"" --network none -v "$PARENT:/models" -v "$HERE:/w:ro" \
  --entrypoint python3 "$IMAGE" -I /w/flash_nvfp4_convert.py "/models/$(basename "$SRC")" "/models/$OUT" \
  --groups gdn --device cuda 2>&1 | tee "$PARENT/$OUT.convert.log"
grep -q '^done: 108 modules NVFP4' "$PARENT/$OUT.convert.log" || { echo "converter did not report 108 modules" >&2; exit 1; }
# The rewritten shards are root-owned; check them as root in a read-only, no-network container.
docker run --rm --network none --read-only -v "$PARENT/$OUT:/m:ro" -v "$HERE:/w:ro" -w /m \
  --entrypoint md5sum "$IMAGE" -c /w/dn4g-shards.md5 /w/dn4g-json.md5 \
  || { echo "dn4g differs from the measured checkpoint: do not serve it as this stack" >&2; exit 1; }
echo "dn4g OK: $PARENT/$OUT matches the measured checkpoint (10 shards, 3 JSON files)"
