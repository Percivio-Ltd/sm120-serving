#!/usr/bin/env bash
# run_all.sh -- the README's measurement on your card: fetch the data, then
#   1. bench: 64K/128K/256K prompts, three tasks each, 2,048 output tokens, prefix caching off (as measured);
#   2. quality: needle retrieval at 64K/128K/200K (6 per size) and the first 200 GSM8K problems (4 at a time),
#      prefix caching on.
# Takes about 1.5 hours on an RTX PRO 4500. Results go to ./results. Pass MODEL=/path/to/local/copy to skip the download.
set -euo pipefail
cd "$(dirname "$0")"
bash fetch_data.sh
LABEL=${LABEL:-$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | tr ' ' '-')}
mkdir -p results
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv > "results/$LABEL.gpu.csv"
PREFIX_CACHE=0 bash serve.sh
python3 -I bench.py --label "$LABEL-bench" --long-src data/django/django --contexts 64,128,256 --out results
PREFIX_CACHE=1 MAX_SEQS=4 bash serve.sh
python3 -I quality.py --label "$LABEL-quality" --long-src data/django/django --gsm8k-file data/gsm8k_test.jsonl \
  --needles 6 --gsm8k 200 --out results
docker logs "${NAME:-q27}" > "results/$LABEL.server.log" 2>&1
docker rm -f "${NAME:-q27}" > /dev/null
echo "done; results in $(pwd)/results"
