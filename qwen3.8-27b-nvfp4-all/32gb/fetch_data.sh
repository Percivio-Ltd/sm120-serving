#!/usr/bin/env bash
# fetch_data.sh -- the two public inputs of bench.py and quality.py, into ./data (or $DATA):
#   the Django 5.1 source (git tag 5.1), from which the 64K/128K/256K prompts and needle haystacks are built, and
#   the GSM8K test split (1,319 problems) from openai/grade-school-math, checked against its SHA-256.
set -euo pipefail
DATA=${DATA:-data}
mkdir -p "$DATA"
[ -d "$DATA/django/.git" ] || git clone -q --depth 1 --branch 5.1 https://github.com/django/django "$DATA/django"
if [ ! -f "$DATA/gsm8k_test.jsonl" ]; then
  curl -fsSL -o "$DATA/gsm8k_test.jsonl.part" \
    https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
  mv "$DATA/gsm8k_test.jsonl.part" "$DATA/gsm8k_test.jsonl"
fi
echo "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14  $DATA/gsm8k_test.jsonl" | sha256sum -c -
echo "data ready in $DATA"
