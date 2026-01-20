#!/bin/bash
# Retry f16 models that OOMed
# Using smaller batch sizes
set -e

source .venv/bin/activate
export PYTHONPATH=/Users/philippe/vitok-release

echo "=== Retrying f16 models ==="

# 1024px (smaller bs=64)
echo "  5B-f16x32 @ 1024px (bs=64)"
modal run scripts/eval_vae.py --model 5B-f16x32 --data div8k \
  --max-size 1024 --batch-size 64 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-1024/5B-f16x32 \
  --output-json results/div8k-1024/5B-f16x32.json &

echo "  5B-f16x64 @ 1024px (bs=64)"
modal run scripts/eval_vae.py --model 5B-f16x64 --data div8k \
  --max-size 1024 --batch-size 64 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-1024/5B-f16x64 \
  --output-json results/div8k-1024/5B-f16x64.json &

# 2048px (smaller bs=16)
echo "  5B-f16x32 @ 2048px (bs=16)"
modal run scripts/eval_vae.py --model 5B-f16x32 --data div8k \
  --max-size 2048 --batch-size 16 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-2048/5B-f16x32 \
  --output-json results/div8k-2048/5B-f16x32.json &

echo "  5B-f16x64 @ 2048px (bs=16)"
modal run scripts/eval_vae.py --model 5B-f16x64 --data div8k \
  --max-size 2048 --batch-size 16 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-2048/5B-f16x64 \
  --output-json results/div8k-2048/5B-f16x64.json &

# 4096px 5B-f16x32 (bs=1, swa=1024)
echo "  5B-f16x32 @ 4096px (bs=1, swa=1024)"
modal run scripts/eval_vae.py --model 5B-f16x32 --data div8k \
  --max-size 4096 --batch-size 1 --swa-window 1024 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-4096/5B-f16x32 \
  --output-json results/div8k-4096/5B-f16x32.json &

# 8192px f16 (bs=1, swa=1024, NO float8)
echo "  5B-f16x32 @ 8192px (bs=1, swa=1024, no fp8)"
modal run scripts/eval_vae.py --model 5B-f16x32 --data div8k \
  --max-size 8192 --batch-size 1 --swa-window 1024 \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-8192/5B-f16x32 \
  --output-json results/div8k-8192/5B-f16x32.json &

echo "  5B-f16x64 @ 8192px (bs=1, swa=1024, no fp8)"
modal run scripts/eval_vae.py --model 5B-f16x64 --data div8k \
  --max-size 8192 --batch-size 1 --swa-window 1024 \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-8192/5B-f16x64 \
  --output-json results/div8k-8192/5B-f16x64.json &

echo ""
echo "7 f16 retry jobs launched!"
wait
echo "Done!"
