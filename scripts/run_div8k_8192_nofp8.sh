#!/bin/bash
# 8192px jobs WITHOUT float8 (float8 causes OOM)
# Using bs=1, swa=1024, flash attention
set -e

source .venv/bin/activate
export PYTHONPATH=/Users/philippe/vitok-release

echo "=== 8192px jobs (NO float8, bs=1, swa=1024) ==="

# ViTok models without float8
echo "  5B-f16x32 @ 8192px"
modal run scripts/eval_vae.py --model 5B-f16x32 --data div8k \
  --max-size 8192 --batch-size 1 --swa-window 1024 \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-8192/5B-f16x32 \
  --output-json results/div8k-8192/5B-f16x32.json &

echo "  5B-f16x64 @ 8192px"
modal run scripts/eval_vae.py --model 5B-f16x64 --data div8k \
  --max-size 8192 --batch-size 1 --swa-window 1024 \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-8192/5B-f16x64 \
  --output-json results/div8k-8192/5B-f16x64.json &

echo "  5B-f32x64 @ 8192px"
modal run scripts/eval_vae.py --model 5B-f32x64 --data div8k \
  --max-size 8192 --batch-size 1 --swa-window 1024 \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-8192/5B-f32x64 \
  --output-json results/div8k-8192/5B-f32x64.json &

echo "  5B-f32x128 @ 8192px"
modal run scripts/eval_vae.py --model 5B-f32x128 --data div8k \
  --max-size 8192 --batch-size 1 --swa-window 1024 \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-8192/5B-f32x128 \
  --output-json results/div8k-8192/5B-f32x128.json &

# Baselines (these will likely OOM - record for blog)
echo "  flux @ 8192px"
modal run scripts/eval_vae.py --baseline flux --data div8k \
  --max-size 8192 --batch-size 1 \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-8192/flux \
  --output-json results/div8k-8192/baseline-flux.json &

echo "  sd @ 8192px"
modal run scripts/eval_vae.py --baseline sd --data div8k \
  --max-size 8192 --batch-size 1 \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-8192/sd \
  --output-json results/div8k-8192/baseline-sd.json &

echo ""
echo "6 jobs launched. Baselines may OOM (that's expected)."
wait
echo "Done!"
