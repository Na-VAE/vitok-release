#!/bin/bash
# Rerun failed DIV8K jobs with corrected batch sizes
# Using bs=1 for 4096/8192 ViTok, flash backend, SWA=1024
set -e

source .venv/bin/activate
export PYTHONPATH=/Users/philippe/vitok-release

echo "=== Rerunning failed DIV8K jobs ==="
echo ""

# =============================================================================
# Missing 1024px jobs (f16 models + f32x128)
# =============================================================================
echo "=== 1024px missing (3 jobs) ==="

echo "  5B-f16x32 @ 1024px (bs=128)"
modal run scripts/eval_vae.py --model 5B-f16x32 --data div8k \
  --max-size 1024 --batch-size 128 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-1024/5B-f16x32 \
  --output-json results/div8k-1024/5B-f16x32.json &

echo "  5B-f16x64 @ 1024px (bs=128)"
modal run scripts/eval_vae.py --model 5B-f16x64 --data div8k \
  --max-size 1024 --batch-size 128 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-1024/5B-f16x64 \
  --output-json results/div8k-1024/5B-f16x64.json &

echo "  5B-f32x128 @ 1024px (bs=256)"
modal run scripts/eval_vae.py --model 5B-f32x128 --data div8k \
  --max-size 1024 --batch-size 256 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-1024/5B-f32x128 \
  --output-json results/div8k-1024/5B-f32x128.json &

# =============================================================================
# Missing 2048px jobs (f16 models only)
# =============================================================================
echo ""
echo "=== 2048px missing (2 jobs) ==="

echo "  5B-f16x32 @ 2048px (bs=32)"
modal run scripts/eval_vae.py --model 5B-f16x32 --data div8k \
  --max-size 2048 --batch-size 32 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-2048/5B-f16x32 \
  --output-json results/div8k-2048/5B-f16x32.json &

echo "  5B-f16x64 @ 2048px (bs=32)"
modal run scripts/eval_vae.py --model 5B-f16x64 --data div8k \
  --max-size 2048 --batch-size 32 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-2048/5B-f16x64 \
  --output-json results/div8k-2048/5B-f16x64.json &

# =============================================================================
# All 4096px jobs (bs=1 for ViTok, swa=1024 for f16)
# =============================================================================
echo ""
echo "=== 4096px all models (6 jobs, bs=1) ==="

# f16 with SWA
echo "  5B-f16x32 @ 4096px (bs=1, swa=1024)"
modal run scripts/eval_vae.py --model 5B-f16x32 --data div8k \
  --max-size 4096 --batch-size 1 --swa-window 1024 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-4096/5B-f16x32 \
  --output-json results/div8k-4096/5B-f16x32.json &

echo "  5B-f16x64 @ 4096px (bs=1, swa=1024)"
modal run scripts/eval_vae.py --model 5B-f16x64 --data div8k \
  --max-size 4096 --batch-size 1 --swa-window 1024 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-4096/5B-f16x64 \
  --output-json results/div8k-4096/5B-f16x64.json &

# f32 with SWA
echo "  5B-f32x64 @ 4096px (bs=1, swa=1024)"
modal run scripts/eval_vae.py --model 5B-f32x64 --data div8k \
  --max-size 4096 --batch-size 1 --swa-window 1024 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-4096/5B-f32x64 \
  --output-json results/div8k-4096/5B-f32x64.json &

echo "  5B-f32x128 @ 4096px (bs=1, swa=1024)"
modal run scripts/eval_vae.py --model 5B-f32x128 --data div8k \
  --max-size 4096 --batch-size 1 --swa-window 1024 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-4096/5B-f32x128 \
  --output-json results/div8k-4096/5B-f32x128.json &

# Baselines
echo "  flux @ 4096px (bs=1)"
modal run scripts/eval_vae.py --baseline flux --data div8k \
  --max-size 4096 --batch-size 1 \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-4096/flux \
  --output-json results/div8k-4096/baseline-flux.json &

echo "  sd @ 4096px (bs=1)"
modal run scripts/eval_vae.py --baseline sd --data div8k \
  --max-size 4096 --batch-size 1 \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-4096/sd \
  --output-json results/div8k-4096/baseline-sd.json &

# =============================================================================
# All 8192px jobs (bs=1 for all, swa=1024 for f16)
# =============================================================================
echo ""
echo "=== 8192px all models (6 jobs, bs=1) ==="

# f16 with SWA
echo "  5B-f16x32 @ 8192px (bs=1, swa=1024)"
modal run scripts/eval_vae.py --model 5B-f16x32 --data div8k \
  --max-size 8192 --batch-size 1 --swa-window 1024 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-8192/5B-f16x32 \
  --output-json results/div8k-8192/5B-f16x32.json &

echo "  5B-f16x64 @ 8192px (bs=1, swa=1024)"
modal run scripts/eval_vae.py --model 5B-f16x64 --data div8k \
  --max-size 8192 --batch-size 1 --swa-window 1024 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-8192/5B-f16x64 \
  --output-json results/div8k-8192/5B-f16x64.json &

# f32 with SWA
echo "  5B-f32x64 @ 8192px (bs=1, swa=1024)"
modal run scripts/eval_vae.py --model 5B-f32x64 --data div8k \
  --max-size 8192 --batch-size 1 --swa-window 1024 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-8192/5B-f32x64 \
  --output-json results/div8k-8192/5B-f32x64.json &

echo "  5B-f32x128 @ 8192px (bs=1, swa=1024)"
modal run scripts/eval_vae.py --model 5B-f32x128 --data div8k \
  --max-size 8192 --batch-size 1 --swa-window 1024 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-8192/5B-f32x128 \
  --output-json results/div8k-8192/5B-f32x128.json &

# Baselines
echo "  flux @ 8192px (bs=1)"
modal run scripts/eval_vae.py --baseline flux --data div8k \
  --max-size 8192 --batch-size 1 \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-8192/flux \
  --output-json results/div8k-8192/baseline-flux.json &

echo "  sd @ 8192px (bs=1)"
modal run scripts/eval_vae.py --baseline sd --data div8k \
  --max-size 8192 --batch-size 1 \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-8192/sd \
  --output-json results/div8k-8192/baseline-sd.json &

echo ""
echo "============================================="
echo "All 17 rerun jobs launched!"
echo "Monitor: modal app list"
echo "============================================="

wait

echo ""
echo "All jobs complete!"
echo "Results: ls results/div8k-*/*.json"
