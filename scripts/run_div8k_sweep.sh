#!/bin/bash
# Run DIV8K evaluation sweep (24 parallel Modal jobs)
# ViTok: 4 models x 4 resolutions = 16 jobs
# Baselines: 2 baselines (flux, sd) x 4 resolutions = 8 jobs
set -e

source .venv/bin/activate
export PYTHONPATH=/Users/philippe/vitok-release

echo "Starting DIV8K sweep (24 parallel jobs)..."
echo "============================================="

# =============================================================================
# ViTok f16 models (5B-f16x32, 5B-f16x64) - 8 jobs
# SWA only at 4096/8192, no SWA at 1024/2048
# =============================================================================
echo ""
echo "=== ViTok f16 models (8 jobs) ==="

# 5B-f16x32
echo "  5B-f16x32 @ 1024px (bs=128)"
modal run scripts/eval_vae.py --model 5B-f16x32 --data div8k \
  --max-size 1024 --batch-size 128 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-1024/5B-f16x32 \
  --output-json results/div8k-1024/5B-f16x32.json &

echo "  5B-f16x32 @ 2048px (bs=32)"
modal run scripts/eval_vae.py --model 5B-f16x32 --data div8k \
  --max-size 2048 --batch-size 32 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-2048/5B-f16x32 \
  --output-json results/div8k-2048/5B-f16x32.json &

echo "  5B-f16x32 @ 4096px (bs=8, swa=1024)"
modal run scripts/eval_vae.py --model 5B-f16x32 --data div8k \
  --max-size 4096 --batch-size 8 --swa-window 1024 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-4096/5B-f16x32 \
  --output-json results/div8k-4096/5B-f16x32.json &

echo "  5B-f16x32 @ 8192px (bs=2, swa=1024)"
modal run scripts/eval_vae.py --model 5B-f16x32 --data div8k \
  --max-size 8192 --batch-size 2 --swa-window 1024 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-8192/5B-f16x32 \
  --output-json results/div8k-8192/5B-f16x32.json &

# 5B-f16x64
echo "  5B-f16x64 @ 1024px (bs=128)"
modal run scripts/eval_vae.py --model 5B-f16x64 --data div8k \
  --max-size 1024 --batch-size 128 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-1024/5B-f16x64 \
  --output-json results/div8k-1024/5B-f16x64.json &

echo "  5B-f16x64 @ 2048px (bs=32)"
modal run scripts/eval_vae.py --model 5B-f16x64 --data div8k \
  --max-size 2048 --batch-size 32 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-2048/5B-f16x64 \
  --output-json results/div8k-2048/5B-f16x64.json &

echo "  5B-f16x64 @ 4096px (bs=8, swa=1024)"
modal run scripts/eval_vae.py --model 5B-f16x64 --data div8k \
  --max-size 4096 --batch-size 8 --swa-window 1024 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-4096/5B-f16x64 \
  --output-json results/div8k-4096/5B-f16x64.json &

echo "  5B-f16x64 @ 8192px (bs=2, swa=1024)"
modal run scripts/eval_vae.py --model 5B-f16x64 --data div8k \
  --max-size 8192 --batch-size 2 --swa-window 1024 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-8192/5B-f16x64 \
  --output-json results/div8k-8192/5B-f16x64.json &

# =============================================================================
# ViTok f32 models (5B-f32x64, 5B-f32x128) - 8 jobs
# No SWA (f32 stride = 4x fewer tokens than f16)
# Larger batch sizes: 256/64/8/2
# =============================================================================
echo ""
echo "=== ViTok f32 models (8 jobs) ==="

# 5B-f32x64
echo "  5B-f32x64 @ 1024px (bs=256)"
modal run scripts/eval_vae.py --model 5B-f32x64 --data div8k \
  --max-size 1024 --batch-size 256 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-1024/5B-f32x64 \
  --output-json results/div8k-1024/5B-f32x64.json &

echo "  5B-f32x64 @ 2048px (bs=64)"
modal run scripts/eval_vae.py --model 5B-f32x64 --data div8k \
  --max-size 2048 --batch-size 64 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-2048/5B-f32x64 \
  --output-json results/div8k-2048/5B-f32x64.json &

echo "  5B-f32x64 @ 4096px (bs=8)"
modal run scripts/eval_vae.py --model 5B-f32x64 --data div8k \
  --max-size 4096 --batch-size 8 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-4096/5B-f32x64 \
  --output-json results/div8k-4096/5B-f32x64.json &

echo "  5B-f32x64 @ 8192px (bs=2)"
modal run scripts/eval_vae.py --model 5B-f32x64 --data div8k \
  --max-size 8192 --batch-size 2 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-8192/5B-f32x64 \
  --output-json results/div8k-8192/5B-f32x64.json &

# 5B-f32x128
echo "  5B-f32x128 @ 1024px (bs=256)"
modal run scripts/eval_vae.py --model 5B-f32x128 --data div8k \
  --max-size 1024 --batch-size 256 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-1024/5B-f32x128 \
  --output-json results/div8k-1024/5B-f32x128.json &

echo "  5B-f32x128 @ 2048px (bs=64)"
modal run scripts/eval_vae.py --model 5B-f32x128 --data div8k \
  --max-size 2048 --batch-size 64 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-2048/5B-f32x128 \
  --output-json results/div8k-2048/5B-f32x128.json &

echo "  5B-f32x128 @ 4096px (bs=8)"
modal run scripts/eval_vae.py --model 5B-f32x128 --data div8k \
  --max-size 4096 --batch-size 8 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-4096/5B-f32x128 \
  --output-json results/div8k-4096/5B-f32x128.json &

echo "  5B-f32x128 @ 8192px (bs=2)"
modal run scripts/eval_vae.py --model 5B-f32x128 --data div8k \
  --max-size 8192 --batch-size 2 --float8 inference \
  --num-samples 1500 --save-visuals 25 \
  --output-dir /output/visuals/div8k-8192/5B-f32x128 \
  --output-json results/div8k-8192/5B-f32x128.json &

# =============================================================================
# Baselines (flux, sd) - 8 jobs (all bs=1)
# =============================================================================
echo ""
echo "=== Baselines (8 jobs) ==="

for baseline in flux sd; do
  for res in 1024 2048 4096 8192; do
    echo "  ${baseline} @ ${res}px (bs=1)"
    modal run scripts/eval_vae.py --baseline $baseline --data div8k \
      --max-size $res --batch-size 1 \
      --num-samples 1500 --save-visuals 25 \
      --output-dir /output/visuals/div8k-${res}/${baseline} \
      --output-json results/div8k-${res}/baseline-${baseline}.json &
  done
done

echo ""
echo "============================================="
echo "All 24 jobs launched! Waiting for completion..."
echo "Monitor progress: modal app list"
echo "============================================="

wait

echo ""
echo "============================================="
echo "All jobs complete!"
echo ""
echo "Results: ls results/div8k-*/*.json"
echo "Visuals: modal volume get vitok-output /output/visuals ./results/visuals"
echo "============================================="
