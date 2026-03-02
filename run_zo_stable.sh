#!/bin/bash
# Stable version of ZO RGE to avoid NaN issues
# Uses more conservative hyperparameters

echo "🔹 Running ZO RGE with STABLE hyperparameters (to avoid NaN)"
echo ""

uv run python zo_rge_main.py \
    --dataset=mnist \
    --epoch=20 \
    --num-pert=10 \
    --lr=5e-6 \
    --mu=1e-2 \
    --momentum=0.9 \
    --seed=42 \
    --grad-estimate-method=rge-central

echo ""
echo "✅ Experiment completed!"
