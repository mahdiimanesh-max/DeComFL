#!/bin/bash
# Run ZO RGE WITHOUT Federated Learning (Single Machine)
# 20 epochs with MNIST

echo "🔹 Running ZO RGE WITHOUT FL (Single Machine)"
echo "Dataset: MNIST, Epochs: 20"
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
