#!/bin/bash
# Run ZO RGE WITH Federated Learning
# Optimized hyperparameters to enable learning in FL setting

echo "🔹 Running ZO RGE WITH FL (Federated Learning)"
echo "Dataset: MNIST, Iterations: 50, Clients: 3"
echo "AGGRESSIVE hyperparameters for faster FL learning"
echo ""

uv run python decomfl_main.py \
    --dataset=mnist \
    --iterations=50 \
    --num-clients=3 \
    --num-sample-clients=3 \
    --local-update-steps=20 \
    --num-pert=100 \
    --lr=2e-4 \
    --mu=1e-2 \
    --momentum=0.9 \
    --eval-iterations=5 \
    --grad-estimate-method=rge-central \
    --seed=42 \
    --no-cuda \
    --no-mps

echo ""
echo "✅ Experiment completed!"
