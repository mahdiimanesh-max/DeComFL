#!/bin/bash
# Run Evolution Strategies (ES) on MNIST
# Optimized hyperparameters for faster convergence

echo "🔹 Running Evolution Strategies (ES) on MNIST"
echo "Population Size: 30, Sigma: 0.01, Alpha: 0.002"
echo "Fixed: Using same perturbations for evaluation and update"
echo ""

uv run python es_mnist_main.py \
    --dataset=mnist \
    --epoch=20 \
    --num-pert=30 \
    --sigma=0.01 \
    --alpha=0.002 \
    --seed=42 \
    --no-cuda \
    --no-mps

echo ""
echo "✅ Experiment completed!"
