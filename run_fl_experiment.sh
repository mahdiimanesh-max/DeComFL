#!/bin/bash
# DeComFL Federated Learning Experiment Script
# This script runs a complete federated learning experiment with DeComFL

echo "🚀 Starting DeComFL Federated Learning Experiment..."
echo ""

# Full FL setup command based on README example
uv run python decomfl_main.py \
    --large-model=opt-125m \
    --dataset=sst2 \
    --iterations=100 \
    --train-batch-size=32 \
    --test-batch-size=200 \
    --eval-iterations=25 \
    --num-clients=3 \
    --num-sample-clients=2 \
    --local-update-steps=1 \
    --num-pert=5 \
    --lr=1e-5 \
    --mu=1e-3 \
    --grad-estimate-method=rge-forward \
    --no-optim \
    --no-cuda \
    --no-mps

echo ""
echo "✅ Experiment completed!"
