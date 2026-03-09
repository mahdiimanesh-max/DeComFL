#!/bin/bash
# Run Evolution Strategies (ES) WITH Federated Learning
# AGGRESSIVE hyperparameters for FL convergence

echo "🔹 Running Evolution Strategies (ES) WITH FL (Federated Learning)"
echo "Dataset: MNIST, Iterations: 100, Clients: 3"
echo "FIXED: Update scaling bug (now divides by num_pert, not total_perturbations)"
echo "Population Size: 50, Local Steps: 40, Sigma: 0.03, Alpha: 0.02"
echo "Increased perturbations and local steps for better estimates and optimization"
echo ""

uv run python es_fl_main.py \
    --dataset=mnist \
    --iterations=100 \
    --num-clients=3 \
    --num-sample-clients=3 \
    --local-update-steps=40 \
    --num-pert=50 \
    --sigma=0.03 \
    --alpha=0.02 \
    --lr=0.02 \
    --eval-iterations=5 \
    --byz-type=no_byz \
    --aggregation=mean \
    --seed=42 \
    --no-cuda \
    --no-mps

echo ""
echo "✅ Experiment completed!"
