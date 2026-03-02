#!/bin/bash
# Script to run ZO RGE with and without FL for comparison
# This runs both experiments and saves results for comparison

set -e

echo "🔬 ZO RGE Comparison Experiment: With FL vs Without FL"
echo "=================================================="
echo ""

# Common hyperparameters for fair comparison (stable settings to avoid NaN)
DATASET="mnist"
NUM_PERT=10
LR=5e-6
MU=1e-2
MOMENTUM=0.9
EPOCHS_OR_ITERATIONS=20
SEED=42
GRAD_METHOD="rge-central"

# Create results directory
RESULTS_DIR="results/zo_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "📁 Results will be saved to: $RESULTS_DIR"
echo ""

# Experiment 1: Without FL (Single Machine)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔹 Experiment 1: ZO RGE WITHOUT Federated Learning (Single Machine)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

uv run python zo_rge_main.py \
    --dataset=$DATASET \
    --epoch=$EPOCHS_OR_ITERATIONS \
    --num-pert=$NUM_PERT \
    --lr=$LR \
    --mu=$MU \
    --momentum=$MOMENTUM \
    --seed=$SEED \
    --grad-estimate-method=$GRAD_METHOD \
    2>&1 | tee "$RESULTS_DIR/zo_rge_no_fl.log"

echo ""
echo "✅ Experiment 1 completed!"
echo ""

# Experiment 2: With FL
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔹 Experiment 2: ZO RGE WITH Federated Learning"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

uv run python decomfl_main.py \
    --dataset=$DATASET \
    --iterations=$EPOCHS_OR_ITERATIONS \
    --num-clients=3 \
    --num-sample-clients=2 \
    --local-update-steps=1 \
    --num-pert=$NUM_PERT \
    --lr=$LR \
    --mu=$MU \
    --momentum=$MOMENTUM \
    --eval-iterations=5 \
    --grad-estimate-method=$GRAD_METHOD \
    --seed=$SEED \
    --no-cuda \
    --no-mps \
    2>&1 | tee "$RESULTS_DIR/zo_rge_with_fl.log"

echo ""
echo "✅ Experiment 2 completed!"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Running comparison analysis..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run comparison script
uv run python compare_zo_results.py "$RESULTS_DIR"

echo ""
echo "✅ All experiments completed! Results saved in: $RESULTS_DIR"
echo ""
