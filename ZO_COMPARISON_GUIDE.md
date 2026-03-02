# ZO RGE Comparison Guide: With FL vs Without FL

This guide helps you run and compare zeroth-order random gradient estimation (ZO RGE) experiments with and without federated learning.

## Overview

You'll run two experiments:
1. **Without FL**: Single-machine training using `zo_rge_main.py`
2. **With FL**: Federated learning using `decomfl_main.py`

Both use the same hyperparameters for fair comparison.

## Quick Start

### Option 1: Run Both Experiments Automatically

```bash
./run_zo_comparison.sh
```

This will:
- Run both experiments sequentially
- Save logs to `results/zo_comparison_<timestamp>/`
- Automatically generate a comparison report

### Option 2: Run Experiments Separately

**Run without FL:**
```bash
./run_zo_no_fl.sh
```

**Run with FL:**
```bash
./run_zo_with_fl.sh
```

**Then compare results manually:**
```bash
# Create a results directory and copy logs there
mkdir -p results/my_comparison
# Copy your log files to results/my_comparison/
# Then run:
uv run python compare_zo_results.py results/my_comparison
```

## Experiment Configuration

Both experiments use these settings:
- **Dataset**: MNIST
- **Epochs/Iterations**: 20
- **Number of perturbations**: 10
- **Learning rate**: 1e-5
- **Perturbation step (mu)**: 1e-3
- **Momentum**: 0.9
- **Seed**: 42 (for reproducibility)

### Without FL (Single Machine)
- Uses `zo_rge_main.py`
- Trains for 20 epochs
- All data on one machine
- No client-server communication

### With FL (Federated Learning)
- Uses `decomfl_main.py`
- Trains for 20 FL iterations
- 3 clients total
- 2 clients sampled per round
- 1 local update step per client
- Evaluates every 5 iterations

## Understanding the Results

The comparison script extracts:
- **Evaluation Loss**: Lower is better
- **Evaluation Accuracy**: Higher is better
- **Round-by-round comparison**: See how each method performs over time

### Expected Differences

**Without FL (Single Machine):**
- ✅ All data available at once
- ✅ No communication overhead
- ✅ Typically faster convergence
- ❌ Not realistic for distributed scenarios

**With FL (Federated Learning):**
- ✅ Realistic distributed setting
- ✅ Dimension-free communication (only scalars)
- ✅ Privacy-preserving
- ❌ May converge slower due to data heterogeneity
- ❌ Communication overhead (even if minimal)

## Output Files

After running `run_zo_comparison.sh`, you'll find:

```
results/zo_comparison_<timestamp>/
├── zo_rge_no_fl.log          # Log from single-machine experiment
├── zo_rge_with_fl.log        # Log from FL experiment
└── comparison_report.txt     # Detailed comparison report
```

## Customizing Experiments

### Change Number of Epochs/Iterations

Edit the scripts and change:
- `EPOCHS_OR_ITERATIONS=20` → your desired number

### Change Hyperparameters

Edit the scripts and modify:
- `NUM_PERT=10` → number of perturbations
- `LR=1e-5` → learning rate
- `MU=1e-3` → perturbation step size
- `MOMENTUM=0.9` → momentum

### Change FL Settings

In `run_zo_with_fl.sh`, modify:
- `--num-clients=3` → total number of clients
- `--num-sample-clients=2` → clients per round
- `--local-update-steps=1` → local training steps
- `--eval-iterations=5` → evaluation frequency

## Manual Comparison

If you want to compare results manually:

1. **Extract metrics from logs:**
   ```bash
   # Look for lines like:
   # "Evaluation(round X): Eval Loss:Y, Accuracy:Z%"
   grep "Evaluation" results/zo_comparison_*/zo_rge_no_fl.log
   grep "Evaluation" results/zo_comparison_*/zo_rge_with_fl.log
   ```

2. **Compare final results:**
   - Check the last evaluation round in each log
   - Compare loss and accuracy values

## Troubleshooting

### Issue: "Command not found: uv"
- Make sure `uv` is in your PATH: `export PATH="$HOME/.local/bin:$PATH"`

### Issue: Out of memory
- Reduce batch size: add `--train-batch-size=4`
- Use `--no-optim` flag for FL experiment

### Issue: Results don't match expectations
- Check that both experiments use the same seed (`--seed=42`)
- Verify hyperparameters are identical
- Note: FL may have different convergence due to data distribution

## Next Steps

After comparison, you can:
1. **Visualize results**: Create plots comparing loss/accuracy curves
2. **Experiment with hyperparameters**: Try different learning rates, perturbations, etc.
3. **Scale up**: Run longer experiments (more epochs/iterations)
4. **Try different datasets**: CIFAR-10, Fashion-MNIST, etc.

## Notes

- **Fair Comparison**: Both experiments use the same hyperparameters, but note that:
  - 20 epochs (single machine) ≠ 20 iterations (FL) in terms of data seen
  - FL splits data across clients, which may affect convergence
  - For more fair comparison, you might want to match total data samples seen

- **Evaluation Frequency**: 
  - Without FL: Evaluates every epoch (20 evaluations total)
  - With FL: Evaluates every 5 iterations (4 evaluations total at iterations 5, 10, 15, 20)
