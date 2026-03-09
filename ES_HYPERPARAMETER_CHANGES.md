# ES Hyperparameter Changes for Faster Convergence

## Problem
The ES training was showing very slow convergence:
- Accuracy stuck at ~10% (random guessing for 10 classes)
- Loss around 2.3-2.4 (high, not decreasing)
- No learning progress over multiple epochs

## Root Cause
The hyperparameters were too conservative:
- **Alpha (learning rate)**: 0.0005 was too small - model updates were negligible
- **Sigma (perturbation scale)**: 0.001 was too small - perturbations didn't explore enough
- **Population size**: 30 was reasonable but could be larger for better estimates

## Changes Made

### 1. Increased Learning Rate (Alpha)
- **Before**: `alpha=0.0005`
- **After**: `alpha=0.01`
- **Impact**: 20x increase - allows model to learn much faster
- **Rationale**: ES needs larger learning rates than gradient-based methods because updates are based on reward-weighted perturbations, not direct gradients

### 2. Increased Perturbation Scale (Sigma)
- **Before**: `sigma=0.001`
- **After**: `sigma=0.01`
- **Impact**: 10x increase - larger exploration of parameter space
- **Rationale**: Larger perturbations help ES explore the loss landscape more effectively

### 3. Population Size (Num Perturbations)
- **Kept**: `num_pert=30`
- **Rationale**: 30 is sufficient for good estimates while keeping computational cost reasonable

## Updated Files
1. `run_es_mnist.sh` - Updated default hyperparameters
2. `es_mnist_main.py` - Updated default values in `ESSetting` class

## Expected Results
With these changes, ES should:
- Start learning immediately (accuracy should increase from ~10%)
- Converge faster (loss should decrease more rapidly)
- Reach higher accuracy within 20 epochs

## ES Update Rule
The model is updated using:
```
θ ← θ + (α/P) Σ_p (r_norm_p · ε_p)
```
where:
- `α` (alpha) = 0.01 (learning rate)
- `P` = 30 (population size)
- `r_norm_p` = normalized reward for perturbation p
- `ε_p` = perturbation vector for p

## Notes
- These hyperparameters are more aggressive and should enable learning
- If training becomes unstable (NaN, exploding loss), reduce alpha slightly
- For even faster convergence, can try: `alpha=0.02`, `sigma=0.02`, `num_pert=50` (but 30 is sufficient)
