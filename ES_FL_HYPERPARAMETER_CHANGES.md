# ES FL Hyperparameter Changes for Convergence

## Problem
ES with FL was not converging:
- Accuracy stuck at ~9% (random guessing)
- Loss around 2.32-2.33 (not decreasing)
- No learning progress over 50 iterations

## Root Cause Analysis
FL requires more aggressive hyperparameters than non-FL because:
1. **Reward Aggregation**: Rewards from multiple clients are averaged, which can reduce signal strength
2. **Distributed Updates**: Updates are less frequent (only at server aggregation)
3. **Data Heterogeneity**: Different clients have different data distributions

## Changes Made

### 1. Increased Learning Rate (Alpha)
- **Before**: `alpha=0.002`
- **After**: `alpha=0.01`
- **Impact**: 5x increase - allows model to learn faster in FL setting
- **Rationale**: FL needs larger learning rates because updates are aggregated and less frequent

### 2. Increased Population Size (Num Perturbations)
- **Before**: `num_pert=30`
- **After**: `num_pert=50`
- **Impact**: More robust reward estimates
- **Rationale**: Larger population provides better statistics when aggregating across clients

### 3. Updated Optimizer Learning Rate
- **Before**: `lr=0.002`
- **After**: `lr=0.01`
- **Impact**: Matches alpha for consistency
- **Rationale**: ES uses optimizer's lr as alpha, so they should match

## Comparison with ZO RGE FL

| Hyperparameter | ZO RGE FL | ES FL (New) | ES FL (Old) |
|----------------|-----------|-------------|-------------|
| num_pert       | 100       | 50          | 30          |
| lr/alpha       | 2e-4      | 0.01        | 0.002       |
| mu/sigma       | 1e-2      | 0.01        | 0.01        |
| local_steps    | 20        | 20          | 20          |

## Expected Results
With these changes, ES FL should:
- Start learning immediately (accuracy should increase from ~9%)
- Converge faster (loss should decrease)
- Reach meaningful accuracy within 50 iterations

## Notes
- These hyperparameters are more aggressive than non-FL ES
- If training becomes unstable (NaN, exploding loss), reduce alpha to 0.005
- Can try even more aggressive: `alpha=0.02`, `num_pert=100` if still not converging
