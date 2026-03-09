# ES Hyperparameter Comparison: With FL vs Without FL

## Summary Table

| Hyperparameter | ES **Without FL** | ES **With FL** | Ratio (FL/NoFL) | Notes |
|----------------|-------------------|----------------|-----------------|-------|
| **ES-Specific Parameters** |
| `num_pert` (Population Size) | 30 | 100 | 3.33x | More perturbations needed for FL aggregation |
| `sigma` (Perturbation Scale) | 0.01 | 0.01 | 1.0x | Same - exploration scale |
| `alpha` (ES Learning Rate) | 0.002 | 0.02 | 10x | Much higher for FL convergence |
| **Training Parameters** |
| `epoch` / `iterations` | 20 epochs | 100 iterations | 5x | FL needs more rounds |
| `lr` (Optimizer LR) | Default* | 0.02 | - | Explicitly set for FL |
| **Federated Learning Parameters** |
| `num_clients` | N/A | 3 | - | Total clients |
| `num_sample_clients` | N/A | 3 | - | Clients per round (all) |
| `local_update_steps` | N/A | 30 | - | Local training steps per client |
| **Other Parameters** |
| `eval_iterations` | Every epoch | 5 | - | Evaluation frequency |
| `aggregation` | N/A | mean | - | Aggregation method |
| `byz_type` | N/A | no_byz | - | No Byzantine attacks |
| `seed` | 42 | 42 | 1.0x | Same seed for reproducibility |

\* *ES without FL uses default optimizer learning rate (typically from dataset-specific settings)*

## Key Observations

### 1. **Population Size (num_pert)**
- **Without FL**: 30 perturbations
- **With FL**: 100 perturbations (3.33x more)
- **Reason**: FL aggregates rewards across clients, so more perturbations provide better statistics

### 2. **ES Learning Rate (alpha)**
- **Without FL**: 0.002 (conservative)
- **With FL**: 0.02 (10x higher)
- **Reason**: FL updates are less frequent and aggregated, requiring larger steps

### 3. **Training Duration**
- **Without FL**: 20 epochs
- **With FL**: 100 iterations
- **Note**: These are different units (epochs vs iterations), but FL typically needs more rounds

### 4. **Local Update Steps**
- **Without FL**: N/A (single machine)
- **With FL**: 30 steps per client
- **Impact**: Each client trains locally before sending updates

## Performance Comparison

| Metric | ES Without FL | ES With FL | ZO Without FL | ZO With FL |
|--------|---------------|------------|---------------|------------|
| **Final Accuracy** | ~?% | **~85%** | 84.43% | 9.08% |
| **Status** | Need to check logs | ✅ Good | ✅ Good | ❌ Poor |

## Hyperparameter Tuning Recommendations

### For ES With FL (Current: 85% accuracy)

**If you want to improve further:**

1. **Increase `alpha`** (0.02 → 0.03-0.05)
   - Risk: May become unstable
   - Benefit: Faster convergence if stable

2. **Increase `num_pert`** (100 → 150-200)
   - Risk: Slower per iteration
   - Benefit: Better gradient estimates

3. **Increase `iterations`** (100 → 150-200)
   - Risk: Longer training time
   - Benefit: More time to converge

4. **Adjust `local_update_steps`** (30 → 20-40)
   - Test different values to find optimal balance

5. **Tune `sigma`** (0.01 → 0.005-0.02)
   - Smaller = more precise, larger = more exploration

### For ES Without FL

**If accuracy is lower than expected:**

1. **Increase `alpha`** (0.002 → 0.005-0.01)
   - Similar to FL, may need more aggressive learning

2. **Increase `num_pert`** (30 → 50-100)
   - Better gradient estimates

3. **Increase `epoch`** (20 → 30-50)
   - More training time

## Notes

- ES with FL achieved **85% accuracy**, which is excellent compared to ZO with FL (9.08%)
- The 10x increase in `alpha` for FL is critical for convergence
- The 3.33x increase in `num_pert` helps with aggregated gradient estimates
- Current FL settings appear well-tuned for MNIST
