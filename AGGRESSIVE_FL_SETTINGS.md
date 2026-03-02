# Aggressive FL Hyperparameters

## Updated Settings for 50 Rounds

### Changes Made:

**Iterations:**
- `--iterations=50` (was 10) - 5x more training rounds

**Learning Rate:**
- `--lr=2e-4` (was 5e-5) - **4x higher** - Much more aggressive

**Perturbations:**
- `--num-pert=100` (was 50) - **2x more** - Better gradient estimates

**Local Update Steps:**
- `--local-update-steps=20` (was 10) - **2x more** - More local training

**Evaluation:**
- `--eval-iterations=5` (was 1) - Evaluate every 5 iterations (10 evaluations total)

### Complete Settings:

```bash
--iterations=50
--num-clients=3
--num-sample-clients=3
--local-update-steps=20
--num-pert=100
--lr=2e-4
--mu=1e-2
--momentum=0.9
--eval-iterations=5
--grad-estimate-method=rge-central
```

## Expected Results

With these aggressive settings over 50 iterations, you should see:
- ✅ Much faster learning
- ✅ Accuracy should reach 30-50%+ (hopefully higher)
- ✅ Loss should decrease significantly
- ✅ Better convergence toward non-FL performance

## Run Command

```bash
./run_zo_with_fl.sh
```

## Notes

- **Higher learning rate (2e-4)**: Risk of instability, but necessary for FL
- **More perturbations (100)**: Slower per iteration, but better gradients
- **More local steps (20)**: Each client trains more before aggregation
- **50 iterations**: Should give enough time to see meaningful progress

## If Still Slow

If accuracy is still low after 50 iterations, consider:
1. Even higher LR: `--lr=5e-4` (but watch for NaN)
2. More perturbations: `--num-pert=200` (slower but better)
3. Check data distribution: Ensure IID split
