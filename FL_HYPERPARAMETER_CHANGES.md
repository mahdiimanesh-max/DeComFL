# FL Hyperparameter Optimization

## Changes Made to Enable Learning in FL

### Previous Settings (Not Learning):
- `--lr=5e-6` ❌ Too small
- `--num-pert=10` ❌ Too few perturbations
- `--local-update-steps=1` ❌ Too few local steps
- `--num-sample-clients=2` ❌ Only 2 out of 3 clients
- `--eval-iterations=5` ❌ Evaluates infrequently

### New Settings (Should Learn):
- `--lr=1e-5` ✅ **2x higher** - More aggressive learning
- `--num-pert=20` ✅ **2x more** - Better gradient estimates
- `--local-update-steps=5` ✅ **5x more** - More local training
- `--num-sample-clients=3` ✅ **All clients** - Use all available data
- `--eval-iterations=1` ✅ **Every iteration** - Better monitoring

## Why These Changes Help

### 1. Higher Learning Rate (1e-5)
- FL has more noise due to aggregation
- Need larger steps to overcome noise
- Previous lr=5e-6 was too conservative

### 2. More Perturbations (20)
- Gradient estimates are aggregated across clients (more noise)
- More perturbations = better gradient estimate
- Reduces variance in gradient estimation

### 3. More Local Update Steps (5)
- Each client does more local training before sending updates
- Better local optimization
- Reduces communication frequency (but improves quality)

### 4. Use All Clients (3)
- More data per round
- Better gradient aggregation
- More stable training

### 5. Evaluate Every Iteration (1)
- Better monitoring of progress
- Can catch issues early
- More detailed comparison with non-FL version

## Expected Results

With these settings, you should see:
- ✅ Loss decreasing over iterations
- ✅ Accuracy improving from ~9% (random) to higher values
- ✅ Model actually learning

## Run Command

```bash
./run_zo_with_fl.sh
```

Or manually:
```bash
uv run python decomfl_main.py \
    --dataset=mnist \
    --iterations=20 \
    --num-clients=3 \
    --num-sample-clients=3 \
    --local-update-steps=5 \
    --num-pert=20 \
    --lr=1e-5 \
    --mu=1e-2 \
    --momentum=0.9 \
    --eval-iterations=1 \
    --grad-estimate-method=rge-central \
    --seed=42 \
    --no-cuda \
    --no-mps
```

## If Still Not Learning

If accuracy is still stuck at ~9%, try:
1. **Even higher learning rate**: `--lr=2e-5` or `--lr=5e-5`
2. **More perturbations**: `--num-pert=30` or `--num-pert=50`
3. **More local steps**: `--local-update-steps=10`
4. **Check data distribution**: Add `--iid` flag for IID data split
