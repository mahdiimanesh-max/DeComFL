# Diagnosing NaN Loss Issue in ZO RGE

## Problem
Loss becomes NaN and accuracy drops to 0% during training.

## Root Causes

### 1. **Learning Rate vs Perturbation Step Size (mu) Mismatch**
Your settings:
- `--lr=1e-5` (learning rate)
- `--mu=1e-3` (perturbation step size)

**Issue**: The gradient estimate formula is:
```
dir_grad = (loss(x+μ·u) - loss(x)) / μ
gradient = dir_grad × u
update = -lr × gradient = -lr × dir_grad × u
```

If `lr` is too large relative to `mu`, or if the gradient estimate is unstable, updates can explode.

### 2. **Numerical Instability in Gradient Estimation**
The gradient is computed as:
```python
dir_grad = (pert_plus_loss - pert_minus_loss) / (self.mu * denominator_factor)
```

Problems:
- If `mu` is too small (1e-3), division by small number amplifies errors
- If loss difference is very small, numerical precision issues
- If loss becomes NaN during perturbation, it propagates

### 3. **Gradient Explosion**
The estimated gradient is:
```python
grad = pb_norm.mul_(dir_grad)  # Random vector × scalar
grad.div_(num_pert)  # Average
```

If `dir_grad` is very large (due to unstable loss), the gradient can explode.

### 4. **Model Weights Becoming NaN**
Once model weights become NaN, all subsequent computations are NaN.

## Solutions

### Solution 1: Reduce Learning Rate (Recommended)
```bash
uv run python zo_rge_main.py \
    --dataset=mnist \
    --epoch=20 \
    --num-pert=10 \
    --lr=1e-6 \          # Reduced from 1e-5
    --mu=1e-3 \
    --momentum=0.9
```

### Solution 2: Increase Perturbation Step Size
```bash
uv run python zo_rge_main.py \
    --dataset=mnist \
    --epoch=20 \
    --num-pert=10 \
    --lr=1e-5 \
    --mu=1e-2 \          # Increased from 1e-3
    --momentum=0.9
```

### Solution 3: Use Gradient Clipping (if code supports it)
Add gradient clipping to prevent explosion.

### Solution 4: Reduce Number of Perturbations
```bash
uv run python zo_rge_main.py \
    --dataset=mnist \
    --epoch=20 \
    --num-pert=5 \       # Reduced from 10
    --lr=1e-5 \
    --mu=1e-3 \
    --momentum=0.9
```

### Solution 5: Use Central Difference (More Stable)
```bash
uv run python zo_rge_main.py \
    --dataset=mnist \
    --epoch=20 \
    --num-pert=10 \
    --lr=1e-5 \
    --mu=1e-3 \
    --momentum=0.9 \
    --grad-estimate-method=rge-central  # More stable than forward
```

## Recommended Fix

Try this combination (most stable):
```bash
uv run python zo_rge_main.py \
    --dataset=mnist \
    --epoch=20 \
    --num-pert=10 \
    --lr=5e-6 \              # Reduced learning rate
    --mu=1e-2 \              # Increased mu for stability
    --momentum=0.9 \
    --grad-estimate-method=rge-central
```

## Why This Happens

1. **Epoch 22-23**: Model weights likely became NaN around epoch 22
2. **NaN Propagation**: Once NaN appears, it propagates through all computations
3. **0% Accuracy**: Model outputs are NaN, so predictions are invalid

## Prevention

1. **Monitor gradients**: Check if gradients are becoming too large
2. **Start with smaller LR**: Begin with very small learning rate and increase gradually
3. **Use stable mu**: mu=1e-2 to 1e-1 is often more stable than 1e-3
4. **Warmup**: The code has warmup epochs - make sure they're working correctly
