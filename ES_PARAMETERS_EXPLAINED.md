# Understanding ES Parameters: Alpha, Sigma, and Learning Rate

## Quick Summary

| Parameter | Role | Controls | Typical Values |
|-----------|------|----------|----------------|
| **Sigma (σ)** | Exploration | How far to perturb parameters for evaluation | 0.01 |
| **Alpha (α)** | Exploitation | How much to update based on rewards | 0.002-0.02 |
| **Learning Rate (lr)** | Same as Alpha in FL | In FL: used as alpha. In non-FL: separate optimizer LR | 0.02 (FL) |

## Detailed Explanation

### 1. **Sigma (σ) - Perturbation Scale**

**What it does:**
- Controls how much we **explore** the parameter space
- Determines the magnitude of random perturbations added to model parameters

**In the code:**
```python
# Perturb model: θ + σ·ε
param.data.add_(noise, alpha=es_estimator.sigma)
```

**How it works:**
1. Generate random noise vector `ε` (standard normal)
2. Add `σ × ε` to model parameters: `θ_new = θ + σ·ε`
3. Evaluate the perturbed model to get reward
4. Restore original parameters: `θ = θ_new - σ·ε`

**Effect:**
- **Small sigma (0.001)**: Small perturbations → fine-grained exploration, but may miss better regions
- **Large sigma (0.1)**: Large perturbations → broad exploration, but may be too noisy

**Current values:**
- ES without FL: `sigma = 0.01`
- ES with FL: `sigma = 0.01`

**Analogy:** Like the step size when exploring a mountain - small steps explore carefully, large steps explore broadly.

---

### 2. **Alpha (α) - ES Learning Rate**

**What it does:**
- Controls how much we **exploit** the information from rewards
- Scales the model update based on reward-weighted perturbations

**ES Update Rule:**
```
θ ← θ + (α/P) Σ_p (r_norm_p · ε_p)
```

Where:
- `α` = alpha (ES learning rate)
- `P` = population size (num_pert)
- `r_norm_p` = normalized reward for perturbation p
- `ε_p` = perturbation vector for p

**In the code:**
```python
# Apply update: θ ← θ + (α/P) · update
param.data.add_(update, alpha=alpha / es_estimator.num_pert)
```

**How it works:**
1. Evaluate P perturbations, get rewards
2. Normalize rewards: `r_norm = (r - mean(r)) / std(r)`
3. Weight each perturbation by its normalized reward
4. Update: `θ_new = θ + (α/P) × Σ(reward_weighted_perturbations)`

**Effect:**
- **Small alpha (0.001)**: Small updates → slow learning, stable
- **Large alpha (0.1)**: Large updates → fast learning, may be unstable

**Current values:**
- ES without FL: `alpha = 0.002` (conservative)
- ES with FL: `alpha = 0.02` (10x higher, needed for FL)

**Analogy:** Like the learning rate in gradient descent - controls how big steps you take toward better solutions.

---

### 3. **Learning Rate (lr) - Optimizer Learning Rate**

**What it does:**
- In **FL version**: This IS alpha! The optimizer's `lr` is used as `alpha`
- In **non-FL version**: Separate from alpha (used by optimizer if needed)

**In the FL code:**
```python
# Get learning rate from optimizer (alpha in ES update rule)
alpha = optimizer.defaults["lr"]  # ES learning rate
```

**Current values:**
- ES without FL: Uses default optimizer LR (separate from alpha)
- ES with FL: `lr = 0.02` (this becomes alpha)

**Why they're the same in FL:**
- ES doesn't use traditional gradient-based optimization
- The optimizer's LR is repurposed as the ES learning rate (alpha)
- So `--lr=0.02` in FL means `alpha=0.02`

---

## Visual Comparison

### ES Training Process

```
1. EXPLORATION (Sigma):
   For each perturbation p in population:
     θ_perturbed = θ + σ·ε_p          ← Sigma controls this
     reward_p = evaluate(θ_perturbed)
   
2. EXPLOITATION (Alpha):
   r_norm_p = normalize(rewards)
   update = Σ_p (r_norm_p · ε_p)
   θ_new = θ + (α/P) · update          ← Alpha controls this
```

### Parameter Roles

```
Sigma (σ):  "How far should I look?"     → Exploration radius
Alpha (α):   "How much should I move?"    → Update magnitude
LR (FL):     "Same as alpha in FL"        → Update magnitude
```

---

## Key Differences

### Sigma vs Alpha

| Aspect | Sigma (σ) | Alpha (α) |
|--------|-----------|-----------|
| **Phase** | Exploration | Exploitation |
| **When used** | During evaluation | During update |
| **Controls** | Perturbation size | Update step size |
| **Effect if too small** | Misses good regions | Learns too slowly |
| **Effect if too large** | Too noisy, unstable | Updates too aggressive |

### Alpha vs Learning Rate

| Context | Alpha | Learning Rate (lr) |
|--------|-------|-------------------|
| **ES without FL** | Separate parameter (`--alpha=0.002`) | Optimizer default (if used) |
| **ES with FL** | Same as `lr` (`alpha = optimizer.lr`) | Explicitly set (`--lr=0.02`) |

**In FL:** `--lr=0.02` means `alpha=0.02` (they're the same!)

---

## Tuning Guidelines

### If learning is too slow:
1. **Increase alpha** (0.02 → 0.03-0.05)
   - Makes updates more aggressive
   - Risk: May become unstable

2. **Increase sigma** (0.01 → 0.02)
   - Explores larger regions
   - Risk: May be too noisy

### If learning is unstable:
1. **Decrease alpha** (0.02 → 0.01)
   - Smaller, more stable updates

2. **Decrease sigma** (0.01 → 0.005)
   - More precise exploration

### If not finding good solutions:
1. **Increase sigma** (0.01 → 0.02-0.05)
   - Explore more broadly

2. **Increase num_pert** (100 → 150-200)
   - Better reward estimates

---

## Current Configuration Summary

### ES Without FL
- `sigma = 0.01` (exploration scale)
- `alpha = 0.002` (update step size)
- `lr = default` (optimizer, separate)

### ES With FL
- `sigma = 0.01` (exploration scale)
- `alpha = 0.02` (update step size, from lr)
- `lr = 0.02` (same as alpha)

**Note:** In FL, `alpha` and `lr` are the same value because the optimizer's learning rate is used as the ES learning rate.
