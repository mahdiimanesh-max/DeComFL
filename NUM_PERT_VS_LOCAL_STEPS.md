# Difference: `num_pert` vs `local_update_steps`

## Quick Summary

| Parameter | What It Controls | When It Happens |
|-----------|------------------|-----------------|
| **`num_pert`** | How many perturbations per step | **Within** each local step |
| **`local_update_steps`** | How many local steps per round | **Between** local steps |

## Detailed Explanation

### `num_pert` (Number of Perturbations)

**What it is:**
- How many **perturbations** are evaluated in **one single local step**
- Each perturbation = one evaluation of the model with added noise

**What happens:**
```
One Local Step:
  1. Get one batch of data
  2. Evaluate num_pert perturbations:
     - Perturbation 1: θ + σ·ε₁ → evaluate → get reward₁
     - Perturbation 2: θ + σ·ε₂ → evaluate → get reward₂
     - Perturbation 3: θ + σ·ε₃ → evaluate → get reward₃
     - ... (repeat num_pert times)
     - Perturbation 50: θ + σ·ε₅₀ → evaluate → get reward₅₀
  3. Aggregate rewards and update model
```

**Example with `num_pert=50`:**
- In **one local step**, you evaluate **50 different perturbations**
- Each perturbation tests the model with slightly different parameters
- You get 50 rewards, normalize them, and update the model

### `local_update_steps` (Number of Local Steps)

**What it is:**
- How many **local training steps** each client does before sending updates
- Each step = one batch + one model update

**What happens:**
```
One FL Round:
  1. Server sends model to clients
  2. Each client does local_update_steps steps:
     - Local Step 1: Get batch → Evaluate num_pert perturbations → Update
     - Local Step 2: Get batch → Evaluate num_pert perturbations → Update
     - Local Step 3: Get batch → Evaluate num_pert perturbations → Update
     - ... (repeat local_update_steps times)
     - Local Step 30: Get batch → Evaluate num_pert perturbations → Update
  3. Client sends all updates to server
```

**Example with `local_update_steps=30`:**
- Each client does **30 local training steps**
- Each step processes **one batch** and updates the model
- After 30 steps, client sends updates to server

## Key Differences

### 1. **Scope**

| Parameter | Scope | Level |
|-----------|-------|-------|
| `num_pert` | **Within** one local step | Micro-level |
| `local_update_steps` | **Across** multiple local steps | Macro-level |

### 2. **What They Control**

| Parameter | Controls | Purpose |
|-----------|---------|---------|
| `num_pert` | **Quality** of each update | Better gradient/reward estimates |
| `local_update_steps` | **Quantity** of local training | More local optimization |

### 3. **When They Happen**

```
FL Round:
  ├─ Client receives model
  ├─ Local Step 1:
  │   ├─ Get batch
  │   ├─ Evaluate num_pert perturbations (50 perturbations)
  │   └─ Update model
  ├─ Local Step 2:
  │   ├─ Get batch
  │   ├─ Evaluate num_pert perturbations (50 perturbations)
  │   └─ Update model
  ├─ ... (repeat local_update_steps times = 30 steps)
  └─ Send updates to server
```

## Visual Comparison

### With `num_pert=50` and `local_update_steps=30`:

```
One FL Round:
  Client 1:
    Step 1: [50 perturbations] → update
    Step 2: [50 perturbations] → update
    Step 3: [50 perturbations] → update
    ...
    Step 30: [50 perturbations] → update
    Total: 30 steps × 50 perturbations = 1,500 evaluations

  Client 2: Same (1,500 evaluations)
  Client 3: Same (1,500 evaluations)
  
  Total per round: 3 clients × 1,500 = 4,500 evaluations
```

### If you change `num_pert` to 100:

```
One FL Round:
  Client 1:
    Step 1: [100 perturbations] → update  ← More perturbations per step
    Step 2: [100 perturbations] → update
    ...
    Step 30: [100 perturbations] → update
    Total: 30 steps × 100 perturbations = 3,000 evaluations
```

### If you change `local_update_steps` to 40:

```
One FL Round:
  Client 1:
    Step 1: [50 perturbations] → update
    Step 2: [50 perturbations] → update
    ...
    Step 40: [50 perturbations] → update  ← More steps
    Total: 40 steps × 50 perturbations = 2,000 evaluations
```

## Effect on Performance

### `num_pert` (Number of Perturbations)

**What it affects:**
- **Quality** of each update
- **Variance** in reward estimates
- **Stability** of training

**More perturbations:**
- ✅ Better reward estimates (less noise)
- ✅ More stable updates
- ✅ Better convergence
- ❌ Slower per step

**Fewer perturbations:**
- ✅ Faster per step
- ❌ Noisier estimates
- ❌ Less stable

### `local_update_steps` (Number of Local Steps)

**What it affects:**
- **Amount** of local training
- **Data diversity** per round
- **Client optimization** before aggregation

**More local steps:**
- ✅ Better local optimization
- ✅ More data seen per round
- ✅ Potentially better updates
- ❌ Slower per round
- ❌ More client drift

**Fewer local steps:**
- ✅ Faster rounds
- ✅ Less client drift
- ❌ Less local optimization

## Total Computations

### Formula:
```
Total Evaluations per FL Round = 
  num_clients × local_update_steps × num_pert
```

### Examples:

| num_pert | local_update_steps | num_clients | Total Evaluations |
|----------|-------------------|-------------|-------------------|
| 30 | 30 | 3 | 2,700 |
| 50 | 30 | 3 | 4,500 |
| 50 | 40 | 3 | 6,000 |
| 100 | 30 | 3 | 9,000 |
| 100 | 40 | 3 | 12,000 |

## Which One to Increase?

### To Improve Update Quality:
**Increase `num_pert`** (50 → 100)
- Better reward estimates per step
- More stable training
- Better convergence

### To Improve Local Optimization:
**Increase `local_update_steps`** (30 → 40)
- More local training before aggregation
- Better client optimization
- More data per round

### To Maximize Performance:
**Increase both** (but slower!)
- `num_pert=100` + `local_update_steps=40`
- Best quality and optimization
- But much slower

## Current vs Proposed

### Current:
- `num_pert=50`: 50 perturbations per step
- `local_update_steps=30`: 30 steps per round
- Total: 4,500 evaluations per round

### If you increase `num_pert` to 100:
- `num_pert=100`: 100 perturbations per step (better quality)
- `local_update_steps=30`: 30 steps per round (unchanged)
- Total: 9,000 evaluations per round (2x more)

### If you increase `local_update_steps` to 40:
- `num_pert=50`: 50 perturbations per step (unchanged)
- `local_update_steps=40`: 40 steps per round (more training)
- Total: 6,000 evaluations per round (1.33x more)

## Summary

| Aspect | `num_pert` | `local_update_steps` |
|--------|-----------|---------------------|
| **What** | Perturbations per step | Steps per round |
| **Controls** | Update quality | Training quantity |
| **Level** | Within one step | Across multiple steps |
| **Effect** | Better estimates | More optimization |
| **Trade-off** | Slower per step | Slower per round |

**Key Point**: 
- `num_pert` = How thoroughly you evaluate in **one step**
- `local_update_steps` = How many steps you do **per round**

Both affect total computation, but in different ways!
