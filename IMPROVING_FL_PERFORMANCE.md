# Strategies to Improve FL Performance vs Non-FL

## Current Status
- **ES without FL**: 90% accuracy
- **ES with FL**: 81.5% accuracy (with 30 perturbations)
- **Gap**: -8.5%

## Updated Configuration (num_pert=50)
- **Expected improvement**: 83-85% accuracy
- **Gap reduction**: From -8.5% to -5-7%

## Can FL Match Non-FL Performance?

### Short Answer: **Probably Not, But You Can Get Close**

**Why FL Will Always Have a Gap:**
1. **Data Heterogeneity**: Each client sees different data distributions
2. **Aggregation Noise**: Averaging across clients introduces variance
3. **Less Frequent Updates**: Updates only happen at server aggregation
4. **Communication Constraints**: Only scalars are communicated (not full gradients)

### Realistic Target: **85-87%** (3-5% gap from non-FL)

## Strategies to Close the Gap

### 1. **Increase Perturbations** ✅ (You're doing this!)
```bash
--num-pert=50  # Current change
--num-pert=100 # If 50 helps, try 100
```
**Why**: More perturbations = better reward estimates = better updates
**Expected**: +1-3% accuracy improvement
**Trade-off**: Slower per iteration

### 2. **Tune Sigma** (Exploration vs Precision)
```bash
--sigma=0.02  # Try between 0.01 and 0.03
```
**Why**: 
- Too large (0.03): Broad exploration but noisy estimates
- Too small (0.01): Precise but may miss good regions
- Sweet spot: 0.015-0.025
**Expected**: +0.5-1% accuracy improvement

### 3. **Increase Local Update Steps**
```bash
--local-update-steps=40  # or 50
```
**Why**: More local optimization before aggregation = better local models
**Expected**: +0.5-1% accuracy improvement
**Trade-off**: Slower per round

### 4. **Increase Training Iterations**
```bash
--iterations=150  # or 200
```
**Why**: More time to converge, especially important for FL
**Expected**: +0.5-1.5% accuracy improvement
**Trade-off**: Longer training time

### 5. **Tune Alpha (Learning Rate)**
```bash
--alpha=0.025  # or 0.03 (carefully!)
```
**Why**: Slightly higher learning rate might help FL converge better
**Expected**: +0.5-1% if stable
**Risk**: May become unstable (watch for NaN)

### 6. **Use All Clients Every Round** (You're already doing this!)
```bash
--num-clients=3 \
--num-sample-clients=3  # All clients participate
```
**Why**: More data per round = better aggregation
**Status**: ✅ Already optimal

### 7. **Better Aggregation Methods** (If available)
- **Current**: Mean aggregation
- **Alternatives**: Weighted average, median (for robustness)
- **Expected**: +0.5-1% if data is non-IID

## Recommended Progression

### Step 1: Test with 50 Perturbations (Current)
```bash
./run_es_with_fl.sh  # num_pert=50
```
**Target**: 83-85% accuracy

### Step 2: If < 84%, Try Sigma Tuning
```bash
--num-pert=50 \
--sigma=0.02 \  # Reduced from 0.03
```
**Target**: 84-85% accuracy

### Step 3: If Still < 85%, Increase Perturbations
```bash
--num-pert=100 \
--sigma=0.02
```
**Target**: 85-86% accuracy

### Step 4: If Still < 86%, Increase Local Steps
```bash
--num-pert=100 \
--sigma=0.02 \
--local-update-steps=40
```
**Target**: 86-87% accuracy

### Step 5: If Still < 87%, More Iterations
```bash
--num-pert=100 \
--sigma=0.02 \
--local-update-steps=40 \
--iterations=150
```
**Target**: 87-88% accuracy

## Theoretical Limits

### Why FL Can't Match Non-FL Perfectly:

1. **Information Loss**: 
   - Non-FL: Full gradients, all data visible
   - FL: Only scalar rewards, distributed data

2. **Heterogeneity**:
   - Non-FL: Single data distribution
   - FL: Multiple client distributions (even if IID split)

3. **Update Frequency**:
   - Non-FL: Updates every batch
   - FL: Updates only at aggregation

4. **Communication Constraints**:
   - Non-FL: No communication needed
   - FL: Only scalars communicated (by design)

### Best Case Scenario:
- **Non-FL**: 90%
- **FL (optimized)**: 87-88% (2-3% gap)
- **Your current FL**: 81.5% (8.5% gap)
- **With optimizations**: 85-87% (3-5% gap) ← **Realistic target**

## Quick Wins (Priority Order)

1. ✅ **Increase num_pert to 50** (You're doing this!)
2. **Tune sigma to 0.02** (Easy, quick test)
3. **Increase iterations to 150** (If time allows)
4. **Increase local_update_steps to 40** (If 50 perturbations help)

## Expected Results

| Configuration | Expected Accuracy | Gap from Non-FL |
|---------------|------------------|-----------------|
| **Current (30 pert)** | 81.5% | -8.5% |
| **50 pert** | 83-85% | -5-7% |
| **50 pert + sigma=0.02** | 84-86% | -4-6% |
| **100 pert + tuned** | 85-87% | -3-5% |
| **Fully optimized** | 86-88% | -2-4% |

## Bottom Line

**Can you match 90%?** Probably not - and that's okay!

**Can you get to 85-87%?** Yes, with the right hyperparameters!

**Is 81.5% bad?** No! It's actually quite good for FL.

**Next step**: Run with 50 perturbations and see if you get 83-85%. Then tune from there!
