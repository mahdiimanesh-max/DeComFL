# Client Count Analysis for ES with FL

## Current Performance
- **Validation Accuracy**: 81.5%
- **Previous Best**: ~85%
- **Current Setup**: 3 clients, all sampled, 30 perturbations

## Data Distribution (MNIST: 60,000 training samples)

| Num Clients | Samples per Client | Num Sample Clients | Samples per Sampled Client |
|-------------|-------------------|-------------------|---------------------------|
| **2** | ~30,000 | 2 | ~30,000 |
| **3** (current) | ~20,000 | 3 | ~20,000 |
| **5** | ~12,000 | 3 | ~12,000 |
| **5** | ~12,000 | 5 | ~12,000 |
| **10** | ~6,000 | 3 | ~6,000 |
| **10** | ~6,000 | 10 | ~6,000 |

## Analysis: More Clients

### ❌ **Likely to HURT Performance**

**Reasons:**
1. **Less data per client** → Noisier reward evaluations
   - With 30 perturbations, each client needs enough data for stable estimates
   - 6k samples (10 clients) vs 20k (3 clients) = 3x less data per client
   
2. **Limited perturbations** → More clients don't help much
   - You only have 30 perturbations per client
   - More clients = more diversity, but each client's estimate is noisier
   - Better to have fewer clients with more data each

3. **Aggregation dilution**
   - More clients = more variance in rewards
   - Mean aggregation may dilute good signals

### ✅ **Might Help IF:**
- You increase `num_pert` to 50-100 (better estimates per client)
- You use fewer sampled clients (e.g., 5 total, sample 3)
- You increase `local_update_steps` (better local optimization)

## Recommendations

### Option 1: **Keep 3 Clients** (Recommended)
- **Why**: Best balance of data per client and diversity
- **Action**: Focus on other hyperparameters

### Option 2: **Try 5 Clients, Sample 3**
```bash
--num-clients=5 \
--num-sample-clients=3 \
```
- **Why**: More diversity, but sampled clients still have reasonable data
- **Trade-off**: Some clients unused each round

### Option 3: **Increase Perturbations Instead**
```bash
--num-clients=3 \
--num-sample-clients=3 \
--num-pert=50 \  # or 100
```
- **Why**: Better reward estimates with same client count
- **Trade-off**: Slower per iteration

## Expected Impact

| Change | Expected Effect | Risk |
|--------|----------------|------|
| **3 → 5 clients (sample 3)** | Slight improvement or neutral | Low |
| **3 → 10 clients (sample 3)** | Likely worse (less data per client) | High |
| **3 → 5 clients (sample 5)** | Likely worse (less data per client) | Medium |
| **Keep 3, increase num_pert to 50** | Likely better (better estimates) | Low |
| **Keep 3, increase num_pert to 100** | Likely better (best estimates) | Low (slower) |

## My Recommendation

**Don't increase client count.** Instead:

1. **Increase `num_pert` to 50-100** (better reward estimates)
2. **Keep 3 clients** (optimal data per client)
3. **Consider increasing `local_update_steps`** if needed

The 81.5% → 85% gap is likely due to:
- Reduced perturbations (30 vs previous 100)
- Increased sigma (0.03 vs 0.01) making estimates noisier
- Not client count

## Quick Test

If you want to test, try:
```bash
--num-clients=5 \
--num-sample-clients=3 \
--num-pert=50 \  # Increase this too
```

This gives you more diversity while maintaining good data per sampled client.
