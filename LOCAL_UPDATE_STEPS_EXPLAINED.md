# What is `--local-update-steps`?

## Simple Explanation

**`--local-update-steps`** = **How many training steps each client does locally before sending updates to the server**

## In Federated Learning Flow

### Without FL (Single Machine):
```
For each batch:
  1. Get batch
  2. Compute gradient
  3. Update model
  4. Repeat
```
- Updates happen **every batch**
- Model updates **immediately**

### With FL (Federated Learning):
```
For each FL round:
  1. Server sends model to clients
  2. Each client does LOCAL training:
     - Step 1: Get batch, compute gradient, update locally
     - Step 2: Get batch, compute gradient, update locally
     - Step 3: Get batch, compute gradient, update locally
     - ... (repeat `local_update_steps` times)
  3. Clients send updates (gradient scalars) to server
  4. Server aggregates updates
  5. Server updates global model
```
- Updates happen **every FL round** (after local steps)
- Each client trains **locally** before communicating

## Example with `local_update_steps=30`

**What happens:**
1. Server sends current model to 3 clients
2. **Each client does 30 local training steps:**
   - Client 1: Trains on 30 batches locally
   - Client 2: Trains on 30 batches locally  
   - Client 3: Trains on 30 batches locally
3. Each client sends their updates (30 gradient scalars) to server
4. Server aggregates the updates from all 3 clients
5. Server updates the global model
6. Repeat for next FL round

## In Your ES Code

Looking at the code:

```python
# In server.py, line 165:
seeds = [random.randint(0, 1000000) for _ in range(self.local_update_steps)]
# Creates 30 random seeds (if local_update_steps=30)

# In client.py, line 115-126:
def local_update(self, seeds: Sequence[int]) -> LocalUpdateResult:
    for seed in seeds:  # Loop 30 times (if 30 seeds)
        # Get batch
        batch_inputs, labels = next(self.data_iterator)
        # Compute ES rewards (for each perturbation)
        grad_scalars = self.grad_estimator.compute_grad(...)
        # Update model locally
        self.grad_estimator.update_model_given_seed_and_grad(...)
    # Return all 30 gradient scalars
```

**For ES specifically:**
- Each local step evaluates `num_pert` perturbations (e.g., 50)
- With `local_update_steps=30`, each client does:
  - 30 local steps × 50 perturbations = 1,500 perturbation evaluations per client
  - 3 clients × 1,500 = 4,500 total evaluations per FL round

## Current vs Proposed

| Setting | Local Steps | What It Means |
|---------|-------------|---------------|
| **Current** | `--local-update-steps=30` | Each client does 30 local training steps |
| **Proposed** | `--local-update-steps=40` | Each client does 40 local training steps |

## Why Increase It?

### Benefits of More Local Steps:
1. **Better Local Optimization**: Each client trains more before aggregation
2. **More Data Used**: Each client sees more batches locally
3. **Better Updates**: More local training = better gradient/reward estimates
4. **Reduced Communication**: Fewer FL rounds needed (but more computation per round)

### Trade-offs:
1. **Slower Per Round**: More computation per FL round
2. **Client Drift**: Clients may diverge more from global model
3. **More Memory**: Need to store more gradient scalars

## Effect on ES

With `local_update_steps=40`:
- Each client: 40 steps × 50 perturbations = **2,000 evaluations per client**
- 3 clients: **6,000 total evaluations per FL round**
- More evaluations = better reward estimates = better updates

## Typical Values

| Scenario | Typical Value | Reason |
|----------|---------------|--------|
| **Fast training** | 1-10 | Quick rounds, frequent aggregation |
| **Balanced** | 20-30 | Good balance (your current) |
| **Thorough** | 40-50 | More local training, better estimates |
| **Very thorough** | 50-100 | Maximum local optimization |

## Recommendation

**Current**: `--local-update-steps=30` ✅ Good balance

**Try**: `--local-update-steps=40` if:
- You want better local optimization
- You have time for slower rounds
- You're trying to close the FL gap

**Expected improvement**: +0.5-1% accuracy (small but helpful)

## Summary

- **`local_update_steps=30`**: Each client trains locally for 30 steps before sending updates
- **`local_update_steps=40`**: Each client trains locally for 40 steps (more thorough)
- **Effect**: Better local optimization, potentially better FL performance
- **Trade-off**: Slower per round, but fewer rounds may be needed
