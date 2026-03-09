# When Are Updates Sent to Server?

## Answer: **After ALL 30 Steps Complete**

The client does **all 30 local steps**, collects **all gradient scalars**, and sends them **all at once** to the server.

## Detailed Flow

### Step-by-Step Process:

```
FL Round (One Iteration):

1. Server creates 30 seeds:
   seeds = [seed1, seed2, ..., seed30]

2. Server sends model + seeds to clients:
   - Client 1 receives: model + [seed1, seed2, ..., seed30]
   - Client 2 receives: model + [seed1, seed2, ..., seed30]
   - Client 3 receives: model + [seed1, seed2, ..., seed30]

3. Each client does ALL 30 local steps LOCALLY:
   Client 1:
     Step 1:  Get batch → Evaluate 50 perturbations → Update locally → Store grad_scalar₁
     Step 2:  Get batch → Evaluate 50 perturbations → Update locally → Store grad_scalar₂
     Step 3:  Get batch → Evaluate 50 perturbations → Update locally → Store grad_scalar₃
     ...
     Step 30: Get batch → Evaluate 50 perturbations → Update locally → Store grad_scalar₃₀
     
     Collects: [grad_scalar₁, grad_scalar₂, ..., grad_scalar₃₀]
     ⬇️
     Sends ALL 30 gradient scalars to server at once

4. Server receives from all clients:
   - Client 1: [grad_scalar₁, grad_scalar₂, ..., grad_scalar₃₀]
   - Client 2: [grad_scalar₁, grad_scalar₂, ..., grad_scalar₃₀]
   - Client 3: [grad_scalar₁, grad_scalar₂, ..., grad_scalar₃₀]

5. Server aggregates ALL updates:
   - Aggregates across clients (mean of 3 clients)
   - Aggregates across steps (all 30 steps)
   - Updates global model ONCE

6. Next FL round begins...
```

## Code Evidence

### In `server.py` (line 165):
```python
seeds = [random.randint(0, 1000000) for _ in range(self.local_update_steps)]
# Creates 30 seeds at once
```

### In `client.py` (line 126-170):
```python
def local_update(self, seeds: Sequence[int]) -> LocalUpdateResult:
    iteration_local_update_grad_vectors: list[torch.Tensor] = []
    
    for seed in seeds:  # Loops through ALL 30 seeds
        # ... do one local step ...
        grad_scalars = self.grad_estimator.compute_grad(...)
        iteration_local_update_grad_vectors.append(grad_scalars)  # Collects each one
    
    return LocalUpdateResult(
        grad_tensors=iteration_local_update_grad_vectors,  # Returns ALL 30
    )
```

### In `server.py` (line 177):
```python
global_grad_scalar = self.aggregation_func(local_grad_scalar_list)
# Aggregates ALL gradient scalars from all clients and all steps
```

## Visual Timeline

```
Time →
┌─────────────────────────────────────────────────────────────┐
│ FL Round 1                                                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ Server: Send model + 30 seeds to clients                    │
│                                                              │
│ Client 1:                                                    │
│   Step 1  → Step 2  → Step 3  → ... → Step 30              │
│   (all done locally, no communication)                      │
│   ⬇️                                                         │
│   Send ALL 30 gradient scalars to server                     │
│                                                              │
│ Client 2: (same)                                            │
│ Client 3: (same)                                            │
│                                                              │
│ Server: Receive ALL updates → Aggregate → Update model ONCE │
│                                                              │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│ FL Round 2 (starts with updated model)                      │
└─────────────────────────────────────────────────────────────┘
```

## Key Points

### ✅ What Happens:
1. **Client does ALL 30 steps locally** (no communication during steps)
2. **Client collects all 30 gradient scalars**
3. **Client sends ALL 30 at once** to server
4. **Server aggregates ALL** (from all clients, all steps)
5. **Server updates model ONCE** per FL round

### ❌ What Does NOT Happen:
- Client does NOT send after each step
- Server does NOT update after each step
- No communication during the 30 local steps

## Why This Design?

### Benefits:
1. **Reduced Communication**: Only one communication per FL round
2. **Better Aggregation**: Server sees all local steps before aggregating
3. **Efficient**: Batch all updates together

### Trade-offs:
1. **Client Drift**: Clients may diverge during 30 steps
2. **Delayed Updates**: Server only updates after all steps complete

## Communication Pattern

| Event | Communication | Frequency |
|-------|---------------|-----------|
| **Send model to clients** | Server → Clients | Once per FL round |
| **30 local steps** | None (local only) | 30 times (no communication) |
| **Send updates to server** | Clients → Server | Once per FL round (all 30 together) |
| **Server aggregation** | Server (internal) | Once per FL round |
| **Server model update** | Server (internal) | Once per FL round |

## Summary

**Question**: Does client send update after each step or after all 30 steps?

**Answer**: **After ALL 30 steps complete**

- Client does 30 local steps (no communication)
- Client collects all 30 gradient scalars
- Client sends all 30 at once to server
- Server aggregates all and updates model once

This is the standard federated learning pattern: **local training → batch communication → aggregation**.
