# ZO RGE Comparison Summary: With FL vs Without FL

## Key Findings

### **Without FL (Single Machine) - ✅ LEARNING SUCCESSFULLY**

**Performance:**
- **Initial**: Loss=2.28, Accuracy=13.45% (Round 0)
- **Final**: Loss=0.67, Accuracy=84.39% (Round 18)
- **Best**: Accuracy=87.86% (Round 6)

**Learning Pattern:**
1. **Rapid improvement** (Rounds 0-6): Accuracy jumps from 13% → 88%
2. **Peak performance** (Round 6): Reaches 87.86% accuracy
3. **Slight degradation** (Rounds 7-18): Accuracy drops to ~84-85%, loss increases
   - This suggests possible overfitting or learning rate issues

**Training Characteristics:**
- ✅ Model learns effectively
- ✅ Loss decreases significantly (2.28 → 0.67)
- ✅ Accuracy improves dramatically (13% → 84%)
- ⚠️ Some overfitting after round 6

---

### **With FL (Federated Learning) - ❌ NOT LEARNING**

**Performance:**
- **Initial**: Loss=2.32, Accuracy=9.13% (Round 0)
- **Final**: Loss=2.32, Accuracy=9.08% (Round 3)
- **All rounds**: Accuracy stuck at ~9% (essentially random for 10-class MNIST)

**Learning Pattern:**
1. **No learning**: Model performance remains constant
2. **Loss barely changes**: 2.3247 → 2.3239 (minimal decrease)
3. **Accuracy stuck**: ~9% throughout (random guessing)

**Training Characteristics:**
- ❌ Model is NOT learning
- ❌ Loss not decreasing
- ❌ Accuracy stuck at random performance
- ❌ Only 4 evaluation points (evaluates every 5 iterations)

---

## Critical Issues with FL Version

### 1. **Model Not Learning**
The FL version shows no improvement, suggesting:
- Gradient estimates may be too noisy
- Learning rate might be too small for FL setting
- Data distribution across clients might be problematic
- Communication/aggregation issues

### 2. **Evaluation Frequency Mismatch**
- **Without FL**: Evaluates every epoch (20 evaluations)
- **With FL**: Evaluates every 5 iterations (4 evaluations)
- This makes direct comparison difficult

### 3. **Performance Gap**
- **Without FL**: 84-88% accuracy
- **With FL**: 9% accuracy (random)
- **Gap**: ~75 percentage points difference

---

## Recommendations

### Immediate Actions:

1. **Increase Learning Rate for FL**
   ```bash
   --lr=1e-5  # Try higher learning rate for FL
   ```

2. **Increase Number of Perturbations**
   ```bash
   --num-pert=20  # More perturbations = better gradient estimates
   ```

3. **Adjust Evaluation Frequency**
   ```bash
   --eval-iterations=1  # Evaluate every iteration for better monitoring
   ```

4. **Check Data Distribution**
   - Verify data is being split correctly across clients
   - Consider using IID data distribution (`--iid` flag)

5. **Try Different FL Settings**
   - Increase `--local-update-steps` (currently 1)
   - Increase `--num-sample-clients` (currently 2 out of 3)

### Suggested FL Command:
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

---

## Conclusion

**Without FL**: ✅ **Success** - Model learns effectively, reaches 84-88% accuracy

**With FL**: ❌ **Failure** - Model not learning, stuck at random performance (9%)

**Root Cause**: The FL version needs different hyperparameters. The current settings (lr=5e-6, num-pert=10, local-update-steps=1) are too conservative for the federated setting where:
- Gradient estimates are aggregated across clients (more noise)
- Data is split across clients (less data per client)
- Communication happens via scalars (approximate gradients)

**Next Steps**: Adjust hyperparameters specifically for FL to enable learning.
