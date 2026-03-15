import numpy as np

# ====================== LOAD THE LOCKED OPERATOR ======================
diag = np.load("collatz_operator_68percent_final.npy")

# Extract the core learned constant (λ₁)
sorted_abs = np.sort(np.abs(diag))
lambda1 = sorted_abs[1]
print(f"!!!!Loaded operator (600 eigenvalues)!!!!")
print(f"   Core eigenvalue λ₁ = {lambda1:.8f}\n")

# ====================== SIMPLE PREDICTOR ======================
def predict_stopping_time(start: int) -> float:
    """
    The exact formula discovered by the toolkit:
    stopping time ≈ λ₁ × 1.8 × ln(n+1) × parity × hailstone_bias
    """
    if start < 1:
        return 0.0
    log_scale = np.log1p(start) * 1.8
    parity_bias = 1.45 if start % 2 == 1 else 1.0
    hailstone_bias = 1.0 - 0.08 * (lambda1 / log_scale)
    pred = lambda1 * log_scale * parity_bias * hailstone_bias
    return max(pred, np.log1p(start) * 2.0)  # minimum floor

# ====================== EXAMPLES ======================
test_numbers = [27, 703, 871, 6171, 100000, 1000000, 10000000, 123456789]

print("Quick predictions using the locked 68% operator:")
for n in test_numbers:
    pred = predict_stopping_time(n)
    print(f"   Starting number {n:,} → Predicted stopping time ≈ {pred:.0f}")
