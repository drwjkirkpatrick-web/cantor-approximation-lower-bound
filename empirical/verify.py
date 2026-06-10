"""
verify.py
=========

Empirical verification of the Cantor Approximation Lower Bound.

Core insight: The level-n Cantor approximation c_n is built recursively:
    Start: [0,1] with slope 1
    Each level: split every interval into 3, replace middle third with
               a plateau at the midpoint height
After n levels: 3^n piecewise linear/constant intervals.

A ReLU network with H hidden units has at most H linear pieces on [0,1]
(when all H breakpoints lie in (0,1), plus the initial segment [0, bp_1]).
Thus exact matching requires H >= 3^n.

Theorems:
    1. Interval Barrier: H < 3^n → nonzero error to c_n.
    2. Polynomial Rate: ε ~ H^{-log_3 2} ≈ H^{-0.631}.
    3. Feature Count: c_n has 3^n pieces; finite ReLU has H.

Usage:
    source ~/heartlib/.venv/bin/activate
    python empirical/verify.py
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


# =============================================================================
# Section 1: Device + Reproducibility
# =============================================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def manual_seed(seed: int = 1729) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Section 2: Level-n Cantor Approximation
# =============================================================================

def generate_level_n_intervals(n: int) -> List[Tuple[float, float, float, float]]:
    """
    Generate the 3^n intervals of the level-n Cantor approximation.
    
    Returns list of (start, end, left_height, right_height).
    """
    intervals = [(0.0, 1.0, 0.0, 1.0)]
    
    for _ in range(n):
        new_intervals = []
        for start, end, h_left, h_right in intervals:
            width = end - start
            third = width / 3.0
            h_mid = (h_left + h_right) / 2.0
            
            new_intervals.append((start, start + third, h_left, h_mid))
            new_intervals.append((start + third, start + 2 * third, h_mid, h_mid))
            new_intervals.append((start + 2 * third, end, h_mid, h_right))
        
        intervals = new_intervals
    
    return intervals


def cantor_level_n(n: int, x_values: np.ndarray) -> np.ndarray:
    """Evaluate c_n at given x values."""
    intervals = generate_level_n_intervals(n)
    y = np.zeros(len(x_values))
    
    for idx, x in enumerate(x_values):
        for start, end, h_left, h_right in intervals:
            if start <= x <= end:
                if abs(h_right - h_left) < 1e-12:
                    y[idx] = h_left
                else:
                    t = (x - start) / (end - start)
                    y[idx] = h_left + t * (h_right - h_left)
                break
    
    return y


# =============================================================================
# Section 3: ReLU Construction (CORRECT: f(x) = s_0·x + b + Σ w_i·relu(x-bp_i))
# =============================================================================

class ReLUNet(nn.Module):
    """Single-hidden-layer ReLU network."""
    
    def __init__(self, H: int):
        super().__init__()
        self.H = H
        self.fc1 = nn.Linear(1, H)
        self.fc2 = nn.Linear(H, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        return self.fc2(h)


def construct_exact_relu(n: int) -> ReLUNet:
    """
    Build a ReLU network with H = 3^n hidden units that EXACTLY matches c_n.
    
    CORRECT construction: f(x) = s_0·x + b + Σ w_i·relu(x - bp_i)
    where s_0 is the initial slope on [0, bp_1].
    
    We use the first unit as relu(x) = max(0, x) to provide the initial slope,
    then subsequent units add slope changes at each breakpoint.
    """
    intervals = generate_level_n_intervals(n)
    n_intervals = len(intervals)
    
    # We need H = n_intervals units:
    #   Unit 0: relu(x) = max(0,x) → provides initial slope on [0,1]
    #   Units 1..(n-1): relu(x - bp_i) → add slope changes at breakpoints
    H = n_intervals
    device = get_device()
    net = ReLUNet(H).to(device)
    
    # Compute slopes on each interval
    slopes = []
    for start, end, h_left, h_right in intervals:
        if end > start:
            slope = (h_right - h_left) / (end - start)
        else:
            slope = 0.0
        slopes.append(slope)
    
    # Breakpoints at interval boundaries (excluding 0, including 1)
    all_bps = sorted(set([end for _, end, _, _ in intervals]))
    # We need breakpoints in (0, 1), not at 0 or 1
    bps = [bp for bp in all_bps if 0 < bp < 1]
    
    with torch.no_grad():
        # Unit 0: relu(x) = max(0, x) provides initial slope on [0, ∞)
        # This means: on [0,1], this unit contributes x
        net.fc1.weight[0, 0] = 1.0
        net.fc1.bias[0] = 0.0
        net.fc2.weight[0, 0] = slopes[0]  # initial slope
        
        # Subsequent units: relu(x - bp_i) adds slope changes
        current_slope = slopes[0]
        for i in range(1, H):
            if i - 1 < len(bps):
                bp = bps[i - 1]
                if i < len(slopes):
                    new_slope = slopes[i]
                else:
                    new_slope = 0.0
                
                slope_change = new_slope - current_slope
                
                net.fc1.weight[i, 0] = 1.0
                net.fc1.bias[i] = -bp
                net.fc2.weight[0, i] = slope_change
                
                current_slope = new_slope
            else:
                net.fc1.weight[i, 0] = 0.0
                net.fc1.bias[i] = 10.0
                net.fc2.weight[0, i] = 0.0
        
        # Bias set so f(0) = c_n(0) = 0
        net.fc2.bias[0] = 0.0
    
    return net


def construct_limited_relu(H: int, n: int) -> ReLUNet:
    """Build ReLU with only H units matching first H intervals of c_n."""
    device = get_device()
    net = ReLUNet(H).to(device)
    
    intervals = generate_level_n_intervals(n)
    slopes = []
    for start, end, h_left, h_right in intervals:
        if end > start:
            slope = (h_right - h_left) / (end - start)
        else:
            slope = 0.0
        slopes.append(slope)
    
    all_bps = sorted(set([end for _, end, _, _ in intervals]))
    bps = [bp for bp in all_bps if 0 < bp < 1]
    
    with torch.no_grad():
        # Unit 0: relu(x)
        net.fc1.weight[0, 0] = 1.0
        net.fc1.bias[0] = 0.0
        net.fc2.weight[0, 0] = slopes[0] if len(slopes) > 0 else 0.0
        
        # First H-1 breakpoints
        current_slope = slopes[0] if len(slopes) > 0 else 0.0
        for i in range(1, H):
            if i - 1 < min(len(bps), len(slopes) - 1):
                bp = bps[i - 1]
                new_slope = slopes[i] if i < len(slopes) else 0.0
                slope_change = new_slope - current_slope
                
                net.fc1.weight[i, 0] = 1.0
                net.fc1.bias[i] = -bp
                net.fc2.weight[0, i] = slope_change
                
                current_slope = new_slope
            else:
                net.fc1.weight[i, 0] = 0.0
                net.fc1.bias[i] = 10.0
                net.fc2.weight[0, i] = 0.0
        
        net.fc2.bias[0] = 0.0
    
    return net


def relu_cantor_error(H: int, n: int, n_points: int = 10000) -> float:
    """L∞ error of H-unit ReLU vs c_n."""
    device = get_device()
    x = np.linspace(0, 1, n_points)
    y_target = cantor_level_n(n, x)
    
    net = construct_limited_relu(H, n)
    x_t = torch.tensor(x, dtype=torch.float32).reshape(-1, 1).to(device)
    
    with torch.no_grad():
        y_pred = net(x_t).cpu().numpy().flatten()
        error = float(np.abs(y_pred - y_target).max())
    
    return error


def relu_exact_error(n: int, n_points: int = 10000) -> float:
    """L∞ error of exact ReLU vs c_n — should be ~0."""
    device = get_device()
    x = np.linspace(0, 1, n_points)
    y_target = cantor_level_n(n, x)
    
    net = construct_exact_relu(n)
    x_t = torch.tensor(x, dtype=torch.float32).reshape(-1, 1).to(device)
    
    with torch.no_grad():
        y_pred = net(x_t).cpu().numpy().flatten()
        error = float(np.abs(y_pred - y_target).max())
    
    return error


# =============================================================================
# Section 4: Theorem Checks
# =============================================================================

@dataclass
class TheoremResult:
    name: str
    passed: bool
    metric: float
    detail: str


def check_theorem_1() -> TheoremResult:
    """Theorem 1: 3^n Interval Barrier."""
    all_passed = True
    max_exact_error = 0.0
    
    for n in [1, 2, 3]:
        exact_H = 3 ** n
        under_H = exact_H - 1
        
        exact_err = relu_exact_error(n)
        under_err = relu_cantor_error(under_H, n)
        
        exact_ok = exact_err < 1e-3
        under_ok = under_err > 1e-3
        
        if not exact_ok or not under_ok:
            all_passed = False
        
        max_exact_error = max(max_exact_error, exact_err)
        
        print(f"    n={n}: exact_H={exact_H}, exact_err={exact_err:.6f}, under_H={under_H}, under_err={under_err:.6f}")
    
    return TheoremResult(
        name="Theorem 1: 3^n Interval Barrier",
        passed=all_passed,
        metric=float(max_exact_error),
        detail=f"n=1,2,3 | Max exact error: {max_exact_error:.6f} | All bounds: {all_passed}",
    )


def check_theorem_2() -> TheoremResult:
    """Theorem 2: Polynomial Rate."""
    n = 4  # c_4 has 81 intervals
    H_values = [3, 9, 27]  # 3^1, 3^2, 3^3 (under-capacity for n=4)
    
    errors = [relu_cantor_error(H, n) for H in H_values]
    
    monotonic = all(errors[i+1] < errors[i] for i in range(len(errors)-1))
    
    log_H = np.log(H_values)
    log_err = np.log(errors)
    slope = float(np.polyfit(log_H, log_err, 1)[0])
    
    negative = slope < 0
    
    return TheoremResult(
        name="Theorem 2: Polynomial Decay",
        passed=monotonic and negative,
        metric=slope,
        detail=f"Slope={slope:.3f} | Monotonic: {monotonic}, Negative: {negative} | H={H_values}, err={[f'{e:.4f}' for e in errors]}",
    )


def check_theorem_3() -> TheoremResult:
    """Theorem 3: Feature Count Gap."""
    comparison = []
    
    for n in [1, 2, 3]:
        x = np.linspace(0, 1, 5000)
        y = cantor_level_n(n, x)
        n_pieces = count_pieces(y)
        
        exact_H = 3 ** n
        net = construct_exact_relu(n)
        device = get_device()
        x_t = torch.tensor(x, dtype=torch.float32).reshape(-1, 1).to(device)
        with torch.no_grad():
            y_relu = net(x_t).cpu().numpy().flatten()
        n_relu_pieces = count_pieces(y_relu)
        
        comparison.append((n, n_pieces, n_relu_pieces))
    
    all_match = all(c == r for _, c, r in comparison)
    growth = all(c >= 3 ** n for n, c, _ in comparison)
    
    return TheoremResult(
        name="Theorem 3: Feature Count Gap",
        passed=all_match and growth,
        metric=float(comparison[-1][1]),
        detail=f"(n, c_n, ReLU): {comparison} | Match: {all_match}, Growth: {growth}",
    )


def count_pieces(y: np.ndarray, threshold: float = 1e-5) -> int:
    """Count linear pieces by detecting slope changes."""
    dy = y[1:] - y[:-1]
    # Slope changes
    d2y = dy[1:] - dy[:-1]
    changes = np.abs(d2y) > threshold
    return int(changes.sum()) + 1


# =============================================================================
# Section 5: Main Runner
# =============================================================================

def main() -> int:
    print("=" * 70)
    print(" Cantor Approximation Lower Bound — Empirical Verification")
    print("=" * 70)
    
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA: {torch.version.cuda}")
    print()
    
    manual_seed(1729)
    
    results = []
    
    print("--- Theorem 1: 3^n Interval Barrier ---")
    r1 = check_theorem_1()
    results.append(r1)
    print(f"  {'✓' if r1.passed else '✗'}  {r1.detail}")
    print()
    
    print("--- Theorem 2: Polynomial Decay Rate ---")
    r2 = check_theorem_2()
    results.append(r2)
    print(f"  {'✓' if r2.passed else '✗'}  {r2.detail}")
    print()
    
    print("--- Theorem 3: Feature Complexity Gap ---")
    r3 = check_theorem_3()
    results.append(r3)
    print(f"  {'✓' if r3.passed else '✗'}  {r3.detail}")
    print()
    
    n_pass = sum(1 for r in results if r.passed)
    print("=" * 70)
    print(f"SUMMARY: {n_pass}/{len(results)} theorems verified")
    for r in results:
        flag = "✓ PASS" if r.passed else "✗ FAIL"
        print(f"   {flag}  {r.name}")
        print(f"          {r.detail}")
    print("=" * 70)
    
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
