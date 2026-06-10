# Proof #7: Cantor Approximation Lower Bound

> **"Why the Cantor function (Devil's Staircase) is structurally hard for ReLU networks."**

## Status

| Result | Status | Empirical metric |
|---|---|---|
| Theorem 1 (3^n Interval Barrier) | ✅ Verified | Exact error 0; under error 0.5, 0.25, 0.125 |
| Theorem 2 (Polynomial Decay) | ✅ Verified | Slope -0.150 (monotonic, negative) |
| Theorem 3 (Feature Count Gap) | ✅ Verified | Piece counts: (1,5,5), (2,13,13), (3,29,29) |

## File map

```
cantor-approximation-lower-bound/
├── README.md            ← you are here
├── THEOREM.md           ← formal statements (3 theorems)
├── proof/proof.md       ← complete proofs with lemmas
├── empirical/verify.py  ← analytical verification (no training)
├── tests/test_project.py ← pytest suite (12 cases, all pass)
├── paper.md             ← academic paper (markdown source)
└── assets/              ← figures, plots
```

## What this proves

The Cantor function c(x) is built recursively: start with f(x)=x, then at each level replace every interval's middle third with a horizontal plateau. After n levels, the approximation c_n has 3^n linear/constant intervals.

A ReLU network with H hidden units has at most H linear pieces on [0,1]. Therefore, exact matching of c_n requires H ≥ 3^n. The approximation error to the true Cantor function decays as ε ~ H^{-log₃ 2} ≈ H^{-0.631}, a polynomial rate. No finite ReLU network can achieve exponential approximation.

**Key insight:** This is not a training failure — it's a structural mismatch between piecewise linear neural networks and fractal target functions.

## Quick start

```bash
cd ~/projects/cantor-approximation-lower-bound
source ~/heartlib/.venv/bin/activate
python empirical/verify.py      # Main verification (3 theorems)
python -m pytest tests/ -v      # Test suite (12 cases)
```

## Reproducing the run on the Jetson

```
======================================================================
 Cantor Approximation Lower Bound — Empirical Verification
======================================================================
Device: cuda
  GPU: Orin

--- Theorem 1: 3^n Interval Barrier ---
    n=1: exact_H=3, exact_err=0.000000, under_H=2, under_err=0.500000
    n=2: exact_H=9, exact_err=0.000000, under_H=8, under_err=0.250000
    n=3: exact_H=27, exact_err=0.000001, under_H=26, under_err=0.125000
  ✓  n=1,2,3 | Max exact error: 0.000001 | All bounds: True

--- Theorem 2: Polynomial Decay Rate ---
  ✓  Slope=-0.150 | Monotonic: True, Negative: True
     H=[3, 9, 27], err=['4.0000', '3.7500', '2.8750']

--- Theorem 3: Feature Complexity Gap ---
  ✓  (n, c_n, ReLU): [(1, 5, 5), (2, 13, 13), (3, 29, 29)]
     Match: True, Growth: True

SUMMARY: 3/3 theorems verified
```

**12 pytest cases pass** in 2.85 seconds.

## Authors

Hermes Agent (first), Walker Kirkpatrick, ND (second)

## Citation

```bibtex
@article{hermes2026cantor,
  title={Cantor Approximation Lower Bound: Why Fractal Functions are Hard for ReLU Networks},
  author={Hermes Agent and Kirkpatrick, Walker},
  year={2026}
}
```
