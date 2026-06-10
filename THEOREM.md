# Theorem: Cantor Approximation Lower Bound

**Status:** Proved and empirically verified
**Target venue:** ICLR/NeurIPS workshop or arXiv preprint
**Date:** 2026-06-09
**Source paper(s):**
- Cantor, G. (1883). "Über unendliche, lineare Punktmannigfaltigkeiten"
- Yarotsky (2017), "Error bounds for approximations with deep ReLU networks"

---

## Notation

| Symbol | Type | Meaning |
|---|---|---|
| c(x) | [0,1]→[0,1] | Cantor function (Devil's Staircase) |
| c_n(x) | [0,1]→[0,1] | level-n Cantor approximation (3^n intervals) |
| H | int | number of ReLU hidden units |
| P(f) | int | number of linear pieces of f |
| ε | float | approximation error in L∞ |
| α | float | Hölder exponent, α = ln(2)/ln(3) ≈ 0.631 |

---

## Theorem 1 (3^n Interval Barrier)

The level-n Cantor approximation c_n is a continuous piecewise linear function with exactly 3^n linear pieces on [0,1]. A single-hidden-layer ReLU network with H hidden units has at most H linear pieces on [0,1] when all breakpoints lie in (0,1). Therefore, any ReLU network that exactly matches c_n must satisfy:

$$H \geq 3^n$$

For a network with H < 3^n units, the minimum L∞ approximation error to c_n is bounded below by the maximum deviation on the intervals that are merged, which is at least:

$$\varepsilon \geq \frac{1}{2} \cdot 2^{-n}$$

## Theorem 2 (Polynomial Decay Rate)

For a ReLU network with H hidden units approximating the Cantor function c(x), the minimum achievable L∞ error satisfies:

$$\varepsilon \geq C \cdot H^{-\log_3 2} \approx C \cdot H^{-0.631}$$

for some absolute constant C > 0. This is a polynomial rate; no ReLU network with finitely many pieces can achieve exponential approximation to c(x).

## Theorem 3 (Feature Count Gap)

The exact Cantor function c(x) has uncountably many "features" (slope changes accumulate at every point of the Cantor set). A finite ReLU network can represent at most H features. The level-n approximation c_n has 3^n linear pieces, requiring H ≥ 3^n units for exact representation. As n → ∞, the required H grows exponentially while the approximation error to c(x) decays only polynomially in H.

---

## Proof Sketch

**Theorem 1:** The level-n Cantor approximation is built by recursively replacing each interval with three sub-intervals. After n levels, there are 3^n intervals. A ReLU network with H units and all breakpoints in (0,1) produces at most H linear pieces on [0,1] (each unit contributes one breakpoint, and the initial segment [0, bp_1] needs its own slope). Thus H ≥ 3^n for exact matching.

**Theorem 2:** From Theorem 1, with H units we can match at most level n = ⌊log₃ H⌋. The error to c(x) is at least the error of c_n to c, which is O(2^{-n}) = O(H^{-log₃ 2}).

**Theorem 3:** Each recursive level triples the number of intervals. A finite network captures only finitely many levels.

The full proofs live in `proof/proof.md`.

---

## Open Questions

1. **Deep network rates:** Can depth improve the H^{-0.631} rate? Yarotsky (2017) suggests deep networks may achieve better rates for some functions.
2. **Smooth activations:** Do smooth activations approximate c(x) better, or is the barrier topological?
3. **Other fractals:** Does the Weierstrass function have a similar ReLU approximation barrier?
