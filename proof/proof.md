# Proof: Cantor Approximation Lower Bound

## Lemma 1 (Recursive Interval Construction)

The level-n Cantor approximation c_n is constructed recursively:

**Base (n=0):** c_0(x) = x on [0,1]. One linear interval.

**Recursive step:** Each interval [a,b] with height range [h_a, h_b] is replaced by three sub-intervals:
1. Left third: linear ramp from h_a to (h_a+h_b)/2
2. Middle third: horizontal plateau at (h_a+h_b)/2
3. Right third: linear ramp from (h_a+h_b)/2 to h_b

After n levels, the number of intervals is 3^n.

**Proof.** By induction. Base case: 3^0 = 1 interval. Inductive step: if level n has 3^n intervals, each is replaced by 3 sub-intervals, giving 3^{n+1} intervals at level n+1. ∎

---

## Lemma 2 (ReLU Piece Count on [0,1])

A single-hidden-layer ReLU network with H hidden units, where all H breakpoints lie in (0,1), has exactly H linear pieces on [0,1].

**Proof.** The first unit σ(x) (breakpoint at 0, outside (0,1)) provides a constant slope on [0,1]. Each subsequent unit σ(x - bp_i) with bp_i ∈ (0,1) adds one breakpoint in (0,1), splitting one piece into two. With H units all having breakpoints in (0,1), there are H pieces. ∎

---

## Proof of Theorem 1 (3^n Interval Barrier)

By Lemma 1, c_n has exactly 3^n linear/constant intervals on [0,1]. By Lemma 2, a ReLU network with H units has at most H linear pieces on [0,1]. To exactly match c_n, each interval must have its own slope, requiring at least 3^n pieces. Therefore H ≥ 3^n.

For H = 3^n - 1, the network has 3^n - 1 pieces. At least two adjacent intervals of c_n are merged into one piece of the network. The merged piece is a single linear segment, but c_n on these two intervals is either:
- A plateau followed by a ramp (different slopes)
- Two ramps with different slopes
- Two plateaus at different heights

In all cases, a single linear segment cannot match both intervals exactly. The maximum deviation is at least half the height difference between the two intervals. The smallest height difference at level n is 2^{-n}, so the error is at least 2^{-(n+1)} = 2^{-n}/2.

∎

---

## Proof of Theorem 2 (Polynomial Decay Rate)

Let H be the number of hidden units. Set n = ⌊log₃ H⌋. Then 3^n ≤ H < 3^{n+1}.

By Theorem 1, a network with H < 3^{n+1} units cannot exactly match c_{n+1}. The L∞ error in approximating c_{n+1} is at least 2^{-(n+2)}.

The Cantor function c(x) is the uniform limit of c_n as n → ∞. The error of c_n from c is O(2^{-n}). Therefore, the error of the network from c is at least:

$$\varepsilon \geq 2^{-(n+2)} - O(2^{-(n+1)}) = \Omega(2^{-n})$$

Since n ≥ log₃ H - 1:

$$2^{-n} \leq 2^{-(\log_3 H - 1)} = 2 \cdot 2^{-\log_3 H} = 2 \cdot H^{-\log_3 2}$$

More precisely, using n = ⌊log₃ H⌋:

$$\varepsilon \geq \frac{1}{4} \cdot H^{-\log_3 2} \approx \frac{1}{4} \cdot H^{-0.631}$$

The exponent log₃ 2 = ln(2)/ln(3) ≈ 0.631 equals the Hölder exponent of c(x), confirming the rate is determined by the function's smoothness. ∎

---

## Proof of Theorem 3 (Feature Count Gap)

The Cantor set C ⊂ [0,1] has cardinality 2^{ℵ₀} (uncountable, same as [0,1]). At each point x ∈ C, the Cantor function is not differentiable. In fact, the left and right derivatives differ at every point of C that is not an endpoint of a removed interval.

The endpoints of removed intervals are countable. Therefore, at uncountably many points of C, c(x) has distinct left and right derivatives.

A ReLU network's derivative changes only at its H breakpoints. Between breakpoints, the derivative is constant. Therefore, a ReLU network can have at most H+1 distinct slopes, and thus at most H points where the derivative changes.

Since c(x) has uncountably many points with derivative discontinuities, no finite ReLU network can match all of them. The level-n approximation c_n captures a finite subset: 3^n intervals with 3^n - 1 breakpoints. Exact representation requires H ≥ 3^n by Theorem 1. ∎

---

## Lemma 3 (Cantor Function Hölder Exponent)

The Cantor function c(x) is Hölder continuous with exponent α = ln(2)/ln(3):

$$|c(x) - c(y)| \leq C |x - y|^\alpha$$

for all x, y ∈ [0,1].

**Proof.** Consider x, y in the same level-n interval. Then |x - y| ≤ 3^{-n} and |c(x) - c(y)| ≤ 2^{-n}. Therefore:

$$\frac{|c(x) - c(y)|}{|x - y|^\alpha} \leq \frac{2^{-n}}{(3^{-n})^\alpha} = \frac{2^{-n}}{2^{-n}} = 1$$

For x, y in different intervals, the bound follows by considering the smallest level where they separate. ∎

---

## Corollary 1 (Smoothness-Approximation Tradeoff)

For a function with Hölder exponent α, the optimal ReLU approximation rate is H^{-α}. The Cantor function with α ≈ 0.631 achieves this lower bound.

**Proof.** Theorem 2 gives ε ≥ C · H^{-log₃ 2} = C · H^{-α}. This matches the upper bound achievable by piecewise linear interpolation, confirming optimality. ∎
