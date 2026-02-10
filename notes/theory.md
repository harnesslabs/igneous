# 📐 Mathematical Foundations

## 1. Conformal Geometric Algebra (CGA)
We use the **Cl(4, 1)** signature.
- **Basis Vectors:** $e_1, e_2, e_3$ (Spatial), $e_+$ (Origin), $e_-$ (Infinity).
- **Null Basis:**
    - $n_o = \frac{1}{2}(e_- - e_+)$ (Origin, point at zero)
    - $n_\infty = e_- + e_+$ (Infinity)
- **Geometric Primitives:**
    - **Point:** $P = x + \frac{1}{2}x^2 n_\infty + n_o$
    - **Sphere:** $S = P - \frac{1}{2}r^2 n_\infty$
    - **Plane:** $\pi = n + d n_\infty$

## 2. Discrete Morse Theory (DMT)
DMT allows us to study the topology of a simplicial complex using a discrete function $f$.

### The Hasse Diagram
We represent the mesh as a Directed Acyclic Graph (DAG) where nodes are simplices.
- **Edge:** A connection exists from $\sigma^{(p)}$ to $\tau^{(p-1)}$ if $\tau$ is a face of $\sigma$.

### Discrete Vector Field $V$
A collection of disjoint pairs $(\alpha^{(p)}, \beta^{(p+1)})$ such that $\alpha$ is a face of $\beta$.
- **Interpretation:** We "flow" from $\alpha$ to $\beta$.
- **Critical Simplex:** Any simplex not involved in a pair.
    - Critical Vertex -> Minimum
    - Critical Edge -> Saddle
    - Critical Triangle -> Maximum

### Persistence & Cancellation
We can simplify the topology by "canceling" a pair of critical points connected by a unique gradient path. This is key for **Topological Data Analysis (TDA)**.