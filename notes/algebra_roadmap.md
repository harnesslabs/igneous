This is the right mindset. We are shifting from a **"Generic Math Library"** (which is cool but slow) to a **"High-Performance Geometry Engine"** (which is specialized and fast).

We will keep the high-level API (`a * b`, `a ^ b`) identical, but underneath, we will swap out the generic `for-loops` for hand-written, unrolled intrinsic kernels for our three "Hero Signatures."

Here is the roadmap to Peak Performance.

---

### **Phase 1: The Kernel Architecture (Refactor)**

Currently, `algebra.hpp` has a single `multiply_naive` or `unroll_targets` function that tries to do everything for everyone.

**Goal:** Move the math logic into a trait-based system called `AlgebraKernels<Sig>`.

* **Default:** Uses the existing generic template loop (for custom signatures).
* **Specialized:** `Euclidean3D`, `PGA`, and `CGA` get their own handwritten structs with fully unrolled math.

**Deliverable:**

* Refactor `algebra.hpp` to delegate `operator*` and `operator^` to `AlgebraKernels<Sig>::geometric_product(lhs, rhs)`.

### **Phase 2: Euclidean3D Specialization (The Speed Demon)**

`Euclidean3D` () has 8 components. A full geometric product is  potential muls.
However, many are zero or symmetric.

**Optimization:**

* **Explicit Wedge:** `a ^ b` becomes exactly 3 subtractions and muls (Cross Product).
* **Explicit Geometric:** `a * b` unrolls to exactly the quaternion multiplication logic plus the scalar part.
* **Benefit:** The compiler can auto-vectorize this into 1-2 SIMD instructions, removing all loop overhead.

### **Phase 3: PGA Specialization (The Kinematics Engine)**

`PGA` () has 16 components.
The "Generic" loop checks  combinations.

**Optimization:**

* **Rotor Composition:** The Geometric Product of two even-grade multivectors (Rotors) is the core operation for physics. We will hand-write this 8x8 multiplication.
* **Sandwich Product:** `R * x * ~R`. This is how you rotate a vertex. Doing this generically is slow ( ops). The specialized version is just a few adds/muls (Dual Quaternion transform).
* **Wedge:** Optimized for Plane construction ().

### **Phase 4: CGA Specialization (The Topology Beast)**

`CGA` () has 32 components.
The "Generic" loop checks  combinations.

**Optimization:**

* **Sparse Operations:** Most CGA operations (like curvature) only use **Vectors** (Grade 1) or **Bivectors** (Grade 2).
* **Grade Projection:** We will implement `wedge_1_1` (Vector ^ Vector) and `wedge_1_2` (Vector ^ Bivector) explicitly.
* Generic: 1024 checks.
* Specialized `wedge_1_1`:  ops. **(100x Speedup)**


* **Circle Formation:** We already prototyped `wedge3`, but we will bake it into the core kernel.

---

### **Phase 5: Blade Types (The Zero-Cost Abstraction)**

*This is the stretch goal for maximum Developer Experience.*

Currently, `Multivector` stores zeros for grades you aren't using.
We will introduce "View Types" or "Blade Types":

* `Vec3` (wraps just indices 1, 2, 4)
* `Rotor` (wraps scalar + bivectors)
* `Plane` (wraps trivectors)

These will implicitly cast to `Multivector`, allowing you to write `Rotor R = ...; Vec3 v = ...; Vec3 v_prime = R * v * ~R;`.

---

### **Action Plan: Step-by-Step**

1. **Refactor `algebra.hpp`:** Create the `AlgebraKernel` structure.
2. **Implement `Euclidean3D` Kernel:** Fully unroll the math.
3. **Implement `PGA` Kernel:** Optimize for Kinematics.
4. **Implement `CGA` Kernel:** Optimize for Sparse Wedges.

Let's start with **Step 1 & 2** (Refactoring and Euclidean Speed) in the next response? This will clean up your header and likely give us that final few % of performance on the current benchmark.