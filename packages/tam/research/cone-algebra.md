
## 0) What is a “cone” in your current system?

Fix a situation \(s\) (in code: \(s_0\)) and a latent commitment \(z\). Your tube network induces parameters

\[
(\mu_{1:T}(s,z),\;\sigma_{1:T}(s,z),\;p_{\text{stop}}(s,z)),
\]

where \(\mu_t \in \mathbb{R}^D\), \(\sigma_t \in \mathbb{R}^D_{>0}\), and halting weights are the truncated geometric weights

\[
w_t(s,z) \;\;\text{for}\;\; t=1..T,\;\; \sum_t w_t=1.
\]

Define the **hard \(k\)-tube cone set** (your predicate’s geometry):

\[
\mathcal{C}_k(s,z)
=\Big\{ x_{1:T} \in \mathbb{R}^{T\times D} \;\Big|\; \forall t,\; |x_t-\mu_t|\_\infty \le k\,\sigma_t \Big\}.
\]

If you want to incorporate halting into the *set* notion, treat it as a **weighted cone** \((\mathcal{C}_k(s,z), w(s,z))\) where \(w\) determines which time-slices “matter” for cost/predicate.

So: **a cone is induced**, not stored. The algebra will act on the *inducing representation*.

---

## 1) The cone space and the main preorder (cone containment)

You need at least a notion of “this commitment is weaker/stronger than that one.”

For *same* \(s\) and \(T\), define a **pointwise containment preorder**:

\[
(s,z)\preceq (s,z')
\quad\Longleftrightarrow\quad
\mathcal{C}_k(s,z)\subseteq \mathcal{C}_k(s,z')
\]

A sufficient (checkable) condition is:

- \(\mu_t(s,z)=\mu_t(s,z')\) and \(\sigma_t(s,z)\le \sigma_t(s,z')\) elementwise for all \(t\).

But you’ll often change \(\mu\) too, so you’ll want a more forgiving “dominance” relation:

**Dominance (shift-aware containment):**
\[
(s,z)\preceq (s,z')
\;\;\text{if}\;\;
\forall t,d:\;
k\sigma'_{t,d} \ge k\sigma_{t,d} + |\mu_{t,d}-\mu'_{t,d}|.
\]

Interpretation: cone \(z'\) is wide enough to cover cone \(z\) even if the mean moved.

This preorder is what makes “tighten/widen” meaningful.

---

## 2) The algebra: operators on commitments with cone semantics

A cone operator is a map on commitments:
\[
\Omega:\mathcal{Z}\to\mathcal{Z}
\]
with induced effect on cones:
\[
\mathcal{C}_k(s,\Omega(z)).
\]

We want operators that correspond to the actions you keep describing:

### (A) Identity / No-op
\[
\mathrm{Id}(z)=z
\]
Semantics: keep the same cone.

### (B) Widen / Relax commitment
A *monotone* operator that should produce a cone that dominates the old one:
\[
\mathrm{Widen}_\alpha:\; z\mapsto z'
\quad\text{s.t. ideally}\quad
(s,z)\preceq (s,z')
\]
You can implement this two ways:

**(B1) Direct parameter-space widening (cleanest)**
Don’t change \(z\); change the tube output deterministically:
- \(\log\sigma_{t,d} \leftarrow \log\sigma_{t,d} + \alpha\)
- optionally adjust \(p_{\text{stop}}\) too.

This makes the semantics *exact*.

**(B2) z-space widening (learned)**
Define a learned direction field \(u(s,z)\) and do:
\[
z' = z + \alpha\,u(s,z).
\]
Then enforce monotonicity softly by adding a penalty if widening fails, e.g.
\[
\sum_{t,d}\max(0,\; (\log\sigma_{t,d}(s,z) - \log\sigma_{t,d}(s,z'))).
\]

### (C) Narrow / Strengthen commitment
Dual to widen:
\[
\mathrm{Narrow}_\alpha(z)=z'
\quad\text{with ideally}\quad
(s,z')\preceq (s,z)
\]
Same two implementation options (direct logσ shift or learned z-step).

### (D) Extend horizon / Commit further in time
This should *push mass later*:
\[
\mathrm{Extend}_\beta:\; p_{\text{stop}} \downarrow,\;\; \mathbb{E}[T]\uparrow
\]
Implementation in parameter space:
- stop_logit \(\leftarrow\) stop_logit \(-\beta\)  (decrease \(p_{\text{stop}}\))

### (E) Contract horizon / Hedge earlier
\[
\mathrm{Contract}_\beta:\; p_{\text{stop}} \uparrow,\;\; \mathbb{E}[T]\downarrow
\]
Implementation:
- stop_logit \(\leftarrow\) stop_logit \(+\beta\)

### (F) Recenter / Re-aim mean while keeping “strength”
Often you want to change what you’re predicting without changing commitment strength:
\[
\mathrm{Recenter}_{\delta}(z):\; \mu \text{ shifts},\; \sigma \text{ preserved (ideally)}
\]
In practice this is a z-space operator:
- \(z' = z + \delta\,v(s,z)\)
with an auxiliary penalty to keep \(\sigma\) stable.

This is the operator that lets the agent “change its mind” about *what will happen* without changing how strict it intends to be.

---

## 3) Composition rules (what makes it an “algebra”)

Let \(\mathfrak{O}\) be your set of operators. Use function composition \(\circ\) as the product:

- **Closure:** if \(\Omega_1,\Omega_2\in\mathfrak{O}\), then \(\Omega_2\circ\Omega_1\in\mathfrak{O}\)
- **Associativity:** \((\Omega_3\circ\Omega_2)\circ\Omega_1=\Omega_3\circ(\Omega_2\circ\Omega_1)\)
- **Identity:** \(\mathrm{Id}\circ \Omega=\Omega\circ \mathrm{Id}=\Omega\)

So \((\mathfrak{O},\circ,\mathrm{Id})\) is at least a **monoid**.

You also get some “approx inverses” in practice:

- \(\mathrm{Narrow}_\alpha\) is an approximate inverse of \(\mathrm{Widen}_\alpha\)
- \(\mathrm{Extend}_\beta\) is an approximate inverse of \(\mathrm{Contract}_\beta\)

Not exact, because the networks are nonlinear, but good enough for control.

---

## 4) The key: monotonic operators + a measurable “strength” functional

To make tit-for-tat / bargaining *legible*, define two scalar functionals:

### Cone “strength” / tightness
A simple one aligned with your current objectives:
\[
\mathrm{Tight}(s,z) \;=\; -\sum_{t} w_t \log\Big(\prod_d \sigma_{t,d}\Big)
\]
Higher = tighter.

### Reliability proxy (soft bind rate)
Your soft bind is already perfect:
\[
\mathrm{Rel}(s,z) \;=\; \sum_t w_t \prod_d \sigma\!\left(\frac{k\sigma_{t,d}-|x_{t,d}-\mu_{t,d}|}{\tau}\right)
\]

Now you can state the bargaining dynamics as movements in z that trade off these functionals.

---

## 5) Minimal “cone algebra” you can implement right now (no new nets)

If you want this algebra *today* without adding any new networks, do it in **parameter space**:

- \(\mathrm{Widen}_\alpha\): `logsig_knots += α`
- \(\mathrm{Narrow}_\alpha\): `logsig_knots -= α`
- \(\mathrm{Extend}_\beta\): `stop_logit -= β`
- \(\mathrm{Contract}_\beta\): `stop_logit += β`
- \(\mathrm{Id}\): nothing

This makes the semantics exact and avoids the “learned operator violates monotonicity” problem.

Then later you can “push the operator into z-space” by learning vector fields \(u,v\) and distilling these parameter-space moves into latent moves.

---

## **Endogenous latent moves**

1) Pick a small set of **scalar cone summaries** you care about (cone width / volume, horizon, maybe “reliability margin”), each a differentiable function of \((s_0, z)\).

2) Define a few **latent “operator directions”** in z-space (vector fields) that are supposed to *monotonically* move those summaries in the right direction.

3) Enforce monotonicity with a **finite-difference hinge loss**: “if I move a little in the widen direction, cone width must not go down.”

This gives you endogenous control *but with identifiable semantics*.

---

## 0) Choose your “cone summaries” (differentiable)

You already compute everything needed.

Let \(w\) be your geometric halting weights for a horizon \(T\) (or a fixed Teval for regularization), and let \(\sigma_t(z)\) come from the tube.

**Cone-width summary (scalar):** (log is nicer)
\[
C(s_0,z) := \sum_{t=1}^{T} w_t \cdot \log\big(\sigma_{t,0}\sigma_{t,1} + \epsilon\big)
\]
In code you already have `std` and `cv_t = std[:,0]*std[:,1]`. Use `log(cv_t)` and weight by `w`.

**Compute summary (expected horizon):**
\[
H(s_0,z) := \mathbb{E}[T] \text{ from } p_{\text{stop}}(s_0,z)
\]

**Optional “safety margin” summary (for calibration):**
\[
M(s_0,z) := \sum_t w_t \cdot \text{mean}_d \; \frac{k\sigma_{t,d} - |y_{t,d}-\mu_{t,d}|}{k\sigma_{t,d}+\epsilon}
\]
(You already compute a margin in older code; it’s useful as a “don’t widen unless you need to” signal.)

For monotone cone algebra, **C and H are the two big ones**.

---

## 1) Introduce latent operator directions in z-space

Start simple: **global directions** (learned parameters), not state-dependent.

- \(u_{\text{wide}} \in \mathbb{R}^{z\_dim}\): moving along this should widen cones (increase C)
- \(u_{\text{narrow}} = -u_{\text{wide}}\) (force inverse)
- \(u_{\text{extend}} \in \mathbb{R}^{z\_dim}\): moving along this should increase horizon (increase H)
- \(u_{\text{contract}} = -u_{\text{extend}}\)

You can add more later, but these four cover what you’ve been doing behaviorally.

Implementation:
```python
self.u_wide = nn.Parameter(torch.randn(z_dim) * 0.05)
self.u_ext  = nn.Parameter(torch.randn(z_dim) * 0.05)
```
Normalize in forward usage (so “step size” is meaningful):
```python
def _unit(v, eps=1e-8): return v / (v.norm() + eps)
uw = _unit(self.u_wide)
ue = _unit(self.u_ext)
```

---

## 2) Define monotonicity constraints as finite differences

Pick a small scalar step `delta` in z-space, e.g. `delta=0.25`.

For a summary \(S\), directional finite difference:
\[
\Delta_u S := S(z + \delta u) - S(z)
\]

### Widen should not *decrease* cone width
\[
\Delta_{u_{\text{wide}}} C \ge m_C
\]
Use a hinge:
\[
L_{\text{mono,wide}} = \text{ReLU}\big(m_C - \Delta_{u_{\text{wide}}} C\big)
\]

### Narrow should not *increase* cone width
Because narrow is \(-u_{\text{wide}}\), this is automatically enforced if the widen constraint holds with enough slack — but it’s often worth including explicitly:
\[
L_{\text{mono,narrow}} = \text{ReLU}\big(m_C + \Delta_{u_{\text{wide}}} C\big)
\]
(where \(\Delta_{u_{\text{wide}}} C = C(z+\delta u)-C(z)\); for the negative direction you get the sign flip.)

### Extend should not decrease horizon
\[
\Delta_{u_{\text{ext}}} H \ge m_H
\]
and similarly for contract.

**Margins \(m_C, m_H\)** can be small (like `m_C=0.0`, `m_H=0.0`) at first. If you want “strong” monotonicity, make them positive.

---

## 3) Where to compute these (cheaply)

Don’t use the full sampled horizon \(T\) (it’s noisy, expensive).

Use a **fixed Teval**, like `Teval = 24` or `32`, and compute:

- \(C\) using tube interpolation at Teval
- \(H\) using your `truncated_geometric_weights(p_stop, Teval)` or directly closed form-ish approximation

This adds **2 extra tube forward passes** per episode (for z+δuw and z+δue). That’s manageable and still way cheaper than enumerating many ports.

---

## 4) Concrete code patch

Add this helper to compute summaries given \((s0,z)\):

```python
def cone_summaries(self, s0_t: torch.Tensor, z: torch.Tensor, Teval: int = 32):
    # tube params
    muK, sigK, p_stop = self._tube_params(z, s0_t)
    mu, log_var = self._tube_traj(muK, sigK, Teval)
    std = torch.exp(0.5 * log_var)              # [Teval,D]
    cv_t = std[:, 0] * std[:, 1]                # [Teval]

    w, E_T = truncated_geometric_weights(p_stop, Teval)  # w:[Teval]
    C = (w * torch.log(cv_t + 1e-8)).sum()       # scalar
    H = E_T                                      # scalar
    return C, H
```

Then inside `train_on_episode` after you have `z` and `s0_t`:

```python
# ---- monotone cone algebra regularizer ----
Teval = min(32, self.maxH)
delta = 0.25
mC = 0.0
mH = 0.0
w_mono = 0.1   # start small, increase later

uw = self.u_wide / (self.u_wide.norm() + 1e-8)
ue = self.u_ext  / (self.u_ext.norm()  + 1e-8)

C0, H0 = self.cone_summaries(s0_t, z, Teval=Teval)

Cw, _  = self.cone_summaries(s0_t, z + delta * uw, Teval=Teval)
_,  He = self.cone_summaries(s0_t, z + delta * ue, Teval=Teval)

dC_wide = Cw - C0
dH_ext  = He - H0

L_wide  = torch.relu(mC - dC_wide)
L_narrow= torch.relu(mC + dC_wide)      # optional: enforce inverse explicitly

L_ext   = torch.relu(mH - dH_ext)
L_contr = torch.relu(mH + dH_ext)       # optional inverse

mono_loss = (L_wide + 0.5*L_narrow) + (L_ext + 0.5*L_contr)

loss = loss + w_mono * mono_loss
```

Add `u_wide` and `u_ext` to your optimizer params (they’re nn.Parameters, so your existing optimizer will pick them up if they’re attributes of the module/class you optimize; if not, include them explicitly).

---

## 5) Prevent degenerate solutions (important)

Without extra structure, the model can satisfy monotonicity in dumb ways.

Add two small regularizers:

### (A) “Independence” between widen and extend
Encourage \(u_{\text{wide}}\) and \(u_{\text{ext}}\) to be orthogonal:
\[
L_{\perp} = \left(\frac{u_w^\top u_e}{\|u_w\|\|u_e\|}\right)^2
\]
```python
cos = torch.dot(uw, ue)
loss = loss + 0.01 * (cos * cos)
```

### (B) Local smoothness (stop discontinuities)
Penalize huge second-order effects:
\[
L_{\text{smooth}} = \|S(z+\delta u) - 2S(z) + S(z-\delta u)\|
\]
This is optional but really helps keep “operator meaning” stable.

---

## 6) Why this gives you “monotone cone algebra”

Once you enforce:
- there exists a direction that *always* widens (increases C)
- there exists a direction that *always* extends (increases H)
- inverses exist (via -direction)
- they are disentangled (orthogonality)

…then you’ve turned z-space into a **semantically-typed control manifold**.

The agent can still pick arbitrary z endogenously, but now:
- “more committed” is an *identifiable direction*
- “more compute” is an *identifiable direction*
- those meanings can’t drift without paying loss

That’s exactly what you want.
