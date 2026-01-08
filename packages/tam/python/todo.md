# TODO: Refactor `Actor` to `nn.Module` + add monotone cone algebra

## 0) Refactor `Actor` into a proper `nn.Module` (so params register)

- [ ] Change class definition to subclass `torch.nn.Module`
- [ ] Call `super().__init__()` first thing in `__init__`
- [ ] Keep `lambda_bind`, `lambda_T` as **plain floats** (or 0-d tensors with `requires_grad=False`) since they’re updated manually
- [ ] Move device handling to `.to(device)` usage where possible; keep `self.device` only if you like

**Patch sketch**
```python
import torch.nn as nn

class Actor(nn.Module):
    def __init__(...):
        super().__init__()
        ...
        self.actor = ActorNet(...)
        self.pol   = SharedPolicy(...)
        self.tube  = SharedTube(...)
```

### Optimizer change (simplify)
- [ ] Replace manual parameter list with `self.parameters()` so new algebra params auto-include

```python
self.optimizer = optim.Adam(self.parameters(), lr=lr)
```

(If you keep a separate runner that owns the optimizer, delete `self.optimizer` from `Actor` entirely and create it outside.)

---

## 1) Add learnable operator directions (algebra generators)

### Add in `__init__`
- [ ] Add two learnable vectors in z-space:
  - `u_wide`: should **increase cone width summary** `C(s0,z)`
  - `u_ext`: should **increase horizon summary** `H(s0,z)=E[T]`
- [ ] Define inverses implicitly by negation:
  - `narrow = -u_wide`, `contract = -u_ext`
- [ ] Add hyperparameters for the algebra checks

```python
# --- cone algebra (latent operators) ---
self.algebra_Teval = min(32, self.maxH)
self.algebra_delta = 0.25
self.algebra_margin_C = 0.0
self.algebra_margin_H = 0.0
self.w_algebra = 0.02          # start small

self.w_algebra_ortho = 0.01    # optional disentanglement
self.w_algebra_comm  = 0.00    # optional (mostly 0 for now)

self.u_wide = nn.Parameter(torch.randn(self.z_dim) * 0.05)
self.u_ext  = nn.Parameter(torch.randn(self.z_dim) * 0.05)
```

---

## 2) Add helper to compute cone summaries `C(s0,z)` and `H(s0,z)`

- [ ] Add a method on `Actor`:

**Definition**
- `C`: weighted log cone volume (log of σ-product per step, weighted by halting weights)
- `H`: expected horizon proxy `E[T]` from `truncated_geometric_weights`

```python
def _cone_summaries(self, s0_t: torch.Tensor, z: torch.Tensor, Teval: int):
    muK, sigK, p_stop = self._tube_params(z, s0_t)
    _, log_var = self._tube_traj(muK, sigK, Teval)
    std = torch.exp(0.5 * log_var)         # [Teval,D]
    cv_t = std[:, 0] * std[:, 1]           # [Teval]

    w, E_T = truncated_geometric_weights(p_stop, Teval)  # w:[Teval]
    C = (w * torch.log(cv_t + 1e-8)).sum()               # scalar
    H = E_T                                              # scalar
    return C, H
```

(Weighted-by-halting matches your anti-exploit semantics.)

---

## 3) Add monotone algebra regularizer inside `train_on_episode`

### Where
- [ ] In `train_on_episode`, after you have `s0_t` and `z`, before `loss.backward()`

### What it enforces (finite differences)
- [ ] `+u_wide` should not decrease `C`
- [ ] `+u_ext` should not decrease `H`
- [ ] Optionally enforce inverse checks using `-u_wide` / `-u_ext`
- [ ] Optionally add orthogonality between directions

**Insert**
```python
Teval = self.algebra_Teval
delta = self.algebra_delta

uw = self.u_wide / (self.u_wide.norm() + 1e-8)
ue = self.u_ext  / (self.u_ext.norm()  + 1e-8)

C0, H0 = self._cone_summaries(s0_t, z, Teval)

Cw, _  = self._cone_summaries(s0_t, z + delta * uw, Teval)
_,  He = self._cone_summaries(s0_t, z + delta * ue, Teval)

dC_wide = Cw - C0
dH_ext  = He - H0

L_wide = torch.relu(self.algebra_margin_C - dC_wide)
L_ext  = torch.relu(self.algebra_margin_H - dH_ext)

# optional inverse checks
Cn, _  = self._cone_summaries(s0_t, z - delta * uw, Teval)
_,  Hc = self._cone_summaries(s0_t, z - delta * ue, Teval)
L_narrow = torch.relu(self.algebra_margin_C - (C0 - Cn))
L_contr  = torch.relu(self.algebra_margin_H - (H0 - Hc))

mono_loss = (L_wide + L_ext + 0.5 * (L_narrow + L_contr))

# optional orthogonality (disentangle widen vs extend)
ortho_loss = torch.tensor(0.0, device=s0_t.device)
if self.w_algebra_ortho > 0.0:
    cos = torch.dot(uw, ue)
    ortho_loss = cos * cos

algebra_loss = mono_loss + self.w_algebra_ortho * ortho_loss
loss = loss + self.w_algebra * algebra_loss
```

---

## 4) Keep dual controller semantics (don’t delete yet)

- [ ] Keep `lambda_bind` and `lambda_T` **as-is** while you introduce algebra
- [ ] Treat algebra loss as **structural** (defines axes), while dual variables do **closed-loop regulation**
- [ ] After it’s stable, do ablations:
  - remove compute controller (`lambda_T`) first
  - only then test removing reliability controller (`lambda_bind`) if bind rate stays near target

---

## 5) Add logging so you can verify the algebra “took”

### Add fields
- [ ] Extend `history` with:
  - `algebra_loss`
  - `dC_wide`
  - `dH_ext`

```python
"algebra_loss": [],
"dC_wide": [],
"dH_ext": [],
```

### Append each episode (detached)
```python
with torch.no_grad():
    self.history["algebra_loss"].append(float(algebra_loss.detach().item()))
    self.history["dC_wide"].append(float(dC_wide.detach().item()))
    self.history["dH_ext"].append(float(dH_ext.detach().item()))
```

Success criterion: `dC_wide` and `dH_ext` become consistently ≥ 0 (or ≥ margin).

---

## 6) Later upgrade path (optional)

- [ ] Once global `u_wide/u_ext` works, consider making them **vector fields**:
  - `u_wide = Uwide(s0)` or `Uwide(s0,z)`
  - keep the same monotonicity checks
  - only then consider commutator penalties (becomes meaningful when directions aren’t constant)
