# **Derivative-Sensitive Adaptive Pondering**.

### 1. The Mathematical Intuition

We want the agent to measure the "rate of clarification."
Let  be the total cone volume (uncertainty) at reasoning step .
The derivative (improvement rate) is:


* **High :** The plan is tightening rapidly. Keep thinking.
* **Low :** The plan has stabilized. Stop thinking to save compute.

### 2. Architectural Changes

We need to move the `PonderHead` *inside* the refinement loop in `Actor.infer_tube` so it can observe the evolving tube.

#### **A. Update `PonderHead` (networks.py)**

Instead of just , it now accepts the current reasoning state.

```python
class DynamicPonderHead(nn.Module):
    def __init__(self, state_dim, z_dim, hidden_dim=64):
        super().__init__()
        # Input: state, z, current_volume, volume_derivative
        self.fc1 = nn.Linear(state_dim + z_dim + 2, hidden_dim)
        self.stop_head = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, s0, z, current_vol, delta_vol):
        # Normalize inputs roughly to [-1, 1] range for stability
        vol_feats = torch.stack([current_vol, delta_vol], dim=-1)
        
        x = torch.cat([s0, z, vol_feats], dim=-1)
        h = self.relu(self.fc1(x))
        
        # Output probability of STOPPING
        return torch.sigmoid(self.stop_head(h))

```

#### **B. Update `infer_tube` (actor.py)**

We change the loop to be dynamic. Instead of iterating exactly `Hr` times, we iterate until the `PonderHead` says stop (or we hit a hard cap).

```python
def infer_tube_dynamic(self, s0_t, z, max_steps=10):
    # 1. Initial Guess
    mu, logsig, stop_logit = self._tube_init(z, s0_t)
    
    # Track volume
    def get_vol(ls): return torch.exp(ls).sum() # Simplified proxy
    
    prev_vol = get_vol(logsig)
    current_vol = prev_vol
    
    refined_steps = 0
    halting_probs = []

    for k in range(max_steps):
        # Calculate derivative (improvement)
        delta_vol = prev_vol - current_vol
        
        # 2. DECIDE: Should we stop?
        # Note: We detach gradients for the input to PonderHead to avoid 
        # "sabotaging" the tube just to make the PonderHead happy.
        p_stop = self.ponder_head(
            s0_t, z, 
            current_vol.detach(), 
            delta_vol.detach()
        )
        halting_probs.append(p_stop)
        
        # Soft halting logic (Bernoulli sampling during inference)
        if self.training:
            # During training, we might run fixed steps and weight losses
            pass 
        else:
            if torch.rand(1) < p_stop:
                break

        # 3. ACT: Refine
        d_mu, d_sig, d_stop = self.refiner(s0_t, z, mu, logsig, stop_logit)
        
        mu = mu + self.refine_step_scale * d_mu
        logsig = logsig + self.refine_step_scale * d_sig
        
        # Update stats
        prev_vol = current_vol
        current_vol = get_vol(logsig)
        refined_steps += 1

    return mu, logsig, refined_steps, halting_probs

```

### 3. Interpreting the Result

If this works, your **Chart 4 (Horizon vs Cone Volume)** should change significantly:

* **Current State:** Vertical stripes (indicating the model picks integer "modes" of depth 1, 2, 3, etc. beforehand).
* **New State:** You should see a smooth efficiency curve. The model will learn to take exactly as many steps as needed to bring the Cone Volume down to the "Safe" threshold (Top-left of Chart 3), and then immediately cut the computation.

### 4. Training the Dynamic Mechanism

This is the hardest part. You need a loss function that encourages the PonderHead to stop **only when** returns diminish.

You can add a **"Regret" Loss**:


* If it stops too early ( but  is high), it pays a penalty for high volume.
* If it thinks too long ( but  isn't changing), it pays the compute cost  ().
