# TAM Python Implementation

Action-Binding TAM with Continuous Ports and Dual Controller.

## Architecture

### Core Concepts

- **Continuous ports**: Latent commitments `z ~ q(z|s0)` via ActorNet (VAE-style encoder)
- **Policy**: `a = pi(x, z)` - actions conditioned on commitment
- **Tube prediction**: Predicts `mu_t, sigma_t` for `t=1..T` via knot values + linear interpolation
- **Learned horizon**: Tube predicts `p_stop` (geometric halting distribution)
- **Binding predicate**: Realized trajectory stays within k-sigma tube

### Training: Dual Tit-for-Tat Controllers

Two-constraint optimization via dual controllers:

1. **Reliability constraint** (bind success rate):
   - Cone minimization: `w_cone * weighted_cone_volume` (pushes to tighten)
   - Constraint penalty: `lambda_bind * (target_bind - soft_bind_rate)`
   - Dual update: `lambda_bind += lambda_lr * (target_bind - soft_bind_rate)`

2. **Compute constraint** (expected horizon):
   - Ponder cost: `(lambda_h + lambda_T) * E[T]`
   - Dual update: `lambda_T += lambda_T_lr * (E[T] - target_ET)`

**Key exploit prevention**:
- Cone volume weighted by geometric halting distribution (prevents "early tight, late wide")
- Soft bind rate weighted by geometric halting distribution (consistent with exp_nll)
- Result: oscillatory homeostasis at the Pareto boundary of tightest cone + shortest horizon

## Module Structure

```
python/
├── __init__.py              # Package exports
├── utils.py                 # Utility functions (NLL, KL, geometric weights, interpolation)
├── networks.py              # Neural network architectures (ActorNet, Policy, Tube)
├── tam_continuous.py        # Main Actor class
├── experiments.py           # Environments and visualization
├── tam_net.py               # Backward compatibility wrapper
└── README.md                # This file
```

## Usage

### Basic Usage

```python
from tam.python import Actor, ControlledPendulumEnv, run_experiment

# Run a full experiment with visualization
run_experiment(seed=0, steps=8000)
```

### Custom Training Loop

```python
import torch
import numpy as np
from tam.python import Actor, ControlledPendulumEnv

# Create environment and agent
env = ControlledPendulumEnv()
agent = Actor(
    state_dim=2,
    z_dim=8,
    maxH=64,
    minT=2,
    M=16,  # number of knots
)

env.reset()

for step in range(1000):
    s0 = env.state.copy()

    # Sample commitment z for this episode
    s0_t = torch.tensor(s0, dtype=torch.float32).unsqueeze(0)
    z, z_mu, z_logstd = agent.sample_z(s0_t)

    # Sample horizon
    T, E_T, _ = agent.sample_horizon(z, s0)

    # Execute policy for T steps
    def policy_fn(state_np):
        st = torch.tensor(state_np, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            a = agent._policy_action(z, st).numpy().squeeze()
        return float(a + np.random.normal(0, 0.05))  # exploration

    states, actions = env.rollout(regime_id=0, policy_fn=policy_fn, horizon=T)

    # Train
    agent.train_on_episode(
        step=step,
        regime=0,
        s0=s0,
        states=states,
        actions=actions,
        z=z,
        z_mu=z_mu,
        z_logstd=z_logstd,
        E_T_imagine=E_T,
    )
```

### Visualization

```python
from tam.python import plot_training_results

# After training
plot_training_results(agent, env)
```

## Configuration

### Actor Parameters

```python
agent = Actor(
    # Architecture
    state_dim=2,          # State dimensionality
    z_dim=8,              # Latent commitment dimension
    hidden_dim=64,        # Hidden layer size
    M=16,                 # Number of knots for tube interpolation

    # Training
    lr=7e-4,              # Learning rate
    maxH=64,              # Maximum horizon
    minT=2,               # Minimum horizon
    k_sigma=2.0,          # Tube width (k-sigma)

    # Regularization
    cone_reg=0.001,       # Cone regularization weight
    goal_weight=0.25,     # Terminal state cost weight
    action_l2=0.01,       # Action L2 penalty
    lambda_h=0.002,       # Base ponder cost
    beta_kl=3e-4,         # KL regularization weight

    # Dual controller (set automatically)
    # target_bind=0.90    # Target bind success rate
    # lambda_lr=0.05      # Reliability dual update rate
    # target_ET=16.0      # Target expected horizon
    # lambda_T_lr=0.02    # Compute dual update rate
    # w_cone=0.05         # Cone volume minimization weight
)
```

## Backward Compatibility

The original `tam_net.py` is maintained as a compatibility wrapper. Existing code can continue to import from it:

```python
# Old style (still works)
from tam.python.tam_net import Actor, run_experiment

# New style (preferred)
from tam.python import Actor, run_experiment
```

## Development

To run the experiment directly:

```bash
cd packages/tam/python
python -m experiments
# or
python tam_net.py
```

## Key Files

- **`utils.py`**: Pure utility functions with no dependencies on other modules
- **`networks.py`**: PyTorch neural network definitions
- **`tam_continuous.py`**: Core TAM logic, training loop, and dual controller
- **`experiments.py`**: Environment implementations and visualization code
- **`tam_net.py`**: Legacy entry point (imports from other modules)

## Theory

This implementation demonstrates:

1. **Episodic binding**: Commitment `z` sampled once per episode and held fixed
2. **Geometric tubes**: Trajectory predictions via knot interpolation, not NN rollouts
3. **Learned horizons**: Ports learn their own stopping time via geometric halting
4. **Dual controllers**: Two tit-for-tat bargaining agents for reliability and compute
5. **Exploit prevention**: Consistent weighting across objectives prevents gaming

The result is an agent that converges to the **tightest cone** the environment allows while using **minimal compute** to achieve the target bind success rate.
