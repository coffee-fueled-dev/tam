# Simple 4D Environment

A fresh implementation of the TAM `Environment` interface demonstrating how to create a new environment that works with the generic `train_tam_system()` training loop.

## Files

- **`environment.py`**: `Simple4DEnvironment` - A clean 4D point-mass environment implementation
- **`train.py`**: Training script that uses `train_tam_system()` with the new environment
- **`__init__.py`**: Module exports

## Features

The `Simple4DEnvironment` implements the abstract `Environment` interface:

- **`get_context()`**: Returns raw context observation with obstacles, goals, boundaries, and energy
- **`bind_port()`**: Executes a tube trajectory and returns the actual path taken (with obstacle/boundary physics)
- **`get_intent()`**: Returns the relative goal vector pointing to the nearest goal
- **Properties**: `state_dim` and `context_dim`

## Usage

Run the training script:

```bash
python -m v3.d4.train
```

Or import and use programmatically:

```python
from v3.d4.environment import Simple4DEnvironment
from v3.system_impl import TAMSystemWrapper
from v3.train_tam import train_tam_system

# Create environment
env = Simple4DEnvironment(
    state_dim=4,
    bounds={'min': [-5.0, -5.0, -5.0, -5.0], 'max': [5.0, 5.0, 5.0, 5.0]},
    obstacles=[([1.0, 1.0, 1.0, 1.0], 0.5), ([2.0, -1.0, 0.5, -0.5], 0.3)],
    goals=[torch.tensor([3.0, 3.0, 3.0, 3.0]), torch.tensor([-2.0, 4.0, -1.0, 2.0])]
)

# Create system (using existing TAMSystemWrapper)
system = TAMSystemWrapper(...)

# Train using generic training loop
train_tam_system(system=system, environment=env, total_moves=500)
```

## Design

This environment is simpler than `SimulationWrapper` to demonstrate:
1. How to implement the `Environment` interface from scratch
2. The flexibility of the abstract contracts - any environment works with `train_tam_system()`
3. A clean, minimal implementation focused on the TAM formalism

The environment handles:
- Obstacle collisions (pushes away)
- Boundary constraints (clamps to bounds)
- Energy consumption (limits movement)
- Goal tracking (finds nearest goal for intent)
