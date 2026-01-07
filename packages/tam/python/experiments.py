"""
Environments and experiments for TAM.
"""

import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Handle both package import and direct execution
try:
    from .tam_continuous import Actor
except ImportError:
    from tam_continuous import Actor


class ControlledPendulumEnv:
    """
    Controlled pendulum environment with regime-dependent actuator semantics.

    This environment implements a pendulum with two different regimes that have
    different physics parameters and action mappings. Useful for testing TAM's
    ability to learn and adapt to different dynamics.
    """

    def __init__(
        self,
        noise_std: float = 0.005,
        dt: float = 0.05,
        amax: float = 2.0,
        action_invert_regime1: bool = True,
        action_gain_regime0: float = 1.0,
        action_gain_regime1: float = 1.25,
        action_deadzone_regime1: float = 0.0,
        switch_mid_episode: bool = True,
        switch_prob_per_step: float = 0.15,
    ):
        self.noise_std = noise_std
        self.dt = dt
        self.amax = amax

        self.action_invert_regime1 = action_invert_regime1
        self.action_gain_regime0 = action_gain_regime0
        self.action_gain_regime1 = action_gain_regime1
        self.action_deadzone_regime1 = action_deadzone_regime1

        self.switch_mid_episode = switch_mid_episode
        self.switch_prob_per_step = switch_prob_per_step

        self.state = np.array([np.pi / 4, 0.0], dtype=np.float64)

    def reset(self):
        """Reset environment to initial state."""
        self.state = np.array([np.pi / 4, 0.0], dtype=np.float64)

    def _effective_action(self, regime_id: int, action: float) -> float:
        """Apply regime-specific action transformations."""
        a = float(np.clip(action, -self.amax, self.amax))

        if regime_id == 0:
            return self.action_gain_regime0 * a

        if abs(a) < self.action_deadzone_regime1:
            a = 0.0
        if self.action_invert_regime1:
            a = -a
        return self.action_gain_regime1 * a

    def step(self, regime_id: int, action: float) -> np.ndarray:
        """Execute one step of the environment."""
        g, f = (9.8, 0.1) if regime_id == 0 else (25.0, 1.5)
        theta, omega = float(self.state[0]), float(self.state[1])

        a_eff = self._effective_action(regime_id, action)
        acc = -g * math.sin(theta) - f * omega + a_eff

        omega = omega + acc * self.dt
        theta = theta + omega * self.dt

        noise = np.random.normal(0, self.noise_std, size=(2,))
        theta += noise[0]
        omega += noise[1]

        self.state = np.array([theta, omega], dtype=np.float64)
        return self.state.copy()

    def rollout(
        self, regime_id: int, policy_fn, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute a full episode rollout with the given policy."""
        states = [self.state.copy()]
        actions = []
        r = int(regime_id)

        for _ in range(horizon):
            if self.switch_mid_episode and (
                np.random.rand() < self.switch_prob_per_step
            ):
                r = 1 - r

            a = float(policy_fn(states[-1]))
            actions.append([a])
            s_next = self.step(r, a)
            states.append(s_next)

        return np.asarray(states), np.asarray(actions)


def run_experiment(seed: int = 0, steps: int = 4000):
    """
    Run a TAM experiment with continuous ports and dual controller.

    This demonstrates the tit-for-tat controllers maintaining homeostasis
    at the boundary of tightest feasible cone with bounded compute.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = ControlledPendulumEnv(noise_std=0.005, dt=0.05, amax=2.0)
    agent = Actor(
        maxH=64,
        minT=2,
        M=16,  # number of knots (increase to 24 or 32 for sharper switching)
        z_dim=8,  # latent commitment dimension
        k_sigma=2.0,
        bind_success_frac=0.85,  # hard predicate threshold (used for logging only)
        lambda_h=0.002,  # ponder cost: try 0.001..0.01
        beta_kl=3e-4,  # KL regularization: try 1e-4..1e-3
        halt_bias=-1.0,  # stop bias: try -2.0 (longer), 0.0 (shorter)
        # Dual controller (automatically computed, not passed as args):
        # target_bind=0.90 (target soft bind rate for tit-for-tat)
        # lambda_lr=0.05 (dual update step size)
        # w_cone=0.05 (cone volume minimization weight)
        # margin_temp=0.25 (softness for differentiable bind)
    )

    env.reset()
    print("Starting Action-Binding TAM (Continuous Ports + Dual Controller)...")
    print(
        f"Reliability constraint: target_bind={agent.target_bind:.2f}, λ_lr={agent.lambda_lr}"
    )
    print(
        f"Compute constraint: target_ET={agent.target_ET:.1f}, λ_T_lr={agent.lambda_T_lr}"
    )
    print(f"Cone minimization weight: {agent.w_cone}")

    for i in range(steps):
        regime = 0 if (i // 1000) % 2 == 0 else 1
        s0 = env.state.copy()

        # commit to a latent port z for the whole episode
        s0_t = torch.tensor(s0, dtype=torch.float32, device=agent.device).unsqueeze(0)
        z, z_mu, z_logstd = agent.sample_z(s0_t)

        T, E_T, _pstop = agent.sample_horizon(z, s0)

        # bind: execute policy for T steps
        def policy_fn(state_np):
            st = torch.tensor(
                state_np, dtype=torch.float32, device=agent.device
            ).unsqueeze(0)
            with torch.no_grad():
                a = agent._policy_action(z, st).cpu().numpy().squeeze()
            a = float(a + np.random.normal(0, 0.05))  # exploration
            return float(np.clip(a, -env.amax, env.amax))

        states, actions = env.rollout(regime_id=regime, policy_fn=policy_fn, horizon=T)

        agent.train_on_episode(
            step=i,
            regime=regime,
            s0=s0,
            states=states,
            actions=actions,
            z=z,
            z_mu=z_mu,
            z_logstd=z_logstd,
            E_T_imagine=E_T,
        )

    # Visualization
    plot_training_results(agent, env)


def plot_training_results(agent: Actor, env):
    """
    Plot comprehensive training results showing dual controller behavior.
    """
    h = agent.history
    steps_arr = np.array(h["step"], dtype=np.int32)

    bind_success = np.array(h["bind_success"], dtype=np.float64)
    cone_volume = np.array(h["cone_volume"], dtype=np.float64)
    exp_nll = np.array(h["exp_nll"], dtype=np.float64)
    avg_abs_action = np.array(h["avg_abs_action"], dtype=np.float64)
    T_samp = np.array(h["T"], dtype=np.float64)
    E_T = np.array(h["E_T"], dtype=np.float64)
    E_T_train = np.array(h["E_T_train"], dtype=np.float64)
    kl = np.array(h["kl"], dtype=np.float64)
    z_norm = np.array(h["z_norm"], dtype=np.float64)
    soft_bind = np.array(h["soft_bind"], dtype=np.float64)
    lambda_bind = np.array(h["lambda_bind"], dtype=np.float64)
    lambda_T = np.array(h["lambda_T"], dtype=np.float64)
    cone_vol = np.array(h["cone_vol"], dtype=np.float64)

    agency = np.exp(-exp_nll)

    fig, axes = plt.subplots(9, 1, figsize=(12, 26), sharex=True)

    axes[0].plot(steps_arr, agency, alpha=0.9)
    axes[0].set_ylabel("Agency exp(-E[NLL])")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(
        "Action-Binding TAM: Continuous Ports + Dual Controller (tightest feasible cone)"
    )

    axes[1].plot(steps_arr, cone_volume, alpha=0.85)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Cone Vol (σ0*σ1)")

    axes[2].plot(
        steps_arr,
        np.convolve(bind_success, np.ones(200) / 200.0, mode="same"),
        alpha=0.9,
    )
    axes[2].set_ylabel("Bind Success")
    axes[2].set_ylim(-0.05, 1.05)

    axes[3].plot(steps_arr, avg_abs_action, alpha=0.85)
    axes[3].set_ylabel("Avg |action|")

    axes[4].plot(
        steps_arr,
        np.convolve(T_samp, np.ones(200) / 200.0, mode="same"),
        alpha=0.9,
        label="T sampled (smoothed)",
    )
    axes[4].plot(
        steps_arr,
        np.convolve(E_T_train, np.ones(200) / 200.0, mode="same"),
        alpha=0.9,
        label="E[T] train (smoothed)",
    )
    axes[4].axhline(
        y=agent.target_ET,
        color="g",
        linestyle="--",
        alpha=0.7,
        label=f"Target E[T] ({agent.target_ET})",
    )
    axes[4].set_ylabel("Horizon")
    axes[4].legend()

    axes[5].plot(
        steps_arr, np.convolve(kl, np.ones(200) / 200.0, mode="same"), alpha=0.9
    )
    axes[5].set_ylabel("KL(q(z|s0)||N(0,I))")

    axes[6].plot(
        steps_arr, np.convolve(z_norm, np.ones(200) / 200.0, mode="same"), alpha=0.9
    )
    axes[6].set_ylabel("||z|| (commitment norm)")

    # Dual controller: soft bind rate vs target
    axes[7].plot(
        steps_arr,
        np.convolve(soft_bind, np.ones(200) / 200.0, mode="same"),
        alpha=0.9,
        label="Soft bind rate",
    )
    axes[7].axhline(
        y=agent.target_bind,
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"Target ({agent.target_bind})",
    )
    axes[7].set_ylabel("Bind Rate")
    axes[7].set_ylim(-0.05, 1.05)
    axes[7].legend()
    axes[7].set_title("Tit-for-tat: hovering at target")

    # Dual variables: λ_bind (reliability) and λ_T (compute)
    axes[8].plot(
        steps_arr,
        np.convolve(lambda_bind, np.ones(200) / 200.0, mode="same"),
        alpha=0.9,
        color="purple",
        label="λ_bind (reliability)",
    )
    axes[8].plot(
        steps_arr,
        np.convolve(lambda_T, np.ones(200) / 200.0, mode="same"),
        alpha=0.9,
        color="orange",
        label="λ_T (compute)",
    )
    axes[8].set_ylabel("λ (dual variables)")
    axes[8].set_xlabel("Training Step")
    axes[8].set_title("Dual prices: reliability & compute constraints")
    axes[8].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_experiment(seed=0, steps=8000)
