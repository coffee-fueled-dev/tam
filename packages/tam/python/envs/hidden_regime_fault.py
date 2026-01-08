"""
Hidden Regime Fault Environment for TAM experiments.

Pendulum-like environment with:
- Hidden regime switching (0/1) with bursty behavior
- Actuator faults (invert, gain drop, deadzone)
- Partial observability (noisy theta-only observations)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FaultEvent:
    """Represents an actuator fault event."""
    kind: str  # "invert" | "gain_drop" | "deadzone"
    duration: int
    strength: float  # meaning depends on kind


class HiddenRegimeFaultEnv:
    """
    Pendulum-like environment with hidden regimes and actuator faults.
    
    Features:
    - Internal hidden regime (0/1), can switch stochastically and in bursts
    - Actuator faults (invert, gain drop, deadzone) applied for random durations
    - Partial observability: observe noisy theta only (optionally theta-only)
    """

    def __init__(
        self,
        noise_std: float = 0.005,
        obs_noise_std: float = 0.02,
        dt: float = 0.05,
        amax: float = 2.0,
        # base regimes: physics
        g0: float = 9.8,
        f0: float = 0.1,
        g1: float = 25.0,
        f1: float = 1.5,
        # base action semantics
        base_invert_regime1: bool = True,
        base_gain_regime0: float = 1.0,
        base_gain_regime1: float = 1.25,
        base_deadzone_regime1: float = 0.0,
        # regime switching
        switch_mid_episode: bool = True,
        switch_prob_per_step: float = 0.15,
        # bursty switching windows
        burst_prob_per_step: float = 0.01,
        burst_len_range: Tuple[int, int] = (8, 20),
        burst_switch_prob: float = 0.6,
        # faults
        fault_prob_per_step: float = 0.01,
        fault_duration_range: Tuple[int, int] = (10, 35),
        # obs
        obs_mode: str = "theta_only",  # "theta_only" or "theta_omega"
        seed: int = 0,
    ):
        self.rng = np.random.RandomState(seed)
        self.noise_std = noise_std
        self.obs_noise_std = obs_noise_std
        self.dt = dt
        self.amax = amax

        self.g0, self.f0 = g0, f0
        self.g1, self.f1 = g1, f1

        self.base_invert_regime1 = base_invert_regime1
        self.base_gain_regime0 = base_gain_regime0
        self.base_gain_regime1 = base_gain_regime1
        self.base_deadzone_regime1 = base_deadzone_regime1

        self.switch_mid_episode = switch_mid_episode
        self.switch_prob_per_step = switch_prob_per_step

        self.burst_prob_per_step = burst_prob_per_step
        self.burst_len_range = burst_len_range
        self.burst_switch_prob = burst_switch_prob
        self._burst_steps_left = 0

        self.fault_prob_per_step = fault_prob_per_step
        self.fault_duration_range = fault_duration_range
        self._fault: Optional[FaultEvent] = None
        self._fault_steps_left = 0

        self.obs_mode = obs_mode

        self.state = np.array([np.pi / 4, 0.0], dtype=np.float64)
        self.regime = 0

        # episode stats for volatility
        self._ep_switches = 0
        self._ep_fault_steps = 0
        self._ep_steps = 0

    @property
    def obs_dim(self) -> int:
        return 1 if self.obs_mode == "theta_only" else 2

    @property
    def state_dim(self) -> int:
        return 2

    def reset(self):
        self.state = np.array([np.pi / 4, 0.0], dtype=np.float64)
        self.regime = int(self.rng.rand() < 0.5)
        self._burst_steps_left = 0
        self._fault = None
        self._fault_steps_left = 0

        self._ep_switches = 0
        self._ep_fault_steps = 0
        self._ep_steps = 0

        return self.observe()

    def observe(self) -> np.ndarray:
        theta, omega = float(self.state[0]), float(self.state[1])
        if self.obs_mode == "theta_only":
            obs = np.array([theta], dtype=np.float64)
        else:
            obs = np.array([theta, omega], dtype=np.float64)

        obs += self.rng.normal(0, self.obs_noise_std, size=obs.shape)
        return obs

    def _maybe_start_burst(self):
        if self._burst_steps_left <= 0 and (self.rng.rand() < self.burst_prob_per_step):
            self._burst_steps_left = self.rng.randint(
                self.burst_len_range[0], self.burst_len_range[1] + 1
            )

    def _maybe_switch_regime(self):
        if not self.switch_mid_episode:
            return

        self._maybe_start_burst()
        p = (
            self.burst_switch_prob
            if self._burst_steps_left > 0
            else self.switch_prob_per_step
        )
        if self.rng.rand() < p:
            self.regime = 1 - self.regime
            self._ep_switches += 1

        if self._burst_steps_left > 0:
            self._burst_steps_left -= 1

    def _maybe_start_fault(self):
        if self._fault_steps_left > 0:
            return
        if self.rng.rand() < self.fault_prob_per_step:
            dur = self.rng.randint(
                self.fault_duration_range[0], self.fault_duration_range[1] + 1
            )
            kind = self.rng.choice(["invert", "gain_drop", "deadzone"])
            if kind == "invert":
                strength = 1.0
            elif kind == "gain_drop":
                strength = float(self.rng.uniform(0.2, 0.7))  # multiply gain by this
            else:
                strength = float(
                    self.rng.uniform(0.2, 0.8)
                )  # deadzone threshold in |a|
            self._fault = FaultEvent(kind=kind, duration=dur, strength=strength)
            self._fault_steps_left = dur

    def _effective_action(self, action: float) -> float:
        a = float(np.clip(action, -self.amax, self.amax))

        # base semantics by regime
        if self.regime == 0:
            gain = self.base_gain_regime0
            invert = False
            deadzone = 0.0
        else:
            gain = self.base_gain_regime1
            invert = self.base_invert_regime1
            deadzone = self.base_deadzone_regime1

        # apply fault modifiers if active
        if self._fault_steps_left > 0 and self._fault is not None:
            self._ep_fault_steps += 1
            if self._fault.kind == "invert":
                invert = not invert
            elif self._fault.kind == "gain_drop":
                gain *= self._fault.strength
            elif self._fault.kind == "deadzone":
                deadzone = max(deadzone, self._fault.strength)

        if abs(a) < deadzone:
            a = 0.0
        if invert:
            a = -a
        return gain * a

    def step(self, action: float) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Execute one step.
        
        Returns:
          obs_next, state_next, info
        """
        self._ep_steps += 1

        # update hidden processes
        self._maybe_switch_regime()
        self._maybe_start_fault()

        # countdown fault if active
        if self._fault_steps_left > 0:
            self._fault_steps_left -= 1
            if self._fault_steps_left == 0:
                self._fault = None

        theta, omega = float(self.state[0]), float(self.state[1])
        g, f = (self.g0, self.f0) if self.regime == 0 else (self.g1, self.f1)

        a_eff = self._effective_action(action)
        acc = -g * math.sin(theta) - f * omega + a_eff

        omega = omega + acc * self.dt
        theta = theta + omega * self.dt

        # process noise on latent state
        theta += self.rng.normal(0, self.noise_std)
        omega += self.rng.normal(0, self.noise_std)

        self.state = np.array([theta, omega], dtype=np.float64)
        obs = self.observe()

        info = {
            "regime": int(self.regime),
            "fault_active": int(self._fault is not None),
        }
        return obs, self.state.copy(), info

    def rollout(
        self, policy_fn, horizon: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Rollout episode.
        
        Args:
            policy_fn: function that takes obs and returns action (float)
            horizon: maximum steps
        
        Returns:
            (obs_seq, state_seq, actions, info)
            - obs:   [T+1, obs_dim]
            - states:[T+1, 2]
            - actions:[T, 1]
            - info: dict with volatility stats
        """
        obs0 = self.observe()
        obs_seq = [obs0]
        state_seq = [self.state.copy()]
        acts = []
        regimes = [int(self.regime)]

        for _ in range(horizon):
            a = float(policy_fn(obs_seq[-1]))
            acts.append([a])
            obs_next, s_next, info = self.step(a)
            obs_seq.append(obs_next)
            state_seq.append(s_next)
            regimes.append(int(info["regime"]))

        obs_arr = np.asarray(obs_seq)
        state_arr = np.asarray(state_seq)
        act_arr = np.asarray(acts)
        regimes = np.asarray(regimes, dtype=np.int32)

        # volatility index: switches per step + fault fraction
        switches = int(self._ep_switches)
        fault_frac = float(self._ep_fault_steps / max(1, self._ep_steps))
        vol = float(switches / max(1, self._ep_steps) + fault_frac)

        info_out = {
            "switches": switches,
            "fault_frac": fault_frac,
            "volatility": vol,
            "regime_path": regimes,
        }
        return obs_arr, state_arr, act_arr, info_out
