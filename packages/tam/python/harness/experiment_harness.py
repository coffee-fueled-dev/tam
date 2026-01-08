"""
Core experiment harness for TAM experiments.

Provides a unified interface for running experiments with any environment and actor.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
import torch

try:
    from actor import Actor
except ImportError:
    try:
        from ..actor import Actor  # type: ignore
    except ImportError:
        Actor = None  # type: ignore


class Environment(Protocol):
    """Protocol for environments compatible with the harness."""
    
    def reset(self) -> None:
        """Reset the environment."""
        ...
    
    def observe(self) -> np.ndarray:
        """Get current observation."""
        ...
    
    def rollout(
        self,
        policy_fn: Callable[[np.ndarray], Any],
        horizon: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Rollout an episode.
        
        Returns:
            obs_seq: [T+1, obs_dim] observations
            state_seq: [T+1, state_dim] states (or observations if state_dim not available)
            actions: [T, action_dim] actions
            info: dict with episode metadata
        """
        ...


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    
    # Experiment metadata
    name: str = "tam_experiment"
    seed: int = 0
    
    # Training parameters
    train_steps: int = 6000
    eval_every: int = 1000
    eval_episodes: int = 200
    
    # Actor parameters
    actor_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Environment parameters (will be passed to env constructor)
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation parameters
    k_sigma_list: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    k_star: float = 2.0
    
    # Output directory
    output_dir: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for saving."""
        return {
            "name": self.name,
            "seed": self.seed,
            "train_steps": self.train_steps,
            "eval_every": self.eval_every,
            "eval_episodes": self.eval_episodes,
            "actor_kwargs": self.actor_kwargs,
            "env_kwargs": self.env_kwargs,
            "k_sigma_list": self.k_sigma_list,
            "k_star": self.k_star,
        }


class ExperimentHarness:
    """
    Unified experiment harness for running TAM experiments.
    
    Handles:
    - Training loop with periodic evaluation
    - Saving results and visualizations
    - Managing experiment state
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        env_factory: Callable[[Dict[str, Any]], Environment],
        actor_factory: Optional[Callable[[Dict[str, Any]], Actor]] = None,
    ):
        """
        Initialize the experiment harness.
        
        Args:
            config: Experiment configuration
            env_factory: Function that creates an environment given kwargs
            actor_factory: Optional function that creates an actor given kwargs.
                          If None, uses default Actor constructor.
        """
        self.config = config
        self.env_factory = env_factory
        self.actor_factory = actor_factory or self._default_actor_factory
        
        # Set random seeds
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        
        # Create output directory
        if config.output_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.run_dir = Path("runs") / f"{config.name}_{timestamp}"
        else:
            self.run_dir = Path(config.output_dir)
        
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "fig").mkdir(exist_ok=True)
        (self.run_dir / "data").mkdir(exist_ok=True)
        
        # Initialize environment and actor
        self.env = self.env_factory(config.env_kwargs)
        self.actor = self.actor_factory(config.actor_kwargs)
        
        # Evaluation snapshots
        self.snapshots_lo: List[Any] = []  # Hr=1 or minimal reasoning
        self.snapshots_hi: List[Any] = []  # Hr=max or full reasoning
        
        # Store for plotting
        self._last_eval_step: Optional[int] = None
    
    @staticmethod
    def _default_actor_factory(kwargs: Dict[str, Any]) -> Actor:
        """Default actor factory using Actor class."""
        if Actor is None:
            raise ImportError("Actor class not available")
        return Actor(**kwargs)
    
    def save_config(self) -> None:
        """Save experiment configuration to JSON."""
        config_dict = self.config.to_dict()
        config_dict["run_dir"] = str(self.run_dir)
        
        with open(self.run_dir / "data" / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
    
    def train(self) -> Tuple[Actor, Environment]:
        """
        Run the training loop with periodic evaluation.
        
        Returns:
            Trained actor and environment
        """
        print(f"Starting experiment: {self.config.name}")
        print(f"Run directory: {self.run_dir}")
        
        self.save_config()
        
        # Training loop
        self.env.reset()
        for step in range(self.config.train_steps):
            # Get initial observation
            obs0 = self.env.observe()
            s0_t = torch.tensor(obs0, dtype=torch.float32, device=self.actor.device).unsqueeze(0)
            
            # Sample commitment
            z, z_mu, z_logstd = self.actor.sample_z(s0_t)
            
            # Sample reasoning steps
            Hr, E_Hr_imagine, p_ref_stop = self.actor.sample_reasoning_steps(z, obs0)
            
            # Sample horizon from refined tube
            T, E_T_imagine, p_stop_val = self.actor.sample_horizon_refined(s0_t, z, Hr)
            
            # Create policy function
            def policy_fn(obs_np: np.ndarray) -> Any:
                ot = torch.tensor(
                    obs_np, dtype=torch.float32, device=self.actor.device
                ).unsqueeze(0)
                action = self.actor._policy_action(z, ot).detach().cpu().numpy().squeeze()
                
                # Handle discrete vs continuous actions
                if self.actor.action_dim == 1:
                    # Continuous: add exploration noise
                    action = float(action + np.random.normal(0, 0.05))
                    if hasattr(self.env, 'amax'):
                        action = np.clip(action, -self.env.amax, self.env.amax)
                    return float(action)
                else:
                    # Discrete: sample from logits
                    logits = action
                    logits = logits + np.random.normal(0, 0.1, size=logits.shape)
                    return int(np.argmax(logits))
            
            # Rollout episode
            obs_seq, state_seq, actions, info = self.env.rollout(
                policy_fn=policy_fn, horizon=T
            )
            
            # Train actor
            self.actor.train_on_episode(
                step=step,
                regime=info.get("rule", 0),
                s0=obs0,
                states=state_seq,
                actions=actions,
                z=z,
                z_mu=z_mu,
                z_logstd=z_logstd,
                E_T_imagine=E_T_imagine,
            )
            
            # Periodic evaluation
            if (step + 1) % self.config.eval_every == 0 or step == 0:
                self._evaluate(step + 1)
        
        return self.actor, self.env
    
    def _evaluate(self, step: int) -> None:
        """Run evaluation at a given step."""
        try:
            from .evaluation import evaluate_agent_generic
        except ImportError:
            try:
                from evaluation import evaluate_agent_generic  # type: ignore
            except ImportError:
                print(f"[eval @ {step:5d}] Evaluation not available (import failed)")
                return
        
        # Determine Hr values based on reasoning mode
        if self.actor.reasoning_mode == "off":
            Hr_eval_lo = 0
            Hr_eval_hi = 0
        else:
            Hr_eval_lo = 1
            Hr_eval_hi = self.actor.max_refine_steps
        
        # Evaluate with minimal reasoning
        try:
            snap_lo = evaluate_agent_generic(
                agent=self.actor,
                env=self.env,
                step=step,
                ks=self.config.k_sigma_list,
                n_episodes=self.config.eval_episodes,
                k_star=self.config.k_star,
                Hr_eval=Hr_eval_lo,
                pred_dim=getattr(self.actor, 'pred_dim', None),
            )
            self.snapshots_lo.append(snap_lo)
        except Exception as e:
            print(f"[eval @ {step:5d}] Warning: Evaluation (Hr={Hr_eval_lo}) failed: {e}")
        
        # Evaluate with full reasoning
        try:
            snap_hi = evaluate_agent_generic(
                agent=self.actor,
                env=self.env,
                step=step,
                ks=self.config.k_sigma_list,
                n_episodes=self.config.eval_episodes,
                k_star=self.config.k_star,
                Hr_eval=Hr_eval_hi,
                pred_dim=getattr(self.actor, 'pred_dim', None),
            )
            self.snapshots_hi.append(snap_hi)
        except Exception as e:
            print(f"[eval @ {step:5d}] Warning: Evaluation (Hr={Hr_eval_hi}) failed: {e}")
            return
        
        # Print evaluation summary
        label_lo = f"Hr={Hr_eval_lo}" if self.actor.reasoning_mode != "off" else "Hr=0"
        label_hi = f"Hr={Hr_eval_hi}" if self.actor.reasoning_mode != "off" else "Hr=0"
        print(
            f"[eval @ {step:5d}] "
            f"{label_lo}: cov(k={self.config.k_star})={snap_lo.empirical_coverage[self.config.k_star]:.3f}, "
            f"logvol={snap_lo.mean_sharp_log_vol:.3f} | "
            f"{label_hi}: cov(k={self.config.k_star})={snap_hi.empirical_coverage[self.config.k_star]:.3f}, "
            f"logvol={snap_hi.mean_sharp_log_vol:.3f}"
        )
    
    def save_results(self, visualizations: Optional[List[Callable]] = None) -> None:
        """
        Save experiment results and generate standard plots.
        
        Args:
            visualizations: Optional list of additional visualization functions to call
        """
        # Save history
        if hasattr(self.actor, 'history'):
            history_dict = {k: np.asarray(v) for k, v in self.actor.history.items()}
            np.savez(self.run_dir / "data" / "history.npz", **history_dict)
        
        # Save memory if available
        if hasattr(self.actor, 'mem') and len(self.actor.mem) > 0:
            try:
                from experiments import extract_memory  # type: ignore
            except ImportError:
                try:
                    from ..experiments import extract_memory  # type: ignore
                except ImportError:
                    extract_memory = None  # type: ignore
            
            if extract_memory is not None:
                zs, soft, cone, lam = extract_memory(self.actor)
                if zs is not None:
                    risk = (1.0 - soft) + 0.2 * np.log(cone + 1e-8) + 0.05 * lam
                    np.savez(
                        self.run_dir / "data" / "memory.npz",
                        zs=zs,
                        soft=soft,
                        cone=cone,
                        lam=lam,
                        risk=risk,
                    )
        
        # Generate standard dashboard plots
        try:
            from .plots import plot_standard_dashboard
            
            # Determine prediction dimension from actor or environment
            D = getattr(self.actor, 'pred_dim', 2)
            
            # Generate standard plots if we have snapshots
            if len(self.snapshots_hi) > 0:
                print("Generating standard dashboard plots...")
                plot_standard_dashboard(
                    agent=self.actor,
                    snapshots_lo=self.snapshots_lo,
                    snapshots_hi=self.snapshots_hi,
                    run_dir=self.run_dir,
                    outcome_fn=None,  # Can be customized via visualizations
                    prefix="dashboard",
                    k_star=self.config.k_star,
                    D=D,
                )
            else:
                # If no snapshots, still try to generate plots that don't need them
                try:
                    from .plots import plot_compute_roi, plot_commitment_atlas
                    plot_compute_roi(self.actor, self.run_dir, prefix="dashboard_compute_roi")
                    plot_commitment_atlas(self.actor, None, self.run_dir, prefix="dashboard_atlas")
                except Exception as e:
                    print(f"Warning: Some plots failed (no snapshots): {e}")
        except Exception as e:
            print(f"Warning: Standard dashboard plots failed: {e}")
        
        # Run additional visualizations if provided
        if visualizations:
            for viz_fn in visualizations:
                try:
                    viz_fn(self.actor, self.env, self.run_dir)
                except Exception as e:
                    print(f"Warning: Visualization failed: {e}")
        
        print(f"\nAll outputs saved to: {self.run_dir}")
        print(f"  Figures: {self.run_dir / 'fig'}")
        print(f"  Data: {self.run_dir / 'data'}")
