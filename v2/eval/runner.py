"""
Universal evaluation runner for TAM actors.
Actor-agnostic, environment-agnostic where possible.
"""

from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, Callable, List
import json
import numpy as np
import torch
import torch.optim as optim

from .dataset import EpisodeBuffer, StratifiedEpisodeBuffer
from .metrics import compute_latent_metrics, compute_contract_metrics


@dataclass
class RunConfig:
    """Configuration for a training/evaluation run."""
    # Required fields (no defaults)
    actor_cls: type
    env_cls: type
    
    # Optional fields with defaults
    actor_kwargs: Dict[str, Any] = field(default_factory=dict)
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Training configuration
    train_epochs: int = 1000
    warmup_epochs: int = 500
    batch_size: int = 64
    
    # Evaluation configuration
    test_episodes: int = 50
    
    # Output configuration
    output_dir: Optional[Path] = None
    seed: Optional[int] = None
    
    # Optional callbacks
    on_epoch: Optional[Callable] = None
    on_warmup_batch: Optional[Callable] = None


class Runner:
    """
    Universal training/evaluation runner.
    """
    
    def __init__(self, config: RunConfig):
        self.config = config
        self.output_dir = config.output_dir or self._make_run_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seed
        if config.seed is not None:
            self._seed_all(config.seed)
        
        # Initialize environment
        self.env = config.env_cls(**config.env_kwargs)
        
        # Initialize actor
        obs_dim = self.env.obs_dim
        pred_dim = self.env.state_dim if hasattr(self.env, 'state_dim') else self.env.config.d
        T = self.env.config.T if hasattr(self.env, 'config') else 16
        
        actor_kwargs = config.actor_kwargs.copy()
        if 'obs_dim' not in actor_kwargs:
            actor_kwargs['obs_dim'] = obs_dim
        if 'pred_dim' not in actor_kwargs:
            actor_kwargs['pred_dim'] = pred_dim
        if 'T' not in actor_kwargs:
            actor_kwargs['T'] = T
        
        self.actor = config.actor_cls(**actor_kwargs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor.to(self.device)
        
        # Training state
        self.training_history = {
            'warmup': [],
            'actor': [],
        }
    
    def _make_run_dir(self) -> Path:
        """Create output directory with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path("artifacts/runs") / f"run_{timestamp}"
    
    def _seed_all(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def train_encoder_stage(
        self,
        n_epochs: int,
        batch_size: int,
        generate_episode_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Stage 1: Encoder-only warmup.
        
        Args:
            n_epochs: Number of warmup epochs
            batch_size: Batch size
            generate_episode_fn: Function that generates episodes
        
        Returns:
            Dict with warmup metrics
        """
        print(f"\nStage 1: Encoder Warmup ({n_epochs} epochs)...")
        
        if generate_episode_fn is None:
            generate_episode_fn = self._default_generate_episode
        
        # Create optimizer for encoder only
        encoder_params = []
        if hasattr(self.actor, 'encoder_mlp'):
            encoder_params.extend(self.actor.encoder_mlp.parameters())
        elif hasattr(self.actor, 'encoder'):
            encoder_params.extend(self.actor.encoder.parameters())
        else:
            raise ValueError("Actor must have encoder_mlp or encoder attribute")
        
        encoder_optimizer = optim.Adam(encoder_params, lr=1e-4)
        
        # Buffer for episodes
        buffer = EpisodeBuffer(max_size=1000)
        
        # Warmup loop
        z_samples = []
        labels = []
        batch_idx = 0
        
        for epoch in range(n_epochs):
            # Generate episodes and add to buffer
            for _ in range(batch_size):
                episode = generate_episode_fn()
                buffer.add(episode)
            
            # Train on batch
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                
                # Prepare batch
                traj_deltas = []
                batch_labels = []
                for ep in batch:
                    s0 = torch.tensor(ep['obs'][0], dtype=torch.float32, device=self.device)
                    traj = torch.tensor(ep['x'][1:], dtype=torch.float32, device=self.device)
                    traj_delta = traj - s0[:self.actor.pred_dim]
                    traj_deltas.append(traj_delta)
                    
                    # Extract label if available
                    k = ep.get('k', 0)
                    if isinstance(k, np.ndarray):
                        k = int(k[-1])
                    batch_labels.append(k)
                
                traj_batch = torch.stack(traj_deltas)
                labels_batch = torch.tensor(batch_labels, dtype=torch.long, device=self.device)
                traj_flat_batch = traj_batch.reshape(batch_size, -1)
                
                # Train encoder
                if hasattr(self.actor, 'train_encoder_batch'):
                    metrics = self.actor.train_encoder_batch(
                        traj_flat_batch, None, labels_batch, encoder_optimizer, epoch=batch_idx
                    )
                else:
                    # Fallback: single-step training
                    metrics = {}
                    for i in range(batch_size):
                        step_metrics = self.actor.train_encoder_step(
                            traj_batch[i], encoder_optimizer,
                            mode_label=batch_labels[i], epoch=epoch
                        )
                        if i == 0:
                            metrics = step_metrics
                
                # Collect z samples
                with torch.no_grad():
                    z_batch = self.actor.encode(traj_flat_batch)
                    z_samples.extend([z.cpu().numpy() for z in z_batch])
                    labels.extend(batch_labels)
                
                self.training_history['warmup'].append(metrics)
                
                if self.config.on_warmup_batch:
                    self.config.on_warmup_batch(batch_idx, metrics)
                
                batch_idx += 1
        
        # Compute final warmup metrics
        warmup_metrics = {}
        if len(z_samples) > 0:
            z_array = np.array(z_samples)
            warmup_metrics = compute_latent_metrics(z_array, labels if labels else None)
        
        return warmup_metrics
    
    def train_actor_stage(
        self,
        n_epochs: int,
        generate_episode_fn: Optional[Callable] = None,
        mode_balanced: bool = False,
    ) -> Dict[str, Any]:
        """
        Stage 2: Actor/tube training with frozen encoder.
        
        Args:
            n_epochs: Number of training epochs
            generate_episode_fn: Function that generates episodes
            mode_balanced: Whether to use mode-balanced sampling
        
        Returns:
            Dict with training metrics
        """
        print(f"\nStage 2: Actor/Tube Training ({n_epochs} epochs)...")
        
        if generate_episode_fn is None:
            generate_episode_fn = self._default_generate_episode
        
        # Freeze encoder
        if hasattr(self.actor, 'encoder_mlp'):
            self.actor.encoder_mlp.requires_grad_(False)
        elif hasattr(self.actor, 'encoder'):
            self.actor.encoder.requires_grad_(False)
        
        # Create optimizer for actor networks
        actor_params = []
        if hasattr(self.actor, 'mu_net'):
            actor_params.extend(self.actor.mu_net.parameters())
        if hasattr(self.actor, 'sigma_net'):
            actor_params.extend(self.actor.sigma_net.parameters())
        if hasattr(self.actor, 'obs_proj'):
            actor_params.extend(self.actor.obs_proj.parameters())
        if hasattr(self.actor, 'z_encoder'):
            actor_params.extend(self.actor.z_encoder.parameters())
        
        actor_optimizer = optim.Adam(actor_params, lr=1e-3)
        
        # Buffer
        if mode_balanced and hasattr(self.env, 'config') and hasattr(self.env.config, 'K'):
            buffer = StratifiedEpisodeBuffer(max_size_per_mode=100, n_modes=self.env.config.K)
            # Pre-fill buffer
            print("  Pre-filling mode buffers...")
            for _ in range(500):
                episode = generate_episode_fn()
                k = episode.get('k', 0)
                if isinstance(k, np.ndarray):
                    k = int(k[-1])
                buffer.add(episode, mode=k)
            print(f"  Buffer sizes: {list(buffer.mode_counts().values())}")
        else:
            buffer = EpisodeBuffer(max_size=1000)
            for _ in range(100):
                buffer.add(generate_episode_fn())
        
        # Training loop
        for epoch in range(n_epochs):
            # Sample episode
            if mode_balanced and isinstance(buffer, StratifiedEpisodeBuffer):
                episodes = buffer.sample_balanced(1)
                if len(episodes) == 0:
                    episode = generate_episode_fn()
                else:
                    episode = episodes[0]
                mode = episode.get('k', 0)
                if isinstance(mode, np.ndarray):
                    mode = int(mode[-1])
            else:
                episodes = buffer.sample(1)
                if len(episodes) == 0:
                    episode = generate_episode_fn()
                else:
                    episode = episodes[0]
                mode = None
            
            s0 = torch.tensor(episode['obs'][0], dtype=torch.float32, device=self.device)
            traj = torch.tensor(episode['x'][1:], dtype=torch.float32, device=self.device)
            
            # Encode trajectory
            traj_delta = traj - s0[:self.actor.pred_dim]
            with torch.no_grad():
                traj_flat = traj_delta.reshape(-1)
                z = self.actor.encode(traj_flat)
            
            # Train actor
            metrics = self.actor.train_step(s0, traj, z, actor_optimizer)
            self.training_history['actor'].append(metrics)
            
            # Add new episode to buffer periodically
            if epoch % 10 == 0:
                new_ep = generate_episode_fn()
                if isinstance(buffer, StratifiedEpisodeBuffer):
                    k = new_ep.get('k', 0)
                    if isinstance(k, np.ndarray):
                        k = int(k[-1])
                    buffer.add(new_ep, mode=k)
                else:
                    buffer.add(new_ep)
            
            if self.config.on_epoch:
                self.config.on_epoch(epoch, metrics, mode)
        
        return {
            'final_metrics': self.training_history['actor'][-1] if self.training_history['actor'] else {},
            'avg_metrics': self._aggregate_metrics(self.training_history['actor']),
        }
    
    def _aggregate_metrics(self, metrics_list: List[Dict]) -> Dict:
        """Aggregate metrics over training history."""
        if not metrics_list:
            return {}
        
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m.get(key, 0) for m in metrics_list]
            aggregated[f'{key}_mean'] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values))
        
        return aggregated
    
    def evaluate(
        self,
        n_episodes: int,
        select_z_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate actor on environment.
        
        Args:
            n_episodes: Number of test episodes
            select_z_fn: Optional function to select z (default: actor.select_z_geometric)
        
        Returns:
            Dict with evaluation results
        """
        print(f"\nEvaluation ({n_episodes} episodes)...")
        
        if select_z_fn is None:
            select_z_fn = lambda s0: self.actor.select_z_geometric(s0, trajectory_delta=None)
        
        all_z_stars = []
        all_rewards = []
        all_modes = []
        all_metrics = []
        
        for episode in range(n_episodes):
            obs = self.env.reset()
            s0 = torch.tensor(obs, dtype=torch.float32, device=self.device)
            
            # Select z
            z_star = select_z_fn(s0)
            all_z_stars.append(z_star.clone())
            
            # Get tube
            mu, sigma = self.actor.get_tube(s0, z_star)
            mu_np = mu.detach().cpu().numpy()[0]
            
            # Execute
            curr_obs = obs
            total_reward = 0
            
            for t in range(self.actor.T):
                action = mu_np[t] - curr_obs[:self.actor.pred_dim]
                curr_obs, reward, done, info = self.env.step(action)
                total_reward += reward
            
            all_rewards.append(total_reward)
            all_modes.append(info.get('k', 0))
            
            # Compute contract metrics for this episode (if available)
            if hasattr(self.actor, 'contract_terms'):
                traj = torch.tensor([info.get('x', curr_obs)], dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    terms = self.actor.contract_terms(s0, traj, mu.squeeze(0), sigma.squeeze(0))
                    all_metrics.append({k: v.item() if isinstance(v, torch.Tensor) else v for k, v in terms.items()})
            else:
                # Fallback: compute basic metrics from tube
                with torch.no_grad():
                    all_metrics.append({
                        'leak': 0.0,  # Would need actual trajectory
                        'vol': sigma.mean().item(),
                        'start_err': (mu[0, 0] - s0[0]).pow(2).mean().item(),
                    })
        
        # Aggregate results
        z_array = np.array([z.detach().cpu().numpy() for z in all_z_stars])
        
        results = {
            'rewards': {
                'mean': float(np.mean(all_rewards)),
                'std': float(np.std(all_rewards)),
                'min': float(np.min(all_rewards)),
                'max': float(np.max(all_rewards)),
            },
            'latent_metrics': compute_latent_metrics(z_array, all_modes if all_modes else None),
            'contract_metrics': compute_contract_metrics(
                [m.get('leak', 0) for m in all_metrics],
                [m.get('vol', 0) for m in all_metrics],
                [m.get('start_err', 0) for m in all_metrics],
                [m.get('dir_loss', 0) for m in all_metrics],
                [m.get('end_err', 0) for m in all_metrics],
            ),
            'mode_distribution': {int(mode): int(sum(1 for m in all_modes if m == mode)) for mode in set(all_modes)},
            'z_samples': z_array,
            'modes': all_modes,
        }
        
        return results
    
    def save_summary(self, results: Dict[str, Any]):
        """Save summary JSON."""
        summary = {
            'config': {
                'actor_cls': self.config.actor_cls.__name__,
                'actor_kwargs': self.config.actor_kwargs,
                'env_cls': self.config.env_cls.__name__,
                'env_kwargs': self.config.env_kwargs,
                'train_epochs': self.config.train_epochs,
                'warmup_epochs': self.config.warmup_epochs,
            },
            'training': self.training_history,
            'results': results,
        }
        
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Saved summary to {summary_path}")
    
    def save_model(self):
        """Save actor model."""
        model_path = self.output_dir / "model.pt"
        torch.save(self.actor.state_dict(), model_path)
        print(f"Saved model to {model_path}")
    
    def _default_generate_episode(self) -> Dict:
        """Default episode generation (environment-specific)."""
        # Try CMG-specific first
        try:
            from v2.environments.cmg.episode import generate_episode as cmg_generate
            return cmg_generate(self.env, policy_mode="goal_seeking")
        except (ImportError, AttributeError):
            # Fallback: generic episode generation
            obs = self.env.reset()
            episode = {'obs': [obs], 'x': [self.env.x.copy() if hasattr(self.env, 'x') else obs]}
            for t in range(self.actor.T):
                action = np.random.randn(self.actor.pred_dim).astype(np.float32)
                obs, reward, done, info = self.env.step(action)
                episode['obs'].append(obs)
                if hasattr(self.env, 'x'):
                    episode['x'].append(self.env.x.copy())
                else:
                    episode['x'].append(obs)
            return episode
    
    def run(self) -> Dict[str, Any]:
        """
        Run full training and evaluation pipeline.
        
        Returns:
            Dict with all results
        """
        # Stage 1: Encoder warmup
        warmup_metrics = self.train_encoder_stage(
            self.config.warmup_epochs,
            self.config.batch_size,
        )
        
        # Stage 2: Actor training
        # Check if env supports mode-balanced sampling
        mode_balanced = (hasattr(self.env, 'config') and 
                        hasattr(self.env.config, 'K') and 
                        self.env.config.K > 1)
        
        actor_metrics = self.train_actor_stage(
            self.config.train_epochs - self.config.warmup_epochs,
            mode_balanced=mode_balanced,
        )
        
        # Evaluation
        eval_results = self.evaluate(self.config.test_episodes)
        
        # Save
        self.save_model()
        self.save_summary({
            'warmup_metrics': warmup_metrics,
            'actor_metrics': actor_metrics,
            'eval_results': eval_results,
        })
        
        return {
            'warmup': warmup_metrics,
            'actor': actor_metrics,
            'eval': eval_results,
        }
