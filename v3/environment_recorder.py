"""
Environment state recorder for tracking agent position, goals, and obstacles
at environment refresh rate (not actor move rate).

Records to JSONL format for visualization and analysis.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import json
import os
from datetime import datetime

try:
    import torch
except ImportError:
    torch = None


class EnvironmentRecorder(ABC):
    """
    Abstract interface for recording environment state.

    Records at environment refresh rate (when apply() is called),
    not at actor move rate.
    """

    @abstractmethod
    def record(
        self,
        agent_position: Any,  # torch.Tensor or list/array
        active_goals: List[Any],
        obstacles: List[Any],
        energy: Optional[float] = None,
        max_energy: Optional[float] = None,
        move_number: Optional[int] = None,
        reached_goals: Optional[List[Any]] = None,
    ):
        """
        Record environment state at current step.

        Args:
            agent_position: Current agent position (state_dim,) tensor or list
            active_goals: List of active goal positions
            obstacles: List of obstacles (position, radius) tuples
            energy: Current energy (if applicable)
            max_energy: Maximum energy (if applicable)
            move_number: Actor move number this step belongs to (for correlation)
            reached_goals: Optional list of reached goal positions (for visualization)
        """
        pass

    @abstractmethod
    def flush(self):
        """Flush any buffered data to disk."""
        pass

    @abstractmethod
    def finalize(self):
        """Finalize recording (close files, write metadata, etc.)."""
        pass


class JSONLEnvironmentRecorder(EnvironmentRecorder):
    """
    JSONL implementation of EnvironmentRecorder.

    Writes environment state to JSONL files at configurable intervals.
    """

    def __init__(self, output_dir: str, flush_interval: int = 10, state_dim: int = 2):
        """
        Initialize JSONL environment recorder.

        Args:
            output_dir: Directory to write output files
            flush_interval: Number of steps between flushes to disk
            state_dim: Dimension of state space
        """
        self.output_dir = output_dir
        self.flush_interval = flush_interval
        self.state_dim = state_dim

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # File path
        self.file_path = os.path.join(output_dir, f"environment_data_{timestamp}.jsonl")

        # Buffer for records
        self.data_buffer: List[Dict[str, Any]] = []

        # Track step number
        self.step_count = 0

        # Track previous obstacles to only record when changed
        self.previous_obstacles = None

        # Track current move number (set by training loop)
        self._current_move_number: Optional[int] = None

    def _convert_position(self, pos: Any) -> List[float]:
        """Convert position to list format."""
        if torch is not None and isinstance(pos, torch.Tensor):
            pos_list = pos.detach().cpu().tolist()
        elif isinstance(pos, (list, tuple)):
            pos_list = list(pos)
        else:
            pos_list = [float(pos)]

        # Pad to 3D if needed
        while len(pos_list) < 3:
            pos_list.append(0.0)

        return pos_list[:3]  # Ensure exactly 3D

    def _convert_goals(self, goals: List[Any]) -> List[List[float]]:
        """Convert goals to list format."""
        converted = []
        for goal in goals:
            converted.append(self._convert_position(goal))
        return converted

    def _convert_obstacles(self, obstacles: List[Any]) -> List[List[float]]:
        """Convert obstacles to list format [x, y, z, radius]."""
        converted = []
        for obs in obstacles:
            if isinstance(obs, tuple) and len(obs) == 2:
                # (position, radius) format
                pos, radius = obs
                pos_list = self._convert_position(pos)
                converted.append(pos_list + [float(radius)])
            elif isinstance(obs, (list, tuple)) and len(obs) >= 4:
                # Already in [x, y, z, radius] format
                converted.append([float(obs[i]) for i in range(4)])
            else:
                # Try to extract position and radius
                if isinstance(obs, (list, tuple)) and len(obs) >= 2:
                    pos_list = self._convert_position(obs[0])
                    radius = float(obs[1]) if len(obs) > 1 else 0.5
                    converted.append(pos_list + [radius])

        return converted

    def record(
        self,
        agent_position: Any,
        active_goals: List[Any],
        obstacles: List[Any],
        energy: Optional[float] = None,
        max_energy: Optional[float] = None,
        move_number: Optional[int] = None,
        reached_goals: Optional[List[Any]] = None,
    ):
        """
        Record environment state at current step.

        Args:
            agent_position: Current agent position (state_dim,) tensor or list
            active_goals: List of active goal positions
            obstacles: List of obstacles (position, radius) tuples
            energy: Current energy (if applicable)
            max_energy: Maximum energy (if applicable)
            move_number: Actor move number this step belongs to (for correlation)
            reached_goals: Optional list of reached goal positions (for visualization)
        """
        # Convert agent position
        agent_pos_list = self._convert_position(agent_position)

        # Convert goals
        goals_list = self._convert_goals(active_goals)

        # Convert reached goals if provided
        reached_goals_list = None
        if reached_goals is not None:
            reached_goals_list = self._convert_goals(reached_goals)

        # Convert obstacles (only include if changed from previous step)
        obstacles_list = self._convert_obstacles(obstacles)
        obstacles_changed = (
            self.previous_obstacles is None or obstacles_list != self.previous_obstacles
        )

        # Create record
        # Note: step is not included - line number = step number
        record = {
            "agent_position": agent_pos_list,
            "active_goals": goals_list,
        }

        # Include reached goals if provided
        if reached_goals_list is not None:
            record["reached_goals"] = reached_goals_list

        # Only include obstacles if changed
        if obstacles_changed:
            record["obstacles"] = obstacles_list
            self.previous_obstacles = obstacles_list

        # Include energy if provided
        if energy is not None:
            record["energy"] = float(energy)
        if max_energy is not None:
            record["max_energy"] = float(max_energy)

        # Include move number for correlation with training_stats
        # Use provided move_number or fall back to tracked current_move_number
        final_move_number = (
            move_number if move_number is not None else self._current_move_number
        )
        if final_move_number is not None:
            record["move_number"] = final_move_number

        # Add to buffer
        self.data_buffer.append(record)
        self.step_count += 1

        # Flush if interval reached
        if len(self.data_buffer) >= self.flush_interval:
            self.flush()

    def flush(self):
        """Flush buffered records to JSONL file."""
        if not self.data_buffer:
            return

        with open(self.file_path, "a") as f:
            for record in self.data_buffer:
                json.dump(record, f)
                f.write("\n")

        self.data_buffer.clear()

    def set_move_number(self, move_number: int):
        """Set the current move number for subsequent records."""
        self._current_move_number = move_number

    def finalize(self):
        """Finalize recording (flush remaining data)."""
        self.flush()
        print(f"Environment data saved to: {self.file_path}")
