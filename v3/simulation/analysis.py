"""
Analysis and plotting utilities for TAM v3 training sessions.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional


def load_goal_stats(goal_stats_file: str) -> List[Dict]:
    """Load goal statistics from JSONL file."""
    goals = []
    with open(goal_stats_file, 'r') as f:
        for line in f:
            if line.strip():
                goals.append(json.loads(line))
    return goals


def plot_training_progress(goal_stats_file: str, loss_history: Optional[List[float]] = None, 
                          output_path: Optional[str] = None):
    """
    Generate comprehensive plots showing training progress and model competence.
    
    Args:
        goal_stats_file: Path to goal_stats JSONL file
        loss_history: Optional list of loss values per move
        output_path: Optional path to save the plot (if None, displays interactively)
    """
    goals = load_goal_stats(goal_stats_file)
    
    if len(goals) == 0:
        print("No goal statistics found. Skipping plot generation.")
        return
    
    # Extract metrics over time
    goal_numbers = [g['goal_number'] for g in goals]
    moves_per_goal = [g['moves_taken'] for g in goals]
    agency_means = [g['agency']['mean'] for g in goals]
    agency_stds = [g['agency']['std'] for g in goals]
    segment_length_means = [g['segment_lengths']['mean'] for g in goals]
    segment_length_stds = [g['segment_lengths']['std'] for g in goals]
    
    # Calculate cumulative moves (total moves up to and including each goal)
    # This will be the x-axis for all plots
    cumulative_moves = np.cumsum(moves_per_goal)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Moves per Goal (Competence: lower is better)
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(cumulative_moves, moves_per_goal, 'o-', color='#2563EB', linewidth=2, markersize=6, alpha=0.8)
    # Add moving average
    if len(moves_per_goal) > 5:
        window = min(5, len(moves_per_goal) // 3)
        moving_avg = np.convolve(moves_per_goal, np.ones(window)/window, mode='valid')
        ax1.plot(cumulative_moves[window-1:], moving_avg, '--', color='#3B82F6', linewidth=2, alpha=0.6, label=f'{window}-goal MA')
        ax1.legend()
    ax1.set_xlabel('Cumulative Moves')
    ax1.set_ylabel('Moves Taken')
    ax1.set_title('Efficiency: Moves per Goal\n(Lower = Better)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(left=0)
    
    # 2. Agency (Sigma) Over Time
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(cumulative_moves, agency_means, 'o-', color='#F59E0B', linewidth=2, markersize=6, alpha=0.8, label='Mean')
    ax2.fill_between(cumulative_moves, 
                     np.array(agency_means) - np.array(agency_stds),
                     np.array(agency_means) + np.array(agency_stds),
                     alpha=0.2, color='#F59E0B', label='±1 std')
    ax2.set_xlabel('Cumulative Moves')
    ax2.set_ylabel('Agency (σ)')
    ax2.set_title('Confidence: Agency Over Time\n(Lower = More Confident)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(bottom=0)
    ax2.set_xlim(left=0)
    
    # 3. Segment Length Statistics
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(cumulative_moves, segment_length_means, 'o-', color='#10B981', linewidth=2, markersize=6, alpha=0.8, label='Mean')
    ax3.fill_between(cumulative_moves,
                     np.array(segment_length_means) - np.array(segment_length_stds),
                     np.array(segment_length_means) + np.array(segment_length_stds),
                     alpha=0.2, color='#10B981', label='±1 std')
    ax3.set_xlabel('Cumulative Moves')
    ax3.set_ylabel('Segment Length')
    ax3.set_title('Path Smoothness: Segment Lengths\n(Stable = Consistent)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(bottom=0)
    ax3.set_xlim(left=0)
    
    # 4. Cumulative Goals vs Cumulative Moves (INVERTED: goals on y-axis)
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(cumulative_moves, goal_numbers, 'o-', color='#8B5CF6', linewidth=2, markersize=6, alpha=0.8)
    ax4.set_xlabel('Cumulative Moves')
    ax4.set_ylabel('Goals Reached')
    ax4.set_title('Total Progress: Goals Reached\n(Inverted: Goals vs Moves)')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)
    ax4.set_xlim(left=0)
    
    # 5. Agency Variability (std of agency)
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(cumulative_moves, agency_stds, 'o-', color='#EF4444', linewidth=2, markersize=6, alpha=0.8)
    ax5.set_xlabel('Cumulative Moves')
    ax5.set_ylabel('Agency Std Dev')
    ax5.set_title('Consistency: Agency Variability\n(Lower = More Consistent)')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(bottom=0)
    ax5.set_xlim(left=0)
    
    # 6. Loss History (averaged per goal, x-axis = cumulative moves)
    ax6 = plt.subplot(2, 3, 6)
    if loss_history and len(loss_history) > 0:
        # Map loss history (per move) to goals
        # Each goal has moves_per_goal[i] moves, so we need to average loss per goal
        goal_losses = []
        move_idx = 0
        for i, moves in enumerate(moves_per_goal):
            if move_idx + moves <= len(loss_history):
                goal_loss = np.mean(loss_history[move_idx:move_idx + moves])
                goal_losses.append(goal_loss)
                move_idx += moves
            else:
                # Handle case where we run out of loss history
                if move_idx < len(loss_history):
                    goal_loss = np.mean(loss_history[move_idx:])
                    goal_losses.append(goal_loss)
                break
        
        if len(goal_losses) > 0:
            # Use cumulative moves for x-axis (match the number of goal_losses)
            plot_cumulative_moves = cumulative_moves[:len(goal_losses)]
            ax6.plot(plot_cumulative_moves, goal_losses, 'o-', color='#EC4899', linewidth=2, markersize=6, alpha=0.8)
            # Add moving average
            if len(goal_losses) > 5:
                window = min(5, len(goal_losses) // 3)
                loss_ma = np.convolve(goal_losses, np.ones(window)/window, mode='valid')
                ma_cumulative_moves = plot_cumulative_moves[window-1:]
                ax6.plot(ma_cumulative_moves, loss_ma, '--', color='#F472B6', linewidth=2, alpha=0.8, label=f'{window}-goal MA')
                ax6.legend()
            ax6.set_xlabel('Cumulative Moves')
            ax6.set_ylabel('Average Loss per Goal')
            ax6.set_title('Training Loss Over Time\n(Lower = Better)')
            ax6.set_yscale('log')  # Log scale for better visualization
            ax6.set_xlim(left=0)
        else:
            ax6.text(0.5, 0.5, 'Loss history\nnot available', 
                    ha='center', va='center', transform=ax6.transAxes,
                    fontsize=14, color='gray')
            ax6.set_title('Training Loss Over Time')
    else:
        ax6.text(0.5, 0.5, 'Loss history\nnot available', 
                ha='center', va='center', transform=ax6.transAxes,
                fontsize=14, color='gray')
        ax6.set_title('Training Loss Over Time')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Training progress plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_training_summary(goal_stats_file: str, output_path: Optional[str] = None) -> Dict:
    """
    Generate a summary statistics dictionary from goal stats.
    
    Args:
        goal_stats_file: Path to goal_stats JSONL file
        output_path: Optional path to save summary JSON
        
    Returns:
        Dictionary with summary statistics
    """
    goals = load_goal_stats(goal_stats_file)
    
    if len(goals) == 0:
        return {}
    
    moves_per_goal = [g['moves_taken'] for g in goals]
    agency_means = [g['agency']['mean'] for g in goals]
    segment_length_means = [g['segment_lengths']['mean'] for g in goals]
    
    # Split into early/mid/late phases
    n = len(goals)
    early_end = n // 3
    mid_end = 2 * n // 3
    
    early_moves = moves_per_goal[:early_end] if early_end > 0 else []
    mid_moves = moves_per_goal[early_end:mid_end] if mid_end > early_end else []
    late_moves = moves_per_goal[mid_end:] if mid_end < n else []
    
    summary = {
        'total_goals': len(goals),
        'total_moves': sum(moves_per_goal),
        'moves_per_goal': {
            'overall': {
                'mean': float(np.mean(moves_per_goal)),
                'std': float(np.std(moves_per_goal)),
                'min': int(np.min(moves_per_goal)),
                'max': int(np.max(moves_per_goal)),
            },
            'early_phase': {
                'mean': float(np.mean(early_moves)) if early_moves else None,
                'std': float(np.std(early_moves)) if early_moves else None,
            },
            'mid_phase': {
                'mean': float(np.mean(mid_moves)) if mid_moves else None,
                'std': float(np.std(mid_moves)) if mid_moves else None,
            },
            'late_phase': {
                'mean': float(np.mean(late_moves)) if late_moves else None,
                'std': float(np.std(late_moves)) if late_moves else None,
            },
        },
        'agency': {
            'overall_mean': float(np.mean(agency_means)),
            'overall_std': float(np.std(agency_means)),
            'early_mean': float(np.mean(agency_means[:early_end])) if early_end > 0 else None,
            'late_mean': float(np.mean(agency_means[mid_end:])) if mid_end < n else None,
        },
        'segment_lengths': {
            'overall_mean': float(np.mean(segment_length_means)),
            'overall_std': float(np.std(segment_length_means)),
        },
        'improvement': {
            'moves_reduction': float(np.mean(early_moves) - np.mean(late_moves)) if early_moves and late_moves else None,
            'agency_change': float(np.mean(agency_means[:early_end]) - np.mean(agency_means[mid_end:])) if early_end > 0 and mid_end < n else None,
        }
    }
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Training summary saved to: {output_path}")
    
    return summary
