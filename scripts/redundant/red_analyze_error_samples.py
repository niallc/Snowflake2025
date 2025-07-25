#!/usr/bin/env python3
"""
Analyze error samples from training to understand data corruption issues.
"""

import argparse
import pickle
import gzip
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import json

def load_error_sample(filepath: Path) -> Dict:
    """Load an error sample from file."""
    with gzip.open(filepath, 'rb') as f:
        return pickle.load(f)

def visualize_board(board_state: np.ndarray, title: str = "Board State"):
    """Visualize a board state."""
    if board_state is None:
        print("No board state available")
        return
    
    # Handle different board formats
    if board_state.shape[0] == 2:
        # 2-channel format (blue, red)
        blue_channel = board_state[0]
        red_channel = board_state[1]
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Blue channel
        im1 = ax1.imshow(blue_channel, cmap='Blues', vmin=0, vmax=1)
        ax1.set_title('Blue Channel')
        ax1.set_xlabel('Column')
        ax1.set_ylabel('Row')
        plt.colorbar(im1, ax=ax1)
        
        # Red channel
        im2 = ax2.imshow(red_channel, cmap='Reds', vmin=0, vmax=1)
        ax2.set_title('Red Channel')
        ax2.set_xlabel('Column')
        ax2.set_ylabel('Row')
        plt.colorbar(im2, ax=ax2)
        
        # Combined view
        combined = np.zeros((*blue_channel.shape, 3))
        combined[:, :, 0] = red_channel  # Red
        combined[:, :, 2] = blue_channel  # Blue
        ax3.imshow(combined)
        ax3.set_title('Combined View')
        ax3.set_xlabel('Column')
        ax3.set_ylabel('Row')
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    elif board_state.shape[0] == 3:
        # 3-channel format (blue, red, player-to-move)
        blue_channel = board_state[0]
        red_channel = board_state[1]
        player_channel = board_state[2]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Blue channel
        im1 = ax1.imshow(blue_channel, cmap='Blues', vmin=0, vmax=1)
        ax1.set_title('Blue Channel')
        plt.colorbar(im1, ax=ax1)
        
        # Red channel
        im2 = ax2.imshow(red_channel, cmap='Reds', vmin=0, vmax=1)
        ax2.set_title('Red Channel')
        plt.colorbar(im2, ax=ax2)
        
        # Player channel
        im3 = ax3.imshow(player_channel, cmap='viridis', vmin=0, vmax=1)
        ax3.set_title('Player-to-Move Channel')
        plt.colorbar(im3, ax=ax3)
        
        # Combined view
        combined = np.zeros((*blue_channel.shape, 3))
        combined[:, :, 0] = red_channel  # Red
        combined[:, :, 2] = blue_channel  # Blue
        ax4.imshow(combined)
        ax4.set_title('Combined View')
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    else:
        print(f"Unknown board format: shape {board_state.shape}")
        return None

def analyze_error_sample(error_data: Dict) -> Dict:
    """Analyze a single error sample."""
    analysis = {
        'error_count': error_data['error_count'],
        'error_msg': error_data['error_msg'],
        'file_path': error_data['file_path'],
        'sample_info': error_data['sample_info'],
        'board_shape': None,
        'blue_count': 0,
        'red_count': 0,
        'total_pieces': 0,
        'policy_target_shape': None,
        'value_target': None
    }
    
    board_state = error_data.get('board_state')
    if board_state is not None:
        analysis['board_shape'] = board_state.shape
        
        if board_state.shape[0] >= 2:
            blue_count = int(np.sum(board_state[0]))
            red_count = int(np.sum(board_state[1]))
            analysis['blue_count'] = blue_count
            analysis['red_count'] = red_count
            analysis['total_pieces'] = blue_count + red_count
    
    policy_target = error_data.get('policy_target')
    if policy_target is not None:
        analysis['policy_target_shape'] = policy_target.shape if hasattr(policy_target, 'shape') else None
    
    value_target = error_data.get('value_target')
    if value_target is not None:
        analysis['value_target'] = float(value_target)
    
    return analysis

def summarize_errors(error_dir: Path) -> Dict:
    """Summarize all error samples."""
    error_files = list(error_dir.glob("error_*.pkl.gz"))
    
    if not error_files:
        print("No error files found")
        return {}
    
    print(f"Found {len(error_files)} error samples")
    
    summaries = []
    error_types = {}
    
    for filepath in error_files:
        try:
            error_data = load_error_sample(filepath)
            analysis = analyze_error_sample(error_data)
            summaries.append(analysis)
            
            # Count error types
            error_msg = analysis['error_msg']
            error_types[error_msg] = error_types.get(error_msg, 0) + 1
            
        except Exception as e:
            print(f"Failed to load {filepath}: {e}")
    
    # Print summary
    print(f"\n=== Error Summary ===")
    print(f"Total errors: {len(summaries)}")
    
    print(f"\nError types:")
    for error_msg, count in error_types.items():
        print(f"  {error_msg}: {count}")
    
    if summaries:
        blue_counts = [s['blue_count'] for s in summaries]
        red_counts = [s['red_count'] for s in summaries]
        total_pieces = [s['total_pieces'] for s in summaries]
        
        print(f"\nBoard statistics:")
        print(f"  Blue pieces: min={min(blue_counts)}, max={max(blue_counts)}, avg={np.mean(blue_counts):.1f}")
        print(f"  Red pieces: min={min(red_counts)}, max={max(red_counts)}, avg={np.mean(red_counts):.1f}")
        print(f"  Total pieces: min={min(total_pieces)}, max={max(total_pieces)}, avg={np.mean(total_pieces):.1f}")
        
        # Find most common file
        file_counts = {}
        for s in summaries:
            file_path = s['file_path']
            file_counts[file_path] = file_counts.get(file_path, 0) + 1
        
        print(f"\nMost problematic files:")
        for file_path, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {file_path}: {count} errors")
    
    return {
        'summaries': summaries,
        'error_types': error_types,
        'total_errors': len(summaries)
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze error samples from training")
    parser.add_argument('--error-dir', type=str, default='checkpoints/errors', 
                       help='Directory containing error samples')
    parser.add_argument('--sample', type=int, help='Visualize specific error sample number')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to files')
    
    args = parser.parse_args()
    
    error_dir = Path(args.error_dir)
    
    if not error_dir.exists():
        print(f"Error directory {error_dir} does not exist")
        return
    
    # Analyze all errors
    summary = summarize_errors(error_dir)
    
    if args.sample is not None:
        # Visualize specific sample
        sample_file = error_dir / f"error_{args.sample:04d}_*.pkl.gz"
        sample_files = list(error_dir.glob(f"error_{args.sample:04d}_*.pkl.gz"))
        
        if sample_files:
            error_data = load_error_sample(sample_files[0])
            print(f"\n=== Error Sample {args.sample} ===")
            print(f"Error: {error_data['error_msg']}")
            print(f"File: {error_data['file_path']}")
            print(f"Sample: {error_data['sample_info']}")
            
            board_state = error_data.get('board_state')
            if board_state is not None:
                fig = visualize_board(board_state, f"Error Sample {args.sample}")
                if args.save_plots:
                    plot_path = error_dir / f"error_{args.sample:04d}_visualization.png"
                    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
                    print(f"Plot saved to {plot_path}")
                else:
                    plt.show()
        else:
            print(f"No error sample {args.sample} found")
    
    # Save summary to JSON
    if summary:
        summary_path = error_dir / "error_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSummary saved to {summary_path}")

if __name__ == "__main__":
    main() 