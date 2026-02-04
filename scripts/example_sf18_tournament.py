#!/usr/bin/env python3
"""
Example script showing how to run SF18 vs SF25 tournaments.

This script demonstrates the usage of the new run_sf18_tournament.py script
with various configurations.
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and print the result."""
    print(f"\n{'='*60}")
    print(f"EXAMPLE: {description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    print()
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✓ Command completed successfully")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print("✗ Command failed")
            if result.stderr:
                print("Error:")
                print(result.stderr)
    except subprocess.TimeoutExpired:
        print("✗ Command timed out (this is expected for long-running tournaments)")
    except Exception as e:
        print(f"✗ Error running command: {e}")

def main():
    """Run example commands."""
    
    # Set up environment
    env_cmd = "source hex_ai_env/bin/activate"
    
    print("SF18 vs SF25 Tournament Examples")
    print("=" * 60)
    print("This script demonstrates how to use the new run_sf18_tournament.py script.")
    print("Note: These are example commands - you'll need to adjust the model paths")
    print("and make sure the SF18 server is running.")
    print()
    
    # Example 1: Basic tournament with model registry
    cmd1 = f"{env_cmd} && python scripts/run_sf18_tournament.py --models=best --strategies=mcts --mcts-sims=30 --num-openings=10 --sf18-difficulty=9"
    run_command(cmd1, "Basic tournament using model registry")
    
    # Example 2: Tournament with direct model files
    cmd2 = f"{env_cmd} && python scripts/run_sf18_tournament.py --model-files=epoch11_mini15.pt.gz,epoch16_mini23.pt.gz --model-dirs=checkpoints/dir1,checkpoints/dir2 --strategies=mcts,mcts --mcts-sims=30,30 --num-openings=20 --sf18-difficulty=8"
    run_command(cmd2, "Tournament with multiple models using direct file paths")
    
    # Example 3: Tournament with custom opening file
    cmd3 = f"{env_cmd} && python scripts/run_sf18_tournament.py --models=best --strategies=mcts --mcts-sims=50 --opening-file=data/deterministic_openings.txt --sf18-difficulty=9"
    run_command(cmd3, "Tournament using custom opening file")
    
    # Example 4: Tournament with different SF18 server
    cmd4 = f"{env_cmd} && python scripts/run_sf18_tournament.py --models=best --strategies=mcts --mcts-sims=30 --num-openings=10 --sf18-difficulty=7 --sf18-server-url=http://localhost:8089"
    run_command(cmd4, "Tournament with custom SF18 server URL")
    
    # Example 5: Tournament with Gumbel AlphaZero
    cmd5 = f"{env_cmd} && python scripts/run_sf18_tournament.py --models=best --strategies=mcts --mcts-sims=100 --enable-gumbel=true --gumbel-sim-threshold=200 --num-openings=15 --sf18-difficulty=9"
    run_command(cmd5, "Tournament with Gumbel AlphaZero enabled")
    
    print(f"\n{'='*60}")
    print("SETUP INSTRUCTIONS")
    print(f"{'='*60}")
    print("To run these tournaments, you need to:")
    print("1. Start the SF18 server:")
    print("   python temp/SF18Interface/HttpGameServer.py \\")
    print("     --valueBuilderPath13=/path/to/value/model \\")
    print("     --policyBuilderPath13=/path/to/policy/model \\")
    print("     --twoHeadBuilderPath=/path/to/twohead/model \\")
    print("     --policyNetworkType=twoHeaded \\")
    print("     --portNum=8088")
    print()
    print("2. Adjust the model paths in the examples above to match your setup")
    print("3. Make sure you have the required model files and directories")
    print()
    print("For more information, see:")
    print("- scripts/run_sf18_tournament.py --help")
    print("- temp/SF18Interface/API_INTERFACE_README.md")

if __name__ == "__main__":
    main()
