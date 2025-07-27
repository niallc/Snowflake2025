#!/usr/bin/env python3
"""
Test script to demonstrate the difference between order-dependent and 
order-independent ELO calculations.

This script creates a simple tournament scenario and shows how the order
of games affects the old ELO calculation but not the new one.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hex_ai.inference.tournament import TournamentResult

def create_test_tournament():
    """Create a simple test tournament with known results."""
    # Create a tournament with 3 players: A, B, C
    participants = ["A", "B", "C"]
    result = TournamentResult(participants)
    
    # Add some test results:
    # A beats B 3 times, loses 1 time
    # B beats C 2 times, loses 2 times  
    # A beats C 4 times, loses 0 times
    
    # A vs B: A wins 3, B wins 1
    for _ in range(3):
        result.record_game("A", "B")
    for _ in range(1):
        result.record_game("B", "A")
    
    # B vs C: B wins 2, C wins 2
    for _ in range(2):
        result.record_game("B", "C")
    for _ in range(2):
        result.record_game("C", "B")
    
    # A vs C: A wins 4, C wins 0
    for _ in range(4):
        result.record_game("A", "C")
    
    return result

def old_elo_calculation(result, base=1500):
    """
    Simulate the old order-dependent ELO calculation.
    This processes games in the order they were recorded.
    """
    ratings = {name: base for name in result.participants}
    k = 32
    
    # Process games in the order they were recorded
    for name in result.participants:
        for op in result.results[name]:
            games = result.results[name][op]['games']
            wins = result.results[name][op]['wins']
            losses = result.results[name][op]['losses']
            
            for _ in range(wins):
                expected = 1 / (1 + 10 ** ((ratings[op] - ratings[name]) / 400))
                ratings[name] += k * (1 - expected)
                ratings[op] += k * (0 - (1 - expected))
            
            for _ in range(losses):
                expected = 1 / (1 + 10 ** ((ratings[op] - ratings[name]) / 400))
                ratings[name] += k * (0 - expected)
                ratings[op] += k * (1 - (0 - expected))
    
    return ratings

def main():
    print("ELO Calculation Comparison Test")
    print("=" * 50)
    
    # Create test tournament
    result = create_test_tournament()
    
    print("Tournament Results:")
    result.print_summary()
    
    print("\nOld ELO Calculation (order-dependent):")
    old_elos = old_elo_calculation(result)
    for name, elo in sorted(old_elos.items(), key=lambda x: -x[1]):
        print(f"  {name}: {elo:.1f}")
    
    print("\nNew ELO Calculation (order-independent):")
    new_elos = result.elo_ratings()
    for name, elo in sorted(new_elos.items(), key=lambda x: -x[1]):
        print(f"  {name}: {elo:.1f}")
    
    print("\nDetailed Analysis:")
    result.print_detailed_analysis()
    
    print("\nKey Differences:")
    print("1. Old method: Processes games sequentially, order matters")
    print("2. New method: Uses maximum likelihood estimation, order-independent")
    print("3. New method: More mathematically sound for tournament settings")
    print("4. New method: Includes fallback for optimization failures")

if __name__ == "__main__":
    main() 