#!/usr/bin/env python3
"""
Utility functions for validating Gumbel algorithm configurations.
"""

import math
from typing import Any, Dict, List, Tuple
from collections import defaultdict

from hex_ai.utils.gumbel_utils import calculate_power_law_candidates
from hex_ai.config import (
    DEFAULT_GUMBEL_CANDIDATE_POWER_SCALE,
    DEFAULT_GUMBEL_CANDIDATE_POWER_RATE,
    DEFAULT_GUMBEL_CANDIDATE_POWER_OFFSET,
    DEFAULT_GUMBEL_CANDIDATE_MIN,
    DEFAULT_GUMBEL_CANDIDATE_MAX
)

def simulate_sequential_halving(
    total_sims: int,
    num_candidates: int,
    verbose: bool = False
) -> Tuple[bool, int, List[Dict]]:
    """
    Simulate the sequential halving algorithm to detect potential issues.
    
    Args:
        total_sims: Total number of simulations available
        num_candidates: Number of candidates to start with
        verbose: Whether to print detailed simulation steps
        
    Returns:
        Tuple of (success, discarded_sims, round_details)
        - success: True if algorithm completes without assertion failure
        - discarded_sims: Number of simulations that would be discarded due to early termination
        - round_details: List of dictionaries with round information
    """
    
    def per_arm_allocation(total_left, rounds_left, num_arms):
        """Simulate the per_arm_allocation function."""
        if num_arms == 0:
            return 0
        
        theoretical_per_arm = total_left // max(1, rounds_left * num_arms)
        max_per_arm = total_left // num_arms
        
        if max_per_arm == 0:
            return 0
        
        return max(1, min(theoretical_per_arm, max_per_arm))
    
    def schedule_round(arms_list, sims_left, rounds_left):
        """Simulate the schedule_round function."""
        per_arm = per_arm_allocation(sims_left, rounds_left, len(arms_list))
        actions = [a for a in arms_list for _ in range(per_arm)]
        return actions
    
    # Initialize
    cand = list(range(num_candidates))
    R = max(1, math.ceil(math.log2(num_candidates)))
    sims_used = 0
    discarded_sims = 0
    round_details = []
    
    if verbose:
        print(f"Simulating sequential halving: {num_candidates} candidates, {total_sims} sims, {R} rounds")
    
    for r in range(R):
        if not cand or sims_used >= total_sims:
            break
            
        rounds_left = R - r
        arms = len(cand)
        sims_left = total_sims - sims_used
        
        # Calculate allocation
        per_arm = per_arm_allocation(sims_left, rounds_left, arms)
        actions_this_round = schedule_round(cand, sims_left, rounds_left)
        total_actions = len(actions_this_round)
        
        round_info = {
            'round': r + 1,
            'candidates': arms,
            'sims_left': sims_left,
            'rounds_left': rounds_left,
            'per_arm': per_arm,
            'total_actions': total_actions,
            'would_exceed_budget': total_actions > sims_left
        }
        round_details.append(round_info)
        
        if verbose:
            print(f"  Round {r+1}: {arms} candidates, {sims_left} sims left, {rounds_left} rounds left")
            print(f"    per_arm={per_arm}, total_actions={total_actions}")
            if total_actions > sims_left:
                print(f"    ❌ WOULD FAIL: {total_actions} > {sims_left}")
            else:
                print(f"    ✅ OK")
        
        # Check for assertion failure
        if total_actions > sims_left:
            # Calculate how many sims would be discarded
            discarded_sims = total_sims - sims_used
            return False, discarded_sims, round_details
        
        # Update sims_used
        sims_used += total_actions
        
        # Halve candidates
        if arms <= 1 or sims_used >= total_sims:
            break
            
        keep = max(1, (arms + 1) // 2)
        cand = cand[:keep]
        
        if verbose:
            print(f"    Keeping top {keep} candidates: {cand}")
    
    # Calculate any remaining discarded sims
    discarded_sims = total_sims - sims_used
    
    return True, discarded_sims, round_details

def validate_gumbel_configurations(
    sim_counts: List[int],
    configs: List[Dict[str, Any]],
    num_legal_actions: int = 164
) -> Dict[str, List[Dict]]:
    """
    Validate Gumbel configurations for potential issues.
    
    Args:
        sim_counts: List of simulation counts to test
        configs: List of configuration dictionaries with gumbel parameters
        num_legal_actions: Number of legal actions (default 164 for Hex)
        
    Returns:
        Dictionary mapping config names to lists of validation results
    """
    results = defaultdict(list)
    
    for config in configs:
        config_name = config.get('name', f"power_scale={config['candidate_power_scale']}, rate={config['candidate_power_rate']}, offset={config['candidate_power_offset']}")
        
        for sims in sim_counts:
            # Calculate number of candidates
            num_candidates = calculate_power_law_candidates(
                sims,
                config['candidate_power_scale'],
                config['candidate_power_rate'],
                config['candidate_power_offset'],
                config['candidate_min'],
                config['candidate_max'],
                num_legal_actions
            )
            
            # Simulate sequential halving
            success, discarded_sims, round_details = simulate_sequential_halving(sims, num_candidates)
            
            result = {
                'sims': sims,
                'num_candidates': num_candidates,
                'success': success,
                'discarded_sims': discarded_sims,
                'round_details': round_details
            }
            
            results[config_name].append(result)
    
    return dict(results)

def print_gumbel_warnings(results: Dict[str, List[Dict]], sim_counts: List[int]):
    """Print warnings for problematic Gumbel configurations."""
    
    print("\n" + "="*80)
    print("GUMBEL ALGORITHM VALIDATION WARNINGS")
    print("="*80)
    
    has_warnings = False
    
    for config_name, config_results in results.items():
        print(f"\nConfiguration: {config_name}")
        print("-" * 60)
        
        for result in config_results:
            if not result['success']:
                has_warnings = True
                print(f"⚠️  SIMS={result['sims']}: FAILURE - {result['discarded_sims']} simulations would be discarded")
                print(f"   Candidates: {result['num_candidates']}")
                
                # Show the failing round
                for round_info in result['round_details']:
                    if round_info['would_exceed_budget']:
                        print(f"   Fails in Round {round_info['round']}: {round_info['total_actions']} actions > {round_info['sims_left']} sims left")
                        break
            elif result['discarded_sims'] > 0:
                print(f"ℹ️  SIMS={result['sims']}: {result['discarded_sims']} simulations would be discarded (early termination)")
                print(f"   Candidates: {result['num_candidates']}")
    
    print("\n" + "="*80)

def check_tournament_gumbel_configs(sim_counts: List[int] = None):
    """Check common tournament simulation counts for Gumbel issues."""
    
    if sim_counts is None:
        sim_counts = [12, 25, 50, 100, 200, 500, 1000]
    
    # Import defaults from config
    from hex_ai.config import (
        DEFAULT_GUMBEL_CANDIDATE_MIN,
        DEFAULT_GUMBEL_CANDIDATE_MAX
    )
    from hex_ai.inference.move_selection import MoveSelectionConfig
    
    # Define the configurations we want to test
    configs = [
        {
            'name': 'Tournament (MoveSelectionConfig)',
            'candidate_power_scale': MoveSelectionConfig.gumbel_candidate_power_scale,
            'candidate_power_rate': MoveSelectionConfig.gumbel_candidate_power_rate,
            'candidate_power_offset': MoveSelectionConfig.gumbel_candidate_power_offset,
            'candidate_min': MoveSelectionConfig.gumbel_candidate_min,
            'candidate_max': MoveSelectionConfig.gumbel_candidate_max
        },
        {
            'name': 'Web App (config.py defaults)',
            'candidate_power_scale': DEFAULT_GUMBEL_CANDIDATE_POWER_SCALE,
            'candidate_power_rate': DEFAULT_GUMBEL_CANDIDATE_POWER_RATE,
            'candidate_power_offset': DEFAULT_GUMBEL_CANDIDATE_POWER_OFFSET,
            'candidate_min': DEFAULT_GUMBEL_CANDIDATE_MIN,
            'candidate_max': DEFAULT_GUMBEL_CANDIDATE_MAX
        }
    ]
    
    results = validate_gumbel_configurations(sim_counts, configs)
    print_gumbel_warnings(results, sim_counts)
    
    return results

def check_gumbel_configurations(strategy_configs):
    """Check for potential Gumbel algorithm issues from resolved strategy configs."""
    
    # Extract unique simulation counts and Gumbel configurations from strategy configs
    sim_counts = set()
    gumbel_configs = []
    config_names = set()
    
    # Check if any strategies use Gumbel
    has_gumbel_strategies = False
    
    # Import here to avoid circular import
    from hex_ai.inference.move_selection import MoveSelectionConfig

    for strategy_config in strategy_configs:
        if strategy_config.strategy_type != "mcts":
            continue

        cfg = strategy_config.config
        gumbel_enabled = bool(cfg["enable_gumbel_root_selection"])
        if not gumbel_enabled:
            continue

        has_gumbel_strategies = True
        sim_counts.add(int(cfg["mcts_sims"]))
        config_dict = {
            "candidate_power_scale": cfg["gumbel_candidate_power_scale"],
            "candidate_power_rate": cfg["gumbel_candidate_power_rate"],
            "candidate_power_offset": cfg["gumbel_candidate_power_offset"],
            "candidate_min": MoveSelectionConfig.gumbel_candidate_min,
            "candidate_max": MoveSelectionConfig.gumbel_candidate_max,
        }

        config_name = (
            f"power_scale={config_dict['candidate_power_scale']}, "
            f"rate={config_dict['candidate_power_rate']}, "
            f"offset={config_dict['candidate_power_offset']}"
        )
        if config_name not in config_names:
            config_dict["name"] = config_name
            gumbel_configs.append(config_dict)
            config_names.add(config_name)
    
    # Convert to sorted list of integers
    sim_counts = sorted([int(s) for s in sim_counts])
    
    # Run validation if we have Gumbel strategies and simulation counts
    if has_gumbel_strategies and sim_counts:
        print("\nChecking Gumbel algorithm configurations...")
        results = validate_gumbel_configurations(sim_counts, gumbel_configs)
        print_gumbel_warnings(results, sim_counts)


if __name__ == "__main__":
    # Test the validation
    check_tournament_gumbel_configs()
