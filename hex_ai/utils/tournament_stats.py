"""
Tournament statistics utilities for calculating and displaying win rates and game counts.
"""

from typing import Dict, List, Any, Tuple


def calculate_head_to_head_stats(
    model_a_name: str,
    model_b_name: str,
    model_a_wins: int,
    model_b_wins: int,
    total_games: int
) -> Dict[str, Any]:
    """
    Calculate head-to-head statistics between two models.
    
    Args:
        model_a_name: Name of model A
        model_b_name: Name of model B
        model_a_wins: Number of games won by model A
        model_b_wins: Number of games won by model B
        total_games: Total number of games played
    
    Returns:
        Dictionary with calculated statistics
    """
    model_a_win_rate = model_a_wins / total_games if total_games > 0 else 0.0
    model_b_win_rate = model_b_wins / total_games if total_games > 0 else 0.0
    
    return {
        'total_games': total_games,
        'model_a_wins': model_a_wins,
        'model_b_wins': model_b_wins,
        'model_a_win_rate': model_a_win_rate,
        'model_b_win_rate': model_b_win_rate,
        'model_a_name': model_a_name,
        'model_b_name': model_b_name
    }


def print_head_to_head_stats(stats: Dict[str, Any]) -> None:
    """
    Print formatted head-to-head statistics.
    
    Args:
        stats: Statistics dictionary from calculate_head_to_head_stats
    """
    print(f"\n{stats['model_a_name']} vs {stats['model_b_name']} Results:")
    print(f"  {stats['model_a_name']}: {stats['model_a_wins']}/{stats['total_games']} wins ({stats['model_a_win_rate']*100:.1f}%)")
    print(f"  {stats['model_b_name']}: {stats['model_b_wins']}/{stats['total_games']} wins ({stats['model_b_win_rate']*100:.1f}%)")


def extract_head_to_head_results_from_tournament_result(
    tournament_result: 'TournamentResult'
) -> List[Dict[str, Any]]:
    """
    Extract head-to-head results from a TournamentResult object.
    
    Args:
        tournament_result: TournamentResult object from hex_ai.inference.tournament
    
    Returns:
        List of head-to-head statistics dictionaries
    """
    head_to_head_stats = []
    
    for model_a in tournament_result.participants:
        for model_b in tournament_result.participants:
            if model_a < model_b:  # Only process each pair once
                model_a_wins = tournament_result.results[model_a][model_b]['wins']
                model_b_wins = tournament_result.results[model_b][model_a]['wins']
                total_games = tournament_result.results[model_a][model_b]['games']
                
                if total_games > 0:  # Only include pairs that actually played
                    stats = calculate_head_to_head_stats(
                        model_a, model_b, model_a_wins, model_b_wins, total_games
                    )
                    head_to_head_stats.append(stats)
    
    return head_to_head_stats


def print_all_head_to_head_results(tournament_result: 'TournamentResult') -> None:
    """
    Print all head-to-head results from a tournament.
    
    Args:
        tournament_result: TournamentResult object from hex_ai.inference.tournament
    """
    head_to_head_stats = extract_head_to_head_results_from_tournament_result(tournament_result)
    
    if not head_to_head_stats:
        print("\nNo head-to-head results to display.")
        return
    
    print("\n" + "="*60)
    print("HEAD-TO-HEAD RESULTS")
    print("="*60)
    
    for stats in head_to_head_stats:
        print_head_to_head_stats(stats)
    
    print("="*60)


def print_win_rates(tournament_result: 'TournamentResult', participant_temperatures: Dict[str, float] = None) -> None:
    """
    Print win rates for all participants.
    
    Args:
        tournament_result: TournamentResult object from hex_ai.inference.tournament
        participant_temperatures: Optional dict mapping participant names to their temperatures
    """
    win_rates = tournament_result.win_rates()
    print("\nWin Rates:")
    for name, rate in sorted(win_rates.items(), key=lambda x: -x[1]):
        games_played = sum(tournament_result.results[name][op]['games'] for op in tournament_result.results[name])
        wins = sum(tournament_result.results[name][op]['wins'] for op in tournament_result.results[name])
        
        # Add temperature info if available
        temp_info = ""
        if participant_temperatures and name in participant_temperatures:
            temp_info = f" (T={participant_temperatures[name]})"
        
        print(f"  {name}: {rate*100:.1f}% ({wins}/{games_played} games){temp_info}")


def print_elo_ratings(tournament_result: 'TournamentResult', participant_temperatures: Dict[str, float] = None) -> None:
    """
    Print ELO ratings for all participants.
    
    Args:
        tournament_result: TournamentResult object from hex_ai.inference.tournament
        participant_temperatures: Optional dict mapping participant names to their temperatures
    """
    elos = tournament_result.elo_ratings()
    print("\nElo Ratings (order-independent calculation):")
    for name, elo in sorted(elos.items(), key=lambda x: -x[1]):
        # Add temperature info if available
        temp_info = ""
        if participant_temperatures and name in participant_temperatures:
            temp_info = f" (T={participant_temperatures[name]})"
        
        print(f"  {name}: {elo:.1f}{temp_info}")


def print_detailed_head_to_head_matrix(tournament_result: 'TournamentResult') -> None:
    """
    Print detailed head-to-head matrix showing all pairwise results.
    
    Args:
        tournament_result: TournamentResult object from hex_ai.inference.tournament
    """
    print("\nHead-to-Head Results:")
    for name in sorted(tournament_result.participants):
        print(f"\n  {name} vs:")
        for op in sorted(tournament_result.participants):
            if op != name:
                wins = tournament_result.results[name][op]['wins']
                losses = tournament_result.results[name][op]['losses']
                games = tournament_result.results[name][op]['games']
                if games > 0:
                    win_rate = wins / games * 100
                    print(f"    {op}: {wins}-{losses} ({win_rate:.1f}%)")


def print_comprehensive_tournament_analysis(
    tournament_result: 'TournamentResult', 
    participant_temperatures: Dict[str, float] = None
) -> None:
    """
    Print comprehensive tournament analysis including all sections.
    
    Args:
        tournament_result: TournamentResult object from hex_ai.inference.tournament
        participant_temperatures: Optional dict mapping participant names to their temperatures
    """
    print("\n" + "="*60)
    print("TOURNAMENT ANALYSIS")
    print("="*60)
    
    print_win_rates(tournament_result, participant_temperatures)
    print_elo_ratings(tournament_result, participant_temperatures)
    print_detailed_head_to_head_matrix(tournament_result)
    
    print("\n" + "="*60)


def print_tournament_summary(tournament_result: 'TournamentResult') -> None:
    """
    Print a simple tournament summary with just win rates.
    
    Args:
        tournament_result: TournamentResult object from hex_ai.inference.tournament
    """
    print("\nTournament Results:")
    for name in tournament_result.participants:
        win_rate = tournament_result.win_rates()[name]
        total_wins = sum(tournament_result.results[name][op]['wins'] for op in tournament_result.results[name])
        total_games = sum(tournament_result.results[name][op]['games'] for op in tournament_result.results[name])
        print(f"{name}: {win_rate*100:.1f}% win rate ({total_wins} wins / {total_games} games)")
    print()
