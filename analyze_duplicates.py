#!/usr/bin/env python3
"""
Analyze duplicate games in trmph files using hash-based detection.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
import time

logger = logging.getLogger(__name__)

def hash_trmph_url(trmph_url: str) -> str:
    """Create a hash of the trmph URL."""
    return hashlib.md5(trmph_url.encode('utf-8')).hexdigest()

def analyze_all_files_for_duplicates(source_dir: str = "data/twoNetGames") -> Dict:
    """Analyze all trmph files for duplicates."""
    source_path = Path(source_dir)
    trmph_files = list(source_path.glob("*.trmph"))
    trmph_files.sort()
    
    if not trmph_files:
        raise FileNotFoundError(f"No .trmph files found in {source_dir}")
    
    logger.info(f"Found {len(trmph_files)} files to analyze")
    
    # Global tracking
    global_hash_counts = Counter()  # hash -> count
    total_games = 0
    files_processed = 0
    
    start_time = time.time()
    
    for file_idx, trmph_file in enumerate(trmph_files, 1):
        logger.info(f"Analyzing file {file_idx}/{len(trmph_files)}: {trmph_file.name}")
        
        file_games = 0
        with open(trmph_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(' ', 1)
                if len(parts) != 2:
                    continue
                
                trmph_url, winner_indicator = parts
                if not trmph_url.startswith("http://www.trmph.com/hex/board#"):
                    continue
                
                game_hash = hash_trmph_url(trmph_url)
                global_hash_counts[game_hash] += 1
                file_games += 1
        
        total_games += file_games
        files_processed += 1
        
        # Log progress every 10 files
        if file_idx % 10 == 0 or file_idx == len(trmph_files):
            unique_games = len(global_hash_counts)
            duplicate_games = total_games - unique_games
            duplicate_rate = duplicate_games / total_games if total_games > 0 else 0
            
            logger.info(f"Progress: {file_idx}/{len(trmph_files)} files, "
                       f"{total_games:,} total games, {unique_games:,} unique games, "
                       f"{duplicate_rate:.2%} duplicate rate")
    
    processing_time = time.time() - start_time
    
    # Calculate statistics
    unique_games = len(global_hash_counts)
    duplicate_games = total_games - unique_games
    duplicate_rate = duplicate_games / total_games if total_games > 0 else 0
    
    # Analyze distribution of game frequencies
    frequency_distribution = Counter(global_hash_counts.values())
    
    # Calculate percentage distribution
    total_unique = len(global_hash_counts)
    frequency_percentages = {}
    for freq, count in sorted(frequency_distribution.items()):
        percentage = (count / total_unique) * 100
        frequency_percentages[freq] = percentage
    
    # Find most common duplicates
    most_common_duplicates = []
    for hash_val, count in global_hash_counts.most_common(10):
        if count > 1:  # Only show actual duplicates
            most_common_duplicates.append((hash_val, count))
    
    results = {
        'total_games': total_games,
        'unique_games': unique_games,
        'duplicate_games': duplicate_games,
        'duplicate_rate': duplicate_rate,
        'processing_time': processing_time,
        'files_analyzed': files_processed,
        'frequency_distribution': dict(frequency_distribution),
        'frequency_percentages': frequency_percentages,
        'most_common_duplicates': most_common_duplicates
    }
    
    return results

def print_analysis_results(results: Dict):
    """Print formatted analysis results."""
    print("\n" + "="*80)
    print("DUPLICATE GAME ANALYSIS RESULTS")
    print("="*80)
    
    print(f"Total games analyzed: {results['total_games']:,}")
    print(f"Unique games: {results['unique_games']:,}")
    print(f"Duplicate games: {results['duplicate_games']:,}")
    print(f"Duplicate rate: {results['duplicate_rate']:.2%}")
    print(f"Processing time: {results['processing_time']:.1f} seconds")
    print(f"Files analyzed: {results['files_analyzed']}")
    
    print(f"\nFREQUENCY DISTRIBUTION:")
    print(f"{'Times Seen':<12} {'Count':<12} {'Percentage':<12}")
    print("-" * 40)
    
    for freq in sorted(results['frequency_distribution'].keys()):
        count = results['frequency_distribution'][freq]
        percentage = results['frequency_percentages'][freq]
        print(f"{freq:<12} {count:<12,} {percentage:<11.3f}%")
    
    # Show most common duplicates
    if results['most_common_duplicates']:
        print(f"\nMost common duplicates:")
        for hash_val, count in results['most_common_duplicates'][:5]:
            print(f"  Hash {hash_val[:8]}...: {count} occurrences")
    
    # Recommendation
    duplicate_rate = results['duplicate_rate']
    if duplicate_rate < 0.01:  # Less than 1%
        print(f"\n✅ Duplicate rate is very low ({duplicate_rate:.2%})")
        print("   No action needed - duplicates are negligible.")
    elif duplicate_rate < 0.05:  # Less than 5%
        print(f"\n⚠️  Moderate duplicate rate ({duplicate_rate:.2%})")
        print("   Consider filtering duplicates in future processing runs.")
    else:  # 5% or more
        print(f"\n🚨 High duplicate rate ({duplicate_rate:.2%})")
        print("   Strongly recommend filtering duplicates.")
    
    print("="*80)

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        logger.info("Starting duplicate analysis...")
        results = analyze_all_files_for_duplicates()
        print_analysis_results(results)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise
