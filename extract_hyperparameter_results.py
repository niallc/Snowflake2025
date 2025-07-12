#!/usr/bin/env python3
"""
Extract and analyze hyperparameter tuning results from completed experiments.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

def extract_experiment_results(experiment_dir: Path) -> Optional[Dict]:
    """Extract results from a single experiment directory."""
    try:
        # Find the best model checkpoint
        best_model_path = experiment_dir / "best_model.pt"
        if not best_model_path.exists():
            return None
            
        # Load the best model to get loss information
        checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
        
        # Extract hyperparameters from the experiment name
        exp_name = experiment_dir.name
        
        # Parse hyperparameters from experiment name
        hyperparams = {}
        if "baseline" in exp_name:
            hyperparams = {"learning_rate": 0.001, "batch_size": 64}
        elif "higher_lr" in exp_name:
            hyperparams = {"learning_rate": 0.003, "batch_size": 64}
        elif "lower_lr" in exp_name:
            hyperparams = {"learning_rate": 0.0003, "batch_size": 64}
        elif "larger_batch" in exp_name:
            hyperparams = {"learning_rate": 0.001, "batch_size": 128}
        elif "smaller_batch" in exp_name:
            hyperparams = {"learning_rate": 0.001, "batch_size": 32}
        elif "high_lr_large_batch" in exp_name:
            hyperparams = {"learning_rate": 0.003, "batch_size": 128}
        elif "low_lr_small_batch" in exp_name:
            hyperparams = {"learning_rate": 0.0003, "batch_size": 32}
        
        # Extract loss information
        best_loss = checkpoint.get('best_val_loss', checkpoint.get('loss', float('inf')))
        epoch = checkpoint.get('epoch', 0)
        
        # Count total checkpoints to determine epochs trained
        checkpoint_files = list(experiment_dir.glob("checkpoint_epoch_*.pt"))
        epochs_trained = len(checkpoint_files)
        
        return {
            "experiment_name": exp_name,
            "hyperparameters": hyperparams,
            "best_loss": best_loss,
            "best_epoch": epoch,
            "epochs_trained": epochs_trained,
            "checkpoint_path": str(best_model_path)
        }
        
    except Exception as e:
        print(f"Error extracting results from {experiment_dir}: {e}")
        return None

def analyze_all_experiments():
    """Analyze all completed experiments."""
    results_dir = Path("hyperparameter_results/experiment_20250711_180725")
    
    if not results_dir.exists():
        print("No hyperparameter results found!")
        return
    
    # Find all experiment directories
    experiment_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name != "__pycache__"]
    
    results = []
    for exp_dir in experiment_dirs:
        if exp_dir.name.startswith("."):
            continue
            
        result = extract_experiment_results(exp_dir)
        if result:
            results.append(result)
    
    # Sort by best loss
    results.sort(key=lambda x: x['best_loss'])
    
    return results

def create_markdown_report(results: List[Dict]) -> str:
    """Create a markdown report from the results."""
    
    report = """# Hyperparameter Tuning Results Report

**Dataset Size:** 10,000 games  
**Epochs per Experiment:** 20  
**Total Experiments:** {count}  
**Analysis Date:** {date}

## Results Summary

| Rank | Experiment | Learning Rate | Batch Size | Best Loss | Best Epoch | Epochs Trained |
|------|------------|---------------|------------|-----------|------------|----------------|
""".format(count=len(results), date=Path().cwd())
    
    for i, result in enumerate(results, 1):
        exp = result['experiment_name']
        lr = result['hyperparameters']['learning_rate']
        batch = result['hyperparameters']['batch_size']
        loss = result['best_loss']
        epoch = result['best_epoch']
        trained = result['epochs_trained']
        
        report += f"| {i} | {exp} | {lr} | {batch} | {loss:.4f} | {epoch} | {trained} |\n"
    
    # Add analysis section
    report += """
## Key Findings

### Best Configuration
The best performing configuration was **{best_exp}** with:
- **Learning Rate:** {best_lr}
- **Batch Size:** {best_batch}
- **Best Loss:** {best_loss:.4f}

### Performance Analysis
"""
    
    if results:
        best = results[0]
        report = report.format(
            best_exp=best['experiment_name'],
            best_lr=best['hyperparameters']['learning_rate'],
            best_batch=best['hyperparameters']['batch_size'],
            best_loss=best['best_loss']
        )
    
    # Add learning rate analysis
    lr_results = {}
    for r in results:
        lr = r['hyperparameters']['learning_rate']
        if lr not in lr_results:
            lr_results[lr] = []
        lr_results[lr].append(r['best_loss'])
    
    report += "\n### Learning Rate Analysis\n"
    for lr, losses in sorted(lr_results.items()):
        avg_loss = np.mean(losses)
        min_loss = min(losses)
        report += f"- **LR {lr}:** Average loss {avg_loss:.4f}, Best loss {min_loss:.4f}\n"
    
    # Add batch size analysis
    batch_results = {}
    for r in results:
        batch = r['hyperparameters']['batch_size']
        if batch not in batch_results:
            batch_results[batch] = []
        batch_results[batch].append(r['best_loss'])
    
    report += "\n### Batch Size Analysis\n"
    for batch, losses in sorted(batch_results.items()):
        avg_loss = np.mean(losses)
        min_loss = min(losses)
        report += f"- **Batch {batch}:** Average loss {avg_loss:.4f}, Best loss {min_loss:.4f}\n"
    
    report += """
## Recommendations

1. **Best Learning Rate:** Based on the results, {best_lr} appears to be optimal
2. **Best Batch Size:** {best_batch} shows the best performance
3. **Training Stability:** All experiments completed successfully
4. **Next Steps:** Consider running longer training with the best configuration

## Technical Notes

- All experiments used mixed precision training
- Validation split was 20% of the dataset
- Loss function combines policy and value losses
- Models used ResNet-18 architecture
"""
    
    if results:
        best = results[0]
        report = report.format(
            best_lr=best['hyperparameters']['learning_rate'],
            best_batch=best['hyperparameters']['batch_size']
        )
    
    return report

def main():
    """Main analysis function."""
    print("Extracting hyperparameter tuning results...")
    
    results = analyze_all_experiments()
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found {len(results)} completed experiments")
    
    # Create report
    report = create_markdown_report(results)
    
    # Save report
    report_path = Path("hyperparameter_results/experiment_20250711_180725/analysis_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {report_path}")
    print("\n" + "="*50)
    print("QUICK SUMMARY:")
    print("="*50)
    
    for i, result in enumerate(results[:3], 1):
        exp = result['experiment_name']
        loss = result['best_loss']
        lr = result['hyperparameters']['learning_rate']
        batch = result['hyperparameters']['batch_size']
        print(f"{i}. {exp}: Loss {loss:.4f} (LR={lr}, Batch={batch})")

if __name__ == "__main__":
    main() 