"""
Tests for training orchestration improvements including multiple data directories,
data weights, resume training, and metadata tracking.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from hex_ai.training_orchestration import (
    discover_and_split_multiple_data,
    save_experiment_metadata,
    save_overall_results,
    find_latest_checkpoint_for_epoch
)


class TestDataWeightsValidation:
    """Test data weights validation logic."""
    
    def test_valid_weights(self):
        """Test that valid weights are accepted."""
        data_dirs = ["dir1", "dir2", "dir3"]
        weights = [0.5, 0.3, 0.2]
        
        # Should not raise an exception
        with patch('hex_ai.data_pipeline.discover_processed_files') as mock_discover:
            mock_discover.return_value = [Path("file1.pkl.gz")]
            with patch('hex_ai.data_pipeline.estimate_dataset_size') as mock_estimate:
                mock_estimate.return_value = 1000
                with patch('hex_ai.data_pipeline.create_train_val_split') as mock_split:
                    mock_split.return_value = ([Path("train1.pkl.gz")], [Path("val1.pkl.gz")])
                    
                    result = discover_and_split_multiple_data(
                        data_dirs=data_dirs,
                        data_weights=weights,
                        train_ratio=0.8,
                        random_seed=42
                    )
                    
                    assert len(result) == 4  # train_files, val_files, all_files, data_source_info
                    assert len(result[3]) == 3  # data_source_info should have 3 entries
    
    def test_invalid_weight_count(self):
        """Test that wrong number of weights raises ValueError."""
        data_dirs = ["dir1", "dir2"]
        weights = [0.5]  # Wrong count
        
        with patch('hex_ai.data_pipeline.discover_processed_files') as mock_discover:
            mock_discover.return_value = [Path("file1.pkl.gz")]
            
            with pytest.raises(ValueError, match="must match number of data directories"):
                discover_and_split_multiple_data(
                    data_dirs=data_dirs,
                    data_weights=weights,
                    train_ratio=0.8,
                    random_seed=42
                )
    
    def test_invalid_weight_sum(self):
        """Test that weights not summing to 1.0 raises ValueError."""
        data_dirs = ["dir1", "dir2"]
        weights = [0.5, 0.3]  # Sums to 0.8, not 1.0
        
        with patch('hex_ai.data_pipeline.discover_processed_files') as mock_discover:
            mock_discover.return_value = [Path("file1.pkl.gz")]
            
            with pytest.raises(ValueError, match="must sum to 1.0"):
                discover_and_split_multiple_data(
                    data_dirs=data_dirs,
                    data_weights=weights,
                    train_ratio=0.8,
                    random_seed=42
                )
    
    def test_equal_weights_default(self):
        """Test that equal weights are used when weights not specified."""
        data_dirs = ["dir1", "dir2", "dir3"]
        
        with patch('hex_ai.data_pipeline.discover_processed_files') as mock_discover:
            mock_discover.return_value = [Path("file1.pkl.gz")]
            with patch('hex_ai.data_pipeline.estimate_dataset_size') as mock_estimate:
                mock_estimate.return_value = 1000
                with patch('hex_ai.data_pipeline.create_train_val_split') as mock_split:
                    mock_split.return_value = ([Path("train1.pkl.gz")], [Path("val1.pkl.gz")])
                    
                    result = discover_and_split_multiple_data(
                        data_dirs=data_dirs,
                        data_weights=None,  # Should use equal weights
                        train_ratio=0.8,
                        random_seed=42
                    )
                    
                    data_source_info = result[3]
                    assert len(data_source_info) == 3
                    for source in data_source_info:
                        assert abs(source['weight'] - 1.0/3) < 1e-6


class TestMetadataSaving:
    """Test experiment metadata saving functionality."""
    
    def test_save_experiment_metadata(self):
        """Test that experiment metadata is saved correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            data_source_info = [
                {
                    'directory': 'data/processed/shuffled',
                    'files': [Path('file1.pkl.gz'), Path('file2.pkl.gz')],
                    'weight': 0.7,
                    'examples_estimated': 1000000
                },
                {
                    'directory': 'data/processed/jul_29_shuffled',
                    'files': [Path('file3.pkl.gz')],
                    'weight': 0.3,
                    'examples_estimated': 500000
                }
            ]
            
            hyperparameters = {
                'learning_rate': 0.001,
                'batch_size': 256,
                'policy_weight': 0.2
            }
            
            training_config = {
                'num_epochs': 2,
                'mini_epoch_batches': 500,
                'enable_augmentation': True
            }
            
            save_experiment_metadata(
                results_path=temp_path,
                experiment_name="test_experiment",
                data_source_info=data_source_info,
                hyperparameters=hyperparameters,
                training_config=training_config
            )
            
            metadata_file = temp_path / "test_experiment" / "experiment_metadata.json"
            assert metadata_file.exists()
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            assert metadata['experiment_name'] == "test_experiment"
            assert metadata['hyperparameters'] == hyperparameters
            assert metadata['training_config'] == training_config
            assert len(metadata['data_sources']) == 2
            assert metadata['total_examples'] == 1500000
            assert metadata['data_weights'] == [0.7, 0.3]
            assert 'created_at' in metadata
    
    def test_save_overall_results_with_data_sources(self):
        """Test that overall results include data source information."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            overall_results = {
                'total_training_time': 3600.0,
                'num_experiments': 3,
                'successful_experiments': 2,
                'experiments': []
            }
            
            data_source_info = [
                {
                    'directory': 'data/processed/shuffled',
                    'weight': 0.7,
                    'examples_estimated': 1000000
                }
            ]
            
            save_overall_results(temp_path, overall_results, data_source_info)
            
            results_file = temp_path / "overall_results.json"
            assert results_file.exists()
            
            with open(results_file, 'r') as f:
                saved_results = json.load(f)
            
            assert 'data_sources' in saved_results
            assert len(saved_results['data_sources']) == 1
            assert saved_results['data_sources'][0]['directory'] == 'data/processed/shuffled'
    
    def test_save_overall_results_without_data_sources(self):
        """Test that overall results work without data source information."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            overall_results = {
                'total_training_time': 3600.0,
                'num_experiments': 3,
                'successful_experiments': 2,
                'experiments': []
            }
            
            save_overall_results(temp_path, overall_results)
            
            results_file = temp_path / "overall_results.json"
            assert results_file.exists()
            
            with open(results_file, 'r') as f:
                saved_results = json.load(f)
            
            assert 'data_sources' not in saved_results
            assert saved_results['total_training_time'] == 3600.0


class TestCheckpointFinding:
    """Test checkpoint finding logic."""
    
    def test_find_latest_checkpoint_for_epoch(self):
        """Test finding the latest checkpoint for a specific epoch."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock checkpoint files
            checkpoint_files = [
                "epoch1_mini10.pt.gz",
                "epoch1_mini20.pt.gz", 
                "epoch2_mini15.pt.gz",
                "epoch2_mini25.pt.gz",
                "epoch3_mini5.pt.gz"
            ]
            
            for filename in checkpoint_files:
                (temp_path / filename).touch()
            
            # Test finding latest checkpoint for epoch 2
            latest_epoch2 = find_latest_checkpoint_for_epoch(temp_path, 2)
            assert latest_epoch2 is not None
            assert "epoch2_mini25" in str(latest_epoch2)  # Should find the latest mini-epoch
            
            # Test finding non-existent epoch
            latest_epoch4 = find_latest_checkpoint_for_epoch(temp_path, 4)
            assert latest_epoch4 is None
    
    def test_find_latest_checkpoint_empty_directory(self):
        """Test checkpoint finding in empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            latest = find_latest_checkpoint_for_epoch(temp_path, 1)
            assert latest is None
    
    def test_find_latest_checkpoint_no_matching_pattern(self):
        """Test checkpoint finding when no files match the pattern."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create files that don't match the pattern
            (temp_path / "model.pt").touch()
            (temp_path / "config.json").touch()
            
            latest = find_latest_checkpoint_for_epoch(temp_path, 1)
            assert latest is None


class TestExperimentNameGeneration:
    """Test experiment name generation functionality."""
    
    def test_experiment_name_with_varying_params(self):
        """Test experiment name generation when parameters are varying."""
        from scripts.hyperparam_sweep import make_experiment_name
        
        # Create a config with varying parameters
        config = {
            'learning_rate': 0.001,
            'batch_size': 256,
            'policy_weight': 0.2,
            'dropout_prob': 0.1
        }
        
        exp_name = make_experiment_name(config, 0, "test_sweep")
        
        assert "test_sweep" in exp_name
        assert "exp0" in exp_name
        # Should contain timestamp (format: YYYYMMDD_HHMMSS)
        assert "_2025" in exp_name or "_2024" in exp_name
    
    def test_experiment_name_no_varying_params(self):
        """Test experiment name generation when no parameters are varying."""
        from scripts.hyperparam_sweep import make_experiment_name
        
        config = {
            'learning_rate': 0.001,
            'batch_size': 256,
            'policy_weight': 0.2
        }
        
        exp_name = make_experiment_name(config, 0, "test_sweep")
        
        assert "test_sweep" in exp_name
        assert "exp0" in exp_name
        # When no parameters vary, should just have tag + hash + timestamp
        assert exp_name.startswith("test_sweep_exp0_")


class TestMemoryEfficiency:
    """Test that the code avoids loading large files into memory."""
    
    def test_estimate_dataset_size_memory_efficient(self):
        """Test that estimate_dataset_size only reads one sample file."""
        from hex_ai.data_pipeline import estimate_dataset_size
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create multiple mock files
            test_files = []
            for i in range(5):
                test_file = temp_path / f"test{i}.pkl.gz"
                test_file.write_bytes(b"x" * 1024 * 100)  # 100KB file
                test_files.append(test_file)
            
            # Mock gzip.open and pickle.load to return sample data
            with patch('gzip.open') as mock_gzip:
                mock_file = MagicMock()
                mock_gzip.return_value.__enter__.return_value = mock_file
                
                with patch('pickle.load') as mock_pickle:
                    # Return sample data with examples
                    mock_pickle.return_value = {
                        'examples': ['example'] * 500  # 500 examples per file
                    }
                    
                    result = estimate_dataset_size(test_files)
                    
                    # Should estimate 500 examples per file * 5 files = 2500 total
                    assert result == 2500
                    
                    # Should only have called gzip.open once (for the sample file)
                    assert mock_gzip.call_count == 1
                    assert mock_pickle.call_count == 1
    
    def test_estimate_dataset_size_fallback_to_file_size(self):
        """Test that estimate_dataset_size falls back to file size estimation."""
        from hex_ai.data_pipeline import estimate_dataset_size
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a mock file
            test_file = temp_path / "test.pkl.gz"
            test_file.write_bytes(b"x" * 1024 * 100)  # 100KB file
            
            # Mock gzip.open to raise an exception
            with patch('gzip.open') as mock_gzip:
                mock_gzip.side_effect = Exception("File read error")
                
                result = estimate_dataset_size([test_file])
                
                # Should fall back to file size estimation
                # 100KB file should estimate roughly 100 examples
                assert 50 <= result <= 200
                
                # Should have tried to read the file once
                assert mock_gzip.call_count == 1
    
    def test_estimate_dataset_size_with_processing_stats(self):
        """Test that estimate_dataset_size uses processing_stats when available."""
        from hex_ai.data_pipeline import estimate_dataset_size
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create multiple mock files
            test_files = []
            for i in range(3):
                test_file = temp_path / f"test{i}.pkl.gz"
                test_file.write_bytes(b"x" * 1024)  # 1KB file
                test_files.append(test_file)
            
            # Mock gzip.open and pickle.load to return processing stats
            with patch('gzip.open') as mock_gzip:
                mock_file = MagicMock()
                mock_gzip.return_value.__enter__.return_value = mock_file
                
                with patch('pickle.load') as mock_pickle:
                    # Return data with processing stats
                    mock_pickle.return_value = {
                        'processing_stats': {'examples_generated': 750}
                    }
                    
                    result = estimate_dataset_size(test_files)
                    
                    # Should estimate 750 examples per file * 3 files = 2250 total
                    assert result == 2250
                    
                    # Should only have called gzip.open once (for the sample file)
                    assert mock_gzip.call_count == 1
                    assert mock_pickle.call_count == 1 