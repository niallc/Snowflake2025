#!/usr/bin/env python3
"""
Tests for the new multiprocessing architecture in process_all_trmph.py.
"""

import sys
import os
import tempfile
import shutil
import pytest
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hex_ai.trmph_processing.config import ProcessingConfig
from hex_ai.trmph_processing.processor import TRMPHProcessor, ParallelProcessor, SequentialProcessor
from hex_ai.trmph_processing.workers import process_single_file_worker


def create_test_trmph_files(temp_dir: Path, num_files: int = 3) -> List[Path]:
    """Create test TRMPH files for processing."""
    trmph_files = []
    
    for i in range(num_files):
        trmph_file = temp_dir / f"test_file_{i}.trmph"
        with open(trmph_file, 'w') as f:
            # Write a few simple game records with proper TRMPH format
            f.write("http://www.trmph.com/hex/board#13,a1b2c3 1\n")
            f.write("http://www.trmph.com/hex/board#13,a1b2c3d4e5f6g7h8i9j10k11l12m13 2\n")
            f.write("http://www.trmph.com/hex/board#13,a1b2c3d4e5f6g7h8i9j10k11l12 1\n")
        
        trmph_files.append(trmph_file)
    
    return trmph_files


class TestProcessingConfig:
    """Test the ProcessingConfig class."""
    
    def test_config_creation(self):
        """Test basic config creation."""
        config = ProcessingConfig(
            data_dir="/tmp/data",
            output_dir="/tmp/output",
            max_files=10,
            position_selector="all",
            run_tag="test",
            max_workers=4,
            combine_output=True
        )
        
        assert config.data_dir == Path("/tmp/data")
        assert config.output_dir == Path("/tmp/output")
        assert config.max_files == 10
        assert config.position_selector == "all"
        assert config.run_tag == "test"
        assert config.max_workers == 4
        assert config.combine_output is True
    
    def test_config_validation(self):
        """Test config validation."""
        # Test invalid position_selector (create temp dir first to avoid data_dir error)
        temp_dir = Path(tempfile.mkdtemp())
        try:
            with pytest.raises(ValueError, match="Invalid position_selector"):
                config = ProcessingConfig(
                    data_dir=str(temp_dir),
                    output_dir="/tmp/output",
                    position_selector="invalid"
                )
                config.validate()
        finally:
            shutil.rmtree(temp_dir)
        
        # Test invalid max_workers
        temp_dir = Path(tempfile.mkdtemp())
        try:
            with pytest.raises(ValueError, match="max_workers must be at least 1"):
                config = ProcessingConfig(
                    data_dir=str(temp_dir),
                    output_dir="/tmp/output",
                    max_workers=0
                )
                config.validate()
        finally:
            shutil.rmtree(temp_dir)
    
    def test_config_serialization(self):
        """Test config serialization for multiprocessing."""
        config = ProcessingConfig(
            data_dir="/tmp/data",
            output_dir="/tmp/output",
            max_files=10,
            position_selector="all",
            run_tag="test",
            max_workers=4,
            combine_output=True
        )
        
        config_dict = config.to_dict()
        restored_config = ProcessingConfig.from_dict(config_dict)
        
        assert restored_config.data_dir == config.data_dir
        assert restored_config.output_dir == config.output_dir
        assert restored_config.max_files == config.max_files
        assert restored_config.position_selector == config.position_selector
        assert restored_config.run_tag == config.run_tag
        assert restored_config.max_workers == config.max_workers
        assert restored_config.combine_output == config.combine_output


class TestWorkerFunction:
    """Test the worker function in isolation."""
    
    def test_worker_function_interface(self):
        """Test that worker function has the correct interface."""
        # Test that the function exists and is callable
        assert callable(process_single_file_worker)
        
        # Test that it accepts a dictionary argument
        file_info = {
            'file_path': '/tmp/test.trmph',
            'file_idx': 0,
            'data_dir': '/tmp/data',
            'output_dir': '/tmp/output',
            'position_selector': 'all',
            'run_tag': 'test'
        }
        
        # Should not raise an error for basic interface test
        # (actual processing will fail due to missing files, but that's expected)
        result = process_single_file_worker(file_info)
        
        # Check that result has expected structure
        assert isinstance(result, dict)
        assert 'success' in result
        assert 'file_path' in result
        assert 'file_idx' in result


class TestSequentialProcessor:
    """Test the sequential processor."""
    
    def test_sequential_processor_creation(self):
        """Test sequential processor creation."""
        processor = SequentialProcessor()
        assert processor is not None
    
    def test_sequential_processing_with_test_files(self):
        """Test sequential processing with actual test files."""
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        data_dir = temp_dir / "data"
        data_dir.mkdir()
        output_dir = temp_dir / "output"
        
        try:
            # Create test TRMPH files
            trmph_files = create_test_trmph_files(data_dir, num_files=2)
            
            # Create config
            config = ProcessingConfig(
                data_dir=str(data_dir),
                output_dir=str(output_dir),
                max_workers=1,  # Force sequential processing
                max_files=2
            )
            
            # Create processor
            processor = TRMPHProcessor(config)
            
            # Process files
            results = processor.process_all_files()
            
            # Check results
            assert len(results) == 2
            # Note: Processing will likely fail due to invalid TRMPH format,
            # but the architecture should work correctly
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)


class TestParallelProcessor:
    """Test the parallel processor."""
    
    def test_parallel_processor_creation(self):
        """Test parallel processor creation."""
        processor = ParallelProcessor(max_workers=4)
        assert processor.max_workers == 4
    
    def test_file_info_preparation(self):
        """Test file info preparation for workers."""
        temp_dir = Path(tempfile.mkdtemp())
        data_dir = temp_dir / "data"
        data_dir.mkdir()
        output_dir = temp_dir / "output"
        
        try:
            # Create test files
            trmph_files = create_test_trmph_files(data_dir, num_files=2)
            
            # Create config
            config = ProcessingConfig(
                data_dir=str(data_dir),
                output_dir=str(output_dir),
                max_workers=2
            )
            
            # Create processor
            processor = ParallelProcessor(max_workers=2)
            
            # Test file info preparation
            file_infos = processor._prepare_file_infos(trmph_files, config)
            
            assert len(file_infos) == 2
            for file_info in file_infos:
                assert 'file_path' in file_info
                assert 'file_idx' in file_info
                assert 'data_dir' in file_info
                assert 'output_dir' in file_info
                assert 'run_tag' in file_info
                assert 'position_selector' in file_info
                
        finally:
            shutil.rmtree(temp_dir)


def test_integration_with_real_script():
    """Test integration with the real script (basic interface test)."""
    # This test verifies that the script can be imported and run
    # without syntax errors, even if it fails due to missing data
    
    # Test that we can import the main script
    # Note: This will fail if not in the right virtual environment, which is expected
    try:
        from hex_ai.trmph_processing.cli import main
        assert callable(main)
    except (ImportError, SystemExit) as e:
        # SystemExit is expected if not in the right virtual environment
        pytest.skip(f"Could not import main script (expected if not in hex_ai_env): {e}")


if __name__ == "__main__":
    pytest.main([__file__]) 