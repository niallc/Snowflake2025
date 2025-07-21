"""
Tests for the TrmphProcessor class and related functionality.

This module tests the large-scale TRMPH file processing pipeline.
"""

import pytest
import tempfile
import os
import gzip
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock

from scripts.process_all_trmph import TrmphProcessor


class TestTrmphProcessor:
    """Test the TrmphProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data"
        self.output_dir = Path(self.temp_dir) / "output"
        self.data_dir.mkdir()
        self.output_dir.mkdir()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_trmph_file(self, filename: str, content: str):
        """Create a test TRMPH file with given content."""
        file_path = self.data_dir / filename
        with open(file_path, 'w') as f:
            f.write(content)
        return file_path
    
    def test_initialization(self):
        """Test TrmphProcessor initialization."""
        processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
        
        assert processor.data_dir == self.data_dir
        assert processor.output_dir == self.output_dir
        assert processor.output_dir.exists()
        assert len(processor.trmph_files) == 0
    
    def test_find_trmph_files(self):
        """Test that TRMPH files are found correctly."""
        # Create test files
        self.create_test_trmph_file("test1.trmph", "content1")
        self.create_test_trmph_file("test2.trmph", "content2")
        self.create_test_trmph_file("test3.txt", "content3")  # Should be ignored
        
        processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
        
        assert len(processor.trmph_files) == 2
        assert all(f.suffix == ".trmph" for f in processor.trmph_files)
    
    def test_process_empty_file(self):
        """Test processing an empty TRMPH file."""
        self.create_test_trmph_file("empty.trmph", "")
        
        processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
        file_path = self.data_dir / "empty.trmph"
        
        stats = processor.process_single_file(file_path)
        
        # Empty files should be handled gracefully with a file error
        assert stats['file_error'] is not None
        assert "Empty file" in stats['file_error']
        assert stats['all_games'] == 0
        assert stats['valid_games'] == 0
        assert stats['skipped_games'] == 0
    
    def test_process_valid_game(self):
        """Test processing a valid TRMPH game."""
        content = "#13,a1b2c3 1\n"  # Simple game with blue winner
        self.create_test_trmph_file("valid.trmph", content)
        
        processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
        file_path = self.data_dir / "valid.trmph"
        
        stats = processor.process_single_file(file_path)
        
        assert stats['file_error'] is None
        assert stats['valid_games'] == 1
        assert stats['skipped_games'] == 0
        assert stats['examples_generated'] > 0
        
        # Check that output file was created
        output_files = list(self.output_dir.glob("*_processed.pkl.gz"))
        assert len(output_files) == 1
    
    def test_process_invalid_game_format(self):
        """Test processing a game with invalid format."""
        content = "#13,a1b2c3\n"  # Missing winner
        self.create_test_trmph_file("invalid_format.trmph", content)
        
        processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
        file_path = self.data_dir / "invalid_format.trmph"
        
        stats = processor.process_single_file(file_path)
        
        assert stats['file_error'] is None  # File-level error handling
        assert stats['valid_games'] == 0
        assert stats['skipped_games'] == 1
        assert stats['examples_generated'] == 0
    
    def test_process_game_with_duplicate_moves(self):
        """Test processing a game with duplicate moves."""
        content = "#13,a1b2a1c3 1\n"  # Duplicate move 'a1'
        self.create_test_trmph_file("duplicate.trmph", content)
        
        processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
        file_path = self.data_dir / "duplicate.trmph"
        
        stats = processor.process_single_file(file_path)
        
        # Should handle gracefully due to remove_repeated_moves
        assert stats['file_error'] is None
        assert stats['valid_games'] == 1
        assert stats['skipped_games'] == 0
    
    def test_process_mixed_valid_invalid_games(self):
        """Test processing a file with both valid and invalid games."""
        content = "#13,a1b2c3 1\n"  # Valid game
        content += "#13,a1b2 2\n"   # Valid game
        content += "invalid_line\n"  # Invalid line
        content += "#13,a1b2c3d4 1\n"  # Valid game
        self.create_test_trmph_file("mixed.trmph", content)
        
        processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
        file_path = self.data_dir / "mixed.trmph"
        
        stats = processor.process_single_file(file_path)
        
        assert stats['file_error'] is None
        assert stats['valid_games'] == 3
        assert stats['skipped_games'] == 1
        assert stats['all_games'] == 4
    
    def test_file_not_found(self):
        """Test handling of missing files."""
        processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
        non_existent_file = self.data_dir / "nonexistent.trmph"
        
        stats = processor.process_single_file(non_existent_file)
        
        assert stats['file_error'] is not None
        assert "File not found" in stats['file_error']
    
    def test_output_file_creation(self):
        """Test that output files are created correctly."""
        content = "#13,a1b2c3 1\n"
        self.create_test_trmph_file("test.trmph", content)
        
        processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
        file_path = self.data_dir / "test.trmph"
        
        processor.process_single_file(file_path)
        
        # Check output file
        output_files = list(self.output_dir.glob("*_processed.pkl.gz"))
        assert len(output_files) == 1
        
        # Load and verify output file
        with gzip.open(output_files[0], 'rb') as f:
            data = pickle.load(f)
        
        assert 'examples' in data
        assert 'source_file' in data
        assert 'processing_stats' in data
        assert 'processed_at' in data
        assert len(data['examples']) > 0
    
    def test_statistics_tracking(self):
        """Test that statistics are tracked correctly."""
        content = "#13,a1b2c3 1\n#13,a1b2 2\n"
        self.create_test_trmph_file("stats_test.trmph", content)
        
        processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
        
        # Process files
        stats = processor.process_all_files()
        
        assert stats['files_processed'] == 1
        assert stats['files_failed'] == 0
        assert stats['all_games'] == 2
        assert stats['valid_games'] == 2
        assert stats['skipped_games'] == 0
        assert stats['total_examples'] > 0
        assert 'elapsed_time' in stats
        assert 'files_per_second' in stats
    
    def test_max_files_parameter(self):
        """Test the max_files parameter."""
        # Create multiple files
        for i in range(5):
            self.create_test_trmph_file(f"test{i}.trmph", "#13,a1b2c3 1\n")
        
        processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
        
        # Process only 3 files
        stats = processor.process_all_files(max_files=3)
        
        assert stats['files_processed'] == 3
        assert len(processor.trmph_files) == 5  # All files found
    
    def test_combined_dataset_creation(self):
        """Test creating a combined dataset."""
        # Create multiple files
        for i in range(3):
            self.create_test_trmph_file(f"test{i}.trmph", "#13,a1b2c3 1\n")
        
        processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
        
        # Process all files
        processor.process_all_files()
        
        # Create combined dataset
        processor.create_combined_dataset()
        
        # Check combined file exists
        combined_file = self.output_dir / "combined_dataset.pkl.gz"
        assert combined_file.exists()
        
        # Load and verify
        with gzip.open(combined_file, 'rb') as f:
            data = pickle.load(f)
        
        assert 'examples' in data
        assert 'total_examples' in data
        assert 'source_files' in data
        assert 'created_at' in data
        assert data['source_files'] == 3
    
    def test_error_handling_during_processing(self):
        """Test error handling during processing."""
        content = "#13,a1b2c3 1\n"
        self.create_test_trmph_file("error_test.trmph", content)
        
        processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
        
        # Mock extract_training_examples_from_game to raise an exception
        with patch('scripts.process_all_trmph.extract_training_examples_from_game') as mock_extract:
            mock_extract.side_effect = Exception("Test error")
            
            file_path = self.data_dir / "error_test.trmph"
            stats = processor.process_single_file(file_path)
            
            assert stats['file_error'] is None  # File-level error handling
            assert stats['all_games'] == 1
            assert stats['valid_games'] == 0
            assert stats['skipped_games'] == 1  # Game was skipped due to exception
    
    def test_large_file_handling(self):
        """Test handling of large files (memory usage)."""
        # Create a large file with many games
        content = ""
        for i in range(1000):
            content += f"#13,a1b2c3 1\n"
        
        self.create_test_trmph_file("large.trmph", content)
        
        processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
        file_path = self.data_dir / "large.trmph"
        
        # Should process without memory issues
        stats = processor.process_single_file(file_path)
        
        assert stats['file_error'] is None
        assert stats['valid_games'] == 1000
        assert stats['examples_generated'] > 0
    
    def test_filename_sanitization_original(self):
        """Test handling of problematic filenames."""
        # Create file with special characters in name
        problematic_name = "test file with spaces and (parentheses).trmph"
        content = "#13,a1b2c3 1\n"
        self.create_test_trmph_file(problematic_name, content)
        
        processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
        file_path = self.data_dir / problematic_name
        
        # Should handle without issues
        stats = processor.process_single_file(file_path)
        
        assert stats['file_error'] is None
        assert stats['valid_games'] == 1
    
    def test_concurrent_processing_simulation(self):
        """Test that output files don't conflict during concurrent processing."""
        # Create multiple files
        for i in range(5):
            self.create_test_trmph_file(f"concurrent{i}.trmph", "#13,a1b2c3 1\n")
        
        processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
        
        # Process all files
        processor.process_all_files()
        
        # Check that all output files were created with unique names
        output_files = list(self.output_dir.glob("*_processed.pkl.gz"))
        assert len(output_files) == 5
        
        # Check that filenames are unique
        filenames = [f.name for f in output_files]
        assert len(filenames) == len(set(filenames))
    
    def test_memory_management_between_files(self):
        """Test that memory is cleared between processing files."""
        import psutil
        import os
        
        # Create multiple large files with valid TRMPH records
        for i in range(3):
            content = ""
            for j in range(100):  # 100 games per file
                # Use the same simple format that works in other tests
                content += f"#13,a1b2c3 1\n"
            self.create_test_trmph_file(f"large{i}.trmph", content)
        
        processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_usage = []
        
        # Process each file and track memory
        for i in range(3):
            file_path = self.data_dir / f"large{i}.trmph"
            stats = processor.process_single_file(file_path)
            
            # Check memory after each file
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage.append(current_memory)
            
            # Verify file was processed successfully
            assert stats['file_error'] is None
            assert stats['valid_games'] == 100
            assert stats['examples_generated'] > 0
        
        # Check that memory usage doesn't grow significantly
        max_memory = max(memory_usage)
        memory_growth = max_memory - initial_memory
        
        # Memory growth should be reasonable (less than 100MB for this test)
        assert memory_growth < 100, f"Memory grew by {memory_growth:.1f}MB, which is too much"
        
        # Log memory usage for debugging
        print(f"Memory usage: {memory_usage}")
        print(f"Memory growth: {memory_growth:.1f}MB")
    
    def test_output_directory_validation(self):
        """Test output directory validation including permissions and space."""
        # Test with valid directory
        processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
        assert processor.output_dir.exists()
        
        # Test with invalid directory (should raise error)
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a read-only directory
            import os
            os.chmod(temp_dir, 0o444)  # Read-only
            
            with pytest.raises(ValueError, match="not writable"):
                TrmphProcessor(data_dir=str(self.data_dir), output_dir=temp_dir)
    
    def test_filename_sanitization(self):
        """Test filename sanitization for safety."""
        processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
        
        # Test various problematic filenames
        test_cases = [
            ("normal_file", "normal_file"),
            ("file with spaces", "file_with_spaces"),
            ("file(with)parentheses", "file_with_parentheses"),  # Fixed: no trailing underscore
            ("file.with.dots", "file.with.dots"),
            ("file-with-dashes", "file-with-dashes"),
            ("file_with_underscores", "file_with_underscores"),
            ("file with special chars!@#$%", "file_with_special_chars"),  # Fixed: no trailing underscores
            ("", "unnamed"),
            ("." * 300, "unnamed"),  # Fixed: dots get removed, empty string becomes "unnamed"
            ("file with unicode éñç", "file_with_unicode_e_n_c"),  # Fixed: unicode gets normalized
        ]
        
        for input_name, expected in test_cases:
            sanitized = processor._sanitize_filename(input_name)
            assert sanitized == expected
    
    def test_filename_uniqueness(self):
        """Test that output filenames are unique."""
        # Create multiple files with same base name
        for i in range(3):
            content = f"#13,a1b2c3 1\n"
            self.create_test_trmph_file(f"same_name.trmph", content)
        
        processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
        
        # Process files - should create unique output names
        processor.process_all_files()
        
        # Check output files
        output_files = list(self.output_dir.glob("*_processed.pkl.gz"))
        assert len(output_files) == 3
        
        # Check that filenames are unique
        filenames = [f.name for f in output_files]
        assert len(filenames) == len(set(filenames))
        
        # Check naming pattern
        assert any("same_name_processed.pkl.gz" in f.name for f in output_files)
        assert any("same_name_processed_1.pkl.gz" in f.name for f in output_files)
        assert any("same_name_processed_2.pkl.gz" in f.name for f in output_files)
    
    def test_data_integrity_validation(self):
        """Test data integrity validation before saving."""
        processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
        
        # Test valid data
        import numpy as np
        valid_examples = [
            {'board': np.zeros((2, 2, 2)), 'policy': np.zeros(4), 'value': 1.0},
            {'board': np.ones((2, 2, 2)), 'policy': np.ones(4), 'value': 0.0},
        ]
        processor._validate_examples_data(valid_examples)  # Should not raise
        
        # Test invalid data types
        with pytest.raises(ValueError, match="must be a list"):
            processor._validate_examples_data("not a list")
        
        with pytest.raises(ValueError, match="cannot be empty"):
            processor._validate_examples_data([])
        
        # Test missing required keys
        invalid_examples = [{'board': np.zeros((2, 2, 2)), 'policy': np.zeros(4)}]  # Missing value
        with pytest.raises(ValueError, match="missing required keys"):
            processor._validate_examples_data(invalid_examples)
        
        # Test non-dictionary examples
        invalid_examples = [{'board': np.zeros((2, 2, 2)), 'policy': np.zeros(4), 'value': 1.0}, "not a dict"]
        with pytest.raises(ValueError, match="not a dictionary"):
            processor._validate_examples_data(invalid_examples)
        
        # Test wrong types
        invalid_examples = [{'board': "not array", 'policy': np.zeros(4), 'value': 1.0}]
        with pytest.raises(ValueError, match="board state must be numpy array"):
            processor._validate_examples_data(invalid_examples)
    
    def test_output_file_structure(self):
        """Test that output files have correct structure and metadata."""
        content = "#13,a1b2c3 1\n"
        self.create_test_trmph_file("structure_test.trmph", content)
        
        processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
        file_path = self.data_dir / "structure_test.trmph"
        
        processor.process_single_file(file_path)
        
        # Check output file
        output_files = list(self.output_dir.glob("*_processed.pkl.gz"))
        assert len(output_files) == 1
        
        # Load and verify structure
        with gzip.open(output_files[0], 'rb') as f:
            data = pickle.load(f)
        
        # Check required fields
        assert 'examples' in data
        assert 'source_file' in data
        assert 'processing_stats' in data
        assert 'processed_at' in data
        assert 'file_size_bytes' in data
        
        # Check data types
        assert isinstance(data['examples'], list)
        assert isinstance(data['source_file'], str)
        assert isinstance(data['processing_stats'], dict)
        assert isinstance(data['processed_at'], str)
        assert isinstance(data['file_size_bytes'], int)
        
        # Check file size matches metadata
        actual_size = output_files[0].stat().st_size
        assert data['file_size_bytes'] == actual_size
        
        # Check examples have required structure
        assert len(data['examples']) > 0
        for example in data['examples']:
            assert isinstance(example, dict)
            assert 'board' in example
            assert 'policy' in example
            assert 'value' in example
            import numpy as np
            assert isinstance(example['board'], np.ndarray)  # board state
            assert isinstance(example['policy'], np.ndarray)  # policy target
            assert isinstance(example['value'], (int, float, np.number))  # value target
    
    def test_atomic_file_writing(self):
        """Test that files are written atomically (temp file then rename)."""
        content = "#13,a1b2c3 1\n"
        self.create_test_trmph_file("atomic_test.trmph", content)
        
        processor = TrmphProcessor(data_dir=str(self.data_dir), output_dir=str(self.output_dir))
        file_path = self.data_dir / "atomic_test.trmph"
        
        # Process file
        processor.process_single_file(file_path)
        
        # Check that no temp files remain
        temp_files = list(self.output_dir.glob("*.tmp"))
        assert len(temp_files) == 0
        
        # Check output file exists and is complete
        output_files = list(self.output_dir.glob("*_processed.pkl.gz"))
        assert len(output_files) == 1
        
        # Verify file can be loaded completely
        with gzip.open(output_files[0], 'rb') as f:
            data = pickle.load(f)
        assert data is not None 