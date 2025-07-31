"""
Tests for the TRMPH processing pipeline.

This module tests the large-scale TRMPH file processing pipeline using the
TRMPHProcessor architecture.
"""

import pytest
import tempfile
import os
import gzip
import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock

from hex_ai.trmph_processing.processor import TRMPHProcessor
from hex_ai.trmph_processing.config import ProcessingConfig
from hex_ai.config import TRMPH_BLUE_WIN, TRMPH_RED_WIN


class TestTRMPHProcessor:
    """Test the TRMPHProcessor class."""
    
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
        """Test TRMPHProcessor initialization."""
        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )
        processor = TRMPHProcessor(config)
        
        assert processor.config.data_dir == self.data_dir
        assert processor.config.output_dir == self.output_dir
        assert processor.config.output_dir.exists()
    
    def test_find_trmph_files(self):
        """Test that TRMPH files are found correctly."""
        # Create test files
        self.create_test_trmph_file("test1.trmph", "content1")
        self.create_test_trmph_file("test2.trmph", "content2")
        self.create_test_trmph_file("test3.txt", "content3")  # Should be ignored
        
        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )
        processor = TRMPHProcessor(config)
        
        trmph_files = processor._find_trmph_files()
        assert len(trmph_files) == 2
        assert all(f.suffix == ".trmph" for f in trmph_files)
    
    def test_process_empty_file(self):
        """Test processing an empty TRMPH file."""
        self.create_test_trmph_file("empty.trmph", "")
        
        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )
        processor = TRMPHProcessor(config)
        
        results = processor.process_all_files()
        
        # Empty files should be handled gracefully
        assert len(results) == 1
        assert not results[0]['success']
        assert "Empty file" in results[0]['error']
    
    def test_process_valid_game(self):
        """Test processing a valid TRMPH game."""
        content = f"#13,a1b2c3 {TRMPH_BLUE_WIN}\n"  # Simple game with blue winner
        self.create_test_trmph_file("valid_game.trmph", content)
        
        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )
        processor = TRMPHProcessor(config)
        
        results = processor.process_all_files()
        
        assert len(results) == 1
        assert results[0]['success']
        assert results[0]['stats']['valid_games'] == 1
        assert results[0]['stats']['skipped_games'] == 0
        assert results[0]['stats']['examples_generated'] > 0
        
        # Check that output file was created
        output_files = list(self.output_dir.glob("*_processed.pkl.gz"))
        assert len(output_files) == 1
    
    def test_process_invalid_game_format(self):
        """Test processing a game with invalid format."""
        content = "#13,a1b2c3\n"  # Missing winner
        self.create_test_trmph_file("invalid_format.trmph", content)
        
        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )
        processor = TRMPHProcessor(config)
        
        results = processor.process_all_files()
        
        assert len(results) == 1
        assert results[0]['success']
        assert results[0]['stats']['valid_games'] == 0
        assert results[0]['stats']['skipped_games'] == 1
        assert results[0]['stats']['examples_generated'] == 0
    
    def test_process_game_with_duplicate_moves(self):
        """Test processing a game with duplicate moves (should be invalid)."""
        content = f"#13,a1a1b2c3 {TRMPH_BLUE_WIN}\n"  # Duplicate move 'a1'
        self.create_test_trmph_file("duplicate_moves.trmph", content)
        
        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )
        processor = TRMPHProcessor(config)
        
        results = processor.process_all_files()
        
        # Should handle gracefully due to remove_repeated_moves
        assert len(results) == 1
        assert results[0]['success']
        assert results[0]['stats']['valid_games'] == 1
        assert results[0]['stats']['skipped_games'] == 0
    
    def test_process_mixed_valid_invalid_games(self):
        """Test processing a file with both valid and invalid games."""
        content = f"#13,a1b2c3 {TRMPH_BLUE_WIN}\n"  # Valid game
        content += f"#13,a1a1b2 {TRMPH_RED_WIN}\n"   # Invalid game (duplicate moves)
        content += f"#13,a1b2c3d4 {TRMPH_BLUE_WIN}\n" # Valid game
        self.create_test_trmph_file("mixed_games.trmph", content)
        
        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )
        processor = TRMPHProcessor(config)
        
        results = processor.process_all_files()
        
        assert len(results) == 1
        assert results[0]['success']
        assert results[0]['stats']['valid_games'] == 3
        assert results[0]['stats']['skipped_games'] == 0  # Duplicate moves are handled gracefully
        assert results[0]['stats']['all_games'] == 3
    
    def test_output_file_creation(self):
        """Test that output files are created with correct naming."""
        content = f"#13,a1b2c3 {TRMPH_BLUE_WIN}\n"
        self.create_test_trmph_file("test_output.trmph", content)
        
        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )
        processor = TRMPHProcessor(config)
        
        processor.process_all_files()
        
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
        """Test that statistics are properly tracked across multiple files."""
        # Create multiple files
        for i in range(3):
            self.create_test_trmph_file(f"test{i}.trmph", f"#13,a1b2c3 {TRMPH_BLUE_WIN}\n")
        
        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )
        processor = TRMPHProcessor(config)
        
        # Process files
        results = processor.process_all_files()
        
        assert len(results) == 3
        successful_results = [r for r in results if r['success']]
        assert len(successful_results) == 3
        
        total_examples = sum(r['stats']['examples_generated'] for r in successful_results)
        assert total_examples > 0
    
    def test_max_files_parameter(self):
        """Test the max_files parameter."""
        # Create multiple files
        for i in range(5):
            self.create_test_trmph_file(f"test{i}.trmph", f"#13,a1b2c3 {TRMPH_BLUE_WIN}\n")
        
        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir),
            max_files=3
        )
        processor = TRMPHProcessor(config)
        
        # Process only 3 files
        results = processor.process_all_files()
        
        assert len(results) == 3
    
    def test_sequential_processing(self):
        """Test sequential processing mode."""
        # Create multiple files
        for i in range(3):
            self.create_test_trmph_file(f"test{i}.trmph", f"#13,a1b2c3 {TRMPH_BLUE_WIN}\n")
        
        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir),
            max_workers=1  # Sequential processing
        )
        processor = TRMPHProcessor(config)
        
        results = processor.process_all_files()
        
        assert len(results) == 3
        assert all(r['success'] for r in results)
    
    def test_parallel_processing(self):
        """Test parallel processing mode."""
        # Create multiple files
        for i in range(3):
            self.create_test_trmph_file(f"test{i}.trmph", f"#13,a1b2c3 {TRMPH_BLUE_WIN}\n")
        
        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir),
            max_workers=2  # Parallel processing
        )
        processor = TRMPHProcessor(config)
        
        results = processor.process_all_files()
        
        assert len(results) == 3
        assert all(r['success'] for r in results)
    
    def test_filename_uniqueness(self):
        """Test that output filenames are unique."""
        # Create multiple files with different names that would produce same output name
        for i in range(3):
            content = f"#13,a1b2c3 {TRMPH_BLUE_WIN}\n"
            # Create files with different names but same stem
            if i == 0:
                self.create_test_trmph_file("same_name.trmph", content)
            elif i == 1:
                self.create_test_trmph_file("same_name_1.trmph", content)
            else:
                self.create_test_trmph_file("same_name_2.trmph", content)
        
        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )
        processor = TRMPHProcessor(config)
        
        # Process files - should create unique output names
        processor.process_all_files()
        
        # Check output files
        output_files = list(self.output_dir.glob("*_processed.pkl.gz"))
        assert len(output_files) == 3
        
        # Check that filenames are unique
        filenames = [f.name for f in output_files]
        assert len(filenames) == len(set(filenames))
    
    def test_output_file_structure(self):
        """Test that output files have correct structure and metadata."""
        content = f"#13,a1b2c3 {TRMPH_BLUE_WIN}\n"
        self.create_test_trmph_file("structure_test.trmph", content)
        
        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )
        processor = TRMPHProcessor(config)
        
        processor.process_all_files()
        
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
        
        # Check data types
        assert isinstance(data['examples'], list)
        assert isinstance(data['source_file'], str)
        assert isinstance(data['processing_stats'], dict)
        assert isinstance(data['processed_at'], str)
        
        # Check examples have required structure
        assert len(data['examples']) > 0
        for example in data['examples']:
            assert isinstance(example, dict)
            assert 'board' in example
            assert 'policy' in example
            assert 'value' in example
            assert 'player_to_move' in example
            assert 'metadata' in example
            import numpy as np
            assert isinstance(example['board'], np.ndarray)
            assert example['policy'] is None or isinstance(example['policy'], np.ndarray)
            assert isinstance(example['value'], (int, float, np.number))
    
    def test_position_selector_all(self):
        """Test position selector 'all' mode."""
        content = f"#13,a1b2c3 {TRMPH_BLUE_WIN}\n"
        self.create_test_trmph_file("test.trmph", content)
        
        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir),
            position_selector="all"
        )
        processor = TRMPHProcessor(config)
        
        results = processor.process_all_files()
        
        assert len(results) == 1
        assert results[0]['success']
        # Should generate multiple examples (one for each position)
        assert results[0]['stats']['examples_generated'] > 1
    
    def test_position_selector_final(self):
        """Test position selector 'final' mode."""
        content = f"#13,a1b2c3 {TRMPH_BLUE_WIN}\n"
        self.create_test_trmph_file("test.trmph", content)
        
        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir),
            position_selector="final"
        )
        processor = TRMPHProcessor(config)
        
        results = processor.process_all_files()
        
        assert len(results) == 1
        assert results[0]['success']
        # Should generate exactly one example (final position)
        assert results[0]['stats']['examples_generated'] == 1
    
    def test_error_handling_during_processing(self):
        """Test error handling during processing."""
        content = "#13,a1b2c3 1\n"  # Invalid format (old format)
        self.create_test_trmph_file("error_test.trmph", content)
        
        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )
        processor = TRMPHProcessor(config)
        
        results = processor.process_all_files()
        
        assert len(results) == 1
        assert results[0]['success']  # File-level success, but game-level failure
        assert results[0]['stats']['valid_games'] == 0
        assert results[0]['stats']['skipped_games'] == 1
    
    def test_large_file_handling(self):
        """Test handling of large files (memory usage)."""
        # Create a large file with many games
        content = ""
        for i in range(100):  # 100 games
            content += f"#13,a1b2c3 {TRMPH_BLUE_WIN}\n"
        
        self.create_test_trmph_file("large_file.trmph", content)
        
        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )
        processor = TRMPHProcessor(config)
        
        results = processor.process_all_files()
        
        assert len(results) == 1
        assert results[0]['success']
        assert results[0]['stats']['valid_games'] == 100
        assert results[0]['stats']['skipped_games'] == 0
    
    def test_atomic_file_writing(self):
        """Test that files are written atomically (temp file then rename)."""
        content = f"#13,a1b2c3 {TRMPH_BLUE_WIN}\n"
        self.create_test_trmph_file("atomic_test.trmph", content)
        
        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )
        processor = TRMPHProcessor(config)
        
        # Process file
        processor.process_all_files()
        
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