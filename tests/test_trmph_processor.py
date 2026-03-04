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
import json
from types import SimpleNamespace
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from hex_ai.data_collection import (
    collect_and_organize_data,
    collect_tournament_data_since_date,
    combine_and_clean_files,
)
from hex_ai.move_provenance import (
    load_move_provenance_sidecar,
    make_move_provenance_record,
    MOVE_PROVENANCE_SCHEMA_VERSION_V2,
    POLICY_TARGET_ENCODING_DENSE_FP16_ZLIB_BASE64,
    sidecar_path_for_trmph,
)
from hex_ai.selfplay.selfplay_engine import SelfPlayEngine
from hex_ai.trmph_processing.processor import TRMPHProcessor
from hex_ai.trmph_processing.config import ProcessingConfig
from hex_ai.config import TRMPH_BLUE_WIN, TRMPH_RED_WIN
from hex_ai.data_utils import extract_training_examples_with_selector_from_game


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

    def create_test_provenance_sidecar(self, trmph_filename: str, move_codes_per_game: list[str]) -> Path:
        """Create a matching provenance sidecar for a test TRMPH file."""
        trmph_path = self.data_dir / trmph_filename
        sidecar_path = sidecar_path_for_trmph(trmph_path)
        with open(sidecar_path, "w", encoding="utf-8") as f:
            for game_index, move_codes in enumerate(move_codes_per_game):
                record = make_move_provenance_record(game_index, move_codes)
                f.write(record.to_json_line())
                f.write("\n")
        return sidecar_path
    
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
    
    def test_process_trmph_file_with_headers(self):
        """Test processing a .trmph file with header lines and TRMPH data."""
        content = "# Self-play games - 2025-07-29T13:34:04.681729\n"
        content += "# Model: checkpoints/model.pt.gz\n"
        content += "# Format: trmph_string winner\n"
        content += "# Example: #13,a4g7e9e8f8f7h7h6j5 r\n"
        content += f"#13,a1b2c3 {TRMPH_BLUE_WIN}\n"  # Valid TRMPH game
        content += f"#13,a1b2c3d4 {TRMPH_RED_WIN}\n"  # Another valid TRMPH game
        
        self.create_test_trmph_file("test_with_headers.trmph", content)
        
        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )
        processor = TRMPHProcessor(config)
        
        results = processor.process_all_files()
        
        assert len(results) == 1
        assert results[0]['success']
        assert results[0]['stats']['valid_games'] == 2
        assert results[0]['stats']['skipped_games'] == 4  # 4 header lines
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
        """Test processing a game with duplicate moves (should be skipped)."""
        content = f"#13,a1a1b2c3 {TRMPH_BLUE_WIN}\n"  # Duplicate move 'a1'
        self.create_test_trmph_file("duplicate_moves.trmph", content)
        
        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir)
        )
        processor = TRMPHProcessor(config)
        
        results = processor.process_all_files()
        
        # Should skip game with duplicate moves entirely
        assert len(results) == 1
        assert results[0]['success']
        assert results[0]['stats']['valid_games'] == 0
        assert results[0]['stats']['duplicate_move_games'] == 1
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
        assert results[0]['stats']['valid_games'] == 2  # Only 2 valid games (duplicate move game is skipped)
        assert results[0]['stats']['duplicate_move_games'] == 1  # 1 game skipped due to duplicate moves
        assert results[0]['stats']['skipped_games'] == 0
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

    def test_policy_provenance_require_masks_policy_targets(self):
        """Require-mode provenance should skip masked policy positions and keep trainable ones."""
        content = (
            f"#13,a1b2c3 {TRMPH_BLUE_WIN}\n"
            f"#13,a1b2 {TRMPH_RED_WIN}\n"
        )
        self.create_test_trmph_file("provenance_ok.trmph", content)
        self.create_test_provenance_sidecar("provenance_ok.trmph", ["VCG", "CC"])

        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir),
            policy_provenance_mode="require",
            max_workers=1,
        )
        processor = TRMPHProcessor(config)
        results = processor.process_all_files()

        assert len(results) == 1
        assert results[0]['success']
        stats = results[0]['stats']
        assert stats['policy_positions_total'] == 5
        assert stats['policy_positions_trainable'] == 2
        assert stats['policy_positions_skipped'] == 3
        assert stats['policy_positions_skipped_by_code'] == {'C': 3}

        output_files = list(self.output_dir.glob("*_processed.pkl.gz"))
        assert len(output_files) == 1
        with gzip.open(output_files[0], 'rb') as f:
            data = pickle.load(f)

        examples = data['examples']
        non_terminal_examples = [
            ex for ex in examples
            if ex['metadata']['position_in_game'] < (ex['metadata']['total_positions'] - 1)
        ]
        assert len(non_terminal_examples) == 2
        assert sum(ex['policy'] is not None for ex in non_terminal_examples) == 2
        assert sum(ex['policy'] is None for ex in non_terminal_examples) == 0

    def test_policy_provenance_require_missing_sidecar_fails(self):
        """Require-mode should fail fast when sidecar is missing."""
        content = f"#13,a1b2c3 {TRMPH_BLUE_WIN}\n"
        self.create_test_trmph_file("missing_sidecar.trmph", content)

        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir),
            policy_provenance_mode="require",
            max_workers=1,
        )
        processor = TRMPHProcessor(config)
        results = processor.process_all_files()

        assert len(results) == 1
        assert not results[0]['success']
        assert "Missing required move provenance sidecar" in results[0]['error']

    def test_policy_provenance_require_move_count_mismatch_fails(self):
        """Require-mode should fail on sidecar/game move-count misalignment."""
        trmph_path = self.create_test_trmph_file(
            "mismatch_sidecar.trmph",
            f"#13,a1b2c3 {TRMPH_BLUE_WIN}\n",
        )
        sidecar_path = sidecar_path_for_trmph(trmph_path)
        bad_record = {
            "schema_version": 1,
            "game_index": 0,
            "move_count": 99,
            "move_codes": "VCG",
            "policy_train_mask": "101",
        }
        with open(sidecar_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(bad_record))
            f.write("\n")

        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir),
            policy_provenance_mode="require",
            max_workers=1,
        )
        processor = TRMPHProcessor(config)
        results = processor.process_all_files()

        assert len(results) == 1
        assert not results[0]['success']
        assert "move_codes length" in results[0]['error']

    def test_policy_provenance_optional_fallback_without_sidecar(self):
        """Optional mode should process files without sidecars as all-policy-trainable."""
        self.create_test_trmph_file(
            "optional_no_sidecar.trmph",
            f"#13,a1b2c3 {TRMPH_BLUE_WIN}\n",
        )

        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir),
            policy_provenance_mode="optional",
            max_workers=1,
        )
        processor = TRMPHProcessor(config)
        results = processor.process_all_files()

        assert len(results) == 1
        assert results[0]['success']
        output_files = list(self.output_dir.glob("*_processed.pkl.gz"))
        assert len(output_files) == 1

        with gzip.open(output_files[0], 'rb') as f:
            data = pickle.load(f)

        non_terminal_examples = [
            ex for ex in data['examples']
            if ex['metadata']['position_in_game'] < (ex['metadata']['total_positions'] - 1)
        ]
        assert non_terminal_examples
        assert all(ex['policy'] is not None for ex in non_terminal_examples)

    def test_combine_and_clean_files_propagates_provenance_sidecars(self):
        """Preprocessing combine/clean should emit aligned sidecars in require mode."""
        self.create_test_trmph_file(
            "combine_1.trmph",
            "#13,a1b2 b\n#13,c3d4 r\n",
        )
        self.create_test_provenance_sidecar("combine_1.trmph", ["VC", "GG"])

        self.create_test_trmph_file(
            "combine_2.trmph",
            "#13,c3d4 r\n#13,e5f6 b\n",
        )
        self.create_test_provenance_sidecar("combine_2.trmph", ["GG", "TT"])

        combine_and_clean_files(
            input_dirs=[self.data_dir],
            output_dir=self.output_dir,
            chunk_size=10,
            policy_provenance_mode="require",
        )

        chunk_path = self.output_dir / "cleaned_chunk_000.trmph"
        assert chunk_path.exists()
        chunk_sidecar = sidecar_path_for_trmph(chunk_path)
        assert chunk_sidecar.exists()

        with open(chunk_path, "r", encoding="utf-8") as f:
            chunk_lines = [line.strip() for line in f.readlines() if line.strip()]
        assert chunk_lines == ["#13,a1b2 b", "#13,c3d4 r", "#13,e5f6 b"]

        records = load_move_provenance_sidecar(chunk_sidecar)
        assert [record.game_index for record in records] == [0, 1, 2]
        assert [record.move_codes for record in records] == ["VC", "GG", "TT"]

    def test_combine_and_clean_files_require_missing_sidecar_fails(self):
        """Preprocessing require mode should fail when an input sidecar is missing."""
        self.create_test_trmph_file(
            "combine_missing_sidecar.trmph",
            "#13,a1b2 b\n",
        )

        with pytest.raises(FileNotFoundError, match="Missing required move provenance sidecar"):
            combine_and_clean_files(
                input_dirs=[self.data_dir],
                output_dir=self.output_dir,
                chunk_size=10,
                policy_provenance_mode="require",
            )

    def test_combine_and_clean_files_optional_writes_fallback_sidecar(self):
        """Preprocessing optional mode should synthesize all-valid sidecars when missing."""
        self.create_test_trmph_file(
            "combine_optional_missing_sidecar.trmph",
            "#13,a1b2 b\n#13,c3d4e5 r\n",
        )

        combine_and_clean_files(
            input_dirs=[self.data_dir],
            output_dir=self.output_dir,
            chunk_size=10,
            policy_provenance_mode="optional",
        )

        chunk_path = self.output_dir / "cleaned_chunk_000.trmph"
        chunk_sidecar = sidecar_path_for_trmph(chunk_path)
        assert chunk_path.exists()
        assert chunk_sidecar.exists()

        records = load_move_provenance_sidecar(chunk_sidecar)
        assert [record.move_codes for record in records] == ["VV", "VVV"]

    def test_combine_and_clean_files_optional_prefers_authoritative_sidecar_over_fallback(self):
        """Optional mode should keep real sidecar provenance when mixed with fallback duplicates."""
        self.create_test_trmph_file(
            "combine_optional_legacy.trmph",
            "#13,a1b2 b\n",
        )
        self.create_test_trmph_file(
            "combine_optional_modern.trmph",
            "#13,a1b2 b\n",
        )
        self.create_test_provenance_sidecar("combine_optional_modern.trmph", ["GC"])

        combine_and_clean_files(
            input_dirs=[self.data_dir],
            output_dir=self.output_dir,
            chunk_size=10,
            policy_provenance_mode="optional",
        )

        chunk_sidecar = sidecar_path_for_trmph(self.output_dir / "cleaned_chunk_000.trmph")
        records = load_move_provenance_sidecar(chunk_sidecar)
        assert [record.move_codes for record in records] == ["GC"]

    def test_combine_and_clean_files_optional_keeps_first_authoritative_on_conflict(self):
        """Optional mode should keep first-seen provenance for conflicting authoritative duplicates."""
        self.create_test_trmph_file(
            "combine_optional_conflict_a.trmph",
            "#13,a1b2 b\n",
        )
        self.create_test_provenance_sidecar("combine_optional_conflict_a.trmph", ["GC"])

        self.create_test_trmph_file(
            "combine_optional_conflict_b.trmph",
            "#13,a1b2 b\n",
        )
        self.create_test_provenance_sidecar("combine_optional_conflict_b.trmph", ["VV"])

        combine_and_clean_files(
            input_dirs=[self.data_dir],
            output_dir=self.output_dir,
            chunk_size=10,
            policy_provenance_mode="optional",
        )

        chunk_sidecar = sidecar_path_for_trmph(self.output_dir / "cleaned_chunk_000.trmph")
        records = load_move_provenance_sidecar(chunk_sidecar)
        assert [record.move_codes for record in records] == ["GC"]

        summary_text = (self.output_dir / "processing_summary.txt").read_text(encoding="utf-8")
        assert "Conflicting authoritative provenance records kept-first: 1" in summary_text

    def test_combine_and_clean_files_optional_prefers_v2_over_v1_when_equivalent(self):
        """Optional mode should prefer v2 payloads over v1 for equivalent duplicate games."""
        self.create_test_trmph_file(
            "combine_optional_v1_first.trmph",
            "#13,a1b2 b\n",
        )
        self.create_test_provenance_sidecar("combine_optional_v1_first.trmph", ["VV"])

        trmph_path = self.create_test_trmph_file(
            "combine_optional_v2_second.trmph",
            "#13,a1b2 b\n",
        )
        targets = np.zeros((2, 169), dtype=np.float32)
        targets[0, 0] = 1.0
        targets[1, 14] = 1.0
        v2_record = make_move_provenance_record(
            game_index=0,
            move_codes="VV",
            policy_targets=targets,
            policy_target_source_codes="VV",
            policy_target_version=1,
        )
        sidecar_path = sidecar_path_for_trmph(trmph_path)
        with open(sidecar_path, "w", encoding="utf-8") as f:
            f.write(v2_record.to_json_line())
            f.write("\n")

        combine_and_clean_files(
            input_dirs=[self.data_dir],
            output_dir=self.output_dir,
            chunk_size=10,
            policy_provenance_mode="optional",
        )

        chunk_sidecar = sidecar_path_for_trmph(self.output_dir / "cleaned_chunk_000.trmph")
        records = load_move_provenance_sidecar(chunk_sidecar)
        assert len(records) == 1
        assert records[0].schema_version == MOVE_PROVENANCE_SCHEMA_VERSION_V2
        decoded = records[0].decode_policy_targets()
        assert decoded is not None
        np.testing.assert_allclose(decoded, targets, rtol=1e-3, atol=1e-3)

    def test_combine_and_clean_files_optional_counts_conflicting_v2_payloads(self):
        """Equivalent duplicate games with different v2 payloads should be counted and keep-first."""
        trmph_a = self.create_test_trmph_file(
            "combine_optional_v2_conflict_a.trmph",
            "#13,a1b2 b\n",
        )
        targets_a = np.zeros((2, 169), dtype=np.float32)
        targets_a[0, 0] = 1.0
        targets_a[1, 1] = 1.0
        record_a = make_move_provenance_record(
            game_index=0,
            move_codes="VV",
            policy_targets=targets_a,
            policy_target_source_codes="VV",
            policy_target_version=1,
        )
        with open(sidecar_path_for_trmph(trmph_a), "w", encoding="utf-8") as f:
            f.write(record_a.to_json_line())
            f.write("\n")

        trmph_b = self.create_test_trmph_file(
            "combine_optional_v2_conflict_b.trmph",
            "#13,a1b2 b\n",
        )
        targets_b = np.zeros((2, 169), dtype=np.float32)
        targets_b[0, 2] = 1.0
        targets_b[1, 3] = 1.0
        record_b = make_move_provenance_record(
            game_index=0,
            move_codes="VV",
            policy_targets=targets_b,
            policy_target_source_codes="VV",
            policy_target_version=1,
        )
        with open(sidecar_path_for_trmph(trmph_b), "w", encoding="utf-8") as f:
            f.write(record_b.to_json_line())
            f.write("\n")

        combine_and_clean_files(
            input_dirs=[self.data_dir],
            output_dir=self.output_dir,
            chunk_size=10,
            policy_provenance_mode="optional",
        )

        chunk_sidecar = sidecar_path_for_trmph(self.output_dir / "cleaned_chunk_000.trmph")
        records = load_move_provenance_sidecar(chunk_sidecar)
        assert len(records) == 1
        assert records[0].schema_version == MOVE_PROVENANCE_SCHEMA_VERSION_V2
        decoded = records[0].decode_policy_targets()
        assert decoded is not None
        np.testing.assert_allclose(decoded, targets_a, rtol=1e-3, atol=1e-3)

        summary_text = (self.output_dir / "processing_summary.txt").read_text(encoding="utf-8")
        assert (
            "Conflicting v2 policy-target payloads (first-seen kept unless newer version): 1"
            in summary_text
        )

    def test_collect_and_organize_data_optional_writes_sidecars(self):
        """Collection mode should emit sidecars in optional mode."""
        self.create_test_trmph_file(
            "collected_optional_sidecar.trmph",
            "#13,a1b2 b\n#13,c3d4e5 r\n",
        )

        stats = collect_and_organize_data(
            source_dirs=[self.data_dir],
            output_dir=self.output_dir,
            chunk_size=10,
            policy_provenance_mode="optional",
        )

        assert stats["provenance_sidecars_written"] is True
        chunk_sidecar = sidecar_path_for_trmph(self.output_dir / "collected_chunk_000.trmph")
        assert chunk_sidecar.exists()
        records = load_move_provenance_sidecar(chunk_sidecar)
        assert [record.move_codes for record in records] == ["VV", "VVV"]

    def test_collect_tournament_data_since_date_optional_writes_sidecars(self):
        """Tournament collection should emit sidecars in optional mode."""
        self.create_test_trmph_file(
            "tournament_optional_sidecar.trmph",
            "#13,a1b2 b\n",
        )

        stats = collect_tournament_data_since_date(
            source_dirs=[self.data_dir],
            output_dir=self.output_dir,
            since_date=datetime.now() - timedelta(days=1),
            chunk_size=10,
            policy_provenance_mode="optional",
        )

        assert stats["provenance_sidecars_written"] is True
        chunk_sidecar = sidecar_path_for_trmph(self.output_dir / "tournament_chunk_000.trmph")
        assert chunk_sidecar.exists()
        records = load_move_provenance_sidecar(chunk_sidecar)
        assert [record.move_codes for record in records] == ["VV"]

    def test_move_provenance_v2_policy_target_roundtrip(self):
        """Schema-v2 records should round-trip encoded policy-target payloads."""
        targets = np.zeros((3, 169), dtype=np.float32)
        targets[0, 0] = 1.0
        targets[1, 1] = 0.7
        targets[1, 2] = 0.3
        targets[2, 168] = 1.0

        record = make_move_provenance_record(
            game_index=0,
            move_codes="VVV",
            policy_targets=targets,
            policy_target_source_codes="VVV",
            policy_target_version=1,
        )
        assert record.schema_version == MOVE_PROVENANCE_SCHEMA_VERSION_V2
        decoded = record.decode_policy_targets()
        assert decoded is not None
        np.testing.assert_allclose(decoded, targets, rtol=1e-3, atol=1e-3)

    def test_load_move_provenance_sidecar_validates_v2_payload_by_default(self):
        """Default sidecar loading should fail fast on malformed v2 payload blobs."""
        trmph_path = self.create_test_trmph_file("invalid_v2_blob.trmph", "#13,a1 b\n")
        sidecar_path = sidecar_path_for_trmph(trmph_path)
        payload = {
            "schema_version": MOVE_PROVENANCE_SCHEMA_VERSION_V2,
            "game_index": 0,
            "move_count": 1,
            "move_codes": "V",
            "policy_train_mask": "1",
            "policy_target_encoding": POLICY_TARGET_ENCODING_DENSE_FP16_ZLIB_BASE64,
            "policy_targets_blob": "!!not_base64!!",
            "policy_target_size": 169,
            "policy_target_source_codes": "V",
            "policy_target_version": 1,
        }
        with open(sidecar_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload))
            f.write("\n")

        with pytest.raises(ValueError, match="base64"):
            load_move_provenance_sidecar(sidecar_path)

    def test_load_move_provenance_sidecar_can_skip_eager_v2_payload_decode(self):
        """Optional lazy mode should defer malformed payload failure until decode call."""
        trmph_path = self.create_test_trmph_file("invalid_v2_blob_lazy.trmph", "#13,a1 b\n")
        sidecar_path = sidecar_path_for_trmph(trmph_path)
        payload = {
            "schema_version": MOVE_PROVENANCE_SCHEMA_VERSION_V2,
            "game_index": 0,
            "move_count": 1,
            "move_codes": "V",
            "policy_train_mask": "1",
            "policy_target_encoding": POLICY_TARGET_ENCODING_DENSE_FP16_ZLIB_BASE64,
            "policy_targets_blob": "!!not_base64!!",
            "policy_target_size": 169,
            "policy_target_source_codes": "V",
            "policy_target_version": 1,
        }
        with open(sidecar_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload))
            f.write("\n")

        records = load_move_provenance_sidecar(
            sidecar_path, validate_policy_targets_payload=False
        )
        assert len(records) == 1
        with pytest.raises(ValueError, match="base64"):
            records[0].decode_policy_targets()

    def test_load_move_provenance_sidecar_can_ignore_truncated_last_line(self):
        """Optional loader mode should skip a non-empty truncated tail line."""
        trmph_path = self.create_test_trmph_file(
            "truncated_sidecar_tail.trmph",
            "#13,a1 b\n#13,a1b2 r\n",
        )
        sidecar_path = sidecar_path_for_trmph(trmph_path)
        first_record = make_move_provenance_record(game_index=0, move_codes="V")
        with open(sidecar_path, "w", encoding="utf-8") as f:
            f.write(first_record.to_json_line())
            f.write("\n")
            # Deliberately write a truncated/incomplete JSON line without newline.
            f.write('{"schema_version":1')

        with pytest.raises(ValueError, match="invalid JSON"):
            load_move_provenance_sidecar(sidecar_path)

        records = load_move_provenance_sidecar(
            sidecar_path,
            allow_truncated_last_line=True,
        )
        assert len(records) == 1
        assert records[0].game_index == 0
        assert records[0].move_codes == "V"

    def test_collect_and_organize_data_optional_recovers_truncated_sidecar_tail(self):
        """Optional provenance mode should recover trailing sidecar truncation via fallback."""
        trmph_path = self.create_test_trmph_file(
            "optional_truncated_sidecar_recovery.trmph",
            "#13,a1 b\n#13,a1b2 r\n",
        )
        sidecar_path = sidecar_path_for_trmph(trmph_path)
        first_record = make_move_provenance_record(game_index=0, move_codes="V")
        with open(sidecar_path, "w", encoding="utf-8") as f:
            f.write(first_record.to_json_line())
            f.write("\n")
            f.write('{"schema_version":1')

        stats = collect_and_organize_data(
            source_dirs=[self.data_dir],
            output_dir=self.output_dir,
            chunk_size=10,
            policy_provenance_mode="optional",
        )
        assert stats["provenance_sidecars_written"] is True

        chunk_sidecar = sidecar_path_for_trmph(self.output_dir / "collected_chunk_000.trmph")
        records = load_move_provenance_sidecar(chunk_sidecar)
        assert [record.move_codes for record in records] == ["V", "VV"]

    def test_extract_training_examples_rejects_zero_mass_trainable_search_target_rows(self):
        """Fail fast when trainable rows carry zero policy_search_target mass."""
        zero_targets = np.zeros((2, 169), dtype=np.float32)
        with pytest.raises(
            ValueError,
            match="policy_search_targets has non-positive probability mass on trainable rows",
        ):
            extract_training_examples_with_selector_from_game(
                trmph_text="#13,a1b2",
                winner_from_file=TRMPH_BLUE_WIN,
                game_id=(0, 1),
                policy_train_mask="11",
                policy_move_codes="VV",
                policy_search_targets=zero_targets,
                policy_target_source_codes="VV",
            )

    def test_combine_and_clean_files_preserves_v2_policy_targets(self):
        """Combine/clean should preserve v2 sidecar payloads when rewriting game_index."""
        trmph_path = self.create_test_trmph_file(
            "combine_v2_payload.trmph",
            "#13,a1b2c3 b\n",
        )
        sidecar_path = sidecar_path_for_trmph(trmph_path)
        targets = np.zeros((3, 169), dtype=np.float32)
        targets[0, 0] = 1.0
        targets[1, 14] = 0.4
        targets[1, 15] = 0.6
        targets[2, 28] = 1.0
        record = make_move_provenance_record(
            game_index=0,
            move_codes="VVV",
            policy_targets=targets,
            policy_target_source_codes="VVV",
            policy_target_version=1,
        )
        with open(sidecar_path, "w", encoding="utf-8") as f:
            f.write(record.to_json_line())
            f.write("\n")

        combine_and_clean_files(
            input_dirs=[self.data_dir],
            output_dir=self.output_dir,
            chunk_size=10,
            policy_provenance_mode="require",
        )

        chunk_sidecar = sidecar_path_for_trmph(self.output_dir / "cleaned_chunk_000.trmph")
        records = load_move_provenance_sidecar(chunk_sidecar)
        assert len(records) == 1
        assert records[0].schema_version == MOVE_PROVENANCE_SCHEMA_VERSION_V2
        decoded = records[0].decode_policy_targets()
        assert decoded is not None
        np.testing.assert_allclose(decoded, targets, rtol=1e-3, atol=1e-3)

    def test_trmph_processor_attaches_policy_search_targets_from_v2_sidecar(self):
        """TRMPH processor should attach per-position policy_search_target when v2 payload exists."""
        trmph_path = self.create_test_trmph_file(
            "processor_v2_payload.trmph",
            "#13,a1b2c3 b\n",
        )
        sidecar_path = sidecar_path_for_trmph(trmph_path)
        targets = np.zeros((3, 169), dtype=np.float32)
        targets[0, 0] = 1.0
        targets[1, 14] = 0.25
        targets[1, 15] = 0.75
        targets[2, 28] = 1.0
        record = make_move_provenance_record(
            game_index=0,
            move_codes="VVV",
            policy_targets=targets,
            policy_target_source_codes="VVV",
            policy_target_version=1,
        )
        with open(sidecar_path, "w", encoding="utf-8") as f:
            f.write(record.to_json_line())
            f.write("\n")

        config = ProcessingConfig(
            data_dir=str(self.data_dir),
            output_dir=str(self.output_dir),
            policy_provenance_mode="require",
            max_workers=1,
        )
        processor = TRMPHProcessor(config)
        results = processor.process_all_files()
        assert len(results) == 1
        assert results[0]["success"]

        output_files = list(self.output_dir.glob("*_processed.pkl.gz"))
        assert len(output_files) == 1
        with gzip.open(output_files[0], "rb") as f:
            data = pickle.load(f)

        non_terminal_examples = [
            ex
            for ex in data["examples"]
            if ex["metadata"]["position_in_game"] < (ex["metadata"]["total_positions"] - 1)
        ]
        by_pos = {
            ex["metadata"]["position_in_game"]: ex
            for ex in non_terminal_examples
        }
        assert sorted(by_pos.keys()) == [0, 1, 2]
        for pos in [0, 1, 2]:
            search_target = by_pos[pos].get("policy_search_target")
            assert isinstance(search_target, np.ndarray)
            assert search_target.shape == (169,)
            np.testing.assert_allclose(search_target, targets[pos], rtol=1e-3, atol=1e-3)

    def test_gumbel_policy_target_vector_uses_final_scores_and_normalizes(self):
        """Gumbel policy target should be softmax(score_without_gumbel) over ranked actions."""
        engine = SelfPlayEngine.__new__(SelfPlayEngine)
        rows = [
            {"tensor_action": 10, "score_without_gumbel": 2.0},
            {"tensor_action": 5, "score_without_gumbel": 1.0},
            {"tensor_action": 7, "score_without_gumbel": 0.0},
        ]
        mcts_result = SimpleNamespace(
            move=(0, 10),  # tensor action 10 on 13x13 board
            stats={"gumbel_final_rank_rows": rows},
        )

        vec = engine._build_policy_target_vector_from_gumbel_final_scores(
            mcts_result, board_size=13
        )
        assert vec.shape == (169,)
        assert np.isclose(float(vec.sum()), 1.0, atol=1e-6)
        assert vec[10] > vec[5] > vec[7]
        assert vec[3] == 0.0

    def test_gumbel_policy_target_vector_fails_if_selected_move_not_top_score(self):
        """Fail fast when selected move does not match top final noise-free Gumbel score."""
        engine = SelfPlayEngine.__new__(SelfPlayEngine)
        rows = [
            {"tensor_action": 10, "score_without_gumbel": 2.0},
            {"tensor_action": 5, "score_without_gumbel": 1.0},
        ]
        mcts_result = SimpleNamespace(
            move=(0, 5),  # selected action is not top score action
            stats={"gumbel_final_rank_rows": rows},
        )

        with pytest.raises(RuntimeError, match="selected move is not the top noise-free"):
            engine._build_policy_target_vector_from_gumbel_final_scores(
                mcts_result, board_size=13
            )

    def test_validate_game_data_fails_for_zero_mass_trainable_policy_row(self):
        """Self-play output validation should fail when a trainable V/G/T row has zero mass."""
        engine = SelfPlayEngine.__new__(SelfPlayEngine)
        engine.board_size = 13
        engine.write_provenance = True

        policy_targets = np.zeros((2, 169), dtype=np.float32)
        # Row 0 corresponds to trainable code 'V' but has zero probability mass.
        # Row 1 corresponds to masked code 'C' and may be zero.
        game_data = {
            "trmph": "#13,a1b2",
            "winner": TRMPH_BLUE_WIN,
            "move_provenance_codes": "VC",
            "policy_targets_matrix": policy_targets,
            "policy_target_source_codes": "VC",
        }

        with pytest.raises(ValueError, match="trainable move codes"):
            engine._validate_game_data(game_data)

    def test_validate_game_data_allows_zero_mass_masked_policy_row(self):
        """Masked C rows may be zero-mass as long as trainable rows have positive mass."""
        engine = SelfPlayEngine.__new__(SelfPlayEngine)
        engine.board_size = 13
        engine.write_provenance = True

        policy_targets = np.zeros((2, 169), dtype=np.float32)
        policy_targets[0, 0] = 1.0
        game_data = {
            "trmph": "#13,a1b2",
            "winner": TRMPH_BLUE_WIN,
            "move_provenance_codes": "VC",
            "policy_targets_matrix": policy_targets,
            "policy_target_source_codes": "VC",
        }

        engine._validate_game_data(game_data)
