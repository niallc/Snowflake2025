"""
Tests for the preprocess_selfplay_data script.

This module tests the data preprocessing functionality for cleaning up
self-play data files.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

# Import the functions we want to test
from scripts.preprocess_selfplay_data import (
    find_trmph_files,
    extract_games_from_file,
    remove_duplicates,
    split_games_into_chunks,
    combine_and_clean_files
)


class TestPreprocessSelfplayData:
    """Test the preprocess_selfplay_data functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.temp_dir) / "input"
        self.output_dir = Path(self.temp_dir) / "output"
        self.input_dir.mkdir()
        self.output_dir.mkdir()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_trmph_file(self, filename: str, content: str):
        """Create a test TRMPH file with given content."""
        file_path = self.input_dir / filename
        with open(file_path, 'w') as f:
            f.write(content)
        return file_path
    
    def test_find_trmph_files(self):
        """Test that TRMPH files are found correctly."""
        # Create test files
        self.create_test_trmph_file("test1.trmph", "content1")
        self.create_test_trmph_file("test2.trmph", "content2")
        self.create_test_trmph_file("test3.txt", "content3")  # Should be ignored
        
        trmph_files = find_trmph_files(self.input_dir)
        assert len(trmph_files) == 2
        assert all(f.suffix == ".trmph" for f in trmph_files)
    
    def test_extract_games_from_file(self):
        """Test extracting games from a TRMPH file."""
        content = """# Header line
# Another header
#13,a1b2c3 b
#13,d4e5f6 r
#13,g7h8i9 b
"""
        self.create_test_trmph_file("test.trmph", content)
        
        games = extract_games_from_file(self.input_dir / "test.trmph")
        assert len(games) == 3
        assert games[0] == "#13,a1b2c3 b"
        assert games[1] == "#13,d4e5f6 r"
        assert games[2] == "#13,g7h8i9 b"
    
    def test_extract_games_empty_file(self):
        """Test extracting games from an empty file."""
        self.create_test_trmph_file("empty.trmph", "")
        
        games = extract_games_from_file(self.input_dir / "empty.trmph")
        assert len(games) == 0
    
    def test_extract_games_header_only(self):
        """Test extracting games from a file with only headers."""
        content = """# Header line
# Another header
# Third header
"""
        self.create_test_trmph_file("headers_only.trmph", content)
        
        games = extract_games_from_file(self.input_dir / "headers_only.trmph")
        assert len(games) == 0
    
    def test_remove_duplicates(self):
        """Test removing duplicate games."""
        games = [
            "#13,a1b2c3 b",
            "#13,d4e5f6 r",
            "#13,a1b2c3 b",  # Duplicate
            "#13,g7h8i9 b",
            "#13,d4e5f6 r",  # Duplicate
        ]
        
        unique_games = remove_duplicates(games)
        assert len(unique_games) == 3
        assert unique_games[0] == "#13,a1b2c3 b"
        assert unique_games[1] == "#13,d4e5f6 r"
        assert unique_games[2] == "#13,g7h8i9 b"
    
    def test_split_games_into_chunks(self):
        """Test splitting games into chunks."""
        games = [f"#13,game{i} b" for i in range(25)]
        
        chunks = split_games_into_chunks(games, chunk_size=10)
        assert len(chunks) == 3
        assert len(chunks[0]) == 10
        assert len(chunks[1]) == 10
        assert len(chunks[2]) == 5
    
    def test_split_games_exact_chunk_size(self):
        """Test splitting games when total is exactly divisible by chunk size."""
        games = [f"#13,game{i} b" for i in range(20)]
        
        chunks = split_games_into_chunks(games, chunk_size=10)
        assert len(chunks) == 2
        assert len(chunks[0]) == 10
        assert len(chunks[1]) == 10
    
    def test_combine_and_clean_files(self):
        """Test the complete combine and clean process."""
        # Create test files with some duplicates
        content1 = """# Header 1
#13,a1b2c3 b
#13,d4e5f6 r
#13,g7h8i9 b
"""
        content2 = """# Header 2
#13,d4e5f6 r  # Duplicate
#13,j1k2l3 b
#13,m4n5o6 r
"""
        content3 = """# Header 3
#13,p7q8r9 b
#13,s1t2u3 r
"""
        
        self.create_test_trmph_file("file1.trmph", content1)
        self.create_test_trmph_file("file2.trmph", content2)
        self.create_test_trmph_file("file3.trmph", content3)
        
        # Process with small chunk size to test splitting
        combine_and_clean_files(self.input_dir, self.output_dir, chunk_size=3)
        
        # Check that output files were created
        chunk_files = list(self.output_dir.glob("cleaned_chunk_*.trmph"))
        assert len(chunk_files) == 3  # 7 unique games split into chunks of 3 (3, 3, 1)
        
        # Check chunks - we should have 3 chunks with 3, 3, and 1 games respectively
        chunk_files_sorted = sorted(chunk_files)
        
        # Check first chunk (should have 3 games)
        with open(chunk_files_sorted[0], 'r') as f:
            chunk1_content = f.read().strip().split('\n')
        assert len(chunk1_content) == 3
        
        # Check second chunk (should have 3 games)
        with open(chunk_files_sorted[1], 'r') as f:
            chunk2_content = f.read().strip().split('\n')
        assert len(chunk2_content) == 3
        
        # Check third chunk (should have 1 game)
        with open(chunk_files_sorted[2], 'r') as f:
            chunk3_content = f.read().strip().split('\n')
        assert len(chunk3_content) == 1
        
        # Check summary file
        summary_file = self.output_dir / "processing_summary.txt"
        assert summary_file.exists()
        
        with open(summary_file, 'r') as f:
            summary_content = f.read()
        assert "Total games extracted: 8" in summary_content
        assert "Unique games after deduplication: 7" in summary_content
        assert "Duplicates removed: 1" in summary_content
    
    def test_combine_and_clean_files_with_duplicates(self):
        """Test the complete process with actual duplicates."""
        # Create test files with duplicates
        content1 = """# Header 1
#13,a1b2c3 b
#13,d4e5f6 r
"""
        content2 = """# Header 2
#13,a1b2c3 b  # Duplicate
#13,g7h8i9 b
"""
        
        self.create_test_trmph_file("file1.trmph", content1)
        self.create_test_trmph_file("file2.trmph", content2)
        
        combine_and_clean_files(self.input_dir, self.output_dir, chunk_size=5)
        
        # Check that duplicates were removed
        chunk_files = list(self.output_dir.glob("cleaned_chunk_*.trmph"))
        assert len(chunk_files) == 1  # 3 unique games in one chunk
        
        with open(chunk_files[0], 'r') as f:
            chunk_content = f.read().strip().split('\n')
        assert len(chunk_content) == 3  # 3 unique games
        
        # Check summary shows duplicates were removed
        summary_file = self.output_dir / "processing_summary.txt"
        with open(summary_file, 'r') as f:
            summary_content = f.read()
        assert "Total games extracted: 4" in summary_content
        assert "Unique games after deduplication: 3" in summary_content
        assert "Duplicates removed: 1" in summary_content
    
    def test_empty_input_directory(self):
        """Test handling of empty input directory."""
        combine_and_clean_files(self.input_dir, self.output_dir)
        
        # Should not create any output files
        chunk_files = list(self.output_dir.glob("cleaned_chunk_*.trmph"))
        assert len(chunk_files) == 0
        
        # Summary should still be created
        summary_file = self.output_dir / "processing_summary.txt"
        assert summary_file.exists() 