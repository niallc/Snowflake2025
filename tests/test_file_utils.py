import pytest
from pathlib import Path
import tempfile
import time
from hex_ai.file_utils import get_unique_checkpoint_path


def test_get_unique_checkpoint_path_new_file():
    """Test that get_unique_checkpoint_path returns the original path for new files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "epoch1_mini1.pt.gz"
        unique_path = get_unique_checkpoint_path(base_path)
        assert unique_path == base_path


def test_get_unique_checkpoint_path_existing_file():
    """Test that get_unique_checkpoint_path appends timestamp for existing files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "epoch1_mini1.pt.gz"
        
        # Create the file
        base_path.touch()
        
        # Get unique path
        unique_path = get_unique_checkpoint_path(base_path)
        
        # Should be different from base path
        assert unique_path != base_path
        
        # Should have timestamp format yymmdd_hrmin
        stem = base_path.stem
        suffix = base_path.suffix
        expected_pattern = f"{stem}_\\d{{6}}_\\d{{4}}{suffix}"
        
        import re
        assert re.match(expected_pattern, unique_path.name), f"Path {unique_path.name} doesn't match expected pattern"


def test_get_unique_checkpoint_path_double_collision():
    """Test that get_unique_checkpoint_path handles double collisions by adding seconds."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "epoch1_mini1.pt.gz"
        
        # Create the base file
        base_path.touch()
        
        # Create a file with timestamp that would be generated
        from datetime import datetime
        timestamp = datetime.now().strftime("%y%m%d_%H%M")
        collision_path = base_path.parent / f"{base_path.stem}_{timestamp}{base_path.suffix}"
        collision_path.touch()
        
        # Get unique path - should add seconds
        unique_path = get_unique_checkpoint_path(base_path)
        
        # Should be different from both base and collision paths
        assert unique_path != base_path
        assert unique_path != collision_path
        
        # Should have timestamp format yymmdd_hrminss
        stem = base_path.stem
        suffix = base_path.suffix
        expected_pattern = f"{stem}_\\d{{6}}_\\d{{6}}{suffix}"
        
        import re
        assert re.match(expected_pattern, unique_path.name), f"Path {unique_path.name} doesn't match expected pattern"


def test_get_unique_checkpoint_path_with_different_extensions():
    """Test that get_unique_checkpoint_path works with different file extensions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test with .pt extension
        base_path = Path(tmpdir) / "epoch1_mini1.pt"
        base_path.touch()
        unique_path = get_unique_checkpoint_path(base_path)
        assert unique_path != base_path
        assert unique_path.suffix == ".pt"
        
        # Test with .pth extension
        base_path2 = Path(tmpdir) / "epoch1_mini1.pth"
        base_path2.touch()
        unique_path2 = get_unique_checkpoint_path(base_path2)
        assert unique_path2 != base_path2
        assert unique_path2.suffix == ".pth"


def test_get_unique_checkpoint_path_pt_collides_with_existing_pt_gz():
    """A base .pt path should collide with an existing .pt.gz checkpoint sibling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "epoch3_mini19.pt"
        compressed_existing = Path(tmpdir) / "epoch3_mini19.pt.gz"
        compressed_existing.touch()

        unique_path = get_unique_checkpoint_path(base_path)

        assert unique_path != base_path
        assert unique_path.suffix == ".pt"
        assert unique_path.name.startswith("epoch3_mini19_")


def test_get_unique_checkpoint_path_pt_gz_collides_with_existing_pt():
    """A base .pt.gz path should collide with an existing .pt checkpoint sibling."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir) / "epoch3_mini19.pt.gz"
        uncompressed_existing = Path(tmpdir) / "epoch3_mini19.pt"
        uncompressed_existing.touch()

        unique_path = get_unique_checkpoint_path(base_path)

        assert unique_path != base_path
        assert unique_path.name.startswith("epoch3_mini19.pt_")
        assert unique_path.suffix == ".gz"
