"""
Pytest configuration for scripts/tests.

This file is automatically loaded by pytest and handles path setup,
eliminating the need for sys.path hacks in test files.
"""
import sys
from pathlib import Path

# Add project root to path so imports work correctly
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
