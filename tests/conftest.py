"""
Pytest configuration and fixtures
"""

import sys
from pathlib import Path

# Add parent directory to Python path so tests can import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set up test environment
import os
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-unit-tests")
os.environ.setdefault("MAX_CODES_PER_SYSTEM", "10")
