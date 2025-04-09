#!/usr/bin/env python3
"""
UTCHS Meta Pattern Detector Test Runner

This script specifically runs tests for the Meta Pattern Detector and infinite pattern
detection implementation in the UTCHS framework.
"""

import os
import sys
import time
import unittest
import subprocess
import importlib
import pytest
from pathlib import Path

# Add the project root directory to the Python path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

def check_dependencies():
    """Check that required dependencies are installed."""
    required_deps = ['numpy', 'pytest', 'scipy']
    missing = []
    
    for dep in required_deps:
        try:
            importlib.import_module(dep)
        except ImportError:
            missing.append(dep)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Please install them with pip before running tests.")
        return False
    
    return True

def run_meta_pattern_tests():
    """Run the UTCHS meta pattern detector tests."""
    print("Running UTCHS Meta Pattern Detector tests...")
    
    # Start with a simple import test to check for conflicts
    print("\n--- Testing basic imports ---")
    try:
        # Test that we can import the meta pattern detector module
        from utchs.core.meta_pattern_detector import MetaPatternDetector
        print("Successfully imported MetaPatternDetector")
        
    except Exception as e:
        print(f"Import test failed: {e}")
        print("Please fix import conflicts before running tests.")
        return False
    
    # Run the tests
    print("\n--- Running meta pattern detector tests ---")
    
    # Define test path
    test_path = Path("tests/test_meta_pattern_detector.py")
    
    # Run pytest
    result = pytest.main([
        "-xvs",
        str(test_path),
    ])
    
    if result == 0:
        print("All meta pattern detector tests passed!")
        return True
    else:
        print(f"Tests failed with code {result}")
        return False

def main():
    """Main function to run tests."""
    print("UTCHS Meta Pattern Detector Test Runner")
    print("--------------------------------------\n")
    
    # Check dependencies
    if not check_dependencies():
        print("Dependency check failed. Please fix the issues and try again.")
        return 1
    
    # Run tests
    if run_meta_pattern_tests():
        print("\nAll tests completed successfully!")
        return 0
    else:
        print("\nTests failed. Please fix the issues and try again.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 