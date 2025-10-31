"""
Test Runner Script
Convenient script to run different test suites
"""

import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and display results"""
    print("\n" + "=" * 80)
    print(f"Running: {description}")
    print("=" * 80)
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def main():
    """Main test runner"""
    if len(sys.argv) < 2:
        print("Usage: python run_tests.py [option]")
        print("\nOptions:")
        print("  all         - Run all tests")
        print("  unit        - Run unit tests only")
        print("  integration - Run integration tests only")
        print("  fast        - Run fast unit tests only")
        print("  coverage    - Run all tests with coverage report")
        print("  api         - Run API integration tests (no OPENAI_API_KEY needed)")
        print("  style       - Check code style with black and flake8")
        print("  types       - Run type checking with mypy")
        print("\nExamples:")
        print("  python run_tests.py unit")
        print("  python run_tests.py coverage")
        sys.exit(1)
    
    option = sys.argv[1].lower()
    success = True
    
    if option == "all":
        success = run_command("pytest -v", "All Tests")
    
    elif option == "unit":
        success = run_command("pytest tests/test_units.py -v", "Unit Tests")
    
    elif option == "integration":
        success = run_command("pytest tests/test_integration.py -v -m integration", 
                             "Integration Tests")
    
    elif option == "fast":
        success = run_command("pytest tests/test_units.py -v -m 'not slow'", 
                             "Fast Unit Tests")
    
    elif option == "coverage":
        success = run_command(
            "pytest --cov=. --cov-report=html --cov-report=term-missing",
            "Tests with Coverage"
        )
        if success:
            print("\nCoverage report generated in: htmlcov/index.html")
    
    elif option == "api":
        success = run_command(
            "pytest tests/test_integration.py::TestClinicalTablesAPIIntegration -v",
            "API Integration Tests"
        )
    
    elif option == "style":
        print("\n" + "=" * 80)
        print("Checking Code Style")
        print("=" * 80)
        
        print("\n1. Running black...")
        black_ok = subprocess.run(["black", ".", "--check"]).returncode == 0
        
        print("\n2. Running flake8...")
        flake8_ok = subprocess.run(["flake8", "."]).returncode == 0
        
        success = black_ok and flake8_ok
    
    elif option == "types":
        success = run_command("mypy agents/ apis/ graph/", "Type Checking")
    
    else:
        print(f"Unknown option: {option}")
        print("Run 'python run_tests.py' for usage information")
        sys.exit(1)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
