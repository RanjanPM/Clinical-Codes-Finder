#!/usr/bin/env python
"""
Simple wrapper for demo purposes
Allows typing: lookup diabetes
Instead of: python main.py "diabetes"
"""

import sys
import subprocess

if __name__ == "__main__":
    # Get all arguments after 'lookup' and join them
    query = " ".join(sys.argv[1:])
    
    if not query:
        print("Usage: lookup <clinical term>")
        print("\nExamples:")
        print("  lookup diabetes")
        print("  lookup blood sugar test")
        print("  lookup metformin 500 mg")
        sys.exit(1)
    
    # Run the actual main.py with the query
    result = subprocess.run(
        ["python", "main.py", query],
        cwd=".",
        capture_output=False  # Let output show directly
    )
    
    sys.exit(result.returncode)
