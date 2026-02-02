#!/usr/bin/env python3
"""
Multi-Agent Goal Collection Planner
Simple wrapper to run the main entry point.
"""
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import main

if __name__ == "__main__":
    main()
