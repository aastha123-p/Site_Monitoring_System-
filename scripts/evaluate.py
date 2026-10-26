"""Convenience runner for scripts/model_training/evaluate_baseline.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model_training.evaluate_baseline import main

if __name__ == "__main__":
    main()
