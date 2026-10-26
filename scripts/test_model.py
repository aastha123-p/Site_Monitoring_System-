"""Convenience runner for scripts/model_training/test_model.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model_training.test_model import main

if __name__ == "__main__":
    main()
