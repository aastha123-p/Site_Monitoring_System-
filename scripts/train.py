"""Convenience runner for scripts/model_training/train_baseline.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model_training.train_baseline import train_model

if __name__ == "__main__":
    train_model()
