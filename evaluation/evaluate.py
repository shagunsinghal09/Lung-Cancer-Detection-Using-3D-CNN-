"""Standalone evaluation helper."""

import json

from training.train import run_dummy_training


if __name__ == "__main__":
    print(json.dumps(run_dummy_training(), indent=2))
