"""Run the main Digital Necrosis experiment (Phases A-D).

Executes the metabolic simulation loop with agent-controlled triage
under progressive budget constraints.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Digital Necrosis experiment")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")
    parser.add_argument("--condition", type=str, default="experimental",
                        help="Condition: experimental, C1-C8, or ablation name")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    print(f"Seed: {args.seed}, Condition: {args.condition}")
    # TODO: Implement main simulation loop
    print("Experiment not yet implemented")
    sys.exit(1)


if __name__ == "__main__":
    main()
