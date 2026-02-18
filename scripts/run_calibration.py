"""Run the pre-experiment quantization calibration curve (Section 6.4).

Measures the mechanical effect of quantization on retrieval fidelity,
independent of agent policy decisions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run quantization calibration curve")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output directory")
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    # TODO: Implement calibration curve
    # 1. For each of 1,000 vectors, compute top-10 neighbors at FP16 (ground truth)
    # 2. Quantize each vector to INT8, INT4, and 1-bit
    # 3. Recompute top-10 neighbors at each tier
    # 4. Report top-k overlap (Jaccard) and rank correlation (Kendall's tau)
    # 5. Report separately for Identity and Utility shards
    print("Calibration curve not yet implemented")
    sys.exit(1)


if __name__ == "__main__":
    main()
