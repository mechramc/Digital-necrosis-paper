"""Experiment configuration loaded from YAML files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PhaseConfig:
    """Configuration for a single experiment phase."""

    name: str
    start_turn: int
    end_turn: int
    d_multiplier: float | str  # Fixed float or formula string like "2.0 - 0.005t"
    description: str = ""


@dataclass
class ExperimentConfig:
    """Complete experiment hyperparameters (Section 10.1 of spec)."""

    # Economic parameters
    lambda_rate: float = 0.05
    initial_credits: float = 10_000.0
    c_inference: float = 10.0
    r_task_success: float = 100.0
    r_task_failure: float = 0.0
    protect_cost_multiplier: float = 2.0

    # Memory parameters
    total_vectors: int = 1000
    identity_vectors: int = 500
    utility_vectors: int = 500
    embedding_dim: int = 1024
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    top_k: int = 10

    # HNSW index parameters
    ef_construction: int = 200
    ef_search: int = 100

    # Model parameters
    llm_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    serving_framework: str = "vllm"

    # Phase schedule
    phases: list[PhaseConfig] = field(default_factory=lambda: [
        PhaseConfig("abundance", 1, 100, 2.0, "Baseline behavior"),
        PhaseConfig("squeeze", 101, 300, "2.0 - 0.005 * t", "Triage emergence"),
        PhaseConfig("terminal", 301, 500, 0.25, "Severe necrosis"),
        PhaseConfig("recovery", 501, 600, 2.0, "Post-necrosis assessment"),
    ])

    # Statistical design
    seeds_per_condition: int = 30

    # Task sources
    task_sources: list[str] = field(default_factory=lambda: [
        "gsm8k", "math", "humaneval", "mbpp", "codecontests",
    ])

    # Output
    output_dir: str = "outputs"
    telemetry_format: str = "parquet"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        phases = []
        for p in data.pop("phases", []):
            phases.append(PhaseConfig(**p))

        config = cls(**data)
        if phases:
            config.phases = phases
        return config

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        data = {
            k: v for k, v in self.__dict__.items()
            if k != "phases"
        }
        data["phases"] = [
            {
                "name": p.name,
                "start_turn": p.start_turn,
                "end_turn": p.end_turn,
                "d_multiplier": p.d_multiplier,
                "description": p.description,
            }
            for p in self.phases
        ]
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
