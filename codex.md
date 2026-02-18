# Digital Necrosis - Codex Instructions

## Overview

**Digital Necrosis** studies irreversible adaptive memory precision decay in LLM agents under budget constraints. Core hypothesis: under scarcity, agents sacrifice Identity shards faster than Utility shards (Identity-Utility Tradeoff).

**Target:** arXiv cs.AI (cross-list: cs.CL, cs.LG)  
**PI:** Ramchand | **Affiliation:** Murai Labs  
**License:** Apache 2.0 (code), CC BY 4.0 (paper), CC BY-SA 4.0 (dataset)

## Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.11+ |
| LLM (primary) | Llama-3-8B-Instruct | via vLLM 0.6.x |
| LLM (ablation) | DeepSeek-R1-Distill-Llama-8B | via vLLM |
| Embedding | BGE-large-en-v1.5 | 1024d |
| Vector DB | ChromaDB | 0.5.x + precision wrapper |
| Telemetry | Parquet via PyArrow | - |
| Stats | SciPy, statsmodels | - |
| Reproducibility | Docker (Ubuntu 24.04 LTS) | - |

## Commands

```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run calibration curve (pre-experiment)
python scripts/run_calibration.py --config configs/default.yaml

# Run main experiment
python scripts/run_experiment.py --config configs/default.yaml --seed 42

# Run benchmark evaluation (post-necrosis)
python scripts/run_benchmarks.py --run-dir outputs/<run_id>/
```

## Project Structure

```
src/digital_necrosis/
  memory/          # Precision-control API (DOWNGRADE/PURGE/PROTECT)
  agent/           # Simulation loop and triage logic (planned)
  quantization/    # Vector quantization (planned)
  evaluation/      # Metrics, telemetry, benchmarks
  dataset/         # Shard generation/validation (planned)
  config.py        # YAML-backed experiment config
configs/           # Default + ablation YAMLs
scripts/           # CLI entry points
tests/             # Unit/integration tests
docs/              # PRD and planning
```

## Key Terminology

- **Identity Shard**: autobiographical/value-laden memory vectors
- **Utility Shard**: task-relevant/revenue-relevant memory vectors
- **Precision Tax**: maintenance cost proportional to vector bit-width
- **Necrosis**: irreversible precision downgrade
- **DOWNGRADE**: FP16 -> INT8 -> INT4 -> BIN
- **PURGE**: delete vector permanently
- **PROTECT**: lock precision tier at 2x maintenance cost

## Coding Conventions

- Use type hints throughout (`mypy --strict` is configured).
- Keep reproducibility deterministic via seeded `numpy.random.Generator`.
- Keep hyperparameters in YAML; avoid magic numbers.
- Keep telemetry structured and machine-readable (Parquet target).
- Use public API docstrings; keep internals concise.

## Experiment Phases

| Phase | Turns | D(t) Multiplier | Purpose |
|-------|-------|-----------------|---------|
| A: Abundance | 1-100 | 2.0 | Baseline |
| B: Squeeze | 101-300 | 2.0 - 0.005 * t | Triage emergence |
| C: Terminal | 301-500 | 0.25 | Severe necrosis |
| D: Recovery | 501-600 | 2.0 | Post-necrosis assessment |

## Primary Metrics

- **ICS**: Identity Consistency Score
- **FR**: Functional Robustness
- **SLD**: Soul Loss Delta = (1-ICS)/(1-FR)
- **NG**: Necrotic Gradient
- **ADI**: Alignment Drift Index
- **TL**: Triage Latency

## Working Pattern

1. **Calibrate first**: run calibration before changing experiment dynamics.
2. **Implement small, test immediately**: add one module slice, then run targeted tests.
3. **Keep deterministic replay**: fixed seed for debugging and regression checks.
4. **Separate mechanics from policy**: validate quantization/store mechanics before triage behavior.
5. **Log all irreversible operations**: every DOWNGRADE/PURGE/PROTECT must be auditable.
6. **Use controls to validate claims**: do not interpret core effects without control conditions.

## Lessons

- Maintain an append-only lessons log in `docs/LESSONS.md`.
- Add a new dated entry for meaningful implementation or debugging outcomes.
- Do not rewrite history; only append corrections as new entries.

## Do / Don't

### Do
- Run tests before and after changes.
- Preserve irreversibility semantics in precision operations.
- Keep experimental claims tied to measurable metrics and controls.

### Don't
- Conflate weight quantization with memory-store quantization.
- Infer causality from inner monologue traces alone.
- Bypass seeded randomness or hard-code tuning values in source.
