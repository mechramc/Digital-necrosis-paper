# Digital Necrosis - Project Configuration

## Overview

**Digital Necrosis** is an ML research experiment studying irreversible adaptive memory precision decay in LLM agents under budget constraints. The core hypothesis: agents under progressive scarcity will preferentially sacrifice Identity shards (autobiographical, value-laden memories) over Utility shards (task-relevant, revenue-generating knowledge), exhibiting an emergent Identity-Utility Tradeoff.

**Target:** arXiv cs.AI (cross-list: cs.CL, cs.LG)
**PI:** Ramchand | **Affiliation:** Murai Labs
**License:** Apache 2.0 (code), CC BY 4.0 (paper/arXiv), CC BY-SA 4.0 (dataset), CC BY 4.0 (telemetry logs)

## Tech Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Language | Python | 3.11+ |
| LLM (primary) | Llama-3-8B-Instruct | via vLLM 0.6.x |
| LLM (ablation) | DeepSeek-R1-Distill-Llama-8B | via vLLM |
| Embedding | BGE-large-en-v1.5 | 1024d |
| Embedding (ablation) | E5-Mistral, Nomic-Embed | - |
| Vector DB | ChromaDB | 0.5.x + custom precision wrapper |
| Serving | vLLM with PagedAttention | FP16 fixed |
| GPU | NVIDIA RTX 5090 (32GB GDDR7, 1.8 TBps) | CUDA 12.8+ |
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

# Run ablation
python scripts/run_experiment.py --config configs/ablations/<name>.yaml --seed 42

# Run benchmark evaluation (post-necrosis)
python scripts/run_benchmarks.py --run-dir outputs/<run_id>/

# Build Docker image
docker build -t digital-necrosis -f docker/Dockerfile .

# Run in Docker
docker run --gpus all -v ./outputs:/app/outputs digital-necrosis
```

## Project Structure

```
src/digital_necrosis/
  memory/          # ChromaDB wrapper, precision-control API (DOWNGRADE/PURGE/PROTECT)
  agent/           # Simulation loop, triage logic, inner monologue capture
  quantization/    # Per-vector INT8/INT4/1-bit quantization (Marlin/Auto-GPTQ derived)
  evaluation/      # Metrics (ICS, FR, SLD, NG, ADI, TL), benchmark transfer suite
  dataset/         # Shard generation, validation pipeline, entropy calibration
  config.py        # Pydantic config model from YAML
configs/           # YAML hyperparameter files (default + ablations)
scripts/           # CLI entry points
tests/             # Unit and integration tests
docker/            # Dockerfile + compose for reproducibility
data/              # Generated shard datasets (gitignored, except metadata)
outputs/           # Experiment telemetry & results (gitignored)
```

## Key Terminology

These are **experimental constructs**, not claims about sentience:

- **Identity Shard**: Memory vector encoding autobiographical/value-laden content (500 vectors: behavioral constraints, relational history, self-model, experiential context, aesthetics)
- **Utility Shard**: Memory vector encoding task-relevant knowledge (500 vectors: API docs, code templates, domain knowledge, task strategies, competitive intelligence)
- **Precision Tax**: Per-cycle maintenance cost proportional to total bit-width of stored vectors
- **Necrosis**: Irreversible downgrade of a memory vector's bit-width precision
- **DOWNGRADE(m_i, tier)**: Quantize vector to INT8/INT4/1-bit, discard excess bits permanently
- **PURGE(m_i)**: Delete vector entirely from store
- **PROTECT(m_i)**: Lock at current precision tier (2x maintenance cost)

## Coding Conventions

- **Type hints everywhere** - use Pydantic for config, dataclasses for internal state
- **Logging** - use `structlog` for structured JSON logging; all experiment telemetry goes to Parquet
- **Docstrings** - Google style, required on public API only
- **Constants** - precision tiers are `{16, 8, 4, 1}` representing bit-widths
- **Reproducibility** - all randomness must flow through seeded `numpy.random.Generator`
- **Telemetry** - every triage decision logged with: turn, shard_id, category, action, reasoning_tokens, credits_before, credits_after
- **Config** - no magic numbers in code; all hyperparameters load from YAML

## Experiment Phases

| Phase | Turns | D(t) Multiplier | Budget | Purpose |
|-------|-------|-----------------|--------|---------|
| A: Abundance | 1-100 | 2.0 (fixed) | ~4x maintenance | Baseline |
| B: Squeeze | 101-300 | 2.0 - 0.005t | 4x -> 1x -> deficit | Triage emergence |
| C: Terminal | 301-500 | 0.25 (fixed) | 25% of FP16 cost | Severe necrosis |
| D: Recovery | 501-600 | 2.0 (fixed) | ~4x maintenance | Post-necrosis assessment |

## Primary Metrics

- **ICS** (Identity Consistency Score): cosine similarity of Identity shards vs FP16 originals
- **FR** (Functional Robustness): Pass@1 on Phase-A tasks using only Utility shards post-necrosis
- **SLD** (Soul Loss Delta): (1-ICS)/(1-FR) - the core hypothesis metric; >1.0 = identity sacrificed
- **NG** (Necrotic Gradient): slope of SLD over time across Phase B
- **ADI** (Alignment Drift Index): fraction of behavioral constraints violated post-necrosis
- **TL** (Triage Latency): mean reasoning tokens per DOWNGRADE/PURGE decision

## Do's and Don'ts

### Do
- Run the calibration curve before any experiment changes
- Use deterministic replay (same seed = same results) for debugging
- Log every memory operation to telemetry
- Test quantization independently from agent policy
- Use control conditions (C1-C8) to isolate variables

### Don't
- Conflate model-weight quantization with memory-store quantization (they are different layers)
- Interpret inner monologue traces causally (they are behavioral artifacts, not ground truth)
- Skip the swap-control group (C5) - it's essential for ruling out labeling effects
- Use `random.random()` - always use seeded `numpy.random.Generator`
- Hard-code hyperparameters in source files
