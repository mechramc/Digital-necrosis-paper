# Digital Necrosis - Project Status

## Current Phase
**Kickoff to Critical Path (P02/P03)**
Foundation baseline exists (config, precision API, metrics, tests). Next execution focus is quantization engine and store integration before simulation loop implementation.

## Overall Progress
- Atomic tasks total: **93**
- Done: **8**
- InProgress: **1**
- Blocked: **0**
- Planned: **84**
- Completion: **8.6%**

## Phase Progress Table
| Phase | Total Atomic Tasks | Done | InProgress | Blocked | Planned | Completion % |
|---|---:|---:|---:|---:|---:|---:|
| P01 Foundations | 7 | 5 | 1 | 0 | 1 | 71.4% |
| P02 Quantization | 6 | 0 | 0 | 0 | 6 | 0.0% |
| P03 Memory Store | 6 | 2 | 0 | 0 | 4 | 33.3% |
| P04 Simulation Loop | 10 | 0 | 0 | 0 | 10 | 0.0% |
| P05 Calibration | 5 | 0 | 0 | 0 | 5 | 0.0% |
| P06 Telemetry | 5 | 0 | 0 | 0 | 5 | 0.0% |
| P07 Controls C1-C8 | 11 | 0 | 0 | 0 | 11 | 0.0% |
| P08 Dataset | 8 | 0 | 0 | 0 | 8 | 0.0% |
| P09 Evaluation | 14 | 1 | 0 | 0 | 13 | 7.1% |
| P10 Analysis/Ablations | 14 | 0 | 0 | 0 | 14 | 0.0% |
| P11 Repro/Infra | 5 | 0 | 0 | 0 | 5 | 0.0% |
| P12 Paper Outputs | 4 | 0 | 0 | 0 | 4 | 0.0% |

## Critical Path Item Status
| Critical Path Item | Status | Owner | ETA | Blocking? |
|---|---|---|---|---|
| DN-P02-T001-S01 Quantization API | Planned | agent | Immediate | No |
| DN-P03-T002-S01 Chroma store wrapper | Planned | agent | After P02 API | No |
| DN-P03-T002-S02 Downgrade wiring | Planned | agent | After store wrapper | No |
| DN-P04-T003-S03 Turn orchestration loop | Planned | agent | After P03 integration | No |
| DN-P05-T001-S01 Calibration baseline | Planned | agent | After loop/store path | No |
| DN-P06-T001-S01 Telemetry schema | Planned | agent | After loop scaffold | No |

## Recent Completed Work
- `DN-P01-T001-S01` and `DN-P01-T001-S02`: config dataclass and YAML round-trip baseline in `src/digital_necrosis/config.py`.
- `DN-P01-T003-S01` and `DN-P01-T003-S02`: operator guidance and lessons workflow (`codex.md`, `docs/LESSONS.md`).
- `DN-P03-T001-S01` and `DN-P03-T001-S02`: precision tier semantics and purge/protect behavior in `src/digital_necrosis/memory/precision_api.py`.
- `DN-P09-T001-S01`: core metrics baseline in `src/digital_necrosis/evaluation/metrics.py`.
- Test baseline currently passes: `pytest tests -q` -> `25 passed`.

## Active Blockers
- No hard blockers at this time.
- Watch item: runtime environment observed at Python 3.10.11 while `pyproject.toml` requires Python 3.11+; track under `DN-P11-T001-S01`.

## Risks
| Risk | Impact | Mitigation | Status |
|---|---|---|---|
| Quantization fidelity mismatch vs assumptions | Could invalidate calibration and downstream claims | Execute P02 + P05 early and lock quality envelopes | Open |
| Loop economics bugs | Distorted triage behavior and invalid metrics | Implement deterministic loop tests before long runs | Open |
| Control conditions delayed | Weak causal claims in paper | Prioritize P07 after P04/P06 baseline | Open |
| Environment/version drift | Non-reproducible results across machines | Complete P11 reproducibility tasks before large runs | Open |

## Decisions Log
| Date | Decision | Rationale |
|---|---|---|
| 2026-02-18 | Use root-level `tasks.md`, `status.md`, `checkpoint.md` | Fast discovery for coding agents |
| 2026-02-18 | Use full-roadmap atomic backlog (P01-P12) | Avoid re-planning overhead per phase |
| 2026-02-18 | Use `Planned/InProgress/Blocked/Done` status model | Consistent, low-friction multi-agent workflow |
| 2026-02-18 | Update checkpoint every work session end | Reliable cross-agent continuity |
| 2026-02-18 | tasks.md v2 spec audit — expanded from 69 to 93 atomic tasks | Closed gaps in controls, forensics, stats, dataset validation, ablations |

## Next 7 Priorities
1. Start `DN-P02-T001-S01` quantization API implementation.
2. Execute `DN-P02-T001-S02` INT8 path and tests.
3. Execute `DN-P02-T001-S03` INT4 path and tests.
4. Execute `DN-P02-T001-S04` BIN path and tests.
5. Complete `DN-P02-T002-S01` quantization test suite.
6. Start `DN-P03-T002-S01` Chroma store wrapper.
7. Start `DN-P03-T002-S02` downgrade integration with quant engine.
