# Lessons Log

Append-only implementation lessons for Digital Necrosis.

## Entry Template

- Date: YYYY-MM-DD
- Area: config | memory | quantization | agent | evaluation | infra | tooling
- Context: What changed or failed?
- Signal: What metric/test/log indicated the issue?
- Root Cause: Why did it happen?
- Decision: What was changed?
- Outcome: What improved or regressed?
- Action Item: Follow-up task (owner + due date)

---

## 2026-02-18 - Baseline Status

- Date: 2026-02-18
- Area: tooling
- Context: Initial repository baseline review and operator guide setup.
- Signal: `pytest` failed at import time without explicit package path; run CLIs are scaffolded with TODO exits.
- Root Cause: Package is in `src/` layout and local env was not installed in editable mode.
- Decision: Add `tests/conftest.py` to inject `src` into `sys.path` for test runs; keep implementation sequencing focused on quantization -> store -> loop.
- Outcome: Local test runs work without manual `PYTHONPATH` export.
- Action Item: Add tests for config validation, quantization fidelity, and loop credit accounting (owner: engineering, due: next implementation sprint).
