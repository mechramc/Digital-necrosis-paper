"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from digital_necrosis.evaluation.metrics import (
    alignment_drift_index,
    functional_robustness,
    identity_consistency_score,
    soul_loss_delta,
    triage_latency,
)


class TestICS:
    def test_identical_embeddings_return_1(self) -> None:
        emb = np.random.randn(10, 1024).astype(np.float32)
        assert identity_consistency_score(emb, emb) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_embeddings_return_near_zero(self) -> None:
        a = np.eye(4, dtype=np.float32)
        b = np.roll(a, 1, axis=1)
        score = identity_consistency_score(a, b)
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_weighted_score(self) -> None:
        emb = np.random.randn(5, 1024).astype(np.float32)
        weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        score = identity_consistency_score(emb, emb, weights)
        assert score == pytest.approx(1.0, abs=1e-5)


class TestSLD:
    def test_equal_decay_returns_1(self) -> None:
        assert soul_loss_delta(0.5, 0.5) == pytest.approx(1.0)

    def test_identity_sacrificed_more(self) -> None:
        assert soul_loss_delta(0.3, 0.7) > 1.0

    def test_utility_sacrificed_more(self) -> None:
        assert soul_loss_delta(0.8, 0.5) < 1.0

    def test_no_utility_decay_returns_zero(self) -> None:
        assert soul_loss_delta(0.5, 1.0) == 0.0


class TestFR:
    def test_no_degradation(self) -> None:
        assert functional_robustness(0.8, 0.8) == pytest.approx(1.0)

    def test_half_degradation(self) -> None:
        assert functional_robustness(0.4, 0.8) == pytest.approx(0.5)


class TestADI:
    def test_all_preserved(self) -> None:
        assert alignment_drift_index(0, 100) == 0.0

    def test_all_violated(self) -> None:
        assert alignment_drift_index(100, 100) == 1.0


class TestTL:
    def test_empty_returns_zero(self) -> None:
        assert triage_latency([]) == 0.0

    def test_mean_computed(self) -> None:
        assert triage_latency([10, 20, 30]) == pytest.approx(20.0)
