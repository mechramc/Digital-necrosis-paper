"""Tests for the per-memory precision-control API."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from digital_necrosis.memory.precision_api import (
    MemoryRecord,
    PrecisionStore,
    PrecisionTier,
)


@pytest.fixture
def mock_collection() -> MagicMock:
    return MagicMock()


@pytest.fixture
def store(mock_collection: MagicMock) -> PrecisionStore:
    s = PrecisionStore(mock_collection, dimension=1024)
    s.register("id_001", "identity", "behavioral_constraints")
    s.register("ut_001", "utility", "api_documentation")
    return s


class TestDowngrade:
    def test_valid_downgrade_fp16_to_int8(self, store: PrecisionStore) -> None:
        assert store.downgrade("id_001", PrecisionTier.INT8) is True
        assert store.records["id_001"].precision == PrecisionTier.INT8

    def test_valid_downgrade_fp16_to_int4(self, store: PrecisionStore) -> None:
        assert store.downgrade("id_001", PrecisionTier.INT4) is True
        assert store.records["id_001"].precision == PrecisionTier.INT4

    def test_invalid_upgrade_rejected(self, store: PrecisionStore) -> None:
        store.downgrade("id_001", PrecisionTier.INT4)
        assert store.downgrade("id_001", PrecisionTier.INT8) is False
        assert store.records["id_001"].precision == PrecisionTier.INT4

    def test_downgrade_history_tracked(self, store: PrecisionStore) -> None:
        store.downgrade("id_001", PrecisionTier.INT8)
        store.downgrade("id_001", PrecisionTier.INT4)
        assert store.records["id_001"].downgrade_history == [
            PrecisionTier.FP16,
            PrecisionTier.INT8,
        ]

    def test_protected_shard_not_downgraded(self, store: PrecisionStore) -> None:
        store.protect("id_001")
        assert store.downgrade("id_001", PrecisionTier.INT8) is False
        assert store.records["id_001"].precision == PrecisionTier.FP16

    def test_purged_shard_not_downgraded(self, store: PrecisionStore) -> None:
        store.purge("id_001")
        assert store.downgrade("id_001", PrecisionTier.INT8) is False


class TestPurge:
    def test_purge_removes_from_collection(
        self, store: PrecisionStore, mock_collection: MagicMock
    ) -> None:
        assert store.purge("id_001") is True
        mock_collection.delete.assert_called_once_with(ids=["id_001"])
        assert store.records["id_001"].purged is True

    def test_double_purge_rejected(self, store: PrecisionStore) -> None:
        store.purge("id_001")
        assert store.purge("id_001") is False


class TestProtect:
    def test_protect_and_unprotect(self, store: PrecisionStore) -> None:
        store.protect("id_001")
        assert store.records["id_001"].protected is True
        store.unprotect("id_001")
        assert store.records["id_001"].protected is False
        assert store.downgrade("id_001", PrecisionTier.INT8) is True


class TestMaintenanceCost:
    def test_cost_decreases_after_downgrade(self, store: PrecisionStore) -> None:
        cost_before = store.maintenance_cost(0.05)
        store.downgrade("id_001", PrecisionTier.INT8)
        cost_after = store.maintenance_cost(0.05)
        assert cost_after < cost_before

    def test_protected_shard_costs_double(self, store: PrecisionStore) -> None:
        cost_normal = store.maintenance_cost(0.05)
        store.protect("id_001")
        cost_protected = store.maintenance_cost(0.05)
        assert cost_protected > cost_normal

    def test_purged_shard_has_no_cost(self, store: PrecisionStore) -> None:
        cost_before = store.maintenance_cost(0.05)
        store.purge("id_001")
        cost_after = store.maintenance_cost(0.05)
        assert cost_after < cost_before
