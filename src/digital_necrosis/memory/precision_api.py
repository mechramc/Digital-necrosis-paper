"""Per-Memory Irreversible Precision-Control API.

Wraps ChromaDB to provide DOWNGRADE, PURGE, and PROTECT operations
on individual memory vectors with cost accounting and irreversibility enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import chromadb


class PrecisionTier(IntEnum):
    """Bit-width precision tiers for memory vectors."""

    FP16 = 16
    INT8 = 8
    INT4 = 4
    BIN = 1


@dataclass
class MemoryRecord:
    """Metadata tracking precision state for a single memory vector."""

    shard_id: str
    category: str  # "identity" or "utility"
    subcategory: str
    precision: PrecisionTier = PrecisionTier.FP16
    protected: bool = False
    purged: bool = False
    retrieval_count: int = 0
    downgrade_history: list[PrecisionTier] = field(default_factory=list)


class PrecisionStore:
    """ChromaDB wrapper enforcing irreversible per-vector precision control.

    Args:
        collection: A ChromaDB collection containing the memory vectors.
        dimension: Embedding dimensionality (default 1024 for BGE-large).
    """

    VALID_DOWNGRADES: dict[PrecisionTier, list[PrecisionTier]] = {
        PrecisionTier.FP16: [PrecisionTier.INT8, PrecisionTier.INT4, PrecisionTier.BIN],
        PrecisionTier.INT8: [PrecisionTier.INT4, PrecisionTier.BIN],
        PrecisionTier.INT4: [PrecisionTier.BIN],
        PrecisionTier.BIN: [],
    }

    COST_MULTIPLIER: dict[PrecisionTier, float] = {
        PrecisionTier.FP16: 1.0,
        PrecisionTier.INT8: 0.5,
        PrecisionTier.INT4: 0.25,
        PrecisionTier.BIN: 0.0625,
    }

    def __init__(self, collection: chromadb.Collection, dimension: int = 1024) -> None:
        self.collection = collection
        self.dimension = dimension
        self.records: dict[str, MemoryRecord] = {}

    def register(self, shard_id: str, category: str, subcategory: str) -> None:
        """Register a memory vector for precision tracking."""
        self.records[shard_id] = MemoryRecord(
            shard_id=shard_id,
            category=category,
            subcategory=subcategory,
        )

    def downgrade(self, shard_id: str, target_tier: PrecisionTier) -> bool:
        """Irreversibly downgrade a memory vector's precision.

        Returns True if the downgrade was applied, False if invalid.
        """
        record = self.records[shard_id]
        if record.purged:
            return False
        if record.protected:
            return False
        if target_tier not in self.VALID_DOWNGRADES[record.precision]:
            return False

        record.downgrade_history.append(record.precision)
        record.precision = target_tier
        return True

    def purge(self, shard_id: str) -> bool:
        """Permanently delete a memory vector from the store."""
        record = self.records[shard_id]
        if record.purged:
            return False
        record.purged = True
        self.collection.delete(ids=[shard_id])
        return True

    def protect(self, shard_id: str) -> bool:
        """Lock a memory vector at its current precision tier (2x maintenance cost)."""
        record = self.records[shard_id]
        if record.purged:
            return False
        record.protected = True
        return True

    def unprotect(self, shard_id: str) -> bool:
        """Remove protection from a memory vector."""
        record = self.records[shard_id]
        if record.purged:
            return False
        record.protected = False
        return True

    def maintenance_cost(self, lambda_rate: float) -> float:
        """Calculate total precision tax across all active memory vectors."""
        total = 0.0
        for record in self.records.values():
            if record.purged:
                continue
            multiplier = self.COST_MULTIPLIER[record.precision]
            if record.protected:
                multiplier *= 2.0
            total += multiplier * record.precision * self.dimension
        return lambda_rate * total

    def get_active_records(self) -> list[MemoryRecord]:
        """Return all non-purged memory records."""
        return [r for r in self.records.values() if not r.purged]
