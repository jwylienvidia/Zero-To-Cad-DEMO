"""Tests for lazy parquet store."""

from __future__ import annotations

from pathlib import Path

from zero_to_cad.dataset.parquet_store import ParquetStore


def test_refresh_index_and_get_row(synthetic_parquet_dir: Path) -> None:
    store = ParquetStore(synthetic_parquet_dir)
    count = store.refresh_index()
    assert count == 2

    uuids = store.list_uuids()
    assert "test-uuid-001" in uuids
    assert "test-uuid-002" in uuids

    row = store.get_row("test-uuid-001")
    assert row.uuid == "test-uuid-001"
    assert len(row.views) == 8
    assert all(v.size == (64, 64) for v in row.views)
    assert "cadquery" in row.cadquery_code.lower()
    assert row.num_faces == 6


def test_export_row(synthetic_parquet_dir: Path, tmp_path: Path) -> None:
    store = ParquetStore(synthetic_parquet_dir)
    store.refresh_index()
    out = store.export_row("test-uuid-001", tmp_path)

    assert out.is_dir()
    assert (out / "code.py").exists()
    assert (out / "view_0.png").exists()
    assert (out / "view_7.png").exists()
