"""Shared pytest fixtures."""

from __future__ import annotations

import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image


@pytest.fixture
def synthetic_parquet_dir(tmp_path: Path) -> Path:
    """Create a minimal parquet shard matching dataset schema."""
    img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    code = b"""
import cadquery as cq
result = cq.Workplane("XY").box(10, 10, 10)
""".strip()

    data = {
        "uuid": ["test-uuid-001", "test-uuid-002"],
        "cadquery_file": [code, code],
        "num_faces": [6, 6],
        "cadquery_ops_count": [1, 1],
    }
    for i in range(8):
        data[f"image_{i}"] = [[img_bytes], [img_bytes]]

    data["step_file"] = [b"", b""]
    data["stl_file"] = [b"", b""]

    table = pa.table(data)
    shard = tmp_path / "test_shard.parquet"
    pq.write_table(table, shard)
    return tmp_path
