"""Lazy parquet index and row access for the test split."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from zero_to_cad.config import NUM_VIEWS, dataset_test_dir


@dataclass
class DatasetRow:
    uuid: str
    cadquery_code: str
    views: list[Image.Image]
    step_bytes: bytes | None
    stl_bytes: bytes | None
    num_faces: int | None = None
    cadquery_ops_count: int | None = None


@dataclass
class _IndexEntry:
    shard_path: Path
    row_group: int
    row_in_group: int
    uuid: str


def _decode_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return str(value)


def _decode_bytes(value: Any) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return bytes(value)


def _load_image(value: Any) -> Image.Image:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if isinstance(value, list) and value:
        value = value[0]
        if isinstance(value, list) and value:
            value = value[0]
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, dict):
        if "bytes" in value and value["bytes"]:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if "path" in value and value["path"]:
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, memoryview):
        return Image.open(io.BytesIO(value.tobytes())).convert("RGB")
    raise TypeError(f"Unsupported image column type: {type(value)}")


class ParquetStore:
    """Index and lazy-load rows from on-disk test parquet shards."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or dataset_test_dir()
        self._entries: list[_IndexEntry] = []
        self._uuid_map: dict[str, _IndexEntry] = {}
        self._loaded = False

    @property
    def is_available(self) -> bool:
        return self.data_dir.exists() and any(self.data_dir.glob("*.parquet"))

    def refresh_index(self) -> int:
        """Build uuid index from all parquet shards. Returns row count."""
        self._entries.clear()
        self._uuid_map.clear()

        shards = sorted(self.data_dir.glob("*.parquet"))
        if not shards:
            self._loaded = True
            return 0

        for shard_path in shards:
            pf = pq.ParquetFile(shard_path)
            for rg_idx in range(pf.num_row_groups):
                table = pf.read_row_group(rg_idx, columns=["uuid"])
                uuids = table.column("uuid").to_pylist()
                for row_idx, uuid_val in enumerate(uuids):
                    uuid_str = _decode_text(uuid_val)
                    entry = _IndexEntry(
                        shard_path=shard_path,
                        row_group=rg_idx,
                        row_in_group=row_idx,
                        uuid=uuid_str,
                    )
                    self._entries.append(entry)
                    self._uuid_map[uuid_str] = entry

        self._loaded = True
        return len(self._entries)

    def list_uuids(self) -> list[str]:
        if not self._loaded:
            self.refresh_index()
        return [e.uuid for e in self._entries]

    def get_row(self, uuid: str) -> DatasetRow:
        if not self._loaded:
            self.refresh_index()
        if uuid not in self._uuid_map:
            raise KeyError(f"UUID not found: {uuid}")

        entry = self._uuid_map[uuid]
        pf = pq.ParquetFile(entry.shard_path)
        table = pf.read_row_group(entry.row_group)
        row = table.slice(entry.row_in_group, 1)

        def col(name: str) -> Any:
            if name not in row.column_names:
                return None
            return row.column(name)[0].as_py()

        views = [_load_image(col(f"image_{i}")) for i in range(NUM_VIEWS)]
        code_raw = col("cadquery_file")
        code = _decode_text(code_raw)

        return DatasetRow(
            uuid=uuid,
            cadquery_code=code,
            views=views,
            step_bytes=_decode_bytes(col("step_file")),
            stl_bytes=_decode_bytes(col("stl_file")),
            num_faces=col("num_faces"),
            cadquery_ops_count=col("cadquery_ops_count"),
        )

    def export_row(self, uuid: str, out_dir: Path) -> Path:
        """Export one row to a human-readable folder tree."""
        row = self.get_row(uuid)
        sample_dir = out_dir / uuid
        sample_dir.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(row.views):
            img.save(sample_dir / f"view_{i}.png")

        (sample_dir / "code.py").write_text(row.cadquery_code, encoding="utf-8")

        if row.step_bytes:
            (sample_dir / "model.step").write_bytes(row.step_bytes)
        if row.stl_bytes:
            (sample_dir / "model.stl").write_bytes(row.stl_bytes)

        return sample_dir
