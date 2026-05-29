"""Save a generated CAD prediction as a self-contained 'asset folder'.

Layout written by :func:`save_asset`::

    <asset_dir>/
        manifest.json
        code.py                  # CadQuery source that produced the asset
        model.step               # parametric solid (if available)
        model.stl                # tessellated mesh (if available)
        model.obj                # OBJ mesh derived from the STL
        model.mtl                # material file referenced by model.obj
        textures/
            albedo.png           # default PBR base-color texture
            view_0.png ...       # the 8 input renders, copied as reference
        views/
            view_0.png ...       # original input views (same as textures/views)

The OBJ/MTL/PNG triple lets common viewers (Blender, three.js, MeshLab,
Substance Painter, game engines) load the mesh together with a material —
which is what most users mean by "save the asset with materials and textures".
STEP is preserved for downstream CAD tooling.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from PIL import Image

_DEFAULT_TEXTURE_SIZE = 512
_DEFAULT_CHECKER_TILES = 16
_DEFAULT_BASE_COLOR = (200, 200, 205)
_DEFAULT_ACCENT_COLOR = (170, 170, 175)


@dataclass
class AssetPaths:
    asset_dir: Path
    manifest_path: Path
    code_path: Path | None = None
    step_path: Path | None = None
    stl_path: Path | None = None
    obj_path: Path | None = None
    mtl_path: Path | None = None
    albedo_path: Path | None = None
    view_paths: list[Path] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        return {
            k: (str(v) if isinstance(v, Path) else v)
            if not isinstance(v, list)
            else [str(p) for p in v]
            for k, v in d.items()
        }


def save_asset(
    out_dir: Path,
    *,
    name: str = "model",
    code: str | None = None,
    step_path: Path | None = None,
    stl_path: Path | None = None,
    views: Iterable[Image.Image] | None = None,
    material_name: str = "ZeroToCadMat",
) -> AssetPaths:
    """Bundle a generated prediction into a portable asset folder.

    Any of ``code`` / ``step_path`` / ``stl_path`` / ``views`` may be ``None``
    — only the components actually provided are written. STEP is copied
    verbatim; the OBJ + MTL + albedo PNG are derived from the STL (so the STL
    is required for the textured-mesh pieces).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    textures_dir = out_dir / "textures"
    views_dir = out_dir / "views"
    textures_dir.mkdir(exist_ok=True)
    views_dir.mkdir(exist_ok=True)

    paths = AssetPaths(asset_dir=out_dir, manifest_path=out_dir / "manifest.json")

    if code:
        paths.code_path = out_dir / "code.py"
        paths.code_path.write_text(code, encoding="utf-8")

    if step_path and Path(step_path).exists():
        paths.step_path = out_dir / f"{name}.step"
        shutil.copyfile(step_path, paths.step_path)

    if stl_path and Path(stl_path).exists():
        paths.stl_path = out_dir / f"{name}.stl"
        shutil.copyfile(stl_path, paths.stl_path)

        paths.albedo_path = textures_dir / "albedo.png"
        _write_checker_texture(paths.albedo_path)

        paths.mtl_path = out_dir / f"{name}.mtl"
        _write_default_mtl(
            paths.mtl_path,
            material_name=material_name,
            albedo_rel_path=paths.albedo_path.relative_to(out_dir).as_posix(),
        )

        paths.obj_path = out_dir / f"{name}.obj"
        _stl_to_obj(
            stl_in=paths.stl_path,
            obj_out=paths.obj_path,
            mtl_rel_path=paths.mtl_path.name,
            material_name=material_name,
        )

    if views:
        for i, img in enumerate(views):
            view_path = views_dir / f"view_{i}.png"
            tex_view_path = textures_dir / f"view_{i}.png"
            img.save(view_path)
            img.save(tex_view_path)
            paths.view_paths.append(view_path)

    manifest = {
        "name": name,
        "material": material_name,
        "components": {
            "code": _rel(paths.code_path, out_dir),
            "step": _rel(paths.step_path, out_dir),
            "stl": _rel(paths.stl_path, out_dir),
            "obj": _rel(paths.obj_path, out_dir),
            "mtl": _rel(paths.mtl_path, out_dir),
            "albedo": _rel(paths.albedo_path, out_dir),
            "views": [_rel(p, out_dir) for p in paths.view_paths],
        },
    }
    paths.manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    return paths


def _rel(p: Path | None, root: Path) -> str | None:
    if p is None:
        return None
    return Path(p).relative_to(root).as_posix()


def _write_checker_texture(path: Path) -> None:
    """Write a soft gray checkerboard PNG as a neutral default albedo."""
    size = _DEFAULT_TEXTURE_SIZE
    tiles = _DEFAULT_CHECKER_TILES
    tile_px = max(1, size // tiles)

    img = Image.new("RGB", (size, size), _DEFAULT_BASE_COLOR)
    pixels = img.load()
    assert pixels is not None
    for y in range(size):
        for x in range(size):
            if ((x // tile_px) + (y // tile_px)) % 2 == 0:
                pixels[x, y] = _DEFAULT_ACCENT_COLOR
    img.save(path)


def _write_default_mtl(
    path: Path, *, material_name: str, albedo_rel_path: str
) -> None:
    """Write a Wavefront MTL with a basic PBR-ish material + albedo map."""
    content = (
        f"# Zero-To-CAD default material\n"
        f"newmtl {material_name}\n"
        f"Ka 0.10 0.10 0.10\n"
        f"Kd 0.80 0.80 0.82\n"
        f"Ks 0.20 0.20 0.20\n"
        f"Ns 32.0\n"
        f"d 1.0\n"
        f"illum 2\n"
        f"map_Kd {albedo_rel_path}\n"
    )
    path.write_text(content, encoding="utf-8")


def _stl_to_obj(
    *, stl_in: Path, obj_out: Path, mtl_rel_path: str, material_name: str
) -> None:
    """Convert an STL mesh to a textured OBJ using pyvista for IO.

    Vertex normals are written when available; UVs use a trivial planar
    projection so the albedo texture is visible without a dedicated unwrap.
    """
    import numpy as np
    import pyvista as pv

    mesh = pv.read(str(stl_in))
    surface = mesh.extract_surface().triangulate()
    points = np.asarray(surface.points, dtype=float)
    faces_flat = np.asarray(surface.faces).reshape(-1, 4)
    if faces_flat.size == 0 or not np.all(faces_flat[:, 0] == 3):
        raise ValueError("Expected a triangulated mesh for OBJ export")
    tris = faces_flat[:, 1:4]

    surface.compute_normals(
        cell_normals=False, point_normals=True, inplace=True, auto_orient_normals=True
    )
    normals = np.asarray(surface.point_data["Normals"], dtype=float)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    extents = np.maximum(maxs - mins, 1e-9)
    uvs = (points[:, :2] - mins[:2]) / extents[:2]

    lines: list[str] = [
        "# Generated by zero_to_cad.export.asset",
        f"mtllib {mtl_rel_path}",
        f"o {obj_out.stem}",
    ]
    lines.extend(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in points)
    lines.extend(f"vt {u:.6f} {v:.6f}" for u, v in uvs)
    lines.extend(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}" for n in normals)
    lines.append(f"usemtl {material_name}")
    lines.append("s 1")
    for a, b, c in tris:
        ia, ib, ic = int(a) + 1, int(b) + 1, int(c) + 1
        lines.append(
            f"f {ia}/{ia}/{ia} {ib}/{ib}/{ib} {ic}/{ic}/{ic}"
        )

    obj_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
