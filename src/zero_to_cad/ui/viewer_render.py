"""Offscreen PyVista rendering for the image-based 3D viewer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraState:
    azimuth: float = 45.0
    elevation: float = 30.0
    zoom: float = 0.85

    def reset(self) -> None:
        self.azimuth = 45.0
        self.elevation = 30.0
        self.zoom = 0.85


# Edges sharper than this angle (degrees) are treated as creases and kept crisp;
# anything smoother is averaged for clean shading on curved surfaces.
_FEATURE_ANGLE = 30.0


def prepare_mesh_for_display(mesh):
    """Clean a tessellated CAD mesh and recompute well-behaved normals.

    Tessellated STL/STEP meshes carry per-triangle (unmerged) vertices, so naive
    smooth shading averages normals across coincident points and across sharp
    edges, producing the muddy/uneven look. We merge coincident points, then
    recompute normals with a feature angle so flat faces stay flat and only truly
    curved surfaces get smoothed.
    """
    import pyvista as pv

    try:
        if isinstance(mesh, pv.PolyData):
            surface = mesh
        elif hasattr(mesh, "extract_surface"):
            surface = mesh.extract_surface()
        else:
            surface = mesh
    except Exception:
        surface = mesh

    try:
        cleaned = surface.clean(
            point_merging=True,
            tolerance=1e-6,
            lines_to_points=False,
            polys_to_lines=False,
            strips_to_polys=False,
        )
    except Exception:
        cleaned = surface

    try:
        return cleaned.compute_normals(
            cell_normals=True,
            point_normals=True,
            split_vertices=True,
            feature_angle=_FEATURE_ANGLE,
            consistent_normals=True,
            auto_orient_normals=True,
            non_manifold_traversal=True,
        )
    except Exception:
        return cleaned


def render_mesh_to_array(
    mesh,
    *,
    show_edges: bool = False,
    camera: CameraState | None = None,
    window_size: tuple[int, int] = (1024, 768),
) -> np.ndarray:
    """Render a PyVista mesh offscreen and return an RGB uint8 image."""
    import pyvista as pv

    cam = camera or CameraState()
    w, h = window_size

    display_mesh = prepare_mesh_for_display(mesh)

    plotter = pv.Plotter(off_screen=True, window_size=(w, h), lighting="three lights")
    try:
        plotter.set_background("white")
        plotter.add_mesh(
            display_mesh,
            show_edges=show_edges,
            color="lightsteelblue",
            smooth_shading=True,
            specular=0.25,
            specular_power=15,
            ambient=0.2,
            diffuse=0.8,
        )
        plotter.show_axes()
        plotter.show_grid()
        plotter.view_isometric()
        plotter.camera.azimuth = cam.azimuth
        plotter.camera.elevation = cam.elevation
        plotter.camera.zoom(cam.zoom)
        return np.asarray(plotter.screenshot(return_img=True))
    finally:
        plotter.close()
