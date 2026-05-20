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

    plotter = pv.Plotter(off_screen=True, window_size=(w, h))
    try:
        plotter.set_background("white")
        plotter.add_mesh(
            mesh,
            show_edges=show_edges,
            color="lightsteelblue",
            smooth_shading=True,
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
