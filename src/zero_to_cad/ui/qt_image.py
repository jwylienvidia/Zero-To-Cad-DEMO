"""Qt helpers for displaying numpy images (no VTK dependency)."""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage, QPixmap


def numpy_to_pixmap(image: np.ndarray) -> QPixmap:
    """Convert an RGB/RGBA numpy image to QPixmap."""
    if image.ndim != 3:
        raise ValueError(f"Expected HxWxC image, got shape {image.shape}")

    h, w, channels = image.shape
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if channels == 4:
        fmt = QImage.Format.Format_RGBA8888
        bytes_per_line = 4 * w
    elif channels == 3:
        fmt = QImage.Format.Format_RGB888
        bytes_per_line = 3 * w
    else:
        raise ValueError(f"Unsupported channel count: {channels}")

    qimage = QImage(image.data, w, h, bytes_per_line, fmt).copy()
    return QPixmap.fromImage(qimage)
