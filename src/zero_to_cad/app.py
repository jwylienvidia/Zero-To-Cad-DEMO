"""Application entry point."""

from __future__ import annotations

import os
import sys


def _configure_vtk_qt() -> None:
    """Use QOpenGLWidget for VTK on Qt6 (must run before pyvistaqt is imported)."""
    if os.environ.get("ZERO_TO_CAD_VIEWER", "image").strip().lower() != "vtk":
        return
    try:
        import vtkmodules.qt as vtk_qt

        vtk_qt.QVTKRWIBase = "QOpenGLWidget"
    except ImportError:
        pass


def main() -> None:
    os.environ.setdefault("QT_API", "pyside6")

    # Load saved API keys / endpoints into the environment before any config
    # (or backend) reads them. Environment values still take precedence.
    from zero_to_cad.settings import apply_settings_to_env

    apply_settings_to_env()

    _configure_vtk_qt()

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QSurfaceFormat
    from PySide6.QtWidgets import QApplication

    # VTK interactive backend only — image backend does not need these.
    if os.environ.get("ZERO_TO_CAD_VIEWER", "image").strip().lower() == "vtk":
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
        if sys.platform.startswith("linux"):
            QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL, True)

        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)
        fmt.setVersion(3, 2)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
        QSurfaceFormat.setDefaultFormat(fmt)

    from zero_to_cad.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Zero-To-CAD")
    app.setOrganizationName("zero-to-cad")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
