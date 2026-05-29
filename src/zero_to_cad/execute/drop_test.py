"""Build a multi-instance 'drop' scene from a single CadQuery script.

The user-supplied script is expected to assign a `cq.Workplane` (or any object
accepted by `cq.Assembly.add`) to a top-level variable named ``result``. We
re-run that script once to obtain the base solid, then arrange ``count`` copies
in a regular grid on the XY plane inside a ``cq.Assembly``. The resulting
compound is exposed again as ``result`` so the existing sandbox runner picks it
up unchanged.

This is intentionally a *placement* drop (no physics) — it's the cheapest way
to visually verify that the same generated solid behaves under repetition,
boolean compounding and STEP/STL export.
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass


@dataclass(frozen=True)
class DropTestParams:
    count: int = 8
    spacing: float | None = None  # None => auto from bounding box
    grid_cols: int | None = None  # None => ceil(sqrt(count))
    z_offset: float = 0.0

    def resolved_cols(self) -> int:
        if self.grid_cols and self.grid_cols > 0:
            return self.grid_cols
        return max(1, math.ceil(math.sqrt(max(1, self.count))))


def build_drop_test_code(base_code: str, params: DropTestParams) -> str:
    """Return a new CadQuery script that builds an N-copy grid of `result`.

    The output script:
      * runs ``base_code`` verbatim (must leave ``result`` defined),
      * captures that ``result`` as the base shape,
      * arranges ``params.count`` copies on a grid using ``cq.Assembly``,
      * exposes the compound as ``result`` for the sandbox to export.
    """
    if not base_code.strip():
        raise ValueError("base_code is empty")
    if params.count < 1:
        raise ValueError("count must be >= 1")

    cols = params.resolved_cols()
    spacing_expr = (
        "None" if params.spacing is None else f"{float(params.spacing)!r}"
    )

    wrapper = textwrap.dedent(
        f"""
        # --- drop test wrapper ---
        import cadquery as _cq_dt

        _dt_base = result
        _dt_count = {int(params.count)}
        _dt_cols = {int(cols)}
        _dt_spacing = {spacing_expr}
        _dt_z = {float(params.z_offset)!r}

        try:
            _dt_bb = _dt_base.val().BoundingBox()  # Workplane
        except AttributeError:
            _dt_bb = _dt_base.BoundingBox()        # Shape/Compound
        _dt_dx = max(_dt_bb.xlen, 1e-6)
        _dt_dy = max(_dt_bb.ylen, 1e-6)
        _dt_default_spacing = max(_dt_dx, _dt_dy) * 1.2
        _dt_step = _dt_default_spacing if _dt_spacing is None else float(_dt_spacing)

        _dt_assembly = _cq_dt.Assembly()
        for _dt_i in range(_dt_count):
            _dt_row, _dt_col = divmod(_dt_i, _dt_cols)
            _dt_loc = _cq_dt.Location(
                _cq_dt.Vector(_dt_col * _dt_step, _dt_row * _dt_step, _dt_z)
            )
            _dt_assembly.add(_dt_base, name=f"copy_{{_dt_i}}", loc=_dt_loc)

        result = _dt_assembly.toCompound()
        """
    ).strip()

    return f"{base_code.rstrip()}\n\n{wrapper}\n"
