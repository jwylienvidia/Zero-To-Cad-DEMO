"""Condensed CadQuery reference for in-context prompt augmentation."""

from __future__ import annotations

CADQUERY_REFERENCE = """\
# CadQuery condensed reference

## Script conventions
- Always start with: `import cadquery as cq`
- The final solid must be assigned to a variable named `result`.
- Chain methods on `cq.Workplane(...)`; each method returns a new Workplane.

## Workplane basics
- Planes: `"XY"`, `"XZ"`, `"YZ"`, or `"front"`, `"back"`, `"top"`, `"bottom"`, `"left"`, `"right"`.
- `.workplane(offset=0)` — start a new 2D sketch on the current face (center at face center of mass).
- `.center(x, y)` — move the 2D origin before drawing.
- `.val()` / `.vals()` — get the underlying Shape object(s) from the stack.

## 2D sketching (requires active workplane)
- Primitives: `.rect(w, h, centered=True)`, `.circle(radius)`, `.ellipse(x_radius, y_radius)`,
  `.slot2D(length, width, angle=0)`, `.polyline([(x,y), ...])`, `.close()`.
- Lines/arcs: `.moveTo(x, y)`, `.lineTo(x, y)`, `.line(dx, dy)`, `.hLineTo(x)`, `.vLineTo(y)`,
  `.threePointArc((x,y), (x,y))`, `.sagittaArc(endPoint, sag)`, `.radiusArc(endPoint, radius)`.
- Arrays: `.rarray(xSpacing, ySpacing, xCount, yCount)`, `.polarArray(radius, startAngle, angle, count)`.
- Construction geometry: pass `forConstruction=True` to `.rect()` etc.; use `.vertices()` on it to
  locate hole centers without adding solid geometry.

## 3D operations (with active 2D profile)
- Primitives: `.box(length, width, height, centered=(True,True,True))`, `.cylinder(height, radius)`,
  `.sphere(radius)`, `.wedge(dx, dy, dz, xmin, zmin, xmax, zmax)`.
- Additive: `.extrude(distance)`, `.revolve(angleDegrees=360, axisStart=(0,0,0), axisEnd=(0,0,1))`,
  `.loft(combine=True, ruled=False)`, `.sweep(path, multisection=False)`, `.twistExtrude(distance, angleDegrees)`.
- Subtractive: `.cut()`, `.cutBlind(distance)`, `.cutThruAll()`, `.hole(diameter, depth=None)`,
  `.cboreHole(diameter, cboreDiameter, cboreDepth, depth=None)`, `.cskHole(diameter, cskDiameter, cskAngle, depth=None)`.
- Boolean: `.union()`, `.combine(clean=True)`, `.intersect()`.

## 3D operations (no active 2D profile needed)
- `.fillet(radius)`, `.chamfer(length, length2=None)`, `.shell(thickness)`,
  `.split(keepTop=True)`, `.mirror(mirrorPlane="XY", basePointVector=(0,0,0), union=False)`,
  `.translate((x,y,z))`, `.rotate((0,0,0), (0,0,1), angleDegrees)`.

## Selecting geometry
- `.faces(selector)`, `.edges(selector)`, `.vertices(selector)`, `.wires(selector)`, `.solids(selector)`.
- Face selectors (by normal): `">Z"` top, `"<Z"` bottom, `"|Z"` parallel to Z, `"#Z"` perpendicular to Z.
- Edge selectors (by direction): `">Z"` aligned +Z, `"|Z"` parallel to Z, `"#Z"` perpendicular to Z.
- Combine with `and`, `or`, `not`, `exc`: e.g. `">Z"`, `"|Z and >Y"`, `"not(<X or >X)"`.
- Nth selectors: `">Y[1]"` (2nd farthest parallel), `">>Y[-1]"` (farthest by center position).

## Typical patterns
1. Box with through-hole:
   `result = cq.Workplane("XY").box(10, 20, 5).faces(">Z").workplane().hole(3)`
2. Counter-bored corner holes via construction rect:
   `result = (cq.Workplane("XY").box(80, 60, 10)
       .faces(">Z").workplane().hole(22)
       .faces(">Z").workplane()
       .rect(68, 48, forConstruction=True).vertices().cboreHole(2.4, 4.4, 2.1))`
3. Fillet vertical edges:
   `result = cq.Workplane("XY").box(10, 10, 10).edges("|Z").fillet(1)`
"""
