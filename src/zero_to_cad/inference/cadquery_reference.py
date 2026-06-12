"""Condensed CadQuery reference for in-context prompt augmentation."""

from __future__ import annotations

CADQUERY_REFERENCE = """\
# CadQuery condensed reference

## Script conventions
- Always start with: `import cadquery as cq`
- The final solid must be assigned to a variable named `result`.
- Chain methods on `cq.Workplane(...)`; each method returns a new Workplane.
- Use named parameters for dimensions; keep operations in logical build order.

## Workplane basics
- Planes: `"XY"`, `"XZ"`, `"YZ"`, or `"front"`, `"back"`, `"top"`, `"bottom"`, `"left"`, `"right"`.
- `.workplane(offset=0)` — start a 2D sketch on the current face (center at face center of mass).
- `.workplane(centerOption="CenterOfMass")` — same, but explicitly anchor at face center of mass.
- `.center(x, y)` — move the 2D origin before drawing.
- `.pushPoints([(x, y), ...])` — place multiple points on the stack (e.g. for repeated holes).
- `.val()` / `.vals()` — get the underlying Shape object(s) from the stack.

## 2D sketching (requires active workplane)
- Primitives: `.rect(w, h, centered=True)`, `.circle(radius)`, `.ellipse(x_radius, y_radius)`,
  `.slot2D(length, width, angle=0)`, `.polyline([(x,y), ...])`, `.close()`.
- Lines/arcs: `.moveTo(x, y)`, `.lineTo(x, y)`, `.line(dx, dy)`, `.vLine(w)`, `.vLineTo(y)`,
  `.hLine(w)`, `.hLineTo(x)`, `.threePointArc((x,y), (x,y))`, `.sagittaArc(endPoint, sag)`,
  `.radiusArc(endPoint, radius)`.
- Symmetry: `.mirrorX()` / `.mirrorY()` — mirror the current wire about the workplane axis.
- Arrays: `.rarray(xSpacing, ySpacing, xCount, yCount)`, `.polarArray(radius, startAngle, angle, count)`.
- Offsets: `.offset2D(distance)` — offset the pending 2D wire inward/outward.
- Construction geometry: pass `forConstruction=True` to `.rect()` etc.; use `.vertices()` on it to
  locate hole centers without adding solid geometry.
- `.toPending()` — move selected wires/edges onto the pending stack for the next 2D operation.

## Sketch API (on a selected face)
- Start with `.faces(">Z").sketch()`, then draw with sketch methods, then `.finalize()` to return
  to Workplane and extrude/cut.
- Sketch ops: `.rect(w, h)`, `.circle(r)`, `.regularPolygon(r, n)`, `.polygon([(x,y), ...])`,
  `.vertices(tag="name")`, `.fillet(r)`, `.chamfer(l)`.
- Boolean modes on sketch faces: default additive; `mode="s"` subtract, `mode="i"` intersect.
- Tagging: `.tag("name")` then `.vertices(tag="name")` to select tagged geometry later.

## 3D operations (with active 2D profile)
- Primitives: `.box(length, width, height, centered=(True,True,True))`, `.cylinder(height, radius)`,
  `.sphere(radius)`, `.wedge(dx, dy, dz, xmin, zmin, xmax, zmax)`.
- Additive: `.extrude(distance, combine=True)`, `.revolve(angleDegrees=360, axisStart=(0,0,0), axisEnd=(0,0,1))`,
  `.loft(combine=True, ruled=False)`, `.sweep(path, transition="round")`, `.twistExtrude(distance, angleDegrees)`.
- Subtractive: `.cut()`, `.cutBlind(distance)`, `.cutThruAll()`, `.hole(diameter, depth=None)`,
  `.cboreHole(diameter, cboreDiameter, cboreDepth, depth=None)`, `.cskHole(diameter, cskDiameter, cskAngle, depth=None)`.
- Boolean: `.union()`, `.combine(clean=True)`, `.intersect()`.
- Use `.extrude(distance, combine=False)` to create a separate boss/shell piece, then `.cut()` it.

## 3D operations (no active 2D profile needed)
- `.fillet(radius)`, `.chamfer(length, length2=None)`, `.shell(thickness)`,
  `.split(keepTop=True, keepBottom=True)`, `.mirror(mirrorPlane="XY", basePointVector=(0,0,0), union=False)`,
  `.translate((x,y,z))`, `.rotate((0,0,0), (0,0,1), angleDegrees)`.
- Fillet order matters on boxes: fillet vertical edges (`"|Z"`) and horizontal edges (`"#Z"`) in an
  order that avoids geometry failures (fillet larger-radius edges first when radii differ).

## Selecting geometry
- `.faces(selector)`, `.edges(selector)`, `.vertices(selector)`, `.wires(selector)`, `.solids(selector)`.
- Face selectors (by normal): `">Z"` top, `"<Z"` bottom, `"|Z"` parallel to Z, `"#Z"` perpendicular to Z.
- Edge selectors (by direction): `">Z"` aligned +Z, `"|Z"` parallel to Z, `"#Z"` perpendicular to Z.
- Type selectors: `"%CIRCLE"`, `"%LINE"`, `"%PLANE"`.
- Combine with `and`, `or`, `not`, `exc`: e.g. `">Z"`, `"|Z and >Y"`, `"not(<X or >X)"`, `"|Z or <Z"`.
- Nth selectors: `">Z[-2]"` (2nd top face), `">Y[1]"` (2nd farthest parallel), `">>Y[-1]"` (farthest by center).

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
4. Profile + mirror + extrude + shell (curved body):
   `result = (cq.Workplane("XY").center(-10, 0).vLine(3)
       .threePointArc((10, 4.5), (20, 3)).vLine(-3).mirrorX().extrude(30)
       .faces(">Z").workplane(centerOption="CenterOfMass").circle(3).extrude(2)
       .faces(">Z").shell(0.3))`
5. Hollow box (outer minus inner shell):
   `outer = cq.Workplane("XY").rect(100, 150).extrude(50).edges("|Z").fillet(10)
   inner = (outer.faces("<Z").workplane(3, True).rect(94, 144)
       .extrude(44, False).edges("|Z").fillet(7))
   result = outer.cut(inner)`
6. Sketch on face then extrude:
   `result = (cq.Workplane().box(5, 5, 1).faces(">Z").sketch()
       .regularPolygon(2, 6).regularPolygon(1.5, 6, mode="s")
       .vertices(tag="outer").fillet(0.2).finalize().extrude(0.5))`
7. Sweep profile along path:
   `path = cq.Workplane().polyline([(0,0), (50,0), (50,50), (0,50)])
   result = cq.Workplane("YZ").rect(4, 4).sweep(path, transition="round")`
"""
