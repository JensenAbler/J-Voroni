# J-Voroni

J-Voroni is a procedural Voronoi and tributary-pattern geometry baker for
Autodesk Maya. It provides a responsive textured viewport preview and bakes
four complementary, slightly extruded meshes: red islands, green islands,
blue islands, and a genuine standalone edge network.

The geometry bake evaluates the mathematical field directly. It does not
convert preview pixels into vertices and does not create or evaluate a V-Ray
texture or V-Ray Scatter node.

## Requirements

- Autodesk Maya 2025.1 on Windows
- Python 3.11 (included with Maya 2025)
- PySide6 / Qt 6 (included with Maya 2025)
- Viewport 2.0

J-Voroni does not require Pillow, NumPy, SciPy, Shapely, OpenCV, or a GLSL
shader plug-in.

## Install and run

Clone or download this repository. In a **Python** tab of Maya's Script
Editor, replace the example path with your clone location and run:

```python
import runpy

runpy.run_path(r"C:\path\to\J-Voroni\run_voronoi_geometry_tool.py")
```

The launcher reloads the core and UI modules, so saved source changes can be
tested without restarting Maya.

## Interactive preview

Click **Create / Refresh Preview** once. J-Voroni creates a one-face Maya
plane with a `surfaceShader` and a temporary PNG file texture. Slider edits
are debounced and update the texture in Viewport 2.0.

The PNG is written with Qt's built-in `QImage`; Pillow is not used. The
preview is solely interactive feedback and is never used by the geometry
bake.

## Geometry bake

Click **Bake Four Meshes** to create a new group containing:

```text
voronoiGeoBake_001
|-- voronoiGeo_RED
|-- voronoiGeo_GREEN
|-- voronoiGeo_BLUE
`-- voronoiGeo_EDGE
```

The RGB objects are combined meshes whose disconnected shells are the inset
cell islands assigned to each color.

EDGE is not a backing plate. Its construction is:

1. Trace every inset RGB boundary by solving the mathematical ownership field.
2. Create an outer rectangular contour and register every RGB contour as a
   hole.
3. Ask Maya's mesh API for a hole-aware planar tessellation.
4. Build explicit bottom triangles, top triangles, outer walls, and cell-hole
   walls as one watertight `MFnMesh` solid.

The tool deliberately does not extrude the original face-with-holes because
Maya can cap those holes during polygon extrusion. Before accepting EDGE, the
baker validates its cap area and confirms that the completed solid has zero
open mesh edges.

Hiding RED, GREEN, and BLUE therefore reveals only the connected edge
channels, filled junctions, and outer border. The RGB tops and EDGE holes use
the same inset loops, so the four pieces are complementary and do not overlap.

Each bake group stores a JSON snapshot of its settings in the custom
`voronoiParameters` attribute.

## Project files

- `voronoi_geometry_core.py` - deterministic field evaluation and boundary tracing
- `voronoi_geometry_tool.py` - Maya UI, preview, materials, and mesh construction
- `run_voronoi_geometry_tool.py` - reloadable Maya launcher

## Suggested first test

1. Save the Maya scene.
2. Launch J-Voroni.
3. Create the preview at the default resolution.
4. Adjust Tributary Bias and Channel Parallelism and confirm that the preview
   responds.
5. Bake the four meshes.
6. Hide RED, GREEN, and BLUE and verify that every island opening in EDGE is
   genuinely empty while the branching network remains connected.

If Maya raises an exception, capture the full traceback from the Script
Editor. The baker's topology validation messages are designed to identify a
filled tessellation or an open solid directly.
