"""Launch J-Voroni from Maya's Python environment."""

import importlib
import os
import sys


TOOL_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

if not os.path.isdir(TOOL_DIRECTORY):
    raise RuntimeError("Voronoi tool directory not found: " + TOOL_DIRECTORY)
if TOOL_DIRECTORY not in sys.path:
    sys.path.insert(0, TOOL_DIRECTORY)

import voronoi_geometry_core
import voronoi_geometry_tool

importlib.reload(voronoi_geometry_core)
importlib.reload(voronoi_geometry_tool)
voronoi_geometry_tool.show()
