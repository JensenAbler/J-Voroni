"""Maya 2025 tool for previewing and baking tributary Voronoi geometry.

The viewport preview is a dynamically generated texture on a Maya plane. The
geometry bake does not use that image: it traces the mathematical field from
voronoi_geometry_core.py and creates four complementary extruded meshes.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from typing import Dict, List, Sequence, Tuple

import maya.api.OpenMaya as om
import maya.cmds as cmds
from maya.app.general.mayaMixin import MayaQWidgetDockableMixin
from PySide6 import QtCore, QtGui, QtWidgets

import voronoi_geometry_core as core


TOOL_VERSION = "0.1.1"
WINDOW_OBJECT = "VoronoiGeometryBakerWindow"
PREVIEW_GROUP = "VORONOI_GEO_PREVIEW_GRP"
PREVIEW_PLANE = "voronoiGeo_previewPlane"
PREVIEW_SHADER = "voronoiGeo_previewSurface"
PREVIEW_SG = "voronoiGeo_previewSG"
PREVIEW_FILE = "voronoiGeo_previewFile"
PREVIEW_PLACE = "voronoiGeo_previewPlace2d"

_WINDOW = None


class FloatControl(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(float)

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        value: float,
        decimals: int = 3,
        step: float = 0.01,
        hard_maximum: float = None,
        parent=None,
    ):
        super().__init__(parent)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.hard_maximum = float(
            maximum if hard_maximum is None else hard_maximum
        )
        self._block = False

        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        self.label = QtWidgets.QLabel(label)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setDecimals(decimals)
        self.spin.setRange(minimum, self.hard_maximum)
        self.spin.setSingleStep(step)
        self.spin.setKeyboardTracking(False)
        self.spin.setMinimumWidth(86)
        layout.addWidget(self.label, 0, 0)
        layout.addWidget(self.slider, 0, 1)
        layout.addWidget(self.spin, 0, 2)
        layout.setColumnStretch(1, 1)

        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self.setValue(value, emit=False)

    def _slider_to_value(self, slider_value: int) -> float:
        return self.minimum + (
            self.maximum - self.minimum
        ) * slider_value / 1000.0

    def _value_to_slider(self, value: float) -> int:
        if self.maximum == self.minimum:
            return 0
        slider_value = (
            1000.0 * (value - self.minimum) / (self.maximum - self.minimum)
        )
        return int(round(max(0.0, min(1000.0, slider_value))))

    def _from_slider(self, slider_value: int):
        if self._block:
            return
        value = self._slider_to_value(slider_value)
        self._block = True
        self.spin.setValue(value)
        self._block = False
        self.valueChanged.emit(self.spin.value())

    def _from_spin(self, value: float):
        if self._block:
            return
        self._block = True
        self.slider.setValue(self._value_to_slider(value))
        self._block = False
        self.valueChanged.emit(float(value))

    def value(self) -> float:
        return float(self.spin.value())

    def setValue(self, value: float, emit: bool = True):
        value = max(self.minimum, min(self.hard_maximum, float(value)))
        self._block = True
        self.spin.setValue(value)
        self.slider.setValue(self._value_to_slider(value))
        self._block = False
        if emit:
            self.valueChanged.emit(value)


class IntControl(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(int)

    def __init__(self, label: str, minimum: int, maximum: int, value: int, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QtWidgets.QLabel(label)
        self.spin = QtWidgets.QSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setValue(value)
        self.spin.setMinimumWidth(90)
        layout.addWidget(self.label)
        layout.addStretch(1)
        layout.addWidget(self.spin)
        self.spin.valueChanged.connect(self.valueChanged.emit)

    def value(self) -> int:
        return int(self.spin.value())


class VoronoiGeometryWindow(MayaQWidgetDockableMixin, QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName(WINDOW_OBJECT)
        self.setWindowTitle("J-Voroni {}".format(TOOL_VERSION))
        self.setMinimumWidth(390)
        self._preview_generation = 0
        self._preview_plane_exists = False

        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(220)
        self.preview_timer.timeout.connect(self.refresh_preview)

        self._build_ui()
        self._connect_controls()
        self._update_estimate()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        content = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(8)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        self.domain_box = self._section("Domain and pattern")
        self.width = self._add_float(self.domain_box, "Width (cm)", 1.0, 200.0, 20.0, 2, 0.5)
        self.depth = self._add_float(self.domain_box, "Depth (cm)", 1.0, 200.0, 20.0, 2, 0.5)
        self.scale = self._add_float(
            self.domain_box,
            "Scale",
            2.0,
            18.0,
            7.0,
            2,
            0.1,
            hard_maximum=10000.0,
        )
        self.edge_width = self._add_float(self.domain_box, "Edge width", 0.01, 0.75, 0.06, 3, 0.005)
        self.smoothness = self._add_float(self.domain_box, "Shape smoothness", 0.0, 1.0, 0.25, 2, 0.01)
        self.variation = self._add_float(
            self.domain_box,
            "Size variation",
            0.0,
            4.0,
            0.70,
            2,
            0.05,
            hard_maximum=1000000.0,
        )
        self.phase = self._add_float(self.domain_box, "Phase", 0.0, math.pi * 2.0, 0.0, 3, 0.02)
        self.seed = IntControl("Seed", 0, 999999, 0)
        self.domain_box.layout().addWidget(self.seed)

        self.flow_box = self._section("Flow")
        self.tributary = self._add_float(self.flow_box, "Tributary bias", 0.0, 1.0, 0.68, 2, 0.01)
        self.parallel = self._add_float(self.flow_box, "Channel parallelism", 0.0, 1.0, 0.38, 2, 0.01)

        self.palette_box = self._section("Pure RGB distribution")
        self.red_ratio = self._add_float(self.palette_box, "Red ratio", 0.0, 1.0, 0.34, 2, 0.01)
        self.green_ratio = self._add_float(self.palette_box, "Green ratio", 0.0, 1.0, 0.33, 2, 0.01)
        self.blue_ratio = self._add_float(self.palette_box, "Blue ratio", 0.0, 1.0, 0.33, 2, 0.01)
        self.red_bias = self._add_float(self.palette_box, "Red size bias", 0.0, 1.0, 0.0, 2, 0.01)
        self.green_bias = self._add_float(self.palette_box, "Green size bias", 0.0, 1.0, 0.0, 2, 0.01)
        self.blue_bias = self._add_float(self.palette_box, "Blue size bias", 0.0, 1.0, 0.0, 2, 0.01)

        self.bake_box = self._section("Bake")
        self.extrusion = self._add_float(self.bake_box, "Extrusion (cm)", 0.01, 5.0, 0.25, 3, 0.01)
        self.tolerance = self._add_float(self.bake_box, "Curve tolerance", 0.002, 0.20, 0.025, 3, 0.002)
        self.initial_rays = IntControl("Initial rays per island", 12, 64, 24)
        self.bake_box.layout().addWidget(self.initial_rays)
        self.max_refinement = IntControl("Max curve refinement", 0, 6, 4)
        self.bake_box.layout().addWidget(self.max_refinement)
        self.preview_resolution = IntControl("Preview resolution", 32, 192, 96)
        self.bake_box.layout().addWidget(self.preview_resolution)

        self.estimate = QtWidgets.QLabel()
        self.estimate.setWordWrap(True)
        self.bake_box.layout().addWidget(self.estimate)

        preview_row = QtWidgets.QHBoxLayout()
        self.preview_button = QtWidgets.QPushButton("Create / Refresh Preview")
        self.hide_preview_button = QtWidgets.QPushButton("Hide Preview")
        preview_row.addWidget(self.preview_button, 1)
        preview_row.addWidget(self.hide_preview_button)
        self.content_layout.addLayout(preview_row)

        self.bake_button = QtWidgets.QPushButton("Bake Four Meshes")
        self.bake_button.setMinimumHeight(34)
        self.content_layout.addWidget(self.bake_button)

        self.status = QtWidgets.QLabel(
            "Preview is rasterized for speed; bake vertices come directly from the field equations."
        )
        self.status.setWordWrap(True)
        self.content_layout.addWidget(self.status)
        self.content_layout.addStretch(1)

    def _section(self, title: str) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)
        self.content_layout.addWidget(box)
        return box

    @staticmethod
    def _add_float(
        box: QtWidgets.QGroupBox,
        label: str,
        minimum: float,
        maximum: float,
        value: float,
        decimals: int,
        step: float,
        hard_maximum: float = None,
    ) -> FloatControl:
        control = FloatControl(
            label,
            minimum,
            maximum,
            value,
            decimals,
            step,
            hard_maximum,
            box,
        )
        box.layout().addWidget(control)
        return control

    def _connect_controls(self):
        float_controls = [
            self.width,
            self.depth,
            self.scale,
            self.edge_width,
            self.smoothness,
            self.variation,
            self.phase,
            self.tributary,
            self.parallel,
            self.red_ratio,
            self.green_ratio,
            self.blue_ratio,
            self.red_bias,
            self.green_bias,
            self.blue_bias,
            self.extrusion,
            self.tolerance,
        ]
        for control in float_controls:
            control.valueChanged.connect(self._parameters_changed)
        for control in (
            self.seed,
            self.initial_rays,
            self.max_refinement,
            self.preview_resolution,
        ):
            control.valueChanged.connect(self._parameters_changed)

        self.preview_button.clicked.connect(self.create_or_refresh_preview)
        self.hide_preview_button.clicked.connect(self.hide_preview)
        self.bake_button.clicked.connect(self.bake)

    def _parameters_changed(self, *_):
        self._update_estimate()
        if cmds.objExists(PREVIEW_PLANE):
            self._resize_preview_plane()
            self.preview_timer.start()

    def _update_estimate(self):
        estimated_cells = max(
            1,
            int(round(
                self.scale.value()
                * self.scale.value()
                * self.width.value()
                / max(self.depth.value(), 0.001)
            )),
        )
        self.estimate.setText(
            "Estimated visible cells: about {}. Lower tolerance and higher refinement create more vertices.".format(
                estimated_cells
            )
        )

    def parameters(self) -> core.VoronoiParameters:
        return core.VoronoiParameters(
            width=self.width.value(),
            depth=self.depth.value(),
            scale=self.scale.value(),
            edge_width=self.edge_width.value(),
            shape_smoothness=self.smoothness.value(),
            size_variation=self.variation.value(),
            phase=self.phase.value(),
            seed=self.seed.value(),
            tributary_bias=self.tributary.value(),
            channel_parallelism=self.parallel.value(),
            red_ratio=self.red_ratio.value(),
            green_ratio=self.green_ratio.value(),
            blue_ratio=self.blue_ratio.value(),
            red_size_bias=self.red_bias.value(),
            green_size_bias=self.green_bias.value(),
            blue_size_bias=self.blue_bias.value(),
            initial_rays=self.initial_rays.value(),
            curve_tolerance=self.tolerance.value(),
            max_refinement=self.max_refinement.value(),
            root_iterations=16,
        )

    def _resize_preview_plane(self):
        if not cmds.objExists(PREVIEW_PLANE):
            return
        cmds.setAttr(PREVIEW_PLANE + ".scaleX", self.width.value())
        cmds.setAttr(PREVIEW_PLANE + ".scaleZ", self.depth.value())

    def _ensure_preview_nodes(self):
        if not cmds.objExists(PREVIEW_GROUP):
            cmds.group(empty=True, name=PREVIEW_GROUP)

        if not cmds.objExists(PREVIEW_PLANE):
            plane = cmds.polyPlane(
                name=PREVIEW_PLANE,
                width=1.0,
                height=1.0,
                subdivisionsX=1,
                subdivisionsY=1,
                axis=(0.0, 1.0, 0.0),
                constructionHistory=False,
            )[0]
            cmds.parent(plane, PREVIEW_GROUP)
        self._resize_preview_plane()

        if not cmds.objExists(PREVIEW_SHADER):
            cmds.shadingNode("surfaceShader", asShader=True, name=PREVIEW_SHADER)
        if not cmds.objExists(PREVIEW_SG):
            cmds.sets(
                renderable=True,
                noSurfaceShader=True,
                empty=True,
                name=PREVIEW_SG,
            )
        if not cmds.isConnected(
            PREVIEW_SHADER + ".outColor", PREVIEW_SG + ".surfaceShader"
        ):
            cmds.connectAttr(
                PREVIEW_SHADER + ".outColor",
                PREVIEW_SG + ".surfaceShader",
                force=True,
            )
        if not cmds.objExists(PREVIEW_FILE):
            cmds.shadingNode("file", asTexture=True, name=PREVIEW_FILE)
        if not cmds.objExists(PREVIEW_PLACE):
            cmds.shadingNode("place2dTexture", asUtility=True, name=PREVIEW_PLACE)
        scalar_connections = (
            "coverage",
            "translateFrame",
            "rotateFrame",
            "mirrorU",
            "mirrorV",
            "stagger",
            "wrapU",
            "wrapV",
            "repeatUV",
            "offset",
            "rotateUV",
            "noiseUV",
            "vertexUvOne",
            "vertexUvTwo",
            "vertexUvThree",
            "vertexCameraOne",
        )
        for attribute in scalar_connections:
            source = PREVIEW_PLACE + "." + attribute
            destination = PREVIEW_FILE + "." + attribute
            if not cmds.isConnected(source, destination):
                cmds.connectAttr(source, destination, force=True)
        if not cmds.isConnected(PREVIEW_PLACE + ".outUV", PREVIEW_FILE + ".uvCoord"):
            cmds.connectAttr(
                PREVIEW_PLACE + ".outUV", PREVIEW_FILE + ".uvCoord", force=True
            )
        if not cmds.isConnected(
            PREVIEW_PLACE + ".outUvFilterSize",
            PREVIEW_FILE + ".uvFilterSize",
        ):
            cmds.connectAttr(
                PREVIEW_PLACE + ".outUvFilterSize",
                PREVIEW_FILE + ".uvFilterSize",
                force=True,
            )
        if not cmds.isConnected(
            PREVIEW_FILE + ".outColor", PREVIEW_SHADER + ".outColor"
        ):
            cmds.connectAttr(
                PREVIEW_FILE + ".outColor",
                PREVIEW_SHADER + ".outColor",
                force=True,
            )
        if cmds.attributeQuery("colorSpace", node=PREVIEW_FILE, exists=True):
            try:
                cmds.setAttr(PREVIEW_FILE + ".colorSpace", "Raw", type="string")
            except RuntimeError:
                pass
        if cmds.attributeQuery("filterType", node=PREVIEW_FILE, exists=True):
            cmds.setAttr(PREVIEW_FILE + ".filterType", 1)
        cmds.sets(PREVIEW_PLANE, edit=True, forceElement=PREVIEW_SG)
        cmds.setAttr(PREVIEW_GROUP + ".visibility", True)

        for panel in cmds.getPanel(type="modelPanel") or []:
            try:
                cmds.modelEditor(panel, edit=True, displayTextures=True)
            except RuntimeError:
                pass

    def create_or_refresh_preview(self):
        cmds.undoInfo(openChunk=True, chunkName="Voronoi preview setup")
        try:
            self._ensure_preview_nodes()
        finally:
            cmds.undoInfo(closeChunk=True)
        self.refresh_preview()

    def _preview_image_path(self) -> str:
        self._preview_generation += 1
        filename = "voronoi_geo_preview_{}_{}.png".format(
            os.getpid(), self._preview_generation % 3
        )
        return os.path.join(tempfile.gettempdir(), filename)

    def _render_preview(self, parameters: core.VoronoiParameters, path: str):
        longest = max(32, int(self.preview_resolution.value()))
        if parameters.width >= parameters.depth:
            pixel_width = longest
            pixel_height = max(
                32, int(round(longest * parameters.depth / parameters.width))
            )
        else:
            pixel_height = longest
            pixel_width = max(
                32, int(round(longest * parameters.width / parameters.depth))
            )

        field = core.VoronoiField(parameters)
        image = QtGui.QImage(
            pixel_width,
            pixel_height,
            QtGui.QImage.Format.Format_RGB888,
        )
        image.fill(QtGui.QColor(255, 255, 255))
        pixel_size = max(
            parameters.pattern_width / float(pixel_width),
            parameters.pattern_height / float(pixel_height),
        )
        anti_alias = max(pixel_size * 0.70, 0.001)
        half_width = max(parameters.edge_width, 0.001) * 0.5
        palette = ((255, 0, 0), (0, 255, 0), (0, 0, 255))

        for image_y in range(pixel_height):
            pattern_y = parameters.pattern_height * (
                pixel_height - image_y - 0.5
            ) / pixel_height
            for image_x in range(pixel_width):
                pattern_x = parameters.pattern_width * (
                    image_x + 0.5
                ) / pixel_width
                point = (pattern_x, pattern_y)
                sample = field.evaluate(point, with_boundary=True)
                domain_margin = min(
                    pattern_x,
                    parameters.pattern_width - pattern_x,
                    pattern_y,
                    parameters.pattern_height - pattern_y,
                )
                margin = min(sample.boundary, domain_margin) - half_width
                amount = core.smoothstep(-anti_alias, anti_alias, margin)
                fill = palette[field.color_index(sample.winner)]
                rgb = tuple(
                    int(round(255.0 + (channel - 255.0) * amount))
                    for channel in fill
                )
                image.setPixelColor(image_x, image_y, QtGui.QColor(*rgb))
        if not image.save(path, "PNG"):
            raise RuntimeError("Qt could not save the preview PNG: " + path)

    def refresh_preview(self):
        if not cmds.objExists(PREVIEW_PLANE):
            return
        parameters = self.parameters()
        path = self._preview_image_path()
        self.status.setText("Updating viewport preview…")
        QtWidgets.QApplication.processEvents()
        started = time.perf_counter()
        try:
            self._render_preview(parameters, path)
            cmds.setAttr(PREVIEW_FILE + ".fileTextureName", path, type="string")
            cmds.dgdirty(PREVIEW_FILE)
            cmds.refresh(force=True)
        except Exception as exc:
            self.status.setText("Preview failed: {}".format(exc))
            raise
        elapsed = time.perf_counter() - started
        self.status.setText(
            "Preview updated in {:.2f}s. Geometry bake remains equation-based.".format(
                elapsed
            )
        )

    def hide_preview(self):
        if cmds.objExists(PREVIEW_GROUP):
            current = cmds.getAttr(PREVIEW_GROUP + ".visibility")
            cmds.setAttr(PREVIEW_GROUP + ".visibility", not current)
            self.hide_preview_button.setText(
                "Show Preview" if current else "Hide Preview"
            )

    def bake(self):
        parameters = self.parameters()
        field = core.VoronoiField(parameters)
        progress_cancelled = {"value": False}

        def cancelled():
            try:
                progress_cancelled["value"] = bool(
                    cmds.progressWindow(query=True, isCancelled=True)
                )
            except RuntimeError:
                pass
            return progress_cancelled["value"]

        def progress(index, total, cell_id):
            if total <= 0:
                percent = 0
            else:
                percent = int(round(100.0 * index / total))
            cmds.progressWindow(
                edit=True,
                progress=percent,
                status="Tracing island {} of {}".format(index, total),
            )
            QtWidgets.QApplication.processEvents()

        started = time.perf_counter()
        self.bake_button.setEnabled(False)
        self.status.setText("Tracing mathematical boundaries…")
        cmds.progressWindow(
            title="Voronoi Geometry Bake",
            progress=0,
            status="Discovering cells",
            isInterruptable=True,
            maxValue=100,
        )
        try:
            polygons = field.trace_all_cells(progress=progress, cancelled=cancelled)
            if cancelled():
                self.status.setText("Bake cancelled before scene creation.")
                return
            if not polygons:
                raise RuntimeError(
                    "No inset islands survived. Reduce edge width or size variation."
                )

            cmds.progressWindow(
                edit=True, progress=100, status="Creating four meshes"
            )
            QtWidgets.QApplication.processEvents()
            cmds.undoInfo(openChunk=True, chunkName="Bake Voronoi geometry")
            try:
                group = self._create_bake_group(parameters)
                materials = self._ensure_bake_materials()
                by_color = {0: [], 1: [], 2: []}
                for polygon in polygons:
                    by_color[polygon.color_index].append(polygon)

                color_names = (
                    "voronoiGeo_RED",
                    "voronoiGeo_GREEN",
                    "voronoiGeo_BLUE",
                )
                color_meshes = []
                for color_index in range(3):
                    mesh = self._create_color_mesh(
                        by_color[color_index],
                        parameters,
                        color_names[color_index],
                        self.extrusion.value(),
                    )
                    cmds.parent(mesh, group)
                    if cmds.listRelatives(mesh, shapes=True, fullPath=True):
                        cmds.sets(
                            mesh,
                            edit=True,
                            forceElement=materials[color_index],
                        )
                    color_meshes.append(mesh)

                edge_mesh = self._create_edge_mesh(
                    polygons,
                    parameters,
                    "voronoiGeo_EDGE",
                    self.extrusion.value(),
                )
                cmds.parent(edge_mesh, group)
                cmds.sets(
                    edge_mesh,
                    edit=True,
                    forceElement=materials[3],
                )
                cmds.select([edge_mesh] + color_meshes, replace=True)
            finally:
                cmds.undoInfo(closeChunk=True)

            elapsed = time.perf_counter() - started
            vertex_count = sum(len(polygon.points) for polygon in polygons)
            self.status.setText(
                "Baked {} islands and the connected edge network in {:.2f}s ({} traced boundary points).".format(
                    len(polygons), elapsed, vertex_count
                )
            )
        except Exception as exc:
            self.status.setText("Bake failed: {}".format(exc))
            cmds.warning("Voronoi bake failed: {}".format(exc))
            raise
        finally:
            try:
                cmds.progressWindow(endProgress=True)
            except RuntimeError:
                pass
            self.bake_button.setEnabled(True)

    @staticmethod
    def _next_bake_name() -> str:
        index = 1
        while cmds.objExists("voronoiGeoBake_{:03d}".format(index)):
            index += 1
        return "voronoiGeoBake_{:03d}".format(index)

    def _create_bake_group(self, parameters: core.VoronoiParameters) -> str:
        name = self._next_bake_name()
        group = cmds.group(empty=True, name=name)
        cmds.addAttr(group, longName="voronoiToolVersion", dataType="string")
        cmds.setAttr(
            group + ".voronoiToolVersion", TOOL_VERSION, type="string"
        )
        cmds.addAttr(group, longName="voronoiParameters", dataType="string")
        cmds.setAttr(
            group + ".voronoiParameters",
            json.dumps(parameters.__dict__, sort_keys=True),
            type="string",
        )
        return group

    @staticmethod
    def _ensure_surface_shader(name: str, color: Tuple[float, float, float]) -> str:
        shader = name + "_MAT"
        shading_group = name + "_SG"
        if not cmds.objExists(shader):
            cmds.shadingNode("surfaceShader", asShader=True, name=shader)
        cmds.setAttr(shader + ".outColor", *color, type="double3")
        if not cmds.objExists(shading_group):
            cmds.sets(
                renderable=True,
                noSurfaceShader=True,
                empty=True,
                name=shading_group,
            )
        if not cmds.isConnected(
            shader + ".outColor", shading_group + ".surfaceShader"
        ):
            cmds.connectAttr(
                shader + ".outColor",
                shading_group + ".surfaceShader",
                force=True,
            )
        return shading_group

    def _ensure_bake_materials(self) -> Tuple[str, str, str, str]:
        return (
            self._ensure_surface_shader("voronoiGeo_RED", (1.0, 0.0, 0.0)),
            self._ensure_surface_shader("voronoiGeo_GREEN", (0.0, 1.0, 0.0)),
            self._ensure_surface_shader("voronoiGeo_BLUE", (0.0, 0.0, 1.0)),
            self._ensure_surface_shader("voronoiGeo_EDGE", (1.0, 1.0, 1.0)),
        )

    @staticmethod
    def _mobject(node_name: str) -> om.MObject:
        selection = om.MSelectionList()
        selection.add(node_name)
        return selection.getDependNode(0)

    def _create_color_mesh(
        self,
        polygons: Sequence[core.CellPolygon],
        parameters: core.VoronoiParameters,
        name: str,
        height: float,
    ) -> str:
        transform = cmds.createNode("transform", name=name)
        if not polygons:
            return transform

        vertices: List[om.MPoint] = []
        polygon_counts: List[int] = []
        polygon_connects: List[int] = []

        for polygon in polygons:
            world_points = [
                core.pattern_to_world(parameters, point)
                for point in polygon.points
            ]
            center_world = core.pattern_to_world(parameters, polygon.center)
            count = len(world_points)
            base_index = len(vertices)
            for x, z in world_points:
                vertices.append(om.MPoint(x, 0.0, z))
            for x, z in world_points:
                vertices.append(om.MPoint(x, height, z))
            bottom_center = len(vertices)
            vertices.append(om.MPoint(center_world[0], 0.0, center_world[1]))
            top_center = len(vertices)
            vertices.append(om.MPoint(center_world[0], height, center_world[1]))

            for index in range(count):
                next_index = (index + 1) % count
                bottom_a = base_index + index
                bottom_b = base_index + next_index
                top_a = base_index + count + index
                top_b = base_index + count + next_index

                polygon_counts.append(3)
                polygon_connects.extend((bottom_center, bottom_a, bottom_b))
                polygon_counts.append(3)
                polygon_connects.extend((top_center, top_b, top_a))
                polygon_counts.append(4)
                polygon_connects.extend((bottom_a, top_a, top_b, bottom_b))

        mesh_function = om.MFnMesh()
        mesh_object = mesh_function.create(
            vertices,
            polygon_counts,
            polygon_connects,
            parent=self._mobject(transform),
        )
        shape = om.MFnDagNode(mesh_object)
        shape.setName(name + "Shape")
        try:
            cmds.polySoftEdge(transform, angle=30.0, constructionHistory=False)
        except RuntimeError:
            pass
        return transform

    def _create_edge_mesh(
        self,
        polygons: Sequence[core.CellPolygon],
        parameters: core.VoronoiParameters,
        name: str,
        height: float,
    ) -> str:
        half_width = parameters.width * 0.5
        half_depth = parameters.depth * 0.5

        # Clockwise in X/Z gives the outer face a +Y normal. Hole loops use
        # the opposite winding and are exactly the same loops as RGB tops.
        point_data: List[Tuple[float, float, float]] = [
            (-half_width, 0.0, -half_depth),
            (-half_width, 0.0, half_depth),
            (half_width, 0.0, half_depth),
            (half_width, 0.0, -half_depth),
        ]
        for polygon in polygons:
            point_data.append(())
            for point in polygon.points:
                x, z = core.pattern_to_world(parameters, point)
                point_data.append((x, 0.0, z))

        source_result = cmds.polyCreateFacet(
            point=point_data,
            name=name + "_tessellationSource",
            constructionHistory=False,
            texture=0,
        )
        source_mesh = source_result[0]

        try:
            selection = om.MSelectionList()
            selection.add(source_mesh)
            source_path = selection.getDagPath(0)
            if source_path.node().hasFn(om.MFn.kTransform):
                source_path.extendToShape()
            source_function = om.MFnMesh(source_path)

            holes = source_function.getHoles()
            if len(holes) != len(polygons):
                raise RuntimeError(
                    "Maya registered {} cell holes, but {} were supplied.".format(
                        len(holes), len(polygons)
                    )
                )

            source_points = list(source_function.getPoints(om.MSpace.kObject))
            _triangle_counts, triangle_vertices = source_function.getTriangles()
            triangle_vertices = list(triangle_vertices)
            if not triangle_vertices:
                raise RuntimeError("Maya could not tessellate the edge complement.")

            # Retain Maya's exact hole vertex IDs so the caps and vertical hole
            # walls share vertices. That makes the finished edge one watertight
            # connected mesh instead of coincident sticker-like surfaces.
            hole_contours = [list(vertex_ids) for _face, vertex_ids in holes]

            def signed_area(vertex_ids):
                area = 0.0
                for index, vertex_id in enumerate(vertex_ids):
                    next_id = vertex_ids[(index + 1) % len(vertex_ids)]
                    point = source_points[vertex_id]
                    next_point = source_points[next_id]
                    area += point.x * next_point.z - next_point.x * point.z
                return area * 0.5

            # Locate the four outer vertices by position; all remaining
            # contours come directly from getHoles().
            outer_targets = point_data[:4]
            outer_contour = []
            for target_x, _target_y, target_z in outer_targets:
                vertex_id = min(
                    range(len(source_points)),
                    key=lambda candidate: (
                        (source_points[candidate].x - target_x) ** 2
                        + (source_points[candidate].z - target_z) ** 2
                    ),
                )
                distance_squared = (
                    (source_points[vertex_id].x - target_x) ** 2
                    + (source_points[vertex_id].z - target_z) ** 2
                )
                if distance_squared > 1.0e-10:
                    raise RuntimeError("Could not recover the outer edge contour.")
                outer_contour.append(vertex_id)

            # All contours are oriented with the solid edge region on their
            # right. The side-quad winding below consequently points outward.
            if signed_area(outer_contour) > 0.0:
                outer_contour.reverse()
            for contour in hole_contours:
                if signed_area(contour) < 0.0:
                    contour.reverse()

            vertex_count = len(source_points)
            vertices = [om.MPoint(point.x, 0.0, point.z) for point in source_points]
            vertices.extend(
                om.MPoint(point.x, height, point.z) for point in source_points
            )
            polygon_counts: List[int] = []
            polygon_connects: List[int] = []
            cap_area = 0.0

            for index in range(0, len(triangle_vertices), 3):
                a, b, c = triangle_vertices[index : index + 3]
                point_a = source_points[a]
                point_b = source_points[b]
                point_c = source_points[c]
                cross_y = (
                    (point_b.z - point_a.z) * (point_c.x - point_a.x)
                    - (point_b.x - point_a.x) * (point_c.z - point_a.z)
                )
                if abs(cross_y) <= 1.0e-12:
                    continue
                if cross_y < 0.0:
                    b, c = c, b
                    cross_y = -cross_y
                cap_area += cross_y * 0.5
                polygon_counts.extend((3, 3))
                polygon_connects.extend((a, c, b))
                polygon_connects.extend(
                    (a + vertex_count, b + vertex_count, c + vertex_count)
                )

            expected_cap_area = parameters.width * parameters.depth
            for polygon in polygons:
                world_points = [
                    core.pattern_to_world(parameters, point)
                    for point in polygon.points
                ]
                polygon_area = 0.0
                for index, point in enumerate(world_points):
                    next_point = world_points[(index + 1) % len(world_points)]
                    polygon_area += (
                        point[0] * next_point[1] - next_point[0] * point[1]
                    )
                expected_cap_area -= abs(polygon_area) * 0.5

            area_tolerance = max(1.0e-6, abs(expected_cap_area) * 0.005)
            if abs(cap_area - expected_cap_area) > area_tolerance:
                raise RuntimeError(
                    "Maya's edge tessellation covered {:.6g} square units; "
                    "the punched complement should cover {:.6g}.".format(
                        cap_area, expected_cap_area
                    )
                )

            for contour in [outer_contour] + hole_contours:
                for index, bottom_a in enumerate(contour):
                    bottom_b = contour[(index + 1) % len(contour)]
                    polygon_counts.append(4)
                    polygon_connects.extend(
                        (
                            bottom_a,
                            bottom_b,
                            bottom_b + vertex_count,
                            bottom_a + vertex_count,
                        )
                    )
        finally:
            if cmds.objExists(source_mesh):
                cmds.delete(source_mesh)

        edge_mesh = cmds.createNode("transform", name=name)
        mesh_function = om.MFnMesh()
        mesh_object = mesh_function.create(
            vertices,
            polygon_counts,
            polygon_connects,
            parent=self._mobject(edge_mesh),
        )
        om.MFnDagNode(mesh_object).setName(name + "Shape")

        # Every edge must now be shared by two faces. Any border means a cap
        # or wall was lost and the object is not a genuine punched solid.
        selection = om.MSelectionList()
        selection.add(edge_mesh)
        edge_path = selection.getDagPath(0)
        if edge_path.node().hasFn(om.MFn.kTransform):
            edge_path.extendToShape()
        edge_iterator = om.MItMeshEdge(edge_path)
        open_edge_count = 0
        while not edge_iterator.isDone():
            if edge_iterator.onBoundary():
                open_edge_count += 1
            edge_iterator.next()
        if open_edge_count:
            cmds.delete(edge_mesh)
            raise RuntimeError(
                "The explicit edge mesh has {} open boundary edges.".format(
                    open_edge_count
                )
            )

        try:
            cmds.polySoftEdge(edge_mesh, angle=30.0, constructionHistory=False)
        except RuntimeError:
            pass
        return edge_mesh


def show():
    global _WINDOW
    try:
        if _WINDOW is not None:
            _WINDOW.close()
            _WINDOW.deleteLater()
    except RuntimeError:
        pass

    workspace_control = WINDOW_OBJECT + "WorkspaceControl"
    if cmds.workspaceControl(workspace_control, query=True, exists=True):
        cmds.deleteUI(workspace_control)

    _WINDOW = VoronoiGeometryWindow()
    _WINDOW.show(dockable=True, area="right", floating=False)
    return _WINDOW


if __name__ == "__main__":
    show()
