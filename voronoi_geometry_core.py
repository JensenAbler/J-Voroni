"""Dependency-free mathematical core for the Maya Voronoi geometry baker.

This module contains no Maya or Qt imports. It evaluates the same competitive
distance field as the viewport preview and traces colored cell interiors by
solving the field directly. Raster images are not used to create vertices.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import j_voroni_earcut


TAU = math.pi * 2.0
Vec2 = Tuple[float, float]
CellId = Tuple[int, int]


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def mix(a: float, b: float, amount: float) -> float:
    return a + (b - a) * amount


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 == edge0:
        return 0.0 if value < edge0 else 1.0
    t = clamp((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def fract(value: float) -> float:
    return value - math.floor(value)


def length2(vector: Vec2) -> float:
    return vector[0] * vector[0] + vector[1] * vector[1]


def length(vector: Vec2) -> float:
    return math.sqrt(length2(vector))


def normalize(vector: Vec2) -> Vec2:
    magnitude = length(vector)
    if magnitude < 1.0e-12:
        return (0.0, 1.0)
    return (vector[0] / magnitude, vector[1] / magnitude)


def hash22(point: Vec2) -> Vec2:
    px = point[0] * 127.1 + point[1] * 311.7
    py = point[0] * 269.5 + point[1] * 183.3
    return (
        fract(math.sin(px) * 43758.5453),
        fract(math.sin(py) * 43758.5453),
    )


def smooth_minimum(a: float, b: float, radius: float) -> float:
    safe_radius = max(radius, 0.0001)
    h = max(safe_radius - abs(a - b), 0.0) / safe_radius
    return min(a, b) - h * h * safe_radius * 0.25


@dataclass
class VoronoiParameters:
    width: float = 20.0
    depth: float = 20.0
    scale: float = 7.0
    edge_width: float = 0.06
    shape_smoothness: float = 0.25
    size_variation: float = 0.70
    phase: float = 0.0
    seed: int = 0
    tributary_bias: float = 0.68
    channel_parallelism: float = 0.38
    red_ratio: float = 0.34
    green_ratio: float = 0.33
    blue_ratio: float = 0.33
    red_size_bias: float = 0.0
    green_size_bias: float = 0.0
    blue_size_bias: float = 0.0
    initial_rays: int = 24
    curve_tolerance: float = 0.025
    max_refinement: int = 4
    root_iterations: int = 16

    @property
    def aspect(self) -> float:
        return max(self.width, 1.0e-6) / max(self.depth, 1.0e-6)

    @property
    def pattern_width(self) -> float:
        return max(self.scale, 1.0e-4) * self.aspect

    @property
    def pattern_height(self) -> float:
        return max(self.scale, 1.0e-4)

    @property
    def seed_offset(self) -> Vec2:
        seed_value = float(self.seed)
        return (seed_value * 17.173, seed_value * 31.719)

    def normalized_ratios(self) -> Tuple[float, float, float]:
        red = max(self.red_ratio, 0.0)
        green = max(self.green_ratio, 0.0)
        blue = max(self.blue_ratio, 0.0)
        total = red + green + blue
        if total <= 1.0e-9:
            return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        return (red / total, green / total, blue / total)


@dataclass
class FieldSample:
    winner: CellId
    boundary: float
    nearest_distance: float
    nearest_metric: float


@dataclass
class CellPolygon:
    cell_id: CellId
    color_index: int
    center: Vec2
    points: List[Vec2]


@dataclass
class MeshBuffers:
    vertices: List[Tuple[float, float, float]]
    polygon_counts: List[int]
    polygon_connects: List[int]


@dataclass
class _SiteInfo:
    position: Vec2
    weight: float
    color_index: int


class VoronoiField:
    TREE_WIDTH = 7.5
    TREE_HEIGHT = 8.5

    def __init__(self, parameters: VoronoiParameters):
        self.parameters = parameters
        self._site_cache: Dict[CellId, _SiteInfo] = {}
        self._segment_cache: Dict[
            CellId,
            List[Tuple[float, float, float, float, float, float, float, float]],
        ] = {}

    def _hash(self, point: Vec2, salt: Vec2 = (0.0, 0.0)) -> Vec2:
        seed_offset = self.parameters.seed_offset
        return hash22(
            (
                point[0] + salt[0] + seed_offset[0],
                point[1] + salt[1] + seed_offset[1],
            )
        )

    def _tree_node(
        self,
        index: float,
        count: float,
        height: float,
        tile_id: CellId,
    ) -> Vec2:
        spacing = self.TREE_WIDTH / count
        seed = (
            tile_id[0] * 31.7 + index * 17.1 + count * 3.7,
            tile_id[1] * 47.3 + count * 11.9 + height * 23.0,
        )
        jitter = self._hash(seed)
        return (
            (index + 0.5) * spacing + (jitter[0] - 0.5) * spacing * 0.18,
            height * self.TREE_HEIGHT
            + (jitter[1] - 0.5) * self.TREE_HEIGHT * 0.012,
        )

    def _segments_for_tile(
        self, tile_id: CellId
    ) -> List[Tuple[float, float, float, float, float, float, float, float]]:
        cached = self._segment_cache.get(tile_id)
        if cached is not None:
            return cached

        origin = (
            tile_id[0] * self.TREE_WIDTH,
            tile_id[1] * self.TREE_HEIGHT,
        )

        def global_node(index: float, count: float, height: float) -> Vec2:
            local = self._tree_node(index, count, height, tile_id)
            return (origin[0] + local[0], origin[1] + local[1])

        raw_segments: List[Tuple[float, float, float, float, float]] = []
        root = global_node(0.0, 1.0, 0.18)
        raw_segments.append((root[0], origin[1], root[0], root[1], 3.0))

        for child in range(2):
            end = global_node(float(child), 2.0, 0.40)
            raw_segments.append((root[0], root[1], end[0], end[1], 2.0))
        for child in range(4):
            start = global_node(float(child // 2), 2.0, 0.40)
            end = global_node(float(child), 4.0, 0.69)
            raw_segments.append((start[0], start[1], end[0], end[1], 1.0))
        for child in range(8):
            start = global_node(float(child // 2), 4.0, 0.69)
            end = global_node(float(child), 8.0, 0.97)
            raw_segments.append((start[0], start[1], end[0], end[1], 0.0))

        segments = []
        for ax, ay, bx, by, level in raw_segments:
            segment_x = bx - ax
            segment_y = by - ay
            segment_length2 = max(
                segment_x * segment_x + segment_y * segment_y,
                0.0001,
            )
            inverse_segment_length2 = 1.0 / segment_length2
            inverse_segment_length = math.sqrt(inverse_segment_length2)
            direction_x = segment_x * inverse_segment_length
            direction_y = segment_y * inverse_segment_length
            segments.append(
                (
                    ax,
                    ay,
                    segment_x,
                    segment_y,
                    inverse_segment_length2,
                    direction_x * direction_x - direction_y * direction_y,
                    2.0 * direction_x * direction_y,
                    1.0 + level * 0.11,
                )
            )

        self._segment_cache[tile_id] = segments
        return segments

    def flow_frame(self, point: Vec2) -> Tuple[float, float, float]:
        center_tile = (
            int(math.floor(point[0] / self.TREE_WIDTH)),
            int(math.floor(point[1] / self.TREE_HEIGHT)),
        )
        orientation_x = 0.0
        orientation_y = 0.0
        total_weight = 0.0

        for tile_y in range(center_tile[1] - 1, center_tile[1] + 2):
            for tile_x in range(center_tile[0] - 1, center_tile[0] + 2):
                for (
                    ax,
                    ay,
                    segment_x,
                    segment_y,
                    inverse_segment_length2,
                    direction_axis_x,
                    direction_axis_y,
                    level_weight,
                ) in self._segments_for_tile(
                    (tile_x, tile_y)
                ):
                    t = clamp(
                        (
                            (point[0] - ax) * segment_x
                            + (point[1] - ay) * segment_y
                        )
                        * inverse_segment_length2,
                        0.0,
                        1.0,
                    )
                    ox = point[0] - (ax + segment_x * t)
                    oy = point[1] - (ay + segment_y * t)
                    weight = math.exp(-(ox * ox + oy * oy) * 0.82)
                    weight *= level_weight
                    orientation_x += weight * direction_axis_x
                    orientation_y += weight * direction_axis_y
                    total_weight += weight

        angle = 0.5 * math.atan2(orientation_y, orientation_x)
        direction = (math.cos(angle), math.sin(angle))
        if direction[1] < 0.0:
            direction = (-direction[0], -direction[1])
        confidence = smoothstep(0.35, 2.35, total_weight)
        return (direction[0], direction[1], confidence)

    def _site_branch_frame(
        self, point: Vec2
    ) -> Tuple[float, float, float, float]:
        tile_id = (
            int(math.floor(point[0] / self.TREE_WIDTH)),
            int(math.floor(point[1] / self.TREE_HEIGHT)),
        )
        qx = point[0] - tile_id[0] * self.TREE_WIDTH
        qy = point[1] - tile_id[1] * self.TREE_HEIGHT
        normalized_y = qy / self.TREE_HEIGHT

        if normalized_y < 0.18:
            root = self._tree_node(0.0, 1.0, 0.18, tile_id)
            start = (root[0], 0.0)
            end = root
            level = 3.0
        elif normalized_y < 0.40:
            child = int(clamp(math.floor(qx / (self.TREE_WIDTH / 2.0)), 0, 1))
            start = self._tree_node(0.0, 1.0, 0.18, tile_id)
            end = self._tree_node(float(child), 2.0, 0.40, tile_id)
            level = 2.0
        elif normalized_y < 0.69:
            child = int(clamp(math.floor(qx / (self.TREE_WIDTH / 4.0)), 0, 3))
            start = self._tree_node(float(child // 2), 2.0, 0.40, tile_id)
            end = self._tree_node(float(child), 4.0, 0.69, tile_id)
            level = 1.0
        else:
            child = int(clamp(math.floor(qx / (self.TREE_WIDTH / 8.0)), 0, 7))
            start = self._tree_node(float(child // 2), 4.0, 0.69, tile_id)
            end = self._tree_node(float(child), 8.0, 0.97, tile_id)
            level = 0.0

        segment = (end[0] - start[0], end[1] - start[1])
        direction = normalize(segment)
        normal = (-direction[1], direction[0])
        segment_length2 = max(length2(segment), 0.0001)
        t = clamp(
            ((qx - start[0]) * segment[0] + (qy - start[1]) * segment[1])
            / segment_length2,
            0.0,
            1.0,
        )
        projection = (start[0] + segment[0] * t, start[1] + segment[1] * t)
        signed_distance = (
            (qx - projection[0]) * normal[0]
            + (qy - projection[1]) * normal[1]
        )
        return (signed_distance, direction[0], direction[1], level)

    def _channel_value(self, values: Sequence[float], channel: int) -> float:
        return values[channel]

    def _channel_random(self, cell_id: CellId, channel: int) -> float:
        if channel == 0:
            return self._hash(cell_id, (19.19, 7.73))[1]
        if channel == 1:
            return self._hash(cell_id, (53.31, 91.17))[0]
        return self._hash(cell_id, (83.07, 37.61))[1]

    def color_index(self, cell_id: CellId) -> int:
        ratios = self.parameters.normalized_ratios()
        biases = (
            clamp(self.parameters.red_size_bias, 0.0, 1.0),
            clamp(self.parameters.green_size_bias, 0.0, 1.0),
            clamp(self.parameters.blue_size_bias, 0.0, 1.0),
        )
        size_rank = self._hash(cell_id, (41.37, 17.17))[0]

        first, second, third = 0, 1, 2
        if biases[1] > biases[0] and biases[1] >= biases[2]:
            first = 1
            if biases[0] >= biases[2]:
                second, third = 0, 2
            else:
                second, third = 2, 0
        elif biases[2] > biases[0] and biases[2] > biases[1]:
            first = 2
            if biases[0] >= biases[1]:
                second, third = 0, 1
            else:
                second, third = 1, 0
        elif biases[2] > biases[1]:
            second, third = 2, 1

        first_ratio = ratios[first]
        first_bias = biases[first]
        first_selector = mix(
            self._channel_random(cell_id, first), size_rank, first_bias
        )
        if first_selector < first_ratio:
            return first

        remaining_ratio = max(1.0 - first_ratio, 0.000001)
        remapped_size_rank = clamp(
            (size_rank - first_ratio) / remaining_ratio, 0.0, 1.0
        )
        survivor_rank = mix(size_rank, remapped_size_rank, first_bias)
        second_ratio = clamp(ratios[second] / remaining_ratio, 0.0, 1.0)
        second_selector = mix(
            self._channel_random(cell_id, second),
            survivor_rank,
            biases[second],
        )
        return second if second_selector < second_ratio else third

    def _site_info(self, cell_id: CellId) -> _SiteInfo:
        cached = self._site_cache.get(cell_id)
        if cached is not None:
            return cached

        random_site = self._hash(cell_id)
        site = (
            cell_id[0]
            + 0.5
            + 0.42 * math.sin(self.parameters.phase + TAU * random_site[0]),
            cell_id[1]
            + 0.5
            + 0.42 * math.sin(self.parameters.phase + TAU * random_site[1]),
        )

        signed_distance, direction_x, direction_y, _ = self._site_branch_frame(
            (cell_id[0] + 0.5, cell_id[1] + 0.5)
        )
        normal = (-direction_y, direction_x)
        side = -1.0 if signed_distance < 0.0 else 1.0
        if abs(signed_distance) < 0.04:
            side = -1.0 if self._hash(cell_id, (73.1, 29.7))[0] < 0.5 else 1.0
        bank_target = side * (
            0.29 + 0.06 * self._hash(cell_id, (13.7, 81.3))[1]
        )
        lane_spacing = mix(
            0.82, 0.58, clamp(self.parameters.channel_parallelism, 0.0, 1.0)
        )
        lane_target = (math.floor(signed_distance / lane_spacing) + 0.5) * lane_spacing
        target = mix(
            signed_distance,
            bank_target,
            clamp(self.parameters.tributary_bias, 0.0, 1.0),
        )
        target = mix(
            target,
            lane_target,
            clamp(self.parameters.channel_parallelism, 0.0, 1.0) * 0.72,
        )
        displacement = clamp(target - signed_distance, -0.68, 0.68)
        site = (
            site[0] + normal[0] * displacement,
            site[1] + normal[1] * displacement,
        )

        size_random = self._hash(cell_id, (41.37, 17.17))[0]
        info = _SiteInfo(
            position=site,
            weight=size_random * size_random * size_random,
            color_index=self.color_index(cell_id),
        )
        self._site_cache[cell_id] = info
        return info

    def _candidate_ids(self, point: Vec2) -> Iterable[CellId]:
        center_x = int(math.floor(point[0]))
        center_y = int(math.floor(point[1]))
        # A power weight V can let a site influence points roughly sqrt(V)
        # cells away. Expanding this radius keeps positive size variation
        # mathematically unbounded, although very large values are naturally
        # expensive and can produce an enormous bake.
        radius = 2 + int(math.ceil(math.sqrt(
            max(self.parameters.size_variation, 0.0)
        )))
        for y in range(center_y - radius, center_y + radius + 1):
            for x in range(center_x - radius, center_x + radius + 1):
                yield (x, y)

    def evaluate(self, point: Vec2, with_boundary: bool = True) -> FieldSample:
        flow_x, flow_y, confidence = self.flow_frame(point)
        normal_x, normal_y = -flow_y, flow_x
        tributary = clamp(self.parameters.tributary_bias, 0.0, 1.0)
        parallelism = clamp(self.parameters.channel_parallelism, 0.0, 1.0)
        aspect = 1.0 + 8.0 * tributary + 14.0 * parallelism
        area_scale = math.sqrt(aspect)
        alignment = clamp(
            (0.88 * tributary + 0.96 * parallelism) * confidence,
            0.0,
            1.0,
        )

        candidates = []
        nearest_metric = float("inf")
        nearest_distance = float("inf")
        winner = (0, 0)

        for cell_id in self._candidate_ids(point):
            info = self._site_info(cell_id)
            dx = point[0] - info.position[0]
            dy = point[1] - info.position[1]
            distance = math.sqrt(dx * dx + dy * dy)
            point_metric = dx * dx + dy * dy
            along = dx * flow_x + dy * flow_y
            across = dx * normal_x + dy * normal_y
            aligned_metric = (
                across * across * area_scale + along * along / area_scale
            )
            metric = mix(point_metric, aligned_metric, alignment)
            metric -= max(self.parameters.size_variation, 0.0) * info.weight
            candidates.append((cell_id, metric, distance))
            if metric < nearest_metric:
                nearest_metric = metric
                nearest_distance = distance
                winner = cell_id

        if not with_boundary:
            return FieldSample(
                winner=winner,
                boundary=float("inf"),
                nearest_distance=nearest_distance,
                nearest_metric=nearest_metric,
            )

        angular_gap = float("inf")
        rounded_gap = float("inf")
        rounding = max(0.001, 0.28 * clamp(self.parameters.shape_smoothness, 0.0, 1.0))
        for cell_id, metric, distance in candidates:
            if cell_id == winner:
                continue
            gap = (metric - nearest_metric) / max(
                distance + nearest_distance, 0.001
            )
            angular_gap = min(angular_gap, gap)
            rounded_gap = smooth_minimum(rounded_gap, gap, rounding)

        boundary = mix(
            angular_gap,
            rounded_gap,
            clamp(self.parameters.shape_smoothness, 0.0, 1.0),
        )
        return FieldSample(
            winner=winner,
            boundary=boundary,
            nearest_distance=nearest_distance,
            nearest_metric=nearest_metric,
        )

    def interior_margin(self, point: Vec2, cell_id: CellId) -> float:
        half_width = max(self.parameters.edge_width, 0.001) * 0.5
        domain_margin = min(
            point[0] - half_width,
            self.parameters.pattern_width - half_width - point[0],
            point[1] - half_width,
            self.parameters.pattern_height - half_width - point[1],
        )
        if domain_margin < 0.0:
            return domain_margin
        sample = self.evaluate(point, with_boundary=True)
        if sample.winner != cell_id:
            return -max(self.parameters.edge_width, 0.001)
        # edge_width is the full network width. Each adjacent cell contributes
        # one half-width of clearance from the original shared boundary.
        return min(sample.boundary - half_width, domain_margin)

    def in_domain(self, point: Vec2, epsilon: float = 0.0) -> bool:
        return (
            -epsilon <= point[0] <= self.parameters.pattern_width + epsilon
            and -epsilon <= point[1] <= self.parameters.pattern_height + epsilon
        )

    def discover_cells(self) -> Dict[CellId, Vec2]:
        width = self.parameters.pattern_width
        height = self.parameters.pattern_height
        samples_x = max(12, int(math.ceil(width * 2.4)))
        samples_y = max(12, int(math.ceil(height * 2.4)))
        best: Dict[CellId, Tuple[float, Vec2]] = {}

        for sample_y in range(samples_y + 1):
            y = height * sample_y / float(samples_y)
            for sample_x in range(samples_x + 1):
                x = width * sample_x / float(samples_x)
                point = (x, y)
                result = self.evaluate(point, with_boundary=True)
                half_width = max(self.parameters.edge_width, 0.001) * 0.5
                domain_margin = min(
                    point[0] - half_width,
                    width - half_width - point[0],
                    point[1] - half_width,
                    height - half_width - point[1],
                )
                margin = min(result.boundary - half_width, domain_margin)
                previous = best.get(result.winner)
                if previous is None or margin > previous[0]:
                    best[result.winner] = (margin, point)

        min_x = int(math.floor(-2.0))
        max_x = int(math.ceil(width + 2.0))
        min_y = int(math.floor(-2.0))
        max_y = int(math.ceil(height + 2.0))
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                cell_id = (x, y)
                site = self._site_info(cell_id).position
                if not self.in_domain(site):
                    continue
                result = self.evaluate(site, with_boundary=True)
                if result.winner != cell_id:
                    continue
                half_width = max(self.parameters.edge_width, 0.001) * 0.5
                domain_margin = min(
                    site[0] - half_width,
                    width - half_width - site[0],
                    site[1] - half_width,
                    height - half_width - site[1],
                )
                margin = min(result.boundary - half_width, domain_margin)
                previous = best.get(cell_id)
                if previous is None or margin > previous[0]:
                    best[cell_id] = (margin, site)

        return {
            cell_id: point
            for cell_id, (margin, point) in best.items()
            if margin > 0.0
        }

    def _ray_domain_exit(self, center: Vec2, direction: Vec2) -> float:
        distances: List[float] = []
        border = max(self.parameters.edge_width, 0.001) * 0.5
        minimum_x = min(border, self.parameters.pattern_width * 0.49)
        maximum_x = max(
            self.parameters.pattern_width - border,
            self.parameters.pattern_width * 0.51,
        )
        minimum_y = min(border, self.parameters.pattern_height * 0.49)
        maximum_y = max(
            self.parameters.pattern_height - border,
            self.parameters.pattern_height * 0.51,
        )
        if direction[0] > 1.0e-12:
            distances.append((maximum_x - center[0]) / direction[0])
        elif direction[0] < -1.0e-12:
            distances.append((minimum_x - center[0]) / direction[0])
        if direction[1] > 1.0e-12:
            distances.append((maximum_y - center[1]) / direction[1])
        elif direction[1] < -1.0e-12:
            distances.append((minimum_y - center[1]) / direction[1])
        positive = [distance for distance in distances if distance >= 0.0]
        return min(positive) if positive else 0.0

    def _boundary_on_ray(
        self, cell_id: CellId, center: Vec2, angle: float
    ) -> Vec2:
        direction = (math.cos(angle), math.sin(angle))
        maximum = self._ray_domain_exit(center, direction)
        if maximum <= 1.0e-12:
            return center

        # Ownership along a tributary-distorted ray is not necessarily
        # monotonic: a cell can lose a ray and appear again farther away.
        # Bracket the first exit before bisection instead of assuming that the
        # domain endpoint and center surround the desired crossing.
        low = 0.0
        low_margin = self.interior_margin(center, cell_id)
        high = None
        minimum_step = max(self.parameters.edge_width * 0.10, 0.002)
        for _ in range(256):
            step = clamp(low_margin * 0.45, minimum_step, 0.20)
            probe_distance = min(low + step, maximum)
            probe = (
                center[0] + direction[0] * probe_distance,
                center[1] + direction[1] * probe_distance,
            )
            probe_margin = self.interior_margin(probe, cell_id)
            if probe_margin < 0.0:
                high = probe_distance
                break
            if probe_distance >= maximum - 1.0e-12:
                return probe
            low = probe_distance
            low_margin = probe_margin
        if high is None:
            raise RuntimeError(
                "Could not bracket the first boundary exit for cell {}.".format(
                    cell_id
                )
            )

        for _ in range(max(8, self.parameters.root_iterations)):
            midpoint = (low + high) * 0.5
            point = (
                center[0] + direction[0] * midpoint,
                center[1] + direction[1] * midpoint,
            )
            if self.interior_margin(point, cell_id) >= 0.0:
                low = midpoint
            else:
                high = midpoint
        distance = (low + high) * 0.5
        return (
            center[0] + direction[0] * distance,
            center[1] + direction[1] * distance,
        )

    def trace_cell(self, cell_id: CellId, center: Vec2) -> Optional[CellPolygon]:
        if self.interior_margin(center, cell_id) <= 0.0:
            return None

        base_count = max(12, int(self.parameters.initial_rays))
        tolerance = max(self.parameters.curve_tolerance, 0.0001)
        cache: Dict[float, Vec2] = {}

        def point_at(angle: float) -> Vec2:
            normalized_angle = angle % TAU
            key = round(normalized_angle, 12)
            if key not in cache:
                cache[key] = self._boundary_on_ray(
                    cell_id, center, normalized_angle
                )
            return cache[key]

        def refine(
            angle_a: float,
            point_a: Vec2,
            angle_b: float,
            point_b: Vec2,
            depth: int,
        ) -> List[Tuple[float, Vec2]]:
            midpoint_angle = (angle_a + angle_b) * 0.5
            midpoint_point = point_at(midpoint_angle)
            chord_midpoint = (
                (point_a[0] + point_b[0]) * 0.5,
                (point_a[1] + point_b[1]) * 0.5,
            )
            error = length(
                (
                    midpoint_point[0] - chord_midpoint[0],
                    midpoint_point[1] - chord_midpoint[1],
                )
            )
            # A low chord-error alone is insufficient on a concave cell: the
            # straight segment can briefly leave its owner and overlap a
            # neighboring island. Require the chord midpoint to remain inside
            # and permit a few safety subdivisions beyond the visual curve
            # refinement limit when topology needs them.
            chord_is_inside = True
            for chord_amount in (0.25, 0.50, 0.75):
                chord_probe = (
                    point_a[0] + (point_b[0] - point_a[0]) * chord_amount,
                    point_a[1] + (point_b[1] - point_a[1]) * chord_amount,
                )
                if self.interior_margin(chord_probe, cell_id) < -tolerance * 0.001:
                    chord_is_inside = False
                    break
            visual_limit = self.parameters.max_refinement
            topology_limit = visual_limit + 4
            if (error <= tolerance or depth >= visual_limit) and chord_is_inside:
                return [(angle_b, point_b)]
            if depth >= topology_limit:
                return [(angle_b, point_b)]
            return (
                refine(
                    angle_a,
                    point_a,
                    midpoint_angle,
                    midpoint_point,
                    depth + 1,
                )
                + refine(
                    midpoint_angle,
                    midpoint_point,
                    angle_b,
                    point_b,
                    depth + 1,
                )
            )

        first_angle = 0.0
        first_point = point_at(first_angle)
        samples: List[Tuple[float, Vec2]] = [(first_angle, first_point)]
        for index in range(base_count):
            angle_a = TAU * index / float(base_count)
            angle_b = TAU * (index + 1) / float(base_count)
            point_a = point_at(angle_a)
            point_b = point_at(angle_b)
            samples.extend(refine(angle_a, point_a, angle_b, point_b, 0))

        points: List[Vec2] = []
        dedupe_distance = tolerance * 0.15
        for _, point in samples[:-1]:
            if not points or length(
                (point[0] - points[-1][0], point[1] - points[-1][1])
            ) > dedupe_distance:
                points.append(point)
        if len(points) >= 2 and length(
            (points[0][0] - points[-1][0], points[0][1] - points[-1][1])
        ) <= dedupe_distance:
            points.pop()

        if len(points) < 3:
            return None
        area = 0.0
        for index, point in enumerate(points):
            next_point = points[(index + 1) % len(points)]
            area += point[0] * next_point[1] - next_point[0] * point[1]
        if abs(area) < 1.0e-8:
            return None
        if area < 0.0:
            points.reverse()

        return CellPolygon(
            cell_id=cell_id,
            color_index=self._site_info(cell_id).color_index,
            center=center,
            points=points,
        )

    def trace_all_cells(
        self,
        progress: Optional[Callable[[int, int, CellId], None]] = None,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> List[CellPolygon]:
        discovered = self.discover_cells()
        items = sorted(discovered.items(), key=lambda item: (item[0][1], item[0][0]))
        polygons: List[CellPolygon] = []
        total = len(items)
        for index, (cell_id, center) in enumerate(items):
            if cancelled is not None and cancelled():
                break
            if progress is not None:
                progress(index, total, cell_id)
            polygon = self.trace_cell(cell_id, center)
            if polygon is not None:
                polygons.append(polygon)
        if progress is not None:
            progress(total, total, (0, 0))
        return polygons


def pattern_to_world(parameters: VoronoiParameters, point: Vec2) -> Vec2:
    u = point[0] / parameters.pattern_width
    v = point[1] / parameters.pattern_height
    return (
        (u - 0.5) * parameters.width,
        (v - 0.5) * parameters.depth,
    )


def build_punched_edge_mesh(
    parameters: VoronoiParameters,
    polygons: Sequence[CellPolygon],
    height: float,
) -> MeshBuffers:
    """Build a watertight rectangular solid with every cell loop punched out.

    Cap triangulation is performed in Python so Maya receives only ordinary
    triangles and quads. This avoids Maya's version-dependent handling of one
    polygon face carrying many hole contours.
    """

    if not polygons:
        raise ValueError("The edge mesh requires at least one cell polygon.")
    height = max(float(height), 1.0e-8)
    half_width = parameters.width * 0.5
    half_depth = parameters.depth * 0.5
    domain_scale = max(parameters.width, parameters.depth, 1.0)
    root_precision = domain_scale / float(
        2 ** max(1, int(parameters.root_iterations))
    )
    cleanup_epsilon = max(
        domain_scale * 1.0e-10,
        min(domain_scale * 1.0e-4, root_precision * 2.0),
    )
    cleanup_epsilon2 = cleanup_epsilon * cleanup_epsilon

    def clean_ring(points: Sequence[Vec2]) -> List[Vec2]:
        cleaned: List[Vec2] = []
        for point in points:
            candidate = (float(point[0]), float(point[1]))
            if cleaned:
                dx = candidate[0] - cleaned[-1][0]
                dy = candidate[1] - cleaned[-1][1]
                if dx * dx + dy * dy <= cleanup_epsilon2:
                    continue
            cleaned.append(candidate)
        if len(cleaned) > 1:
            dx = cleaned[0][0] - cleaned[-1][0]
            dy = cleaned[0][1] - cleaned[-1][1]
            if dx * dx + dy * dy <= cleanup_epsilon2:
                cleaned.pop()

        changed = True
        while changed and len(cleaned) >= 3:
            changed = False
            result = []
            count = len(cleaned)
            for index, point in enumerate(cleaned):
                previous = cleaned[(index - 1) % count]
                following = cleaned[(index + 1) % count]
                previous_x = point[0] - previous[0]
                previous_y = point[1] - previous[1]
                following_x = following[0] - point[0]
                following_y = following[1] - point[1]
                cross = previous_x * following_y - previous_y * following_x
                same_direction = (
                    previous_x * following_x + previous_y * following_y
                ) >= 0.0
                scale = max(
                    abs(previous_x),
                    abs(previous_y),
                    abs(following_x),
                    abs(following_y),
                    1.0,
                )
                if abs(cross) <= cleanup_epsilon * scale and same_direction:
                    changed = True
                    continue
                result.append(point)
            cleaned = result
        if len(cleaned) < 3:
            raise ValueError("An EDGE contour collapsed below three points.")
        return cleaned

    outer_ring = [
        (-half_width, -half_depth),
        (-half_width, half_depth),
        (half_width, half_depth),
        (half_width, -half_depth),
    ]
    rings = [outer_ring]
    for polygon in polygons:
        rings.append(
            clean_ring(
                [
                    pattern_to_world(parameters, point)
                    for point in polygon.points
                ]
            )
        )

    flat_coordinates: List[float] = []
    flat_points: List[Vec2] = []
    contour_indices: List[List[int]] = []
    hole_indices: List[int] = []
    for ring_index, ring in enumerate(rings):
        if ring_index:
            hole_indices.append(len(flat_points))
        indices = []
        for point in ring:
            indices.append(len(flat_points))
            flat_points.append(point)
            flat_coordinates.extend(point)
        contour_indices.append(indices)

    triangle_indices = j_voroni_earcut.earcut(
        flat_coordinates,
        hole_indices,
        2,
    )
    if not triangle_indices or len(triangle_indices) % 3:
        raise RuntimeError("Earcut did not return complete EDGE cap triangles.")

    def signed_area(indices: Sequence[int]) -> float:
        area = 0.0
        for index, vertex_id in enumerate(indices):
            next_id = indices[(index + 1) % len(indices)]
            point = flat_points[vertex_id]
            next_point = flat_points[next_id]
            area += point[0] * next_point[1] - next_point[0] * point[1]
        return area * 0.5

    expected_cap_area = abs(signed_area(contour_indices[0]))
    for contour in contour_indices[1:]:
        expected_cap_area -= abs(signed_area(contour))
    if expected_cap_area <= 0.0:
        raise RuntimeError(
            "The inset cells consume the entire rectangular EDGE domain."
        )

    # Earcut intentionally removes redundant collinear boundary vertices.
    # Compact to the vertices it actually used, then derive wall contours from
    # the cap's real boundary edges so cap and wall topology always agree.
    used_source_ids = sorted(set(triangle_indices))
    source_to_compact = {
        source_id: compact_id
        for compact_id, source_id in enumerate(used_source_ids)
    }
    flat_points = [flat_points[source_id] for source_id in used_source_ids]
    triangle_indices = [
        source_to_compact[source_id] for source_id in triangle_indices
    ]

    vertex_count = len(flat_points)
    vertices = [(point[0], 0.0, point[1]) for point in flat_points]
    vertices.extend((point[0], height, point[1]) for point in flat_points)
    faces: List[Tuple[int, ...]] = []
    cap_area = 0.0
    cap_edge_use: Dict[Tuple[int, int], int] = {}

    for index in range(0, len(triangle_indices), 3):
        a, b, c = triangle_indices[index : index + 3]
        point_a = flat_points[a]
        point_b = flat_points[b]
        point_c = flat_points[c]
        cross_y = (
            (point_b[1] - point_a[1]) * (point_c[0] - point_a[0])
            - (point_b[0] - point_a[0]) * (point_c[1] - point_a[1])
        )
        if abs(cross_y) <= 1.0e-14:
            continue
        if cross_y < 0.0:
            b, c = c, b
            cross_y = -cross_y
        cap_area += cross_y * 0.5
        faces.append((a, c, b))
        faces.append(
            (a + vertex_count, b + vertex_count, c + vertex_count)
        )
        for vertex_a, vertex_b in ((a, b), (b, c), (c, a)):
            edge = (
                (vertex_a, vertex_b)
                if vertex_a < vertex_b
                else (vertex_b, vertex_a)
            )
            cap_edge_use[edge] = cap_edge_use.get(edge, 0) + 1

    area_tolerance = max(1.0e-7, expected_cap_area * 1.0e-6)
    if abs(cap_area - expected_cap_area) > area_tolerance:
        raise RuntimeError(
            "EDGE cap triangulation covered {:.8g} square units; expected "
            "{:.8g}.".format(cap_area, expected_cap_area)
        )

    boundary_edges = {
        edge for edge, uses in cap_edge_use.items() if uses == 1
    }
    boundary_adjacency: Dict[int, List[int]] = {}
    for vertex_a, vertex_b in boundary_edges:
        boundary_adjacency.setdefault(vertex_a, []).append(vertex_b)
        boundary_adjacency.setdefault(vertex_b, []).append(vertex_a)
    invalid_vertices = [
        vertex_id
        for vertex_id, neighbors in boundary_adjacency.items()
        if len(neighbors) != 2
    ]
    if invalid_vertices:
        raise RuntimeError(
            "EDGE cap boundary does not form closed loops at {} vertices.".format(
                len(invalid_vertices)
            )
        )

    boundary_loops: List[List[int]] = []
    unvisited_edges = set(boundary_edges)
    while unvisited_edges:
        start, current = next(iter(unvisited_edges))
        contour = [start]
        previous = start
        unvisited_edges.remove((min(start, current), max(start, current)))
        while current != start:
            contour.append(current)
            neighbors = boundary_adjacency[current]
            following = neighbors[0] if neighbors[0] != previous else neighbors[1]
            edge = (min(current, following), max(current, following))
            if edge not in unvisited_edges:
                if following != start:
                    raise RuntimeError("EDGE cap boundary loop terminated early.")
            else:
                unvisited_edges.remove(edge)
            previous, current = current, following
        boundary_loops.append(contour)

    outer_loop_index = max(
        range(len(boundary_loops)),
        key=lambda index: abs(signed_area(boundary_loops[index])),
    )
    for contour_index, source_contour in enumerate(boundary_loops):
        contour = list(source_contour)
        contour_area = signed_area(contour)
        if (contour_index == outer_loop_index and contour_area > 0.0) or (
            contour_index != outer_loop_index and contour_area < 0.0
        ):
            contour.reverse()
        for index, bottom_a in enumerate(contour):
            bottom_b = contour[(index + 1) % len(contour)]
            faces.append(
                (
                    bottom_a,
                    bottom_b,
                    bottom_b + vertex_count,
                    bottom_a + vertex_count,
                )
            )

    edge_use: Dict[Tuple[int, int], int] = {}
    for face in faces:
        for index, vertex_a in enumerate(face):
            vertex_b = face[(index + 1) % len(face)]
            edge = (
                (vertex_a, vertex_b)
                if vertex_a < vertex_b
                else (vertex_b, vertex_a)
            )
            edge_use[edge] = edge_use.get(edge, 0) + 1
    non_manifold_edges = [edge for edge, uses in edge_use.items() if uses != 2]
    if non_manifold_edges:
        examples = [
            "{}:{}".format(edge, edge_use[edge])
            for edge in non_manifold_edges[:8]
        ]
        raise RuntimeError(
            "EDGE triangulation produced {} non-watertight edges ({}).".format(
                len(non_manifold_edges),
                ", ".join(examples),
            )
        )

    polygon_counts = [len(face) for face in faces]
    polygon_connects = [vertex_id for face in faces for vertex_id in face]
    return MeshBuffers(vertices, polygon_counts, polygon_connects)


def render_preview_rgb(
    parameters: VoronoiParameters,
    pixel_width: int,
    pixel_height: int,
    flow_samples_per_unit: float = 4.0,
    cancelled: Optional[Callable[[], bool]] = None,
) -> Optional[bytearray]:
    """Render the field into a tightly packed top-down RGB byte buffer.

    This path is deliberately preview-specific. It preserves the exact sites,
    weights, palette selection, edge-width calculation, and rounded Voronoi
    metric, while bilinearly interpolating the smoothly varying flow frame
    from a compact grid. Geometry tracing continues to use ``evaluate()`` and
    is therefore unaffected by this optimization.
    """

    pixel_width = max(1, int(pixel_width))
    pixel_height = max(1, int(pixel_height))
    pattern_width = parameters.pattern_width
    pattern_height = parameters.pattern_height
    field = VoronoiField(parameters)

    flow_columns = min(
        pixel_width + 1,
        max(2, int(math.ceil(pattern_width * flow_samples_per_unit)) + 1),
    )
    flow_rows = min(
        pixel_height + 1,
        max(2, int(math.ceil(pattern_height * flow_samples_per_unit)) + 1),
    )
    flow_grid: List[List[Tuple[float, float, float]]] = []
    for row in range(flow_rows):
        if cancelled is not None and cancelled():
            return None
        pattern_y = pattern_height * row / float(flow_rows - 1)
        flow_row = []
        for column in range(flow_columns):
            pattern_x = pattern_width * column / float(flow_columns - 1)
            flow_row.append(field.flow_frame((pattern_x, pattern_y)))
        flow_grid.append(flow_row)

    x_samples = []
    for image_x in range(pixel_width):
        grid_x = (image_x + 0.5) * (flow_columns - 1) / float(pixel_width)
        x0 = min(int(grid_x), flow_columns - 2)
        x_samples.append((x0, x0 + 1, grid_x - x0))

    y_samples = []
    for image_y in range(pixel_height):
        pattern_y = pattern_height * (
            pixel_height - image_y - 0.5
        ) / pixel_height
        grid_y = pattern_y * (flow_rows - 1) / pattern_height
        y0 = min(int(grid_y), flow_rows - 2)
        y_samples.append((pattern_y, y0, y0 + 1, grid_y - y0))

    variation = max(parameters.size_variation, 0.0)
    candidate_radius = 2 + int(math.ceil(math.sqrt(variation)))
    candidate_cache: Dict[CellId, List[Tuple[float, float, float, int]]] = {}

    def candidates_for(point_x: float, point_y: float):
        tile_id = (int(math.floor(point_x)), int(math.floor(point_y)))
        candidates = candidate_cache.get(tile_id)
        if candidates is not None:
            return candidates
        candidates = []
        for cell_y in range(
            tile_id[1] - candidate_radius,
            tile_id[1] + candidate_radius + 1,
        ):
            for cell_x in range(
                tile_id[0] - candidate_radius,
                tile_id[0] + candidate_radius + 1,
            ):
                info = field._site_info((cell_x, cell_y))
                candidates.append(
                    (
                        info.position[0],
                        info.position[1],
                        info.weight,
                        info.color_index,
                    )
                )
        candidate_cache[tile_id] = candidates
        return candidates

    tributary = clamp(parameters.tributary_bias, 0.0, 1.0)
    parallelism = clamp(parameters.channel_parallelism, 0.0, 1.0)
    metric_aspect = 1.0 + 8.0 * tributary + 14.0 * parallelism
    area_scale = math.sqrt(metric_aspect)
    inverse_area_scale = 1.0 / area_scale
    alignment_amount = 0.88 * tributary + 0.96 * parallelism
    shape_smoothness = clamp(parameters.shape_smoothness, 0.0, 1.0)
    rounding = max(0.001, 0.28 * shape_smoothness)
    half_edge_width = max(parameters.edge_width, 0.001) * 0.5
    pixel_size = max(
        pattern_width / float(pixel_width),
        pattern_height / float(pixel_height),
    )
    anti_alias = max(pixel_size * 0.70, 0.001)
    inverse_anti_alias_width = 1.0 / (2.0 * anti_alias)
    pixels = bytearray(pixel_width * pixel_height * 3)
    metrics = [0.0] * ((candidate_radius * 2 + 1) ** 2)
    distances = [0.0] * len(metrics)

    for image_y, (pattern_y, y0, y1, amount_y) in enumerate(y_samples):
        if cancelled is not None and cancelled():
            return None
        flow_row_0 = flow_grid[y0]
        flow_row_1 = flow_grid[y1]
        row_offset = image_y * pixel_width * 3

        for image_x, (x0, x1, amount_x) in enumerate(x_samples):
            pattern_x = pattern_width * (image_x + 0.5) / pixel_width
            flow_00 = flow_row_0[x0]
            flow_10 = flow_row_0[x1]
            flow_01 = flow_row_1[x0]
            flow_11 = flow_row_1[x1]
            lower_x = flow_00[0] + (flow_10[0] - flow_00[0]) * amount_x
            lower_y = flow_00[1] + (flow_10[1] - flow_00[1]) * amount_x
            lower_confidence = flow_00[2] + (
                flow_10[2] - flow_00[2]
            ) * amount_x
            upper_x = flow_01[0] + (flow_11[0] - flow_01[0]) * amount_x
            upper_y = flow_01[1] + (flow_11[1] - flow_01[1]) * amount_x
            upper_confidence = flow_01[2] + (
                flow_11[2] - flow_01[2]
            ) * amount_x
            flow_x = lower_x + (upper_x - lower_x) * amount_y
            flow_y = lower_y + (upper_y - lower_y) * amount_y
            confidence = lower_confidence + (
                upper_confidence - lower_confidence
            ) * amount_y
            flow_length = math.sqrt(flow_x * flow_x + flow_y * flow_y)
            if flow_length > 1.0e-12:
                flow_x /= flow_length
                flow_y /= flow_length
            else:
                flow_x, flow_y = 0.0, 1.0
            normal_x, normal_y = -flow_y, flow_x
            alignment = clamp(alignment_amount * confidence, 0.0, 1.0)

            candidates = candidates_for(pattern_x, pattern_y)
            nearest_metric = float("inf")
            nearest_distance = float("inf")
            winner_index = 0
            for candidate_index, candidate in enumerate(candidates):
                dx = pattern_x - candidate[0]
                dy = pattern_y - candidate[1]
                distance = math.sqrt(dx * dx + dy * dy)
                point_metric = dx * dx + dy * dy
                along = dx * flow_x + dy * flow_y
                across = dx * normal_x + dy * normal_y
                aligned_metric = (
                    across * across * area_scale
                    + along * along * inverse_area_scale
                )
                metric = point_metric + (
                    aligned_metric - point_metric
                ) * alignment
                metric -= variation * candidate[2]
                metrics[candidate_index] = metric
                distances[candidate_index] = distance
                if metric < nearest_metric:
                    nearest_metric = metric
                    nearest_distance = distance
                    winner_index = candidate_index

            angular_gap = float("inf")
            rounded_gap = float("inf")
            for candidate_index in range(len(candidates)):
                if candidate_index == winner_index:
                    continue
                gap = (metrics[candidate_index] - nearest_metric) / max(
                    distances[candidate_index] + nearest_distance,
                    0.001,
                )
                if gap < angular_gap:
                    angular_gap = gap
                blend = max(rounding - abs(rounded_gap - gap), 0.0) / rounding
                rounded_gap = min(rounded_gap, gap) - (
                    blend * blend * rounding * 0.25
                )

            boundary = angular_gap + (
                rounded_gap - angular_gap
            ) * shape_smoothness
            domain_margin = min(
                pattern_x,
                pattern_width - pattern_x,
                pattern_y,
                pattern_height - pattern_y,
            )
            margin = min(boundary, domain_margin) - half_edge_width
            amount = clamp(
                (margin + anti_alias) * inverse_anti_alias_width,
                0.0,
                1.0,
            )
            amount = amount * amount * (3.0 - 2.0 * amount)
            faded_channel = int(round(255.0 * (1.0 - amount)))
            color_index = candidates[winner_index][3]
            pixel_offset = row_offset + image_x * 3
            pixels[pixel_offset] = 255 if color_index == 0 else faded_channel
            pixels[pixel_offset + 1] = 255 if color_index == 1 else faded_channel
            pixels[pixel_offset + 2] = 255 if color_index == 2 else faded_channel

    return pixels
