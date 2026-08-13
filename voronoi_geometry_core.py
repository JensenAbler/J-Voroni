"""Dependency-free mathematical core for the Maya Voronoi geometry baker.

This module contains no Maya or Qt imports. It evaluates the same competitive
distance field as the viewport preview and traces colored cell interiors by
solving the field directly. Raster images are not used to create vertices.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


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
        self._segment_cache: Dict[CellId, List[Tuple[float, float, float, float, float]]] = {}

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
    ) -> List[Tuple[float, float, float, float, float]]:
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

        segments: List[Tuple[float, float, float, float, float]] = []
        root = global_node(0.0, 1.0, 0.18)
        segments.append((root[0], origin[1], root[0], root[1], 3.0))

        for child in range(2):
            end = global_node(float(child), 2.0, 0.40)
            segments.append((root[0], root[1], end[0], end[1], 2.0))
        for child in range(4):
            start = global_node(float(child // 2), 2.0, 0.40)
            end = global_node(float(child), 4.0, 0.69)
            segments.append((start[0], start[1], end[0], end[1], 1.0))
        for child in range(8):
            start = global_node(float(child // 2), 4.0, 0.69)
            end = global_node(float(child), 8.0, 0.97)
            segments.append((start[0], start[1], end[0], end[1], 0.0))

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
                for ax, ay, bx, by, level in self._segments_for_tile(
                    (tile_x, tile_y)
                ):
                    sx = bx - ax
                    sy = by - ay
                    segment_length2 = max(sx * sx + sy * sy, 0.0001)
                    t = clamp(
                        ((point[0] - ax) * sx + (point[1] - ay) * sy)
                        / segment_length2,
                        0.0,
                        1.0,
                    )
                    ox = point[0] - (ax + sx * t)
                    oy = point[1] - (ay + sy * t)
                    weight = math.exp(-(ox * ox + oy * oy) * 0.82)
                    weight *= 1.0 + level * 0.11
                    direction = normalize((sx, sy))
                    orientation_x += weight * (
                        direction[0] * direction[0]
                        - direction[1] * direction[1]
                    )
                    orientation_y += weight * (
                        2.0 * direction[0] * direction[1]
                    )
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

        edge_point = (
            center[0] + direction[0] * maximum,
            center[1] + direction[1] * maximum,
        )
        probe_distance = max(0.0, maximum - 1.0e-7)
        probe = (
            center[0] + direction[0] * probe_distance,
            center[1] + direction[1] * probe_distance,
        )
        if self.interior_margin(probe, cell_id) >= 0.0:
            return edge_point

        low = 0.0
        high = maximum
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
            if error <= tolerance or depth >= self.parameters.max_refinement:
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
