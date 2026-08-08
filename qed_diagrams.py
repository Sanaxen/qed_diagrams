from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from itertools import permutations
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from feynman import Diagram as FeynmanDiagram
from pyfeyn2.auto.bend import auto_bend
from pyfeyn2.feynmandiagram import (
    FeynmanDiagram as FeynMLDiagram,
    Leg as FeynMLLeg,
    Propagator as FeynMLPropagator,
    Vertex as FeynMLVertex,
)


@dataclass(frozen=True)
class DiagramRequest:
    electron_in: int
    photon_in: int
    electron_out: int
    photon_out: int
    vertices: int
    positron_in: int = 0
    positron_out: int = 0
    loops: int = 0


@dataclass
class Diagram:
    graph: nx.MultiGraph
    external: list[tuple[int, str, str]]


def loop_order_from_counts(external_lines: int, vertices: int) -> int | None:
    numerator = vertices - external_lines + 2
    return numerator // 2 if numerator % 2 == 0 else None


def vertex_count_from_loops(external_lines: int, loops: int) -> int:
    return external_lines + 2 * loops - 2


def validate_request(req: DiagramRequest) -> tuple[bool, str, int, int]:
    ef = req.electron_in + req.electron_out + req.positron_in + req.positron_out
    photons = req.photon_in + req.photon_out
    if req.electron_in + req.positron_out != req.electron_out + req.positron_in:
        return False, "Charge is not conserved. The condition e⁻(in) + e⁺(out) = e⁻(out) + e⁺(in) is required.", 0, 0
    if photons > req.vertices:
        return False, "The number of external photons cannot exceed the number of vertices.", 0, 0
    if ef > 2 * req.vertices:
        return False, "The number of external fermion lines exceeds the capacity of the vertices.", 0, 0
    photon_stubs = req.vertices - photons
    fermion_stubs = 2 * req.vertices - ef
    if photon_stubs % 2:
        return False, "Vertex count minus external photon count must be even.", 0, 0
    if fermion_stubs % 2:
        return False, "Twice the vertex count minus the external fermion count must be even.", 0, 0
    if req.vertices == 0:
        return False, "Vertex count must be at least 1.", 0, 0
    internal_fermions = fermion_stubs // 2
    internal_photons = photon_stubs // 2
    calculated_loops = internal_fermions + internal_photons - req.vertices + 1
    if calculated_loops < 0:
        return False, "These external-line and vertex counts cannot form a connected diagram.", 0, 0
    if req.loops != calculated_loops:
        return False, f"These conditions imply loop order L={calculated_loops}. Change either the vertex count or loop order.", 0, 0
    return True, "The requested perturbative order is valid.", internal_fermions, internal_photons


def _pair_stubs(stubs: list[int], rng: random.Random) -> list[tuple[int, int]] | None:
    for _ in range(120):
        pool = stubs[:]
        rng.shuffle(pool)
        pairs = []
        valid = True
        while pool:
            a = pool.pop()
            choices = [i for i, b in enumerate(pool) if b != a]
            if not choices:
                valid = False
                break
            idx = rng.choice(choices)
            pairs.append((a, pool.pop(idx)))
        if valid:
            return pairs
    return None


def _pair_fermion_stubs(outgoing: list[int], incoming: list[int], rng: random.Random) -> list[tuple[int, int]] | None:
    """Pair fermion arrows while preserving one-in/one-out at every QED vertex."""
    if len(outgoing) != len(incoming):
        return None
    for _ in range(120):
        targets = incoming[:]
        rng.shuffle(targets)
        pairs = list(zip(outgoing, targets))
        if all(source != target for source, target in pairs):
            return pairs
    return None


def _external_assignments(
    vertices: int, external_types: list[tuple[str, str]]
) -> list[list[tuple[int, str, str]]]:
    """Canonical external-leg placements, modulo vertex relabeling."""
    results: list[list[tuple[int, str, str]]] = []
    assigned: list[tuple[int, str, str]] = []
    occupied = {"fermion_in": set(), "fermion_out": set(), "photon": set()}

    def capacity(kind: str, direction: str) -> str:
        if kind == "photon":
            return "photon"
        inward = (kind == "electron" and direction == "in") or (kind == "positron" and direction == "out")
        return "fermion_in" if inward else "fermion_out"

    def visit(index: int, greatest: int) -> None:
        if index == len(external_types):
            results.append(assigned[:])
            return
        kind, direction = external_types[index]
        slot = capacity(kind, direction)
        upper = min(vertices - 1, greatest + 1)
        lower = 0
        if index and external_types[index - 1] == (kind, direction):
            lower = assigned[-1][0]
        for vertex in range(lower, upper + 1):
            if vertex in occupied[slot]:
                continue
            occupied[slot].add(vertex)
            assigned.append((vertex, kind, direction))
            visit(index + 1, max(greatest, vertex))
            assigned.pop()
            occupied[slot].remove(vertex)

    visit(0, -1)
    return results


def _perfect_matchings(stubs: tuple[int, ...]):
    """Yield every unordered photon pairing without self-contractions."""
    if not stubs:
        yield []
        return
    first = stubs[0]
    for index in range(1, len(stubs)):
        second = stubs[index]
        if first == second:
            continue
        rest = stubs[1:index] + stubs[index + 1:]
        for matching in _perfect_matchings(rest):
            yield [(first, second), *matching]


def _incidence_graph(diagram: Diagram) -> nx.DiGraph:
    h = nx.DiGraph()
    for v in diagram.graph.nodes:
        h.add_node(f"v{v}", kind="vertex")
    for i, (a, b, data) in enumerate(diagram.graph.edges(data=True, keys=False)):
        edge_node = f"edge{i}"
        h.add_node(edge_node, kind=data["kind"])
        if data["kind"] == "fermion":
            h.add_edge(f"v{data['source']}", edge_node)
            h.add_edge(edge_node, f"v{data['target']}")
        else:
            h.add_edge(f"v{a}", edge_node)
            h.add_edge(edge_node, f"v{b}")
            h.add_edge(f"v{b}", edge_node)
            h.add_edge(edge_node, f"v{a}")
    for i, (v, kind, direction) in enumerate(diagram.external):
        node = f"ext{i}"
        h.add_node(node, kind=f"{kind}_{direction}")
        h.add_edge(f"v{v}", node)
        h.add_edge(node, f"v{v}")
    return h


def _signature(diagram: Diagram) -> str:
    return nx.weisfeiler_lehman_graph_hash(_incidence_graph(diagram), node_attr="kind")


def _is_equivalent(left: Diagram, right: Diagram) -> bool:
    return nx.is_isomorphic(
        _incidence_graph(left),
        _incidence_graph(right),
        node_match=lambda a, b: a["kind"] == b["kind"],
    )


def is_one_particle_irreducible(diagram: Diagram) -> bool:
    """Return True iff removing any one internal propagator keeps the graph connected."""
    graph = diagram.graph
    if len(graph.nodes) > 1 and not nx.is_connected(nx.Graph(graph)):
        return False
    for source, target, key in list(graph.edges(keys=True)):
        reduced = graph.copy()
        reduced.remove_edge(source, target, key)
        if len(reduced.nodes) > 1 and not nx.is_connected(nx.Graph(reduced)):
            return False
    return True


def passes_furry_theorem(diagram: Diagram) -> bool:
    """Reject closed fermion loops with an odd number of photon vertices."""
    fermions = nx.DiGraph()
    fermions.add_nodes_from(diagram.graph.nodes)
    for source, target, data in diagram.graph.edges(data=True):
        if data["kind"] == "fermion":
            fermions.add_edge(data["source"], data["target"])
    return all(len(cycle) % 2 == 0 for cycle in nx.simple_cycles(fermions))


def generate_diagrams(req: DiagramRequest, limit: int = 12, seed: int = 7) -> list[Diagram]:
    ok, _, _, _ = validate_request(req)
    if not ok:
        return []
    found: dict[str, list[Diagram]] = {}
    vertices = list(range(req.vertices))

    external_types = (
        [("electron", "in")] * req.electron_in
        + [("electron", "out")] * req.electron_out
        + [("positron", "in")] * req.positron_in
        + [("positron", "out")] * req.positron_out
        + [("photon", "in")] * req.photon_in
        + [("photon", "out")] * req.photon_out
    )
    for external in _external_assignments(req.vertices, external_types):
        fermion_in_capacity = {v: 1 for v in vertices}
        fermion_out_capacity = {v: 1 for v in vertices}
        photon_capacity = {v: 1 for v in vertices}
        for v, kind, direction in external:
            if kind == "photon":
                cap = photon_capacity
            else:
                arrow_inward = (kind == "electron" and direction == "in") or (kind == "positron" and direction == "out")
                cap = fermion_in_capacity if arrow_inward else fermion_out_capacity
            cap[v] -= 1

        fermion_in_stubs = [v for v in vertices if fermion_in_capacity[v]]
        fermion_out_stubs = [v for v in vertices if fermion_out_capacity[v]]
        photon_stubs = tuple(v for v in vertices for _ in range(photon_capacity[v]))
        photon_pairings = list(_perfect_matchings(photon_stubs))
        for targets in permutations(fermion_in_stubs):
            fpairs = list(zip(fermion_out_stubs, targets))
            if any(source == target for source, target in fpairs):
                continue
            for ppairs in photon_pairings:
                graph = nx.MultiGraph()
                graph.add_nodes_from(vertices)
                for source, target in fpairs:
                    graph.add_edge(source, target, kind="fermion", source=source, target=target)
                graph.add_edges_from(ppairs, kind="photon")
                if req.vertices > 1 and not nx.is_connected(nx.Graph(graph)):
                    continue
                diagram = Diagram(graph, external[:])
                if not passes_furry_theorem(diagram):
                    continue
                if not is_one_particle_irreducible(diagram):
                    continue
                signature = _signature(diagram)
                bucket = found.setdefault(signature, [])
                if not any(_is_equivalent(diagram, existing) for existing in bucket):
                    bucket.append(diagram)
                    if sum(len(items) for items in found.values()) >= limit:
                        return [item for items in found.values() for item in items]
    return [diagram for bucket in found.values() for diagram in bucket]


def draw_diagram(diagram: Diagram, title: str = ""):
    graph = diagram.graph
    n = len(graph.nodes)
    fermion_flow = nx.DiGraph()
    fermion_flow.add_nodes_from(graph.nodes)
    for _, _, data in graph.edges(data=True):
        if data["kind"] == "fermion":
            fermion_flow.add_edge(data["source"], data["target"])
    fermion_cycles = list(nx.simple_cycles(fermion_flow))
    charged_in = [
        v for v, kind, direction in diagram.external
        if (kind == "electron" and direction == "in") or (kind == "positron" and direction == "out")
    ]
    charged_out = [
        v for v, kind, direction in diagram.external
        if (kind == "electron" and direction == "out") or (kind == "positron" and direction == "in")
    ]

    # Put the open fermion backbone on a horizontal time axis. Closed-loop
    # vertices are lifted away from it, making loop structure immediately visible.
    backbone: list[int] = []
    if len(charged_in) == len(charged_out) == 1 and nx.has_path(fermion_flow, charged_in[0], charged_out[0]):
        backbone = nx.shortest_path(fermion_flow, charged_in[0], charged_out[0])
        remaining = [v for v in graph.nodes if v not in backbone]
        simple = nx.Graph(graph)
        planar, embedding = nx.check_planarity(simple)
        if planar:
            initial = {v: np.asarray(xy, dtype=float) for v, xy in nx.planar_layout(embedding).items()}
        else:
            initial = {v: np.asarray(xy, dtype=float) for v, xy in nx.kamada_kawai_layout(simple).items()}

        # Every vertex on the continuous open fermion chain is fixed to the
        # horizontal time axis. Only the remaining loop vertices are relaxed.
        for vertex, x in zip(backbone, np.linspace(-1.0, 1.0, len(backbone))):
            initial[vertex] = np.array([x, 0.0])
        raw_pos = {
            v: np.asarray(xy, dtype=float)
            for v, xy in nx.spring_layout(
                simple,
                pos=initial,
                fixed=backbone,
                seed=19,
                iterations=300,
                k=0.72,
            ).items()
        }
        if remaining and np.mean([raw_pos[v][1] for v in remaining]) < 0:
            raw_pos = {v: np.array([xy[0], -xy[1]]) for v, xy in raw_pos.items()}
    elif n == 2:
        raw_pos = {0: np.array([0.0, 1.0]), 1: np.array([0.0, -1.0])}
    else:
        simple = nx.Graph(graph)
        planar, embedding = nx.check_planarity(simple)
        raw_pos = nx.planar_layout(embedding) if planar else nx.kamada_kawai_layout(simple)
    raw = np.array(list(raw_pos.values()))
    center_raw = (raw.min(axis=0) + raw.max(axis=0)) / 2
    scale = max(np.ptp(raw[:, 0]), np.ptp(raw[:, 1]), 1.0)
    pos = {v: 0.5 + 0.38 * (np.asarray(xy) - center_raw) / scale for v, xy in raw_pos.items()}

    # A two-vertex vacuum-polarization loop can collapse into an almost
    # vertical pair during force-directed layout.  Give it a stable horizontal
    # diameter so its two fermion arcs and attached photons remain separable.
    for cycle in fermion_cycles:
        if len(cycle) != 2 or any(vertex in backbone for vertex in cycle):
            continue
        left_vertex, right_vertex = sorted(cycle)
        loop_center = 0.5 * (pos[left_vertex] + pos[right_vertex])
        half_width = max(0.085, 0.5 * abs(pos[right_vertex][0] - pos[left_vertex][0]))
        pos[left_vertex] = loop_center + np.array([-half_width, 0.0])
        pos[right_vertex] = loop_center + np.array([half_width, 0.0])
    fig, ax = plt.subplots(figsize=(8.4, 5.2), facecolor="#0e1422")
    ax.set_facecolor("#0e1422")
    renderer = FeynmanDiagram(ax)
    vertex_objects = {
        v: renderer.vertex(xy=pos[v], marker="o", markersize=7, color="#ff6b7a", zorder=20)
        for v in graph.nodes
    }
    center = np.mean(np.array(list(pos.values())), axis=0)
    counters = {(particle, direction): 0 for particle in ("electron", "positron", "photon") for direction in ("in", "out")}
    external_by_vertex: dict[int, list[int]] = {v: [] for v in graph.nodes}
    for index, (v, _, _) in enumerate(diagram.external):
        external_by_vertex[v].append(index)

    def external_route_score(vertex: int, endpoint: np.ndarray) -> float:
        """Penalize external-leg routes that pass through internal geometry."""
        start = pos[vertex]
        route = endpoint - start
        route_length_sq = float(np.dot(route, route))
        score = 0.0
        for other, point in pos.items():
            if other == vertex:
                continue
            t = np.clip(np.dot(point - start, route) / max(route_length_sq, 1e-12), 0.0, 1.0)
            distance = np.linalg.norm(point - (start + t * route))
            score += max(0.0, 0.13 - distance) * 80.0

        def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
            ab, ac = b - a, c - a
            return float(ab[0] * ac[1] - ab[1] * ac[0])

        for left, right in graph.edges():
            if vertex in {left, right}:
                continue
            a, b = pos[left], pos[right]
            o1, o2 = orientation(start, endpoint, a), orientation(start, endpoint, b)
            o3, o4 = orientation(a, b, start), orientation(a, b, endpoint)
            if o1 * o2 < 0 and o3 * o4 < 0:
                score += 100.0
        # In a tie, prefer the side facing away from the diagram center.
        score -= 0.2 * float(np.dot(endpoint - start, start - center))
        return score

    endpoints: dict[int, np.ndarray] = {}
    for v, indices in external_by_vertex.items():
        radial = pos[v] - center
        if backbone and v == charged_in[0]:
            radial = np.array([-1.0, -0.18])
        elif backbone and v == charged_out[0]:
            radial = np.array([1.0, -0.18])
        if np.linalg.norm(radial) < 0.1:
            radial = np.array([0.0, 1.0])
        base_angle = np.arctan2(radial[1], radial[0])
        fan = np.linspace(-0.58, 0.58, len(indices)) if len(indices) > 1 else [0.0]
        for index, angle_offset in zip(indices, fan):
            _, kind, _ = diagram.external[index]
            particle_angle = base_angle
            if kind == "photon":
                fermion_neighbors = []
                for left, right, data in graph.edges(v, data=True):
                    if data["kind"] == "fermion":
                        fermion_neighbors.append(right if left == v else left)
                unique_neighbors = list(dict.fromkeys(fermion_neighbors))
                if len(unique_neighbors) >= 2:
                    tangent = pos[unique_neighbors[-1]] - pos[unique_neighbors[0]]
                elif unique_neighbors:
                    tangent = pos[v] - pos[unique_neighbors[0]]
                else:
                    tangent = np.array([1.0, 0.0])
                normal = np.array([-tangent[1], tangent[0]])
                if np.linalg.norm(normal) > 1e-9:
                    normal /= np.linalg.norm(normal)
                    candidates = (normal, -normal)
                    normal = min(
                        candidates,
                        key=lambda candidate: external_route_score(v, pos[v] + 0.24 * candidate),
                    )
                    particle_angle = np.arctan2(normal[1], normal[0])
            angle = particle_angle + angle_offset
            endpoints[index] = pos[v] + 0.24 * np.array([np.cos(angle), np.sin(angle)])

    # Convert to pyfeyn2/FeynML so topology and bending metadata use the
    # standard open-source representation.
    feynml = FeynMLDiagram().with_id("qed-diagram")
    feynml_vertices = {v: FeynMLVertex().with_id(f"v{v}").with_xy(*pos[v]) for v in graph.nodes}
    feynml.vertices.extend(feynml_vertices.values())
    edge_records = list(graph.edges(data=True, keys=True))
    for edge_index, (a, b, _, data) in enumerate(edge_records):
        source, target = data.get("source", a), data.get("target", b)
        feynml.propagators.append(
            FeynMLPropagator()
            .with_id(f"p{edge_index}")
            .connect(feynml_vertices[source], feynml_vertices[target])
            .with_type("fermion" if data["kind"] == "fermion" else "photon")
        )
    for index, (v, kind, direction) in enumerate(diagram.external):
        leg_type = "photon" if kind == "photon" else ("anti fermion" if kind == "positron" else "fermion")
        leg = FeynMLLeg().with_id(f"l{index}").with_target(feynml_vertices[v]).with_xy(*endpoints[index]).with_type(leg_type)
        (leg.with_incoming() if direction == "in" else leg.with_outgoing())
        feynml.legs.append(leg)
    auto_bend(feynml)
    bend_direction = {}
    for index, propagator in enumerate(feynml.propagators):
        bend = propagator.style.getProperty("bend-direction")
        bend_direction[index] = bend.value if bend is not None else None

    cycle_edges = set()
    for cycle in fermion_cycles:
        for source, target in zip(cycle, cycle[1:] + cycle[:1]):
            cycle_edges.add((source, target))

    grouped: dict[tuple[int, int], list[tuple[int, dict]]] = {}
    for edge_index, (a, b, _, data) in enumerate(edge_records):
        grouped.setdefault(tuple(sorted((a, b))), []).append((edge_index, data))

    # Reserve the side used by an external photon and route internal photons
    # primarily to the opposite side. Crossing intervals alone switch sides.
    backbone_index = {vertex: index for index, vertex in enumerate(backbone)}
    external_photon_sides = []
    for index, (vertex, kind, _) in enumerate(diagram.external):
        if kind == "photon" and vertex in backbone_index:
            external_photon_sides.append(1 if endpoints[index][1] >= pos[vertex][1] else -1)
    preferred_side = -external_photon_sides[0] if external_photon_sides else 1
    photon_lanes: dict[int, int] = {}
    fermion_lanes: dict[int, int] = {}

    def outward_lane(a: int, b: int) -> int:
        """Return the curve side pointing away from the nearest local loop."""
        canonical_a, canonical_b = sorted((a, b))
        segment = pos[canonical_b] - pos[canonical_a]
        normal = np.array([-segment[1], segment[0]])
        midpoint = 0.5 * (pos[canonical_a] + pos[canonical_b])
        containing_cycles = [cycle for cycle in fermion_cycles if a in cycle and b in cycle]
        if containing_cycles:
            local_cycle = min(containing_cycles, key=len)
            reference_center = np.mean([pos[vertex] for vertex in local_cycle], axis=0)
        else:
            reference_center = center
        return 1 if np.dot(normal, midpoint - reference_center) >= 0 else -1

    intervals = []
    for edge_index, (a, b, _, data) in enumerate(edge_records):
        if data["kind"] == "fermion":
            source, target = data.get("source", a), data.get("target", b)
            if (source, target) in cycle_edges and not (a in backbone_index and b in backbone_index):
                fermion_lanes[edge_index] = outward_lane(a, b)
            continue
        if len(grouped[tuple(sorted((a, b)))]) > 1:
            continue
        if a in backbone_index and b in backbone_index:
            left, right = sorted((backbone_index[a], backbone_index[b]))
            intervals.append((edge_index, left, right))
        else:
            # Away from the horizontal fermion backbone, bend each photon
            # toward the outside of the diagram.  Thus photon lines facing one
            # another on opposite sides of a loop bulge away from each other
            # instead of both intruding into the loop's interior.
            photon_lanes[edge_index] = outward_lane(a, b)

    assigned_intervals: list[tuple[int, int, int]] = []
    for edge_index, left, right in sorted(intervals, key=lambda item: (-(item[2] - item[1]), item[0])):
        side = preferred_side
        for old_left, old_right, old_lane in assigned_intervals:
            crosses = (left < old_left < right < old_right) or (old_left < left < old_right < right)
            if crosses:
                side = -1 if old_lane > 0 else 1
                break
        overlapping_levels = [
            abs(old_lane) for old_left, old_right, old_lane in assigned_intervals
            if (old_lane > 0) == (side > 0) and not (right <= old_left or old_right <= left)
        ]
        level = max(overlapping_levels, default=0) + 1
        lane = side * level
        photon_lanes[edge_index] = lane
        assigned_intervals.append((left, right, lane))

    for (a, b), edges in grouped.items():
        parallel_lanes: dict[int, int] = {}
        parallel_spreads: dict[int, float] = {}
        if len(edges) > 1:
            fermion_indices = [edge_index for edge_index, data in edges if data["kind"] == "fermion"]
            photon_indices = [edge_index for edge_index, data in edges if data["kind"] == "photon"]
            if len(fermion_indices) >= 2:
                for lane, edge_index in zip((1, -1, 2, -2), fermion_indices):
                    parallel_lanes[edge_index] = lane
                for lane, edge_index in zip((2, -2, 3, -3), photon_indices):
                    parallel_lanes[edge_index] = lane
                # A vacuum-polarization subgraph has two oppositely directed
                # fermion arcs and one or more photon lines sharing endpoints.
                # Keep the fermion loop compact and put photons clearly outside
                # it; otherwise the wavy line appears to weave through the loop.
                for edge_index in fermion_indices:
                    parallel_spreads[edge_index] = 0.30
                for offset, edge_index in enumerate(photon_indices):
                    parallel_spreads[edge_index] = min(0.82, 0.68 + 0.07 * offset)
            else:
                for lane, (edge_index, _) in zip((1, -1, 2, -2), edges):
                    parallel_lanes[edge_index] = lane
        for local_index, (edge_index, data) in enumerate(edges):
            source, target = data.get("source", a), data.get("target", b)
            curved = len(edges) > 1 or data["kind"] == "photon" or (source, target) in cycle_edges
            common = dict(linewidth=2.1, color="#9ee6ff" if data["kind"] == "fermion" else "#ffd36a")
            if curved:
                lane = parallel_lanes.get(edge_index, photon_lanes.get(edge_index, fermion_lanes.get(edge_index)))
                if lane is not None:
                    side = "up" if lane > 0 else "down"
                    spread = parallel_spreads.get(
                        edge_index,
                        min(0.49, (0.41 if len(edges) > 1 else 0.36) + 0.055 * (abs(lane) - 1)),
                    )
                else:
                    bend = bend_direction.get(edge_index)
                    side = bend if bend in {"left", "right"} else ("up" if (edge_index + local_index) % 2 == 0 else "down")
                    spread = 0.34
                start_node, end_node = (source, target) if data["kind"] == "fermion" else (a, b)
                if lane is not None:
                    # pyfeyn2 interprets the curvature sign relative to the
                    # directed line. Direction-aware signs keep photon and
                    # fermion curves on their intended geometric sides.
                    direction_factor = 1.0 if (start_node, end_node) == (a, b) else -1.0
                    curvature_sign = direction_factor if lane > 0 else -direction_factor
                else:
                    orientation = 1.0 if pos[end_node][0] >= pos[start_node][0] else -1.0
                    curvature_sign = orientation if side in {"left", "up"} else -orientation
                common.update(
                    shape="elliptic",
                    ellipse_spread=spread,
                    ellipse_excentricity=1.18 * curvature_sign,
                )
            if data["kind"] == "fermion":
                renderer.line(vertex_objects[source], vertex_objects[target], flavour="simple", arrow=True,
                              arrow_param={"color": "#9ee6ff", "width": 0.024, "length": 0.065}, **common)
            else:
                renderer.line(vertex_objects[a], vertex_objects[b], flavour="wiggly", arrow=False, amplitude=0.012, nwiggles=7, **common)

    for index, (v, kind, direction) in enumerate(diagram.external):
        counters[(kind, direction)] += 1
        end = endpoints[index]
        external_vertex = renderer.vertex(xy=end, marker="")
        if kind == "photon":
            renderer.line(vertex_objects[v], external_vertex, flavour="wiggly", arrow=False, amplitude=0.012, nwiggles=5,
                          linewidth=2.1, color="#ffd36a")
        else:
            arrow_inward = (kind == "electron" and direction == "in") or (kind == "positron" and direction == "out")
            start, finish = (external_vertex, vertex_objects[v]) if arrow_inward else (vertex_objects[v], external_vertex)
            renderer.line(start, finish, flavour="simple", arrow=True, linewidth=2.1, color="#9ee6ff",
                          arrow_param={"color": "#9ee6ff", "width": 0.024, "length": 0.065})
        particle = {"electron": "e⁻", "positron": "e⁺", "photon": "γ"}[kind]
        arrow = "in" if direction == "in" else "out"
        ax.text(end[0], end[1], f"{particle} {arrow} {counters[(kind, direction)]}", color="#eef3ff", fontsize=9,
                ha="center", va="center", zorder=30,
                bbox=dict(boxstyle="round,pad=.25", fc="#18233a", ec="none"))

    renderer.plot()
    ax.set_title(title, color="#eef3ff", fontsize=14, pad=12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout()
    return fig


def export_figure(fig, fmt: str) -> bytes:
    output = BytesIO()
    fig.savefig(output, format=fmt, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    return output.getvalue()
