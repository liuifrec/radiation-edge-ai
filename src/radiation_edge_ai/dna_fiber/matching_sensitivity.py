"""Alternative matching diagnostics for DNA-fiber characterization.

The legacy greedy matcher remains the primary historical metric.  This module
adds a separately labelled maximum-cardinality sensitivity analysis.  Among
maximum-cardinality matchings it minimizes cost=-IoU, i.e. maximizes total IoU.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class _Edge:
    to: int
    rev: int
    cap: int
    cost: int
    left: int | None = None
    right: int | None = None
    iou: float | None = None


def greedy_iou_matching(
    reference: Sequence[Any], candidate: Sequence[Any], threshold: float = 0.5
) -> list[tuple[int, int, float]]:
    edges: list[tuple[float, int, int]] = []
    for i, ref in enumerate(reference):
        for j, cand in enumerate(candidate):
            iou = float(ref.bbox_iou(cand))
            if iou >= threshold:
                edges.append((iou, i, j))
    edges.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for iou, i, j in edges:
        if i in used_left or j in used_right:
            continue
        used_left.add(i)
        used_right.add(j)
        matches.append((i, j, iou))
    return matches


def maximum_cardinality_iou_matching(
    reference: Sequence[Any], candidate: Sequence[Any], threshold: float = 0.5
) -> list[tuple[int, int, float]]:
    n_left = len(reference)
    n_right = len(candidate)
    source = 0
    left0 = 1
    right0 = left0 + n_left
    sink = right0 + n_right
    n_nodes = sink + 1
    graph: list[list[_Edge]] = [[] for _ in range(n_nodes)]

    def add_edge(u: int, v: int, cap: int, cost: int, *, left=None, right=None, iou=None) -> None:
        fwd = _Edge(v, len(graph[v]), cap, cost, left, right, iou)
        rev = _Edge(u, len(graph[u]), 0, -cost)
        graph[u].append(fwd)
        graph[v].append(rev)

    for i in range(n_left):
        add_edge(source, left0 + i, 1, 0)
    for j in range(n_right):
        add_edge(right0 + j, sink, 1, 0)
    scale = 1_000_000_000
    for i, ref in enumerate(reference):
        for j, cand in enumerate(candidate):
            iou = float(ref.bbox_iou(cand))
            if iou >= threshold:
                add_edge(left0 + i, right0 + j, 1, -round(iou * scale), left=i, right=j, iou=iou)

    # Successive shortest augmenting paths with Bellman-Ford handles negative
    # forward IoU costs and residual reverse edges without optional dependencies.
    while True:
        inf = 10**30
        dist = [inf] * n_nodes
        parent: list[tuple[int, int] | None] = [None] * n_nodes
        dist[source] = 0
        for _ in range(n_nodes - 1):
            changed = False
            for u in range(n_nodes):
                if dist[u] == inf:
                    continue
                for edge_index, edge in enumerate(graph[u]):
                    if edge.cap <= 0:
                        continue
                    new = dist[u] + edge.cost
                    candidate_parent = (u, edge_index)
                    if new < dist[edge.to] or (
                        new == dist[edge.to]
                        and parent[edge.to] is not None
                        and candidate_parent < parent[edge.to]
                    ):
                        dist[edge.to] = new
                        parent[edge.to] = candidate_parent
                        changed = True
            if not changed:
                break
        if parent[sink] is None:
            break
        v = sink
        while v != source:
            item = parent[v]
            if item is None:
                raise RuntimeError("Broken augmenting path")
            u, edge_index = item
            edge = graph[u][edge_index]
            edge.cap -= 1
            graph[v][edge.rev].cap += 1
            v = u

    matches: list[tuple[int, int, float]] = []
    for i in range(n_left):
        u = left0 + i
        for edge in graph[u]:
            if edge.left is None or edge.right is None or edge.iou is None:
                continue
            # Forward matching edge has cap 0 when used (initial cap was 1).
            if edge.cap == 0:
                matches.append((edge.left, edge.right, edge.iou))
    matches.sort(key=lambda item: (item[0], item[1]))
    return matches


def matching_summary(
    n_reference: int, n_candidate: int, matches: Sequence[tuple[int, int, float]]
) -> dict[str, float | int]:
    n_match = len(matches)
    precision = n_match / n_candidate if n_candidate else (1.0 if n_reference == 0 else 0.0)
    recall = n_match / n_reference if n_reference else (1.0 if n_candidate == 0 else 0.0)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "n_reference": n_reference,
        "n_candidate": n_candidate,
        "n_matched": n_match,
        "n_reference_unmatched": n_reference - n_match,
        "n_candidate_unmatched": n_candidate - n_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def reference_defined_nonempty(
    baseline_fiber_count: int, human_fiber_counts: Sequence[int]
) -> bool:
    if baseline_fiber_count < 0 or any(value < 0 for value in human_fiber_counts):
        raise RuntimeError("Fiber counts must be non-negative")
    return baseline_fiber_count > 0 or any(value > 0 for value in human_fiber_counts)
