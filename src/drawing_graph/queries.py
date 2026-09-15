"""Deterministic, evidence-producing query executor for CDG v0."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Any, Mapping

from src.drawing_graph.schema import CanonicalDrawingGraph, DrawingNode


@dataclass(frozen=True)
class QueryResult:
    answer: Any
    evidence_nodes: tuple[str, ...] = ()
    evidence_edges: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "evidence_nodes": list(self.evidence_nodes),
            "evidence_edges": list(self.evidence_edges),
        }


def _matching_nodes(graph: CanonicalDrawingGraph, query: Mapping[str, Any]) -> list[DrawingNode]:
    filters = {
        key: query[key]
        for key in ("kind", "node_type", "semantic_id", "semantic_name")
        if key in query
    }
    return [
        node
        for node in graph.nodes
        if all(getattr(node, key) == value for key, value in filters.items())
    ]


def execute_query(graph: CanonicalDrawingGraph, query: Mapping[str, Any]) -> QueryResult:
    """Execute one versioned query and return answer plus graph evidence."""

    operation = query.get("op")
    if operation == "count_nodes":
        nodes = _matching_nodes(graph, query)
        return QueryResult(len(nodes), tuple(node.id for node in nodes))
    if operation == "list_nodes":
        nodes = _matching_nodes(graph, query)
        return QueryResult([node.id for node in nodes], tuple(node.id for node in nodes))
    if operation == "count_relations":
        relation = query.get("relation")
        edges = [edge for edge in graph.edges if relation is None or edge.relation == relation]
        return QueryResult(len(edges), evidence_edges=tuple(edge.id for edge in edges))
    if operation == "neighbors":
        node_id = str(query["node_id"])
        relation = query.get("relation")
        direction = query.get("direction", "both")
        if direction not in {"in", "out", "both"}:
            raise ValueError("neighbors.direction must be 'in', 'out', or 'both'")
        node_ids: list[str] = []
        edge_ids: list[str] = []
        for edge in graph.edges:
            if relation is not None and edge.relation != relation:
                continue
            if direction in {"out", "both"} and edge.source == node_id:
                node_ids.append(edge.target)
                edge_ids.append(edge.id)
            if direction in {"in", "both"} and edge.target == node_id:
                node_ids.append(edge.source)
                edge_ids.append(edge.id)
        unique_nodes = tuple(dict.fromkeys(node_ids))
        return QueryResult(list(unique_nodes), unique_nodes, tuple(dict.fromkeys(edge_ids)))
    if operation == "shortest_path":
        source = str(query["source"])
        target = str(query["target"])
        relation_filter = query.get("relations")
        if relation_filter is None:
            relations = None
        elif isinstance(relation_filter, str):
            relations = {relation_filter}
        else:
            relations = {str(value) for value in relation_filter}

        node_ids = {node.id for node in graph.nodes}
        if source not in node_ids or target not in node_ids:
            raise ValueError("shortest_path source and target must reference existing nodes")
        adjacency: dict[str, list[tuple[str, str]]] = {node_id: [] for node_id in node_ids}
        for edge in graph.edges:
            if relations is not None and edge.relation not in relations:
                continue
            adjacency[edge.source].append((edge.target, edge.id))
            adjacency[edge.target].append((edge.source, edge.id))

        queue = deque([source])
        previous: dict[str, tuple[str, str] | None] = {source: None}
        while queue and target not in previous:
            current = queue.popleft()
            for neighbor, edge_id in adjacency[current]:
                if neighbor in previous:
                    continue
                previous[neighbor] = (current, edge_id)
                queue.append(neighbor)

        if target not in previous:
            return QueryResult(None)
        path_nodes = [target]
        path_edges: list[str] = []
        while path_nodes[-1] != source:
            link = previous[path_nodes[-1]]
            if link is None:
                raise RuntimeError("broken shortest-path predecessor chain")
            parent, edge_id = link
            path_nodes.append(parent)
            path_edges.append(edge_id)
        path_nodes.reverse()
        path_edges.reverse()
        return QueryResult(path_nodes, tuple(path_nodes), tuple(path_edges))
    raise ValueError(f"unsupported query operation: {operation!r}")
