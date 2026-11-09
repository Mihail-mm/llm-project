"""Общий формат фактов и источников для сводок и поиска."""

import networkx as nx

from assistant.infra.schemas import Community


def fact(graph: nx.DiGraph, source: str, target: str) -> str:
    edge = graph.edges[source, target]
    return f"{graph.nodes[source]['name']} -{edge['type']}-> {graph.nodes[target]['name']}"


def documents(chunks) -> list[str]:
    return sorted({chunk.split("#", 1)[0] for chunk in chunks})


def community_edges(community: Community, graph: nx.DiGraph) -> list[tuple[str, str]]:
    # Каждое ребро принадлежит сообществу своего начала: map/reduce не удваивает частоты.
    nodes = set(community.nodes)
    edges = {(source, target) for source, target in graph.edges if source in nodes}
    replaced = {source for source, target in edges if graph.edges[source, target]["type"] == "REPLACED_BY"}
    edges.update(
        (model, part)
        for part in replaced
        for model in graph.predecessors(part)
        if graph.edges[model, part]["type"] == "HAS_PART"
    )
    # Линейка нужна для интерпретации статистики FAILED даже на границе сообществ.
    models = {source for source, target in edges if graph.edges[source, target]["type"] in {"FAILED", "HAS_PART"}}
    edges.update(
        (model, target)
        for model in models
        for target in graph.successors(model)
        if graph.edges[model, target]["type"] == "BELONGS_TO"
    )
    return sorted(edges)


def community_sources(community: Community, graph: nx.DiGraph) -> list[str]:
    chunks = [
        chunk
        for source, target in community_edges(community, graph)
        for chunk in graph.edges[source, target].get("sources", [])
    ]
    chunks.extend(chunk for node in community.nodes if node in graph for chunk in graph.nodes[node].get("sources", []))
    return documents(chunks)
