"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    graph = nx.DiGraph(conflicts=0)
    for result in results:
        valid = [relation for relation in result.relations if relation.matches_schema()]
        entities = list(result.entities) + [end for relation in valid for end in (relation.subject, relation.object)]
        for entity in entities:
            if entity.node_id not in graph:
                graph.add_node(entity.node_id, type=entity.type, name=entity.name, sources=set())
            graph.nodes[entity.node_id]["sources"].add(result.chunk_id)
        for relation in valid:
            source, target = relation.subject.node_id, relation.object.node_id
            if not graph.has_edge(source, target):
                graph.add_edge(source, target, type=relation.type, weight=0, sources=set())
            edge = graph.edges[source, target]
            edge["sources"].add(result.chunk_id)
            edge["weight"] = len(edge["sources"])

    for source, target in sorted(list(graph.edges)):
        if source >= target or not graph.has_edge(source, target) or not graph.has_edge(target, source):
            continue
        forward, backward = graph.edges[source, target], graph.edges[target, source]
        if forward["type"] == backward["type"] == "REPLACED_BY":
            graph.graph["conflicts"] += 1
            if forward["weight"] <= backward["weight"]:
                graph.remove_edge(source, target)
            if backward["weight"] <= forward["weight"]:
                graph.remove_edge(target, source)

    for _, attrs in graph.nodes(data=True):
        attrs["sources"] = sorted(attrs["sources"])
    for _, _, attrs in graph.edges(data=True):
        attrs["sources"] = sorted(attrs["sources"])
    return graph
