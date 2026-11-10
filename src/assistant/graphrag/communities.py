"""Сообщества графа и их сводки. Требование FR-3."""

import json
from collections import Counter

import networkx as nx

from assistant.graphrag.evidence import community_edges, fact
from assistant.infra.llm import SEED, LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    undirected = nx.Graph()
    undirected.add_nodes_from(sorted(graph.nodes))
    for source, target, attrs in sorted(graph.edges(data=True)):
        weight = attrs.get("weight", 1)
        if undirected.has_edge(source, target):
            weight += undirected.edges[source, target]["weight"]
        undirected.add_edge(source, target, weight=weight)
    if not undirected.number_of_edges():
        groups = [{node} for node in undirected]
    else:
        groups = nx.community.louvain_communities(undirected, weight="weight", seed=SEED)
    return [Community(index, nodes) for index, nodes in enumerate(sorted(sorted(group) for group in groups))]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    edges = community_edges(community, graph)
    failures: Counter = Counter()
    model_failures: Counter = Counter()
    relations: Counter = Counter()
    for source, target in edges:
        attrs = graph.edges[source, target]
        relations[attrs["type"]] += attrs["weight"]
        if attrs["type"] == "FAILED":
            lines = [
                graph.nodes[node]["name"]
                for node in graph.successors(source)
                if graph.edges[source, node]["type"] == "BELONGS_TO"
            ] or ["линейка неизвестна"]
            for line in lines:
                failures[line, graph.nodes[target]["name"]] += attrs["weight"]
                model_failures[line, graph.nodes[source]["name"]] += attrs["weight"]
    statistics = {
        "узлов": len(community.nodes),
        "суммы весов по типам связей": dict(sorted(relations.items())),
        "отказы по линейке и узлу": [
            {"линейка": line, "узел": component, "отказов": count}
            for (line, component), count in sorted(failures.items(), key=lambda row: (-row[1], row[0]))
        ],
        "отказы по линейке и модели": [
            {"линейка": line, "модель": model, "отказов": count}
            for (line, model), count in sorted(model_failures.items(), key=lambda row: (-row[1], row[0]))
        ],
    }
    isolated = [
        {"тип": graph.nodes[node]["type"], "название": graph.nodes[node]["name"]}
        for node in sorted(community.nodes)
        if graph.degree(node) == 0
    ]
    if isolated:
        statistics["сущности без связей"] = isolated
    facts = "\n".join(
        f"{fact(graph, source, target)} (вес {graph.edges[source, target]['weight']})" for source, target in edges
    )
    response = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Напиши сводку сообщества графа, не более 150 слов. Используй только переданные факты. "
                    "Числа уже посчитаны кодом: перенеси их точно, не считай заново. Сохрани частоты отказов "
                    "по линейкам и узлам, цепочки замен и поставщиков. Частоты относятся к этому сообществу, "
                    "а не ко всему корпусу. Связи FAILED отражают отказы, остальные веса - упоминания."
                    + (" Назови каждую сущность без связей." if isolated else "")
                ),
            },
            {"role": "user", "content": json.dumps(statistics, ensure_ascii=False) + "\nФакты:\n" + facts},
        ]
    )
    return " ".join(response.content.split()[:150])
