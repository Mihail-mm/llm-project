"""Дополнительные проверки: pytest src/assistant/graphrag/test_regressions.py."""

import json

import networkx as nx
import pytest

from assistant.graphrag.communities import detect_communities, summarize_community
from assistant.graphrag.extraction import extract_chunk
from assistant.graphrag.graph_builder import build_graph
from assistant.graphrag.search import global_search, local_search
from assistant.infra.schemas import Chunk, Community, Entity, ExtractionResult, KnowledgeBase, Relation
from assistant.infra.testing import FakeLLM


def chain_kb(cycle=False):
    model = Entity("Model", "BZ-110")
    component = Entity("Component", "нагреватель оттайки")
    a, b, c = [Entity("Part", name) for name in ("P-2054", "P-2054B", "P-2054C")]
    supplier = Entity("Supplier", "АвтоматикаСевер")
    relations = [
        Relation(model, "HAS_PART", a),
        Relation(a, "IS_A", component),
        Relation(a, "REPLACED_BY", b),
        Relation(b, "REPLACED_BY", c),
        Relation(c, "SUPPLIED_BY", supplier),
    ]
    if cycle:
        relations.append(Relation(c, "REPLACED_BY", a))
    graph = build_graph([ExtractionResult("manual.md#0", "manual.md", relations=relations)])
    return KnowledgeBase(graph, [], {"Component": [component.name]})


def test_empty_and_isolated_graph_communities():
    assert detect_communities(nx.DiGraph()) == []
    graph = nx.DiGraph()
    graph.add_nodes_from(["b", "a"])
    assert [community.nodes for community in detect_communities(graph)] == [["a"], ["b"]]


def test_summaries_distinguish_isolated_procedures():
    procedures = [Entity("Procedure", code) for code in ("SP-105", "SP-115")]
    graph = build_graph([ExtractionResult("procedures.md#0", "procedures.md", entities=procedures)])
    llm = FakeLLM(["Регламент SP-105 без связей.", "Регламент SP-115 без связей."])
    for community in detect_communities(graph):
        summarize_community(community, graph, llm)
    first, second = [request["messages"][1]["content"] for request in llm.requests]
    assert "SP-105" in first and "SP-115" not in first
    assert "SP-115" in second and "SP-105" not in second


def test_chain_reaches_terminal_part_and_its_supplier():
    llm = FakeLLM(["Актуальную P-2054C поставляет АвтоматикаСевер."])
    answer = local_search("Кто поставляет нагреватель оттайки для BZ-110?", chain_kb(), llm)
    assert not answer.refused
    assert any("P-2054C -SUPPLIED_BY-> АвтоматикаСевер" in line for line in answer.facts)
    assert any("P-2054B -REPLACED_BY-> P-2054C" in line for line in answer.facts)


def test_replacement_cycle_refuses_without_model_call():
    answer = local_search("Какой нагреватель оттайки для BZ-110?", chain_kb(cycle=True), FakeLLM([]))
    assert answer.refused and "цикл" in answer.text


def test_terminal_part_is_preserved_when_the_model_omits_it():
    answer = local_search("Чем заменён P-2054?", chain_kb(), FakeLLM(["Деталь заменена."]))
    assert "P-2054C" in answer.text


def test_local_fact_context_is_bounded():
    graph = nx.DiGraph()
    graph.add_node("Model:AR-210", type="Model", name="AR-210", sources=[])
    for number in range(1000, 1400):
        node = f"Part:P-{number}"
        graph.add_node(node, type="Part", name=f"P-{number}", sources=[])
        graph.add_edge("Model:AR-210", node, type="HAS_PART", weight=1, sources=[f"doc{number}.md#0"])
    llm = FakeLLM(["В модели установлены детали из перечня."])
    answer = local_search("Какие детали в AR-210?", KnowledgeBase(graph, [], {}), llm)
    assert 0 < len("\n".join(answer.facts)) <= 8000
    assert len(answer.sources) == len(answer.facts) < 400


@pytest.mark.parametrize(
    "partial",
    ["[]", "null", '{"answer": "x", "score": true}', '{"answer": 3, "score": 7}', '{"answer": "x", "score": 11}'],
)
def test_malformed_map_values_do_not_reach_reduce(partial):
    kb = chain_kb()
    kb.communities = [Community(0, list(kb.graph), "Сводка.")]
    answer = global_search("Что известно?", kb, FakeLLM([partial]))
    assert answer.refused


def test_global_sources_only_come_from_useful_summaries():
    graph = nx.DiGraph()
    for name in ("a", "b"):
        graph.add_node(name, type="Component", name=name, sources=[f"{name}.md#0"])
    kb = KnowledgeBase(graph, [Community(0, ["a"], "a"), Community(1, ["b"], "b")], {})
    llm = FakeLLM(['{"answer":"Полезно","score":9}', '{"answer":"Нет","score":0}', "Полезно"])
    assert global_search("Что известно?", kb, llm).sources == ["a.md"]


def test_summary_receives_model_totals_not_only_individual_edge_weights():
    model = Entity("Model", "AR-210")
    line = Entity("ProductLine", "Арктика")
    a, b = Entity("Component", "пусковое реле"), Entity("Component", "контроллер")
    results = [
        ExtractionResult(
            f"{number}.md#0",
            f"{number}.md",
            relations=[Relation(model, "FAILED", component), Relation(model, "BELONGS_TO", line)],
        )
        for number, component in enumerate([a, a, b])
    ]
    graph = build_graph(results)
    llm = FakeLLM(["Всего три отказа."])
    summarize_community(Community(0, list(graph)), graph, llm)
    prompt = llm.requests[0]["messages"][1]["content"]
    stats = json.loads(prompt.split("\nФакты:")[0])
    assert stats["отказы по линейке и модели"] == [{"линейка": "Арктика", "модель": "AR-210", "отказов": 3}]


def test_extraction_retries_schema_mismatch_and_keeps_only_grounded_endpoints():
    payload = {
        "entities": [],
        "relations": [
            {
                "subject_type": "Part",
                "subject": "P-2054",
                "type": "REPLACED_BY",
                "object_type": "Part",
                "object": "P-2054C",
            }
        ],
    }
    llm = FakeLLM(['{"entities":[],"relations":[{}]}', json.dumps(payload)])
    result = extract_chunk(Chunk("x#0", "x", "Деталь p 2054 заменена на P-2054C."), llm)
    assert len(result.relations) == 1 and len(result.entities) == 2 and len(llm.requests) == 2


def test_invented_code_in_map_answer_does_not_reach_reduce():
    kb = chain_kb()
    kb.communities = [Community(0, list(kb.graph), "P-2054 заменена на P-2054C.")]
    llm = FakeLLM(['{"answer": "Нужна P-7777.", "score": 9}'])
    answer = global_search("Какие детали заменены?", kb, llm)
    assert answer.refused and len(llm.requests) == 1 and "P-7777" not in answer.text
