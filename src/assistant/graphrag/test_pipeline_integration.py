"""Сквозная проверка модулей через LLMClient с подставным HTTP-ответом, без живой модели."""

import json

import httpx
import networkx as nx

from assistant.graphrag.search import local_search
from assistant.infra.config import Settings
from assistant.infra.indexing import build_index, load_kb
from assistant.infra.llm import LLMClient


def test_index_roundtrip_and_llm_cache(tmp_path):
    data_dir = tmp_path / "data"
    corpus = data_dir / "corpus"
    corpus.mkdir(parents=True)
    (data_dir / "vocabulary.json").write_text("{}", encoding="utf-8")
    (corpus / "bulletin.md").write_text("# Замена\n\nP-1042 заменена на P-1042B.", encoding="utf-8")
    payload = {
        "entities": [{"type": "Part", "name": "P-1042"}, {"type": "Part", "name": "P-1042B"}],
        "relations": [
            {
                "subject_type": "Part",
                "subject": "P-1042",
                "type": "REPLACED_BY",
                "object_type": "Part",
                "object": "P-1042B",
            }
        ],
    }
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["temperature"] == 0 and body["seed"] == 7
        if "response_format" in body:
            content = json.dumps(payload)
        else:
            content = "P-1042 заменена на P-1042B."
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    settings = Settings(data_dir=data_dir, artifacts_dir=tmp_path / "artifacts")
    with httpx.Client(base_url="http://mock/v1/", transport=httpx.MockTransport(respond)) as http:
        llm = LLMClient(
            "http://mock/v1/", "scripted-test", "unused", http=http, cache_dir=settings.artifacts_dir / "llm_cache"
        )
        built = build_index(settings, llm, log=lambda line: None)
        loaded = load_kb(settings)
        assert nx.utils.graphs_equal(built.graph, loaded.graph)
        assert built.communities == loaded.communities
        answer = local_search("Чем заменена P-1042?", loaded, llm)
        assert answer.sources == ["bulletin.md"] and "P-1042B" in answer.text
        network_calls = len(requests)
        assert local_search("Чем заменена P-1042?", loaded, llm) == answer
        assert len(requests) == network_calls and llm.stats.cache_hits == 1
    stats = json.loads((settings.artifacts_dir / "index_stats.json").read_text(encoding="utf-8"))
    assert stats["chunks"] == 1 and stats["nodes"] == 2 and stats["edges"] == 1
    assert stats["llm"]["calls"] == 2
