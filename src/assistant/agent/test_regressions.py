"""Дополнительные проверки: pytest src/assistant/agent/test_regressions.py."""

import networkx as nx
import pytest
from fastapi.testclient import TestClient

from assistant.agent.agent import run_agent
from assistant.agent.grounding import requested_quantities, unsupported
from assistant.agent.tools import build_tools
from assistant.infra.erp_client import ErpClient, ErpUnavailable
from assistant.infra.llm import LLMResponse, ToolCall
from assistant.infra.schemas import KnowledgeBase
from assistant.infra.testing import FakeLLM
from erp.app import create_app

ORDER = {"part_id": "P-1042B", "model": "AR-210", "warehouse": "Казань", "qty": 2}


def call(name, **args):
    return LLMResponse("", [ToolCall("call", name, args)])


@pytest.mark.parametrize(
    "question",
    [
        "Закажи P-1042B для AR-210.",
        "Оформи заказ 09.11.2026 для AR-210.",
        "Закажи за 2630 руб. P-1042B.",
        "Закажи 2630 руб. P-1042B.",
    ],
)
def test_codes_dates_and_prices_are_not_order_quantities(question):
    assert requested_quantities(question) == set()


@pytest.mark.parametrize("question", ["Закажи две детали P-1042B.", "Оформи заказ: 2 штуки.", "Закажи 2 P-1042B."])
def test_explicit_order_quantity(question):
    assert requested_quantities(question) == {2}


@pytest.mark.parametrize("answer", ["Цена 1042 руб.", "Цена 3 руб.", "Остаток 2630 шт.", "Цена 9999."])
def test_code_digits_and_unrelated_erp_fields_do_not_ground_prices_or_stock(answer):
    results = [{"part_id": "P-1042B", "qty": 3, "price_rub": 2630}]
    assert unsupported(answer, "Что известно о P-1042B?", results)


def test_price_with_grouping_and_decimal_comma_is_grounded():
    assert not unsupported("Цена 2 630,00 руб., в наличии 0 шт.", "Что известно?", [{"qty": 0, "price_rub": 2630}])


def test_error_text_cannot_authorize_another_part():
    calls = []

    def price(**args):
        calls.append(args)
        return {"error": "Возможно, нужна P-7777"}

    llm = FakeLLM([call("get_price", part_id="P-1042B"), call("get_price", part_id="P-7777"), "Нет данных."])
    result = run_agent("Сколько стоит P-1042B?", llm, {"get_price": price})
    assert len(calls) == 1 and result.steps[1].error


@pytest.mark.parametrize("qty", [True, "2", 2.5, 0, -1])
def test_invalid_quantity_cannot_even_be_proposed_for_confirmation(qty):
    calls = []
    llm = FakeLLM([call("create_service_order", **{**ORDER, "qty": qty}), "Уточните количество."])
    result = run_agent(
        "Закажи 2 штуки P-1042B для AR-210, Казань.", llm, {"create_service_order": lambda **args: calls.append(args)}
    )
    assert not calls and not result.needs_confirmation and result.steps[0].error


def test_tool_batch_respects_step_budget():
    calls = []
    response = LLMResponse("", [ToolCall(str(index), "get_price", {"part_id": "P-1042B"}) for index in range(5)])
    llm = FakeLLM([response])

    def price(**args):
        calls.append(args)
        return {"part_id": "P-1042B", "price_rub": 2630}

    result = run_agent("Цена P-1042B?", llm, {"get_price": price}, max_steps=2)
    assert len(result.steps) == 2 and len(calls) == 1 and "прервана" in result.answer
    assert "2630" in result.answer


def test_unavailable_erp_is_reported_even_if_model_ignores_the_error():
    def dead(**args):
        raise ErpUnavailable()

    result = run_agent(
        "Цена P-1042B?", FakeLLM([call("get_price", part_id="P-1042B"), "ERP обработала запрос."]), {"get_price": dead}
    )
    assert "Данные ERP получить не удалось" in result.answer


def test_real_erp_adapter_stock_price_and_confirmed_order():
    with TestClient(create_app(fail_rate=0)) as http:
        erp = ErpClient("http://testserver", http=http)
        llm = FakeLLM(
            [
                call("check_stock", part_id="P-1042B", warehouse="Казань"),
                call("get_price", part_id="P-1042B"),
                call("create_service_order", **ORDER),
                call("create_service_order", **ORDER),
                "В Казани 3 шт., цена 2630 руб.",
            ]
        )
        tools = build_tools(KnowledgeBase(nx.DiGraph(), [], {}), llm, erp)
        result = run_agent(
            "Узнай цену и остаток. Закажи 2 штуки P-1042B для AR-210, склад Казань.", llm, tools, confirm=True
        )
        assert "SO-000001" in result.answer and not result.refused
        assert http.get("/orders/SO-000001").status_code == 200
        assert http.get("/orders/SO-000002").status_code == 404


def test_unconfirmed_order_does_not_reach_real_erp():
    with TestClient(create_app(fail_rate=0)) as http:
        llm = FakeLLM([call("create_service_order", **ORDER)])
        tools = build_tools(KnowledgeBase(nx.DiGraph(), [], {}), llm, ErpClient("http://testserver", http=http))
        result = run_agent("Закажи 2 штуки P-1042B для AR-210, склад Казань.", llm, tools)
        assert result.pending_action == {"tool": "create_service_order", "args": ORDER}
        assert http.get("/orders/SO-000001").status_code == 404


@pytest.mark.parametrize(
    "tool_result",
    [
        {"answer": "В базе знаний не найдено.", "refused": True, "sources": []},
        {"error": "Данных в ERP нет", "error_type": "not_found"},
    ],
)
def test_no_data_remains_a_refusal_when_the_step_budget_is_exhausted(tool_result):
    llm = FakeLLM([call("search_knowledge", question="Что означает E-99?")])
    result = run_agent("Что означает E-99?", llm, {"search_knowledge": lambda **args: tool_result}, max_steps=1)
    assert result.refused and "В базе знаний не найдено" in result.answer
    assert "прервана" in result.answer


def test_zero_stock_is_preserved_at_the_step_limit():
    llm = FakeLLM([call("check_stock", part_id="P-1042B", warehouse="Казань")])
    result = run_agent(
        "Есть ли P-1042B в Казани?",
        llm,
        {"check_stock": lambda **args: {**args, "qty": 0, "discontinued": False}},
        max_steps=1,
    )
    assert not result.refused and "0 шт." in result.answer
