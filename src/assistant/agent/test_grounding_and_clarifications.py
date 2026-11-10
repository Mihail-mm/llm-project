"""Регрессии проверки единиц, происхождения количества и первых уточнений."""

from decimal import Decimal

import pytest

from assistant.agent.agent import run_agent
from assistant.agent.grounding import numbers, requested_quantities, unsupported
from assistant.graphrag.search import NOT_FOUND
from assistant.infra.llm import LLMResponse, ToolCall
from assistant.infra.testing import FakeLLM

PART = "P-1042B"
STOCK = {"part_id": PART, "warehouse": "Санкт-Петербург", "qty": 0, "discontinued": False}
ORDER = {"part_id": PART, "warehouse": "Санкт-Петербург", "model": "AR-210", "qty": 2}
QUESTION = "Проверь остаток и закажи 2 штуки P-1042B для AR-210, склад Санкт-Петербург."


def call(name, **args):
    return LLMResponse("", [ToolCall(name, name, args)])


@pytest.mark.parametrize("unit", ["руб.", "рублей", "РУБ.", "₽", "шт.", "штук", "единиц", "деталей", "экземпляров"])
def test_attached_units_do_not_hide_unverified_numbers(unit):
    answer = f"9999{unit}"
    assert numbers(answer) == {Decimal(9999)}
    assert unsupported(answer, "Цена и остаток P-1042B?", [{"price_rub": 2630}, STOCK]) == ["9999"]


@pytest.mark.parametrize("value", ["-9999", "−9999", "+9999", "9 999,50", "9\u00a0999,50", "9999.50"])
def test_signed_and_fractional_numbers_keep_their_value_with_attached_units(value):
    assert numbers(f"{value}руб.") == numbers(f"{value} руб.")
    assert unsupported(f"Цена {value}руб.", "Цена?", [{"price_rub": 2630}])


def test_attached_units_do_not_turn_codes_or_prices_into_order_quantities():
    assert not numbers("P-1042B AR-210 SO-000001 9999abc")
    assert requested_quantities("Закажи 2630руб. P-1042B.") == set()
    assert requested_quantities("Закажи 2шт. P-1042B.") == {2}


def test_agent_repairs_invented_values_with_attached_units():
    llm = FakeLLM(
        [
            call("get_price", part_id=PART),
            call("check_stock", part_id=PART, warehouse="Санкт-Петербург"),
            "Цена 9999руб., остаток 777шт.",
            "Цена 2630руб., остаток 0шт.",
        ]
    )
    result = run_agent(
        "Цена и остаток P-1042B в Санкт-Петербурге?",
        llm,
        {"get_price": lambda **args: {"price_rub": 2630}, "check_stock": lambda **args: STOCK},
    )
    assert result.answer == "Цена 2630руб., остаток 0шт."
    assert not result.refused and len(llm.requests) == 4


@pytest.mark.parametrize("order_text", ["закажи 2 штуки", "закажи 2", "оформи заказ на 2", "закажи две детали"])
def test_requested_quantity_does_not_ground_stock(order_text):
    question = f"Проверь остаток и {order_text} P-1042B для AR-210, склад Санкт-Петербург."
    assert unsupported("В наличии 2 шт.", question, [STOCK]) == ["2"]


@pytest.mark.parametrize("results", [[{"order_id": "SO-000001", **ORDER}], [STOCK, {"order_id": "SO-000001", **ORDER}]])
def test_created_order_quantity_does_not_ground_stock(results):
    assert unsupported("В наличии 2 шт.", QUESTION, results) == ["2"]


@pytest.mark.parametrize(
    "answer",
    [
        "Закажите 2 шт.; остаток 0 шт.",
        "В заказе 2 шт.; в наличии 0 шт.",
        "Заказ на 2шт., остаток 0шт.",
        "Заказ SO-000001 создан: 2 шт.; остаток 0 шт.",
    ],
)
def test_order_quantity_and_stock_can_both_be_reported(answer):
    assert not unsupported(answer, QUESTION, [STOCK, {"order_id": "SO-000001", **ORDER}])


@pytest.mark.parametrize("answer", ["Закажите 2 шт.; в наличии 2 шт.", "В наличии 2 шт.; закажите 2 шт."])
def test_order_context_does_not_authorize_another_stock_claim(answer):
    assert unsupported(answer, QUESTION, [STOCK]) == ["2"]


def test_stock_does_not_authorize_a_different_order_quantity():
    assert unsupported("Закажите 3 шт.", QUESTION, [{**STOCK, "qty": 3}]) == ["3"]


@pytest.mark.parametrize("repaired", ["В наличии 0 шт.", "В наличии 2 шт."])
def test_agent_repairs_or_refuses_stock_copied_from_order_quantity(repaired):
    llm = FakeLLM(
        [call("check_stock", part_id=PART, warehouse="Санкт-Петербург"), "В наличии 2 шт.", repaired]
    )
    result = run_agent(QUESTION, llm, {"check_stock": lambda **args: STOCK})
    assert len(llm.requests) == 3
    assert result.refused == (repaired == "В наличии 2 шт.")
    assert "2 шт." not in result.answer
    if not result.refused:
        assert result.answer == repaired


@pytest.mark.parametrize(
    ("question", "clarification", "expected"),
    [
        ("Есть ли P-1042B?", "На каком складе проверить наличие?", "Укажите склад."),
        ("Есть ли P-1042B?", "Укажите, пожалуйста, склад.", "Укажите склад."),
        ("Есть ли P-1042B?", "Уточните, пожалуйста, на каком складе проверить наличие?", "Укажите склад."),
        ("Закажи P-1042B для AR-210, склад Казань", "Сколько штук заказать?", "Сколько штук заказать?"),
        ("Закажи P-1042B для AR-210, склад Казань", "Уточните количество.", "Сколько штук заказать?"),
        ("Закажи P-1042B для AR-210, склад Казань", "Сколько нужно заказать?", "Сколько штук заказать?"),
        (
            "Закажи 2 штуки P-1042B, склад Казань",
            "Для какой модели оформить заказ?",
            "Для какой модели оформить заказ?",
        ),
    ],
)
def test_initial_clarification_is_preserved_without_tool_calls(question, clarification, expected):
    llm = FakeLLM([clarification])
    result = run_agent(question, llm, {})
    assert result.answer == expected and not result.refused
    assert not result.steps and not result.needs_confirmation and result.pending_action is None
    assert len(llm.requests) == 1


@pytest.mark.parametrize(
    "answer",
    ["", "P-1042B есть в наличии.", NOT_FOUND],
)
def test_no_tool_factual_answers_and_empty_answers_still_refuse(answer):
    result = run_agent("Есть ли P-1042B?", FakeLLM([answer]), {})
    assert result.refused and result.answer == NOT_FOUND


def test_clarifications_are_still_checked_for_invented_codes():
    answer = "На каком складе проверить наличие P-7777?"
    llm = FakeLLM([answer, answer])
    result = run_agent("Есть ли P-1042B?", llm, {})
    assert result.refused and "P-7777" not in result.answer
    assert len(llm.requests) == 2


@pytest.mark.parametrize(
    "answer",
    [
        "В наличии для заказа 2 шт.",
        "На складе доступно для заказа 2 шт.",
        "Остаток для оформления заказа: 2 шт.",
        "В наличии, для заказа: 2 шт.",
        "На складе для вашего заказа 2 шт.",
        "На складе под ваш заказ 2 шт.",
        "На складе, для вашего заказа: 2 шт.",
    ],
)
def test_stock_claim_keeps_stock_provenance_when_it_mentions_an_order(answer):
    assert unsupported(answer, QUESTION, [STOCK]) == ["2"]
    llm = FakeLLM([call("check_stock", part_id=PART, warehouse="Санкт-Петербург"), answer, "В наличии 0 шт."])
    result = run_agent(QUESTION, llm, {"check_stock": lambda **args: STOCK})
    assert result.answer == "В наличии 0 шт." and not result.refused
    assert len(llm.requests) == 3


@pytest.mark.parametrize(
    "answer",
    [
        "Заказ SO-000001 создан: P-1042B — 2 шт.",
        "Создан заказ SO-000001, количество: 2 шт.",
        "Оформлено 2 шт.",
        "Заказ SO-000001 оформлен на складе Санкт-Петербург: 2 шт.",
        "Заказ SO-000001 создан:\nP-1042B — 2 шт.",
        "Заказ SO-000001 создан. Количество: 2 шт.",
        "Для клиента оформлен заказ SO-000001: P-1042B — 2 шт.",
        "Заказ SO-000001 создан: P-1042B — 2 шт.; в наличии 0 шт.",
        "Остаток 0 шт., заказ SO-000001 создан: P-1042B — 2 шт.",
        "Осталось 0 штук и создан заказ SO-000001 на 2 штуки.",
    ],
)
def test_created_order_summaries_accept_quantities_separated_from_order_words(answer):
    created = {"order_id": "SO-000001", "status": "created", **ORDER}
    llm = FakeLLM(
        [call("check_stock", part_id=PART, warehouse="Санкт-Петербург"), call("create_service_order", **ORDER), answer]
    )
    calls = []

    def create(**args):
        calls.append(args)
        return created

    result = run_agent(
        QUESTION, llm, {"check_stock": lambda **args: STOCK, "create_service_order": create}, confirm=True
    )
    assert not result.refused and answer in result.answer
    assert "SO-000001" in result.answer and calls == [ORDER]
    assert len(llm.requests) == 3


@pytest.mark.parametrize(
    "answer",
    [
        "Уточните склад, деталь есть в наличии.",
        "Укажите склад — деталь подходит для ремонта.",
        "На каком складе проверить наличие? Деталь есть в наличии.",
        "P-1042B есть в наличии. На каком складе проверить?",
    ],
)
def test_initial_clarification_cannot_emit_added_factual_claims(answer):
    result = run_agent("Есть ли P-1042B?", FakeLLM([answer]), {})
    assert result.answer == "Укажите склад." and not result.refused
    assert result.steps == []


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Уточните, пожалуйста, сколько штук нужно заказать?", "Сколько штук заказать?"),
        ("На каком складе проверить наличие? Укажите город.", "Укажите склад."),
        ("Уточните склад. Сколько штук нужно заказать?", "Укажите склад. Сколько штук заказать?"),
        ("Укажите номер модели оборудования.", "Для какой модели оформить заказ?"),
        ("Уточните номер модели", "Для какой модели оформить заказ?"),
        ("Какой номер модели?", "Для какой модели оформить заказ?"),
    ],
)
def test_compound_and_indirect_clarifications_are_rendered_as_safe_requests(answer, expected):
    result = run_agent("Закажи P-1042B для AR-210", FakeLLM([answer]), {})
    assert result.answer == expected and not result.refused
    assert result.steps == []


@pytest.mark.parametrize(
    "stock_text",
    [
        "В Санкт-Петербурге осталось {qty} шт.",
        "В Санкт-Петербурге {qty} шт.",
        "Склад Санкт-Петербург: {qty} шт.",
        "По наличию: {qty} шт.",
        "P-1042B — {qty} шт. в наличии.",
        "P-1042B — {qty} шт. на складе.",
        "P-1042B — {qty} шт. осталось.",
    ],
)
@pytest.mark.parametrize("qty", [0, 2])
def test_a_new_stock_statement_cannot_inherit_the_preceding_order_quantity(stock_text, qty):
    answer = "Заказ SO-000001 создан. " + stock_text.format(qty=qty)
    created = {"order_id": "SO-000001", "status": "created", **ORDER}
    assert unsupported(answer, QUESTION, [STOCK, created]) == ([] if qty == 0 else ["2"])
    llm = FakeLLM(
        [
            call("check_stock", part_id=PART, warehouse="Санкт-Петербург"),
            call("create_service_order", **ORDER),
            answer,
            "Заказ SO-000001 создан. В наличии 0 шт.",
        ]
    )
    calls = []

    def create(**args):
        calls.append(args)
        return created

    result = run_agent(
        QUESTION, llm, {"check_stock": lambda **args: STOCK, "create_service_order": create}, confirm=True
    )
    assert not result.refused and "0 шт." in result.answer and "2 шт." not in result.answer
    assert calls == [ORDER]
    assert len(llm.requests) == (3 if qty == 0 else 4)


@pytest.mark.parametrize(
    ("action", "conjunction"),
    [
        ("Оформлено", "и осталось"),
        ("Заказано", "и осталось"),
        ("Заказано", "и доступно"),
        ("Оформлено", "но осталось"),
        ("Заказано", "и на складе Санкт-Петербург есть"),
        ("Оформлено", "а на складе осталось"),
    ],
)
@pytest.mark.parametrize("qty", [0, 2])
def test_conjoined_order_and_stock_facts_keep_separate_quantities(action, conjunction, qty):
    answer = f"{action} 2 штуки {conjunction} {qty} штук."
    created = {"order_id": "SO-000001", "status": "created", **ORDER}
    assert unsupported(answer, QUESTION, [STOCK, created]) == ([] if qty == 0 else ["2"])
    llm = FakeLLM(
        [
            call("check_stock", part_id=PART, warehouse="Санкт-Петербург"),
            call("create_service_order", **ORDER),
            answer,
            answer,
        ]
    )
    calls = []

    def create(**args):
        calls.append(args)
        return created

    result = run_agent(
        QUESTION, llm, {"check_stock": lambda **args: STOCK, "create_service_order": create}, confirm=True
    )
    assert result.refused == (qty == 2)
    assert calls == [ORDER] and "SO-000001" in result.answer
    assert len(llm.requests) == (3 if qty == 0 else 4)
    if qty == 0:
        assert answer in result.answer
