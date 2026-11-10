"""Привязка параметров заказа и ответов ERP к их источникам."""

import pytest

from assistant.agent.agent import run_agent
from assistant.agent.grounding import requested_quantities, requested_warehouses, unsupported
from assistant.infra.llm import LLMResponse, ToolCall
from assistant.infra.testing import FakeLLM

PART = "P-1042B"
ORDER = {"part_id": PART, "model": "AR-210", "warehouse": "Казань", "qty": 2}
STOCK = [
    {"part_id": PART, "warehouse": "Москва", "qty": 0},
    {"part_id": PART, "warehouse": "Казань", "qty": 3},
]


def call(name, **args):
    return LLMResponse("", [ToolCall(name, name, args)])


@pytest.mark.parametrize(
    "question",
    [
        "На складе осталось 8 штук P-1042B. Закажи 2 штуки для AR-210, склад Казань.",
        "На складе осталось восемь деталей P-1042B. Закажи две детали для AR-210, склад Казань.",
        "Закажи 2 штуки P-1042B для AR-210, склад Казань. В наличии 8 штук.",
        "Остаток 8 шт., закажи 2 шт. P-1042B для AR-210, склад Казань.",
    ],
)
def test_only_order_quantity_is_authorized(question):
    assert requested_quantities(question) == {2}


@pytest.mark.parametrize(
    "question",
    [
        "Заказано 8 штук P-1042B вчера. Закажи P-1042B для AR-210, склад Казань.",
        "В наличии 8 штук P-1042B. Закажи P-1042B для AR-210, склад Казань.",
        "Не оформи 8 штук P-1042B.",
        "Вчера просили заказать 8 штук P-1042B. Закажи P-1042B для AR-210, склад Казань.",
    ],
)
def test_observations_and_negated_requests_do_not_authorize_a_quantity(question):
    assert requested_quantities(question) == set()


def test_ambiguous_order_quantity_requires_clarification():
    calls = []
    result = run_agent(
        "Закажи 2 или 8 штук P-1042B для AR-210, склад Казань.",
        FakeLLM([call("create_service_order", **ORDER)]),
        {"create_service_order": lambda **args: calls.append(args)},
        confirm=True,
    )
    assert not calls and result.pending_action is None and "количеств" in result.answer.lower()


def test_user_numbers_do_not_override_erp_stock():
    assert unsupported("В Москве 3 шт. P-1042B", "В Москве остаток 3, проверь P-1042B.", STOCK)


def test_repeated_stock_observation_uses_latest_value_for_the_same_identity():
    records = [STOCK[1], {**STOCK[1], "qty": 1}]
    assert not unsupported("В Казани 1 шт. P-1042B", "Остаток P-1042B?", records)
    assert unsupported("В Казани 3 шт. P-1042B", "Остаток P-1042B?", records)


@pytest.mark.parametrize("confirm", [False, True])
@pytest.mark.parametrize("qty", [2, 8])
def test_stock_quantity_cannot_be_used_for_pending_or_confirmed_order(confirm, qty):
    question = "На складе осталось 8 штук P-1042B. Закажи 2 штуки для AR-210, склад Казань."
    calls = []

    def create(**args):
        calls.append(args)
        return {"order_id": "SO-000001", **args}

    result = run_agent(
        question,
        FakeLLM([call("create_service_order", **{**ORDER, "qty": qty}), "Заказ SO-000001 создан."]),
        {"create_service_order": create},
        confirm=confirm,
    )
    assert calls == ([ORDER] if confirm and qty == 2 else [])
    assert result.needs_confirmation == (not confirm and qty == 2)
    if qty == 8:
        assert result.pending_action is None and "количеств" in result.answer.lower()


@pytest.mark.parametrize("confirm", [False, True])
@pytest.mark.parametrize(
    ("question", "warehouse"),
    [
        ("Закажи 2 штуки P-1042B для AR-210, склад Казань.", "Москва"),
        ("Закажи 2 штуки P-1042B для AR-210.", "Москва"),
        ("Закажи 2 штуки P-1042B для AR-210, Москва или Казань.", "Москва"),
    ],
)
def test_unselected_warehouse_never_reaches_confirmation_or_erp(question, warehouse, confirm):
    calls = []
    result = run_agent(
        question,
        FakeLLM([call("create_service_order", **{**ORDER, "warehouse": warehouse}), "Готово."]),
        {"create_service_order": lambda **args: calls.append(args)},
        confirm=confirm,
    )
    assert not calls and not result.needs_confirmation and result.pending_action is None
    assert "склад" in result.answer.lower()


@pytest.mark.parametrize(
    ("location", "warehouse"),
    [("в Казани", "Казань"), ("из Москвы", "Москва"), ("в Санкт-Петербурге", "Санкт-Петербург")],
)
def test_explicit_warehouse_in_inflected_form_is_preserved(location, warehouse):
    args = {**ORDER, "warehouse": warehouse}
    result = run_agent(
        f"Закажи 2 штуки P-1042B для AR-210 {location}.",
        FakeLLM([call("create_service_order", **args)]),
        {"create_service_order": lambda **kwargs: pytest.fail("Не подтверждён")},
    )
    assert result.needs_confirmation and result.pending_action["args"] == args


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("В Москве осталось 8 штук. Закажи 2 штуки P-1042B для AR-210, склад Казань.", {"Казань"}),
        ("В Москве осталось 8 штук. Закажи 2 штуки P-1042B для AR-210.", set()),
        ("Закажи 2 штуки P-1042B для AR-210. В Москве осталось 8 штук.", set()),
        ("Склад: Казань. Закажи 2 штуки P-1042B для AR-210.", {"Казань"}),
        ("Закажи 2шт. P-1042B для AR-210, склад Казань.", {"Казань"}),
    ],
)
def test_requested_warehouse_is_not_taken_from_a_stock_observation(question, expected):
    assert requested_warehouses(question) == expected


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("swapped", [False, True])
def test_stock_claims_are_bound_to_warehouse(reverse, swapped):
    statements = [f"В Москве {3 if swapped else 0} шт. P-1042B", f"в Казани {0 if swapped else 3} шт. P-1042B"]
    answer = ", ".join(reversed(statements) if reverse else statements) + "."
    assert bool(unsupported(answer, "Остатки P-1042B в Москве и Казани?", STOCK)) == swapped


@pytest.mark.parametrize("field,unit", [("qty", "шт."), ("price_rub", "руб.")])
def test_values_are_bound_to_part(field, unit):
    records = [{"part_id": PART, field: 3}, {"part_id": "P-1043", field: 7}]
    if field == "qty":
        records = [{**record, "warehouse": "Казань"} for record in records]
    question = "Цена и остатки P-1042B и P-1043 в Казани?"
    assert not unsupported(f"P-1042B: 3 {unit}; P-1043: 7 {unit}", question, records)
    assert unsupported(f"P-1042B: 7 {unit}; P-1043: 3 {unit}", question, records)


def test_stock_must_be_supported_for_the_part_and_warehouse_together():
    records = [
        {"part_id": PART, "warehouse": "Москва", "qty": 0},
        {"part_id": PART, "warehouse": "Казань", "qty": 3},
        {"part_id": "P-1043", "warehouse": "Москва", "qty": 3},
    ]
    assert unsupported("В Москве 3 шт. P-1042B", "P-1042B и P-1043 в Москве?", records)


@pytest.mark.parametrize("repair", [True, False])
def test_agent_repairs_or_refuses_swapped_warehouse_stock(repair):
    wrong = "В Москве 3 шт. P-1042B, в Казани 0 шт. P-1042B."
    correct = "В Москве 0 шт. P-1042B, в Казани 3 шт. P-1042B."
    llm = FakeLLM(
        [
            call("check_stock", part_id=PART, warehouse="Москва"),
            call("check_stock", part_id=PART, warehouse="Казань"),
            wrong,
            correct if repair else wrong,
        ]
    )
    result = run_agent(
        "Остатки P-1042B в Москве и Казани?",
        llm,
        {"check_stock": lambda **args: next(item for item in STOCK if item["warehouse"] == args["warehouse"])},
    )
    assert result.refused == (not repair) and len(llm.requests) == 4
    assert wrong not in result.answer
    if repair:
        assert result.answer == correct


@pytest.mark.parametrize("confirm", [False, True])
@pytest.mark.parametrize(
    ("question", "qty", "warehouse"),
    [
        ("Закажи не две детали, а восемь P-1042B для AR-210, склад Казань.", 2, "Казань"),
        ("Закажи 2 или 8 P-1042B для AR-210, склад Казань.", 2, "Казань"),
        ("Закажи две или восемь деталей P-1042B для AR-210, склад Казань.", 8, "Казань"),
        ("Вчера просили заказать 8 штук P-1042B. Закажи P-1042B для AR-210, склад Казань.", 8, "Казань"),
        ("Закажи 2 штуки P-1042B для AR-210, склад не Казань.", 2, "Казань"),
        ("Закажи 2 штуки P-1042B для AR-210, в Москве товар закончился.", 2, "Москва"),
        ("Закажи P-1042B для AR-210, склад Казань, доставка через 2 дня.", 2, "Казань"),
        ("Закажи P-1042B для AR-210, склад Казань, к 15 ноября.", 15, "Казань"),
        ("Закажи 2 штуки P-1042B для AR-210 в Казани. Склад: Москва.", 2, "Казань"),
        ("Закажи 2 штуки P-1042B для AR-210 на склад Казань или Екатеринбург.", 2, "Казань"),
        ("Закажи 2 штуки P-1042B для AR-210 в Казани. Склад Екатеринбург.", 2, "Казань"),
        ("Оформи заказ на 15 ноября: P-1042B для AR-210, склад Казань.", 15, "Казань"),
        ("Оформи заказ на две недели: P-1042B для AR-210, склад Казань.", 2, "Казань"),
        ("Оформи заказ на 15:00: P-1042B для AR-210, склад Казань.", 15, "Казань"),
        ("Оформи заказ на 15/11: P-1042B для AR-210, склад Казань.", 15, "Казань"),
        ("Оформи заказ на 2 ч.: P-1042B для AR-210, склад Казань.", 2, "Казань"),
    ],
)
def test_negated_historical_or_ambiguous_parameters_never_authorize_an_order(question, qty, warehouse, confirm):
    calls = []
    result = run_agent(
        question,
        FakeLLM([call("create_service_order", **{**ORDER, "qty": qty, "warehouse": warehouse}), "Готово."]),
        {"create_service_order": lambda **args: calls.append(args)},
        confirm=confirm,
    )
    assert not calls and not result.needs_confirmation and result.pending_action is None


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Закажи не две детали, а восемь P-1042B для AR-210, склад Казань.", {8}),
        ("Закажи 2 или 8 P-1042B для AR-210, склад Казань.", {2, 8}),
        ("Закажи две или восемь деталей P-1042B для AR-210, склад Казань.", {2, 8}),
        ("Закажи P-1042B для AR-210, склад Казань, количество 2.", {2}),
        ("Хочу заказать 2 P-1042B для AR-210, склад Казань.", {2}),
        ("Закажи P-1042B для AR-210, склад Казань, доставка через 2 дня.", set()),
        ("Закажи P-1042B для AR-210, склад Казань, к 15 ноября.", set()),
        ("Закажи 2 штуки P-1042B для AR-210, склад Казань, доставка через 3 дня.", {2}),
        ("Оформи заказ на 15 ноября: P-1042B для AR-210, склад Казань.", set()),
        ("Оформи заказ на 2 P-1042B для AR-210, склад Казань, доставка 15 ноября.", {2}),
        ("Оформи заказ на две недели: P-1042B для AR-210, склад Казань.", set()),
        ("Оформи заказ на 15:00: P-1042B для AR-210, склад Казань.", set()),
        ("Оформи заказ на 15/11: P-1042B для AR-210, склад Казань.", set()),
        ("Оформи заказ на 2 ч.: P-1042B для AR-210, склад Казань.", set()),
    ],
)
def test_quantity_selection_keeps_alternatives_and_explicit_fields(question, expected):
    assert requested_quantities(question) == expected


def test_explicit_quantity_after_warehouse_still_reaches_confirmation():
    result = run_agent(
        "Закажи P-1042B для AR-210, склад Казань, количество 2.",
        FakeLLM([call("create_service_order", **ORDER)]),
        {"create_service_order": lambda **args: pytest.fail("Нет подтверждения")},
    )
    assert result.needs_confirmation and result.pending_action["args"] == ORDER


@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("field", ["stock", "price"])
def test_markdown_tables_validate_each_record(swapped, field):
    if field == "stock":
        table = "| Деталь | Склад | Остаток, шт. |\n| --- | --- | --- |\n"
        table += f"| P-1042B | Москва | {3 if swapped else 0} |\n| P-1042B | Казань | {0 if swapped else 3} |"
        records = STOCK
    else:
        table = "| Деталь | Цена, руб. |\n| --- | --- |\n"
        table += f"| P-1042B | {200 if swapped else 100} |\n| P-1043 | {100 if swapped else 200} |"
        records = [{"part_id": PART, "price_rub": 100}, {"part_id": "P-1043", "price_rub": 200}]
    assert bool(unsupported(table, "Цена и остатки P-1042B и P-1043?", records)) == swapped


@pytest.mark.parametrize("swapped", [False, True])
def test_full_agent_checks_a_table_without_units_in_each_cell(swapped):
    answer = "| Деталь | Склад | Остаток, шт. |\n| --- | --- | --- |\n"
    answer += f"| P-1042B | Москва | {3 if swapped else 0} |\n| P-1042B | Казань | {0 if swapped else 3} |"
    llm = FakeLLM(
        [call("check_stock", part_id=PART, warehouse="Москва"),
         call("check_stock", part_id=PART, warehouse="Казань"), answer, answer]
    )
    result = run_agent(
        "Остатки P-1042B в Москве и Казани?", llm,
        {"check_stock": lambda **args: next(item for item in STOCK if item["warehouse"] == args["warehouse"])},
    )
    assert result.refused == swapped and len(llm.requests) == (4 if swapped else 3)


@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("field", ["stock", "price"])
def test_coordinated_facts_keep_their_entities(swapped, field):
    if field == "stock":
        answer = f"В Москве {3 if swapped else 0} шт. P-1042B и в Казани {0 if swapped else 3} шт. P-1042B."
        records = STOCK
    else:
        answer = f"P-1042B — {200 if swapped else 100} руб. и P-1043 — {100 if swapped else 200} руб."
        records = [{"part_id": PART, "price_rub": 100}, {"part_id": "P-1043", "price_rub": 200}]
    assert bool(unsupported(answer, "Цена и остатки P-1042B и P-1043?", records)) == swapped


@pytest.mark.parametrize("suffix", ["", " шт."])
def test_created_order_table_does_not_use_stock_quantity(suffix):
    records = [*STOCK, {"order_id": "SO-000001", **ORDER}]
    table = "| Заказ | Деталь | Количество |\n| --- | --- | --- |\n"
    assert not unsupported(table + f"| SO-000001 | P-1042B | 2{suffix} |", "Закажи 2 штуки P-1042B.", records)
    assert unsupported(table + f"| SO-000001 | P-1042B | 3{suffix} |", "Закажи 2 штуки P-1042B.", records)


def test_unsupported_table_layout_cannot_bypass_erp_grounding():
    table = "| Город | Ответ |\n| --- | --- |\n| Москва | 3 |\n| Казань | 0 |"
    assert unsupported(table, "Остатки P-1042B?", STOCK)


def test_an_erp_error_does_not_allow_question_numbers_as_table_prices():
    table = "| Деталь | Цена, руб. |\n| --- | --- |\n| P-1042B | 3 |"
    assert unsupported(table, "Цена для 3 штук P-1042B?", [{"error": "ERP недоступна"}])


def test_non_erp_table_can_still_quote_knowledge_facts():
    table = "| Ошибка | Частота |\n| --- | --- |\n| E-17 | 3 |"
    assert not unsupported(table, "Частота E-17?", [{"answer": "E-17 встречается 3 раза."}])


@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("field", ["stock", "price"])
def test_concise_entity_value_lists_without_units_keep_provenance(swapped, field):
    if field == "stock":
        answer = f"Остатки P-1042B: Москва — {3 if swapped else 0}; Казань — {0 if swapped else 3}."
        records = STOCK
    else:
        answer = f"Цены: P-1042B — {700 if swapped else 2630}; P-1043 — {2630 if swapped else 700}."
        records = [{"part_id": PART, "price_rub": 2630}, {"part_id": "P-1043", "price_rub": 700}]
    assert bool(unsupported(answer, "Цена и остатки P-1042B и P-1043?", records)) == swapped


def test_agent_repairs_a_concise_stock_list_without_repeated_units():
    wrong = "Остатки P-1042B: Москва — 3; Казань — 0."
    correct = "Остатки P-1042B: Москва — 0; Казань — 3."
    llm = FakeLLM(
        [call("check_stock", part_id=PART, warehouse="Москва"),
         call("check_stock", part_id=PART, warehouse="Казань"), wrong, correct]
    )
    result = run_agent(
        "Остатки P-1042B в Москве и Казани?", llm,
        {"check_stock": lambda **args: next(item for item in STOCK if item["warehouse"] == args["warehouse"])},
    )
    assert result.answer == correct and not result.refused and len(llm.requests) == 4
