"""Ссылки и русские падежи сохраняют тип и источник количественного факта."""

from decimal import Decimal

import pytest

from assistant.agent.agent import run_agent
from assistant.agent.grounding import numbers, requested_quantities, unsupported
from assistant.agent.numeric_text import normalize_numeric_text
from assistant.infra.llm import LLMResponse, ToolCall
from assistant.infra.testing import FakeLLM

PART = "P-1042B"
QUESTION = "Цена и остаток P-1042B в Казани?"
PRICE = {"part_id": PART, "price_rub": 2630}
STOCK = {"part_id": PART, "warehouse": "Казань", "qty": 3}


def call(name, **args):
    return LLMResponse("", [ToolCall(name, name, args)])


@pytest.mark.parametrize(
    "template",
    [
        "Цена: [{price}](https://example.com/price) руб., остаток: [{qty}](https://example.com/stock) шт.",
        "**Цена:** [**{price}**](https://example.com/a_(b)) руб., остаток: [`{qty}`](https://example.com/c) шт.",
        'Цена: [{price}](<https://example.com/price> "2630 руб.") руб., остаток: [{qty}](stock) шт.',
        "[Цена: {price} руб.](price) [Остаток: {qty} шт.](stock)",
        "Цена: [{price}][цена] руб., остаток: [{qty}][склад] шт.\n\n"
        "[цена]: https://example.com/price\n[склад]: https://example.com/stock",
        "Цена: [{price}][] руб., остаток: [{qty}][] шт.\n\n"
        "[{price}]: price\n[{qty}]: stock",
        "Цена: [{price}] руб., остаток: [{qty}] шт.\n\n"
        "[{price}]: price\n[{qty}]: stock",
    ],
)
@pytest.mark.parametrize("swapped", [False, True])
def test_link_labels_preserve_price_and_stock_roles(template, swapped):
    price, qty = (3, 2630) if swapped else (2630, 3)
    answer = template.format(price=price, qty=qty)
    assert bool(unsupported(answer, QUESTION, [PRICE, STOCK])) == swapped


@pytest.mark.parametrize(
    "answer",
    [
        "Цена [P-1042B](part): [2630](https://example.com/9999) руб.",
        'Цена: [2630](price "Цена 9999 рублей") руб.',
        "Цена: [2630][9999] руб.\n\n[9999]: https://example.com/7777",
    ],
)
def test_hidden_link_metadata_is_not_numeric_evidence(answer):
    assert numbers(answer) == {Decimal(2630)}
    assert not unsupported(answer, QUESTION, [PRICE])
    assert unsupported("Цена 9999 руб.", answer, [])


def test_reference_definitions_do_not_leak_between_calls():
    assert normalize_numeric_text("[2630][p]\n\n[p]: price") == ("2630", [])
    assert normalize_numeric_text("[2630][p]")[0] == "[2630][p]"


@pytest.mark.parametrize("swapped", [False, True])
def test_links_preserve_table_cells_and_headers(swapped):
    price, qty = (3, 2630) if swapped else (2630, 3)
    answer = (
        "| Деталь | [Цена](price) | [Остаток](stock) |\n| --- | --- | --- |\n"
        f"| [P-1042B](part) | [{price}](price) | [{qty}](stock) |"
    )
    assert bool(unsupported(answer, QUESTION, [PRICE, STOCK])) == swapped


def test_linked_entity_still_binds_stock_to_the_correct_warehouse():
    records = [STOCK, {**STOCK, "warehouse": "Москва", "qty": 0}]
    assert unsupported("В [Москве](location) [3](stock) шт. [P-1042B](part).", QUESTION, records)
    assert not unsupported("В [Казани](location) [3](stock) шт. [P-1042B](part).", QUESTION, records)


@pytest.mark.parametrize(
    ("phrase", "value"),
    [
        ("двадцати двух", 22), ("двадцати двум", 22), ("двадцатью двумя", 22),
        ("двум тысячам", 2000), ("двух тысяч", 2000), ("двумя тысячами", 2000),
        ("двух тысяч шестисот тридцати", 2630), ("двум тысячам шестистам тридцати", 2630),
        ("двумя тысячами шестьюстами тридцатью", 2630),
        ("ста двум", 102), ("ста двумя", 102),
        ("пятидесяти пяти", 55), ("пятьюдесятью пятью", 55),
        ("восьмьюстами восьмьюдесятью восемью", 888),
        ("трёхсот сорока четырёх", 344),
        ("одного миллиарда двух миллионов трёх тысяч четырёхсот пятидесяти шести", 1002003456),
        ("нулю", 0), ("нулём", 0), ("одному", 1), ("одной", 1),
        ("обоих", 2), ("троим", 3), ("десятью", 10), ("одиннадцати", 11),
    ],
)
def test_case_forms_have_one_complete_numeric_value(phrase, value):
    assert normalize_numeric_text(phrase) == (str(value), [])
    assert numbers(phrase) == {Decimal(value)}


@pytest.mark.parametrize(
    ("answer", "field", "value"),
    [
        ("Остаток P-1042B в Казани равен двадцати двум штукам.", "qty", 22),
        ("Цена P-1042B равна двум тысячам рублей.", "price_rub", 2000),
        ("Остаток P-1042B в Казани составляет двадцать две штуки.", "qty", 22),
        ("Цена P-1042B равна двум тысячам шестистам тридцати рублям.", "price_rub", 2630),
    ],
)
def test_case_forms_are_checked_against_the_record(answer, field, value):
    record = {"part_id": PART, field: value}
    if field == "qty":
        record["warehouse"] = "Казань"
    assert not unsupported(answer, QUESTION, [record])
    assert unsupported(answer, QUESTION, [{**record, field: 0}])


def test_inflected_numbers_cannot_borrow_a_value_from_another_field():
    assert unsupported("Цена P-1042B равна трём рублям.", QUESTION, [PRICE, STOCK])
    assert unsupported(
        "Остаток P-1042B в Казани равен двум тысячам шестистам тридцати штукам.", QUESTION, [PRICE, STOCK]
    )


@pytest.mark.parametrize("confirm", [False, True])
@pytest.mark.parametrize("qty", [2, 22])
def test_order_does_not_authorize_the_tail_of_an_inflected_phrase(confirm, qty):
    args = {"part_id": PART, "model": "AR-210", "warehouse": "Казань", "qty": qty}
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return {"order_id": "SO-000001", **kwargs}

    result = run_agent(
        "Закажи P-1042B для AR-210 в количестве двадцати двух штук, склад Казань.",
        FakeLLM([call("create_service_order", **args), "Заказ SO-000001 создан."]),
        {"create_service_order": create},
        confirm=confirm,
    )
    assert calls == ([args] if confirm and qty == 22 else [])
    assert result.needs_confirmation == (not confirm and qty == 22)
    if qty == 2:
        assert result.pending_action is None and "количеств" in result.answer.lower()


def test_link_destinations_cannot_authorize_an_order_quantity():
    assert requested_quantities("Закажи [две](https://example.com/22) штуки P-1042B.") == {2}
    assert requested_quantities("Закажи [деталь](https://example.com/22) P-1042B.") == set()


@pytest.mark.parametrize("marker", ["_", "__"])
@pytest.mark.parametrize("table", [False, True])
@pytest.mark.parametrize("swapped", [False, True])
def test_literal_underscores_next_to_units_do_not_hide_numbers(marker, table, swapped):
    price, qty = (3, 2630) if swapped else (2630, 3)
    price_cell, stock_cell = f"{marker}{price}{marker}руб.", f"{marker}{qty}{marker}шт."
    answer = f"Цена: {price_cell}, остаток: {stock_cell}"
    if table:
        answer = f"| Цена | Остаток |\n| --- | --- |\n| {price_cell} | {stock_cell} |"
    assert numbers(answer) == {Decimal(3), Decimal(2630)}
    assert bool(unsupported(answer, QUESTION, [PRICE, STOCK])) == swapped


@pytest.mark.parametrize(
    "answer",
    [
        "    [Цена]: 9999",
        "```\n\n[Цена]: 9999\n\n```",
        "~~~text\n[Цена]: 9999\n~~~",
        "```\n[Цена]: 9999\n```\n\n[Цена]: https://example.com/2630",
    ],
)
def test_reference_like_content_inside_code_is_visible_evidence(answer):
    assert numbers(answer) == {Decimal(9999)}
    assert unsupported(answer, QUESTION, [PRICE]) == ["9999"]
    assert not unsupported(answer.replace("9999", "2630"), QUESTION, [PRICE])


@pytest.mark.parametrize(
    ("wrong", "qty"),
    [
        ("Цена: [3](https://example.com/price) руб., остаток: [2630](https://example.com/stock) шт.", 3),
        ("Остаток P-1042B в Казани равен двадцати двум штукам.", 0),
        ("Цена P-1042B равна двум тысячам рублей.", 3),
        ("Цена: __3__руб., остаток: __2630__шт.", 3),
        ("```\n\n[Цена]: 9999\n\n```", 3),
    ],
)
@pytest.mark.parametrize("repair", [False, True])
def test_agent_repairs_or_refuses_each_review_scenario(wrong, qty, repair):
    quantity = "трём" if qty else "нулю"
    correct = (
        "Цена P-1042B равна [двум тысячам шестистам тридцати](price) рублям. "
        f"Остаток P-1042B в Казани равен {quantity} штукам."
    )
    llm = FakeLLM([
        call("get_price", part_id=PART), call("check_stock", part_id=PART, warehouse="Казань"),
        wrong, correct if repair else wrong,
    ])
    result = run_agent(
        QUESTION, llm,
        {"get_price": lambda **args: PRICE, "check_stock": lambda **args: {**STOCK, "qty": qty}},
    )
    assert len(llm.requests) == 4
    assert result.refused == (not repair)
    assert wrong not in result.answer
    if repair:
        assert result.answer == correct
