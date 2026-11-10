"""Числительные и оформление не меняют смысл количества, цены и остатка."""

from decimal import Decimal

import pytest

from assistant.agent.agent import run_agent
from assistant.agent.grounding import evidence_text, numbers, requested_quantities, unsupported
from assistant.agent.numeric_text import normalize_numeric_text
from assistant.infra.llm import LLMResponse, ToolCall
from assistant.infra.testing import FakeLLM

PART = "P-1042B"
ORDER = {"part_id": PART, "model": "AR-210", "warehouse": "Казань", "qty": 22}
PRICE = {"part_id": PART, "price_rub": 2630}
STOCK = {"part_id": PART, "warehouse": "Казань", "qty": 3}
QUESTION = "Цена и остаток P-1042B в Казани?"


def call(name, **args):
    return LLMResponse("", [ToolCall(name, name, args)])


@pytest.mark.parametrize(
    ("phrase", "value"),
    [
        ("двадцать две", 22),
        ("сто две", 102),
        ("девятнадцать", 19),
        ("девятьсот девяносто девять", 999),
        ("две тысячи", 2000),
        ("Две Тысячи Шестьсот Тридцать", 2630),
        ("тысячу одну", 1001),
        ("один миллион две тысячи три", 1002003),
        ("2 тысячи три", 2003),
        ("двадцать **две**", 22),
        ("двадцать\nдве", 22),
    ],
)
def test_order_numeral_is_read_as_one_whole_value(phrase, value):
    assert requested_quantities(f"Закажи {phrase} штуки P-1042B для AR-210, склад Казань.") == {value}
    assert numbers(f"Количество: {phrase}") == {Decimal(value)}


@pytest.mark.parametrize("phrase,value", [("двадцать две", 22), ("сто две", 102)])
@pytest.mark.parametrize("confirm", [False, True])
@pytest.mark.parametrize("use_tail", [False, True])
def test_agent_never_orders_only_the_tail_of_a_composite_numeral(phrase, value, confirm, use_tail):
    args = {**ORDER, "qty": 2 if use_tail else value}
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return {"order_id": "SO-000001", **kwargs}

    result = run_agent(
        f"Закажи {phrase} штуки P-1042B для AR-210, склад Казань.",
        FakeLLM([call("create_service_order", **args), "Заказ SO-000001 создан."]),
        {"create_service_order": create},
        confirm=confirm,
    )
    assert calls == ([args] if confirm and not use_tail else [])
    assert result.needs_confirmation == (not confirm and not use_tail)
    if use_tail:
        assert result.pending_action is None and "количеств" in result.answer.lower()
    elif not confirm:
        assert result.pending_action["args"] == args


@pytest.mark.parametrize(
    "phrase",
    ["двадцать две три", "сто ноль две", "две тысячи тысячи две", "полторы тысячи две", "две целых пять десятых"],
)
def test_unparsed_phrase_cannot_authorize_its_tail(phrase):
    assert requested_quantities(f"Закажи {phrase} штуки P-1042B.") == set()
    assert unsupported(f"В наличии {phrase} штуки P-1042B.", QUESTION, [STOCK, {"qty": 2}])


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Закажи двадцать две или сто две штуки P-1042B.", {22, 102}),
        ("Закажи не двадцать две, а сто две штуки P-1042B.", {102}),
        ("Закажи P-1042B за две тысячи рублей, количество двадцать две штуки.", {22}),
        ("Закажи P-1042B через двадцать два дня, количество две штуки.", {2}),
        ("Закажи двадцать второго ноября две штуки P-1042B.", set()),
        ("Закажи минус двадцать две штуки P-1042B.", set()),
    ],
)
def test_normalized_numerals_still_respect_quantity_context(question, expected):
    assert requested_quantities(question) == expected


@pytest.mark.parametrize("marker", ["**", "__", "*", "_", "`", "***", "``"])
@pytest.mark.parametrize("swapped", [False, True])
def test_formatted_numbers_keep_price_and_stock_provenance(marker, swapped):
    price, stock = (3, 2630) if swapped else (2630, 3)
    answer = f"Цена: {marker}{price}{marker} руб., остаток: {marker}{stock}{marker} шт."
    assert normalize_numeric_text(answer) == (f"Цена: {price} руб., остаток: {stock} шт.", [])
    assert bool(unsupported(answer, QUESTION, [PRICE, STOCK])) == swapped


@pytest.mark.parametrize(
    "answer",
    [
        "**Цена:** **3 руб.**, **остаток:** **2630 шт.**",
        "**Цена: 3 руб., остаток: 2630 шт.**",
        "Цена: ***3***руб., остаток: ***2630***шт.",
        "**Цена:** `3`, **остаток:** `2630`.",
        "В **Казани** **2630** шт. **P-1042B**.",
        "В Казани в наличии три штуки P-1042B, цена три рубля.",
        "Цена P-1042B — две тысячи рублей.",
        "Остаток P-1042B: две тысячи шестьсот тридцать шт.",
        "| Деталь | Цена | Остаток |\n| --- | --- | --- |\n| P-1042B | три | **2630** |",
    ],
)
def test_markup_and_words_cannot_swap_fields_or_invent_values(answer):
    assert unsupported(answer, QUESTION, [PRICE, STOCK])


@pytest.mark.parametrize(
    "answer",
    [
        "**Цена:** **2630 руб.**, **остаток:** **3 шт.**",
        "Цена: ***2630***руб., остаток: ***3***шт.",
        "Цена P-1042B — две тысячи шестьсот тридцать рублей. В Казани в наличии три штуки P-1042B.",
        "**Цена:** две тысячи шестьсот тридцать; **остаток:** три.",
        "| Деталь | Цена | Остаток |\n| --- | --- | --- |\n| P-1042B | две тысячи шестьсот тридцать | **три** |",
    ],
)
def test_supported_values_remain_valid_in_words_and_markdown(answer):
    assert not unsupported(answer, QUESTION, [PRICE, STOCK])


def test_word_quantities_do_not_ground_stock_from_the_order():
    question = "Закажи две штуки P-1042B для AR-210, склад Казань."
    assert unsupported("В Казани в наличии две штуки P-1042B.", question, [{**STOCK, "qty": 0}])
    assert not unsupported("В Казани в наличии ноль штук P-1042B.", question, [{**STOCK, "qty": 0}])


def test_normalization_preserves_codes_dates_and_separate_evidence_values():
    text = "P-1042B AR-210 SO-000001 12.11.2026\n2630\n3"
    assert normalize_numeric_text(text) == (text, [])
    assert normalize_numeric_text("Первая причина — первичный перегрев.") == (
        "Первая причина — первичный перегрев.", []
    )
    assert numbers("Цена: две тысячи\nОстаток: три") == {Decimal(2000), Decimal(3)}


def test_independent_evidence_values_do_not_become_a_composite_numeral():
    evidence = evidence_text({"first": "двадцать", "second": "две", "facts": ["сто", "три"]})
    assert numbers(evidence) == {Decimal(20), Decimal(2), Decimal(100), Decimal(3)}
    assert unsupported("Срок: двадцать два дня.", "Срок?", [{"facts": ["двадцать", "две"]}])


@pytest.mark.parametrize(
    ("wrong", "stock_qty"),
    [
        ("Цена: **3** руб., остаток: **2630** шт.", 3),
        ("В Казани в наличии две штуки P-1042B.", 0),
        ("Цена P-1042B — две тысячи рублей.", 3),
    ],
)
@pytest.mark.parametrize("repair", [False, True])
def test_agent_repairs_or_refuses_unsupported_formatted_or_word_facts(wrong, stock_qty, repair):
    quantity = "три" if stock_qty else "ноль"
    correct = f"Цена: **2630** руб. В Казани в наличии {quantity} штук P-1042B."
    llm = FakeLLM([
        call("get_price", part_id=PART),
        call("check_stock", part_id=PART, warehouse="Казань"),
        wrong,
        correct if repair else wrong,
    ])
    result = run_agent(
        QUESTION,
        llm,
        {"get_price": lambda **args: PRICE, "check_stock": lambda **args: {**STOCK, "qty": stock_qty}},
    )
    assert len(llm.requests) == 4
    assert result.refused == (not repair)
    assert wrong not in result.answer
    if repair:
        assert result.answer == correct
