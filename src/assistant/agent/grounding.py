"""Проверки кодов, чисел и количества без обращения к модели."""

import re
from decimal import Decimal, InvalidOperation

from assistant.infra.grounding import find_codes, normalize_code

_CODE_TOKEN = re.compile(r"(?<!\w)[A-Za-zА-Яа-яЁё]{1,2}[- ]?\d{2,4}[A-Za-zА-Яа-яЁё]?(?!\w)")
_ORDER_ID = re.compile(r"\bSO-\d+\b", re.IGNORECASE)
_NUMBER = r"(?<![\w+−-])[-+−]?(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:[.,]\d+)?(?![\w-])"
_LIST_MARKER = re.compile(r"(?m)^[ \t]*\d+[.)](?=[ \t]+\S)")
_PRICE = re.compile(rf"({_NUMBER})\s*(?:руб\w*|₽)", re.IGNORECASE)
_STOCK = re.compile(rf"({_NUMBER})\s*(?:шт\w*|единиц\w*|штук\w*|детал\w*|экземпляр\w*)", re.IGNORECASE)
_PRICE_PREFIX = re.compile(rf"(?:цена|стоимость|стоит)\s*[:=—-]?\s*({_NUMBER})", re.IGNORECASE)
_STOCK_PREFIX = re.compile(rf"(?:остаток|в наличии|на складе)\s*[:=—-]?\s*({_NUMBER})", re.IGNORECASE)
_WORDS = {
    "одну": 1,
    "один": 1,
    "одна": 1,
    "две": 2,
    "два": 2,
    "три": 3,
    "четыре": 4,
    "пять": 5,
    "шесть": 6,
    "семь": 7,
    "восемь": 8,
    "девять": 9,
    "десять": 10,
}


def _without_codes(text: str) -> str:
    text = _CODE_TOKEN.sub(lambda match: " " if normalize_code(match.group()) else match.group(), text)
    return _ORDER_ID.sub(" ", text)


def _number(raw) -> Decimal | None:
    try:
        number = Decimal(re.sub(r"[\s\u00a0\u202f]", "", str(raw)).replace(",", ".").replace("−", "-"))
        return number if number.is_finite() else None
    except InvalidOperation:
        return None


def numbers(text: str) -> set[Decimal]:
    # Номера пунктов не являются фактами ни в ответе, ни в его доказательствах.
    text = _without_codes(_LIST_MARKER.sub("", text))
    return {value for raw in re.findall(_NUMBER, text) if (value := _number(raw)) is not None}


def requested_quantities(question: str) -> set[int]:
    """Не принимаем цифры кода, цену или дату за количество для заказа."""
    text = _without_codes(question).lower()
    text = re.sub(r"\b\d{1,4}[./-]\d{1,2}[./-]\d{1,4}\b", " ", text)
    for pattern in (_PRICE, _PRICE_PREFIX):
        text = pattern.sub(" ", text)
    found = {_number(value) for value in _STOCK.findall(text)}
    found.update(
        _number(value)
        for value in re.findall(
            rf"\b(?:количеств\w*|закаж\w*|заказ(?:ать)?|оформи\w*)\b\s*[:=]?\s*(?:на\s+)?({_NUMBER})", text
        )
    )
    for word, value in _WORDS.items():
        if re.search(rf"\b{word}\s+(?:шт\w*|детал\w*|единиц\w*|экземпляр\w*|реле)\b", text):
            found.add(Decimal(value))
    return {int(value) for value in found if value is not None and value > 0 and value == value.to_integral_value()}


def evidence_text(result: dict) -> str:
    """Ошибки и имена файлов не становятся доказательством номера детали или цены."""
    if result.get("error") or result.get("refused"):
        return ""
    values = []
    for key, value in result.items():
        if key in {"sources", "discontinued"}:
            continue
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            values.append(str(value))
        elif key == "facts" and isinstance(value, list):
            values.extend(str(item) for item in value)
    return "\n".join(values)


def unsupported(answer: str, question: str, results: list[dict]) -> list[str]:
    """Возвращает неподтверждённые коды/числа; цены и остатки проверяет раздельно."""
    evidence = question + "\n" + "\n".join(evidence_text(result) for result in results)
    errors = [code for code in find_codes(answer) if code not in find_codes(evidence)]
    order_ids = {value.upper() for value in _ORDER_ID.findall(evidence)}
    errors.extend(value for value in _ORDER_ID.findall(answer) if value.upper() not in order_ids)
    errors.extend(str(value) for value in sorted(numbers(answer) - numbers(evidence)))
    prices = {_number(value) for pattern in (_PRICE, _PRICE_PREFIX) for value in pattern.findall(question)}
    stock = {_number(value) for pattern in (_STOCK, _STOCK_PREFIX) for value in pattern.findall(question)}
    stock.update(Decimal(value) for value in requested_quantities(question))
    for result in results:
        if result.get("error") or result.get("refused"):
            continue
        if "price_rub" in result:
            prices.add(_number(result["price_rub"]))
        if "qty" in result:
            stock.add(_number(result["qty"]))
    for pattern, allowed in ((_PRICE, prices), (_PRICE_PREFIX, prices), (_STOCK, stock), (_STOCK_PREFIX, stock)):
        errors.extend(value for value in pattern.findall(answer) if _number(value) not in allowed)
    return sorted(set(errors))
