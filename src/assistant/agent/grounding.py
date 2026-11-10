"""Проверки кодов, чисел и количества без обращения к модели."""

import re
from decimal import Decimal, InvalidOperation

from assistant.infra.grounding import find_codes, normalize_code

_CODE_TOKEN = re.compile(r"(?<!\w)[A-Za-zА-Яа-яЁё]{1,2}[- ]?\d{2,4}[A-Za-zА-Яа-яЁё]?(?!\w)")
_ORDER_ID = re.compile(r"\bSO-\d+\b", re.IGNORECASE)
_PRICE_UNIT = r"(?:руб\w*|₽)"
_QUANTITY_UNIT = r"(?:шт\w*|единиц\w*|штук\w*|детал\w*|экземпляр\w*)"
_NUMBER = (
    r"(?<![\w+−-])[-+−]?(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:[.,]\d+)?"
    rf"(?:(?![\w-])|(?=(?i:{_PRICE_UNIT}|{_QUANTITY_UNIT})(?![\w-])))"
)
_LIST_MARKER = re.compile(r"(?m)^[ \t]*\d+[.)](?=[ \t]+\S)")
_PRICE = re.compile(rf"({_NUMBER})\s*{_PRICE_UNIT}", re.IGNORECASE)
_STOCK = re.compile(rf"({_NUMBER})\s*{_QUANTITY_UNIT}", re.IGNORECASE)
_PRICE_PREFIX = re.compile(rf"(?:цена|стоимость|стоит)\s*[:=—-]?\s*({_NUMBER})", re.IGNORECASE)
_STOCK_PREFIX = re.compile(rf"(?:остаток|в наличии|на складе)\s*[:=—-]?\s*({_NUMBER})", re.IGNORECASE)
_ORDER_AMOUNT = re.compile(
    rf"\b(?:закаж\w*|заказ\w*|оформи\w*)\b\s*(?:SO-\d+\s*)?(?:создан\w*\s*)?"
    rf"[:=—-]?\s*(?:на\s+|в количестве\s+)?({_NUMBER})",
    re.IGNORECASE,
)
_STOCK_CONTEXT = re.compile(r"\b(?:остат\w*|остал\w*|в наличии|по наличию|доступн\w*)\b", re.IGNORECASE)
_WAREHOUSE_CONTEXT = re.compile(r"\b(?:на\s+)?склад\w*\b", re.IGNORECASE)
_ORDER_CONTEXT = re.compile(r"\b(?:заказ\w*|закаж\w*|оформи\w*|оформл\w*)\b", re.IGNORECASE)
_ORDER_PREDICATE = re.compile(
    r"\b(?:заказан(?:о|а|ы)?|заказал[аи]?|оформлен(?:о|а|ы)?|оформил[аи]?|создан(?:о|а|ы)?)\b",
    re.IGNORECASE,
)
_CLAUSE_BREAK = re.compile(
    r"(?<!\d),(?!\d)|,(?=\s)|"
    rf"\b(?:и|а|но)\s+(?=(?:{_STOCK_CONTEXT.pattern}|{_WAREHOUSE_CONTEXT.pattern}|"
    rf"{_ORDER_CONTEXT.pattern}|{_ORDER_PREDICATE.pattern}))",
    re.IGNORECASE,
)
_ORDER_CONTINUATION = re.compile(r"^\s*(?:[-*•]\s*)?(?:количеств\w*\b|P[- ]?\d{4}[A-Z]?\b)", re.IGNORECASE)
_REQUEST_START = re.compile(
    r"^(?:пожалуйста,?\s+)?(?:сколько|(?:(?:на|с|для|в)\s+)?как(?:ой|ом|ого|ая|ую|ое)|"
    r"укажите|уточните|подскажите)\b",
    re.IGNORECASE,
)
_REQUEST_FIELDS = (
    (re.compile(r"\b(?:склад\w*|город\w*)\b", re.IGNORECASE), "Укажите склад."),
    (re.compile(r"\b(?:количеств\w*|сколько)\b", re.IGNORECASE), "Сколько штук заказать?"),
    (re.compile(r"\bмодел\w*\b", re.IGNORECASE), "Для какой модели оформить заказ?"),
    (re.compile(r"\bдетал\w*\b", re.IGNORECASE), "Укажите код детали."),
)
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


def clarification_request(answer: str) -> str:
    """Формирует запрос параметров, не копируя возможные утверждения модели."""
    requests = []
    for sentence in re.split(r"[.!?;\n]+", answer):
        sentence = sentence.strip()
        if not _REQUEST_START.match(sentence):
            continue
        fields = [
            (match.start(), prompt)
            for pattern, prompt in _REQUEST_FIELDS
            if (match := pattern.search(sentence)) is not None
        ]
        if fields:
            # Берём предмет запроса, а не параметры в последующей оговорке модели.
            requests.append(min(fields)[1])
        elif re.search(r"\b(?:код|номер)\b", sentence, re.IGNORECASE):
            requests.append("Укажите код детали.")
    return " ".join(dict.fromkeys(requests))


def _quantity_context(text: str, position: int) -> str:
    """Разделяет сведения о заказе и наличии в соседних частях предложения."""
    # Точка сокращения не заканчивает «2 шт. в наличии»; длина строки сохраняется.
    text = re.sub(r"(?i:шт)\.(?=\s+[а-яё])", lambda match: match.group()[:-1] + " ", text)
    end = re.search(r"[;!?\n]|\.(?!\d)|" + _CLAUSE_BREAK.pattern, text[position:], re.IGNORECASE)
    scope = text[:position + end.start()] if end else text
    context = "stock"
    for sentence in re.split(r"[;!?\n]|\.(?!\d)", scope):
        if not sentence.strip():
            continue
        # Наследуют контекст только явно названное количество или строка позиции.
        # Самостоятельное сообщение о наличии не становится продолжением заказа.
        if not _ORDER_CONTINUATION.match(sentence):
            context = "stock"
        for clause in _CLAUSE_BREAK.split(sentence):
            # «В наличии для заказа» всё ещё утверждает наличие, а не размер заказа.
            if _STOCK_CONTEXT.search(clause):
                context = "stock"
            elif _ORDER_PREDICATE.search(clause) or any(
                not re.search(r"\b(?:для|под|к)\s+[\w\s-]*$", clause[:match.start()], re.IGNORECASE)
                for match in _ORDER_CONTEXT.finditer(clause)
            ):
                context = "order"
            elif _WAREHOUSE_CONTEXT.search(clause):
                context = "stock"
    return context


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
    # Количество из просьбы о заказе не подтверждает наличие на складе.
    stock = {_number(value) for value in _STOCK_PREFIX.findall(question)}
    order_quantities = {Decimal(value) for value in requested_quantities(question)}
    for result in results:
        if result.get("error") or result.get("refused"):
            continue
        if "price_rub" in result:
            prices.add(_number(result["price_rub"]))
        if "qty" in result:
            allowed = order_quantities if "order_id" in result else stock
            allowed.add(_number(result["qty"]))
    order_amounts = list(_ORDER_AMOUNT.finditer(answer))
    for match in [*_STOCK.finditer(answer), *order_amounts]:
        allowed = order_quantities if _quantity_context(answer, match.start(1)) == "order" else stock
        if _number(match.group(1)) not in allowed:
            errors.append(match.group(1))
    for pattern, allowed in ((_PRICE, prices), (_PRICE_PREFIX, prices), (_STOCK_PREFIX, stock)):
        errors.extend(value for value in pattern.findall(answer) if _number(value) not in allowed)
    return sorted(set(errors))
