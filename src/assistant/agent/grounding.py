"""Проверки кодов, чисел и количества без обращения к модели."""

import re
from decimal import Decimal, InvalidOperation

from assistant.agent.numeric_text import normalize_numeric_text
from assistant.agent.warehouses import WAREHOUSE_PATTERN, find_warehouses, selected_warehouses, warehouse_label
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
_STOCK_CONTEXT = re.compile(
    r"\b(?:остат\w*|остал\w*|в наличии|по наличию|доступн\w*|законч\w*)\b", re.IGNORECASE
)
_WAREHOUSE_CONTEXT = re.compile(r"\b(?:на\s+)?склад\w*\b", re.IGNORECASE)
_ORDER_CONTEXT = re.compile(r"\b(?:заказ\w*|закаж\w*|оформи\w*|оформл\w*)\b", re.IGNORECASE)
_ORDER_PREDICATE = re.compile(
    r"\b(?:заказан(?:о|а|ы)?|заказал[аи]?|оформлен(?:о|а|ы)?|оформил[аи]?|создан(?:о|а|ы)?)\b",
    re.IGNORECASE,
)
_ORDER_REQUEST = re.compile(
    r"\b(?:закаж(?:и|ите)|оформи(?:те)?)\b|"
    r"(?:^\s*|\b(?:хочу|нужно|необходимо|прошу|давай(?:те)?)\s+)(?:заказать|оформить)\b",
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
_TIME_UNIT = re.compile(
    r"\s*(?:январ\w*|феврал\w*|март\w*|апрел\w*|ма[йяе]|июн\w*|июл\w*|август\w*|"
    r"сентябр\w*|октябр\w*|ноябр\w*|декабр\w*|день|дня|дней|дни|сут\w*|недел\w*|"
    r"месяц\w*|год\w*|лет|час\w*|минут\w*|секунд\w*|ч|мин|сек|дн|нед|мес|г)\b",
    re.IGNORECASE,
)


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
    text, _ = normalize_numeric_text(text)
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


def _quantity_context(text: str, position: int, *, request: bool = False) -> str:
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
            elif request and any(
                not re.search(r"\bне\s*$", clause[:match.start()], re.IGNORECASE)
                for match in _ORDER_REQUEST.finditer(clause)
            ):
                context = "order"
            elif not request and (_ORDER_PREDICATE.search(clause) or any(
                not re.search(r"\b(?:для|под|к)\s+[\w\s-]*$", clause[:match.start()], re.IGNORECASE)
                for match in _ORDER_CONTEXT.finditer(clause)
            )):
                context = "order"
            elif not request and _WAREHOUSE_CONTEXT.search(clause):
                context = "stock"
    return context


def requested_quantities(question: str) -> set[int]:
    """Не принимаем цифры кода, цену или дату за количество для заказа."""
    question, errors = normalize_numeric_text(question)
    if errors:
        return set()
    text = _without_codes(question).lower()
    text = re.sub(r"\b\d{1,4}[./-]\d{1,2}[./-]\d{1,4}\b", " ", text)
    text = re.sub(r"\b\d{1,2}/\d{1,2}\b", " ", text)
    text = re.sub(r"\b\d{1,2}:\d{2}\b", " ", text)
    for pattern in (_PRICE, _PRICE_PREFIX):
        text = pattern.sub(" ", text)
    candidates = re.compile(_NUMBER)
    direct = re.compile(r"\b(?:количеств\w*|закаж\w*|заказать|оформи\w*|заказ)\s*[:=]?\s*(?:на\s+)?(?:не\s+)?$")
    alternative = re.compile(rf"\s*(?:{_QUANTITY_UNIT}|реле)?\.?\s*,?\s*(?:или|либо|а)\s*(?:не\s+)?$")
    found = set()
    previous = None
    for match in candidates.finditer(text):
        if _TIME_UNIT.match(text[match.end():]):
            previous = None
            continue
        anchored = (
            re.match(rf"\s*(?:{_QUANTITY_UNIT}|реле)\b", text[match.end():])
            or direct.search(text[:match.start()])
            or (previous is not None and alternative.fullmatch(text[previous.end():match.start()]))
        )
        if not anchored or _quantity_context(text, match.start(), request=True) != "order":
            previous = None
            continue
        previous = match
        if not re.search(r"\bне\s*$", text[:match.start()]):
            found.add(_number(match.group()))
    return {int(value) for value in found if value is not None and value > 0 and value == value.to_integral_value()}


def requested_warehouses(question: str) -> set[str]:
    """Склад просьбы о заказе или отдельного поля, не склад справочного остатка."""
    question, _ = normalize_numeric_text(question)
    text = re.sub(r"(?i:\bшт)\.", "шт ", question)
    selected, labels = set(), set()
    requested = ambiguous = False
    for sentence in re.split(r"[;!?\n]|\.(?!\d)", text):
        active = False
        for clause in _CLAUSE_BREAK.split(sentence):
            labels.update(warehouse_label(clause))
            command = any(
                not re.search(r"\bне\s*$", clause[:match.start()], re.IGNORECASE)
                for match in _ORDER_REQUEST.finditer(clause)
            )
            if command:
                active = requested = True
            elif _STOCK_CONTEXT.search(clause) or _STOCK.search(clause):
                active = False
            if command and not _STOCK_CONTEXT.search(clause):
                selected.update(selected_warehouses(clause))
            elif active:
                selected.update(warehouse_label(clause))
            if active and find_warehouses(clause) and re.search(r"\b(?:или|либо)\b", clause, re.IGNORECASE):
                ambiguous = True
            if (
                re.match(r"\s*склад\b", clause, re.IGNORECASE)
                and not warehouse_label(clause)
                and not _STOCK_CONTEXT.search(clause)
                and not _STOCK.search(clause)
            ):
                ambiguous = True
    return (selected | labels) if requested and not ambiguous else set()


def evidence_text(result: dict) -> str:
    """Ошибки и имена файлов не становятся доказательством номера детали или цены."""
    if result.get("error") or result.get("refused"):
        return ""
    values = []
    for key, value in result.items():
        if key in {"sources", "discontinued"}:
            continue
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            values.append(normalize_numeric_text(str(value))[0])
        elif key == "facts" and isinstance(value, list):
            values.extend(normalize_numeric_text(str(item))[0] for item in value)
    # Разбираем каждый источник отдельно: соседние поля «двадцать» и «две»
    # не должны после объединения превратиться в доказательство числа 22.
    return "\n".join(values)


def _claim_entities(text: str, position: int) -> tuple[set[str], set[str]]:
    """Идентификаторы количественного факта и его однозначного заголовка."""
    text = re.sub(
        r"(?i:\b(?:шт|руб))\.(?=\s+(?:[а-яё]|[PР][- ]?\d))",
        lambda match: match.group()[:-1] + " ",
        text,
    )
    sentence_break = r"[;!?\n]|\.(?!\d)"
    entity_break = (
        rf"\b(?:и|а|но)\s+(?=(?:(?:(?:в|на|из)\s+)?{WAREHOUSE_PATTERN}|"
        rf"{_CODE_TOKEN.pattern}|цена\b|стоимость\b))"
    )
    boundaries = list(re.finditer(
        sentence_break + "|" + _CLAUSE_BREAK.pattern + "|" + entity_break, text, re.IGNORECASE
    ))
    start = max((match.end() for match in boundaries if match.end() <= position), default=0)
    end = min((match.start() for match in boundaries if match.start() > position), default=len(text))
    clause = text[start:end]
    sentence_start = max((match.end() for match in re.finditer(sentence_break, text[:start])), default=0)
    heading = text[sentence_start:start]
    parts = {code for code in find_codes(clause) if code.startswith("P-")}
    warehouses = find_warehouses(clause)
    if not parts:
        heading_parts = {code for code in find_codes(heading) if code.startswith("P-")}
        if len(heading_parts) == 1:
            parts = heading_parts
    if not warehouses:
        heading_warehouses = find_warehouses(heading)
        if len(heading_warehouses) == 1:
            warehouses = heading_warehouses
    return parts, warehouses


def _records(results: list[dict], field: str) -> list[dict]:
    """Последнее успешное наблюдение каждой детали/склада, без количества заказа."""
    records = {}
    for result in results:
        if result.get("error") or result.get("refused") or field not in result:
            continue
        if field == "qty" and "order_id" in result:
            continue
        key = (result.get("part_id"), result.get("warehouse") if field == "qty" else None)
        records[key] = result
    return list(records.values())


def _matches_value(
    value: Decimal | None, parts: set[str], warehouses: set[str], records: list[dict], field: str
) -> bool:
    if field != "qty":
        warehouses = set()
    candidates = [
        record for record in records
        if (not parts or record.get("part_id") in parts)
        and (not warehouses or record.get("warehouse") in warehouses)
    ]
    if not candidates:
        return False
    # Для перечисленных сущностей требуются записи для каждой пары, а не любое
    # совпадение числового значения в общем наборе результатов ERP.
    for part in parts or {None}:
        for warehouse in warehouses or {None}:
            if not any(
                (part is None or record.get("part_id") == part)
                and (warehouse is None or record.get("warehouse") == warehouse)
                for record in candidates
            ):
                return False
    return all(_number(record[field]) == value for record in candidates)


def _matches_record(answer: str, match: re.Match, records: list[dict], field: str) -> bool:
    return _matches_value(_number(match.group(1)), *_claim_entities(answer, match.start(1)), records, field)


def _table_errors(
    answer: str, stock_records: list[dict], price_records: list[dict], order_records: list[dict]
) -> list[str]:
    """Проверяет числовые ячейки вместе с идентификаторами строки таблицы."""
    has_erp = bool(stock_records or price_records or order_records)
    lines = answer.splitlines()
    header = None
    errors = []
    for index, line in enumerate(lines):
        if "|" not in line:
            header = None
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        if index + 1 < len(lines):
            next_cells = [cell.strip() for cell in lines[index + 1].strip().strip("|").split("|")]
            if len(next_cells) == len(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in next_cells):
                header = cells
                continue
        if header is None or len(header) != len(cells):
            if any(numbers(cell) for cell in cells) and (has_erp or _PRICE.search(line) or _STOCK.search(line)):
                errors.append("неразобранная таблица ERP")
            continue
        row = " ".join(cells)
        parts = {code for code in find_codes(row) if code.startswith("P-")}
        warehouses = find_warehouses(row)
        order_ids = {value.upper() for value in _ORDER_ID.findall(row)}
        for title, cell in zip(header, cells, strict=True):
            values = numbers(cell)
            if not values:
                continue
            if re.search(r"цена|стоимость|price", title, re.IGNORECASE):
                records, field = price_records, "price_rub"
            elif re.search(r"остаток|наличие|количеств|qty", title, re.IGNORECASE):
                records, field = stock_records, "qty"
                if order_ids and not re.search(r"остаток|наличие", title, re.IGNORECASE):
                    records = [record for record in order_records if record["order_id"].upper() in order_ids]
            else:
                if has_erp or _PRICE.search(cell) or _STOCK.search(cell):
                    errors.append("неразобранная таблица ERP")
                continue
            errors.extend(
                str(value) for value in values if not _matches_value(value, parts, warehouses, records, field)
            )
    return errors


def _in_table(text: str, position: int) -> bool:
    start = text.rfind("\n", 0, position) + 1
    end = text.find("\n", position)
    return "|" in text[start:end if end != -1 else len(text)]


def unsupported(answer: str, question: str, results: list[dict]) -> list[str]:
    """Возвращает неподтверждённые коды/числа; цены и остатки проверяет раздельно."""
    order_quantities = {Decimal(value) for value in requested_quantities(question)}
    answer, errors = normalize_numeric_text(answer)
    question, _ = normalize_numeric_text(question)
    evidence = question + "\n" + "\n".join(evidence_text(result) for result in results)
    errors.extend(code for code in find_codes(answer) if code not in find_codes(evidence))
    order_ids = {value.upper() for value in _ORDER_ID.findall(evidence)}
    errors.extend(value for value in _ORDER_ID.findall(answer) if value.upper() not in order_ids)
    errors.extend(str(value) for value in sorted(numbers(answer) - numbers(evidence)))
    prices = {_number(value) for pattern in (_PRICE, _PRICE_PREFIX) for value in pattern.findall(question)}
    # Количество из просьбы о заказе не подтверждает наличие на складе.
    stock = {_number(value) for value in _STOCK_PREFIX.findall(question)}
    stock_records = _records(results, "qty")
    price_records = _records(results, "price_rub")
    order_records = [result for result in results if result.get("order_id") and not result.get("error")]
    errors.extend(_table_errors(answer, stock_records, price_records, order_records))
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
        if _in_table(answer, match.start(1)):
            continue
        if _quantity_context(answer, match.start(1)) == "order":
            valid = _number(match.group(1)) in order_quantities
        else:
            valid = (
                _matches_record(answer, match, stock_records, "qty") if stock_records
                else _number(match.group(1)) in stock
            )
        if not valid:
            errors.append(match.group(1))
    for pattern, allowed in ((_PRICE, prices), (_PRICE_PREFIX, prices), (_STOCK_PREFIX, stock)):
        field = "qty" if pattern is _STOCK_PREFIX else "price_rub"
        records = stock_records if field == "qty" else price_records
        for match in pattern.finditer(answer):
            if _in_table(answer, match.start(1)):
                continue
            valid = _matches_record(answer, match, records, field) if records else _number(match.group(1)) in allowed
            if not valid:
                errors.append(match.group(1))
    # Краткие записи «Москва — 3» и «P-1042B — 2630» также являются фактами,
    # даже когда единица вынесена в заголовок списка.
    units = rf"\s*(?:{_PRICE_UNIT}|{_QUANTITY_UNIT})"
    for match in re.finditer(_NUMBER, answer):
        if _in_table(answer, match.start()) or re.match(units, answer[match.end():], re.IGNORECASE):
            continue
        prefix = answer[:match.start()]
        warehouse_value = re.search(rf"{WAREHOUSE_PATTERN}\s*[:=—-]?\s*$", prefix, re.IGNORECASE)
        part_value = re.search(rf"{_CODE_TOKEN.pattern}\s*[:=—-]?\s*$", prefix, re.IGNORECASE)
        if part_value and not any(code.startswith("P-") for code in find_codes(part_value.group())):
            part_value = None
        if not warehouse_value and not part_value:
            continue
        if not warehouse_value and _quantity_context(answer, match.start()) == "order":
            if _number(match.group()) not in order_quantities:
                errors.append(match.group())
            continue
        contexts = list(re.finditer(r"\b(?:цен\w*|стоимост\w*|остат\w*|наличи\w*)\b", prefix, re.IGNORECASE))
        price_context = contexts and re.match(r"цен|стоимост", contexts[-1].group(), re.IGNORECASE)
        if warehouse_value:
            records, field = stock_records, "qty"
        elif price_context or (price_records and not stock_records):
            records, field = price_records, "price_rub"
        else:
            records, field = stock_records, "qty"
        if not _matches_value(_number(match.group()), *_claim_entities(answer, match.start()), records, field):
            errors.append(match.group())
    return sorted(set(errors))
