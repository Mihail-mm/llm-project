"""Единое представление чисел для проверки фактов и параметров заказа."""

import re

_SMALL = {
    "ноль": 0, "нуль": 0,
    "один": 1, "одна": 1, "одно": 1, "одну": 1,
    "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5,
    "шесть": 6, "семь": 7, "восемь": 8, "девять": 9,
    "десять": 10, "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13,
    "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16,
    "семнадцать": 17, "восемнадцать": 18, "девятнадцать": 19,
}
_TENS = {
    "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50,
    "шестьдесят": 60, "семьдесят": 70, "восемьдесят": 80, "девяносто": 90,
}
_HUNDREDS = {
    "сто": 100, "двести": 200, "триста": 300, "четыреста": 400,
    "пятьсот": 500, "шестьсот": 600, "семьсот": 700,
    "восемьсот": 800, "девятьсот": 900,
}
_SCALES = {
    "тысяча": 1000, "тысячу": 1000, "тысячи": 1000, "тысяч": 1000,
    "миллион": 10**6, "миллиона": 10**6, "миллионов": 10**6,
    "миллиард": 10**9, "миллиарда": 10**9, "миллиардов": 10**9,
}
# Эти конструкции пока требуют записи цифрами. Включаем их в целую фразу,
# чтобы, например, из «полторы тысячи две» не получить разрешение на две штуки.
_FRACTION = r"полтора|полторы|полутора|половин(?:а|у|ы|ой)|четверть|цел(?:ая|ое|ых)"
_ORDINAL = (
    r"(?:перв|втор|четв[её]рт|пят|шест|седьм|восьм|девят|десят|сот|тысячн|миллионн|"
    r"двадцат|тридцат)(?:ый|ой|ая|ое|ую|ого|ому|ом|ые|ых)|треть(?:е|его|я|ю|им)"
)
_UNSUPPORTED = rf"{_FRACTION}|{_ORDINAL}"
_WORDS = "|".join((*_SMALL, *_TENS, *_HUNDREDS, *_SCALES, "минус", "плюс"))
_TOKEN = rf"(?:{_WORDS}|{_UNSUPPORTED}|[-+−]?\d+(?:[.,]\d+)?)"
# Перенос строки внутри одного числительного не делает его хвост отдельным числом.
_PHRASE = re.compile(rf"(?<![\w+−-]){_TOKEN}(?![\w-])(?:\s+{_TOKEN}(?![\w-]))*", re.IGNORECASE)
_FORMATTING = re.compile(r"(?P<mark>\*{1,3}|_{1,3}|`+)(?=\S)(?P<text>[^\n]*?\S)(?P=mark)")


def _group(words: list[str]) -> int | None:
    """Одна группа до тысячи: сотни, десятки, единицы — в этом порядке."""
    if len(words) == 1 and words[0].isdigit():
        value = int(words[0])
        return value if value < 1000 else None
    index = value = 0
    if words and words[0] in _HUNDREDS:
        value = _HUNDREDS[words[0]]
        index += 1
    if index < len(words) and words[index] in _TENS:
        value += _TENS[words[index]]
        index += 1
        if index < len(words) and 0 < _SMALL.get(words[index], 100) < 10:
            value += _SMALL[words[index]]
            index += 1
    elif index < len(words) and words[index] in _SMALL:
        # Ноль не является хвостом составного числа («сто ноль»).
        if value and _SMALL[words[index]] == 0:
            return None
        value += _SMALL[words[index]]
        index += 1
    return value if words and index == len(words) else None


def _cardinal(words: list[str]) -> int | None:
    sign = 1
    if words[0] in {"минус", "плюс"}:
        sign = -1 if words[0] == "минус" else 1
        words = words[1:]
    if not words:
        return None
    total, previous_scale, start = 0, 10**12, 0
    for index, word in enumerate(words):
        if word not in _SCALES:
            continue
        scale = _SCALES[word]
        group = _group(words[start:index]) if index > start else 1
        if scale >= previous_scale or group is None or group <= 0:
            return None
        total += group * scale
        previous_scale, start = scale, index + 1
    if start < len(words):
        group = _group(words[start:])
        if group is None or (total and group == 0):
            return None
        total += group
    return sign * total


def normalize_numeric_text(text: str) -> tuple[str, list[str]]:
    """Убирает парное оформление и заменяет целые числительные на цифры.

    Неразобранные числовые фразы возвращаются как ошибки, а в тексте заменяются
    неделимым маркером: их хвосты нельзя принять за отдельное количество.
    Исходный ответ пользователю не изменяется.
    """
    while True:
        plain = _FORMATTING.sub(lambda match: match["text"], text)
        if plain == text:
            break
        text = plain
    errors = []

    def replace(match: re.Match) -> str:
        phrase = match.group()
        if not re.search(r"[а-яё]", phrase, re.IGNORECASE):
            return phrase  # Форматирование десятичных чисел обрабатывается отдельно.
        if re.fullmatch(_ORDINAL, phrase, re.IGNORECASE):
            return phrase  # «Первая причина» не является количественным фактом.
        value = _cardinal(phrase.lower().split())
        if value is None:
            errors.append(phrase)
            return "неразобранное_число"
        return str(value)

    return _PHRASE.sub(replace, text), errors
