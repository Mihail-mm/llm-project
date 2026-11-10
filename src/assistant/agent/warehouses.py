"""Закрытый справочник складов и формы их названий в русском тексте."""

import re

_NAMES = {
    "Москва": r"москв(?:а|ы|е|у|ой|ою)",
    "Санкт-Петербург": r"(?:санкт[-–—\s]петербург(?:а|е|у|ом)?|спб)",
    "Казань": r"казан(?:ь|и|ью)",
    "Новосибирск": r"новосибирск(?:а|е|у|ом)?",
}
WAREHOUSES = list(_NAMES)
WAREHOUSE_PATTERN = rf"\b(?:{'|'.join(_NAMES.values())})\b"
_PATTERNS = {name: re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE) for name, pattern in _NAMES.items()}


def find_warehouses(text: str) -> set[str]:
    return {name for name, pattern in _PATTERNS.items() if pattern.search(text)}


def selected_warehouses(text: str) -> set[str]:
    """Не превращает запрет склада в положительный выбор."""
    return {
        name for name, pattern in _PATTERNS.items()
        for match in pattern.finditer(text)
        if not re.search(r"\b(?:не|кроме)(?:\s+(?:в|из|на|склад\w*))?\s*$", text[:match.start()], re.IGNORECASE)
    }


def warehouse_label(text: str) -> set[str]:
    """Отдельное поле «склад: Казань», без утверждения об остатке."""
    return {
        name for name, pattern in _NAMES.items()
        if re.fullmatch(
            rf"\s*(?:(?:в|на|из|для|с|со)\s+)?(?:склад\w*\s*[:=—-]?\s*)?(?:{pattern})\s*",
            text,
            re.IGNORECASE,
        )
    }
