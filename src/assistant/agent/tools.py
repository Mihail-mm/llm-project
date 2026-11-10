"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.agent.warehouses import WAREHOUSES
from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.grounding import code_type, normalize_code
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

_QUESTION = {"type": "string", "minLength": 1, "description": "Полный вопрос с кодами и названием узла"}
_PART = {"type": "string", "pattern": r"^P-\d{4}[A-Z]?$", "description": "Код из вопроса или поиска знаний"}
_WAREHOUSE = {"type": "string", "enum": WAREHOUSES}


def _schema(name: str, description: str, properties: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _schema(
        "search_knowledge",
        "Найти узел, деталь, полную цепочку замен и источники в документах.",
        {"question": _QUESTION},
    ),
    _schema(
        "summarize_topic",
        "Ответить на обзорный вопрос по сводкам графа, включая частоты отказов.",
        {"question": _QUESTION},
    ),
    _schema(
        "check_stock",
        "Получить остаток и discontinued из ERP для детали и склада.",
        {"part_id": _PART, "warehouse": _WAREHOUSE},
    ),
    _schema("get_price", "Получить цену в рублях и discontinued из ERP.", {"part_id": _PART}),
    _schema(
        "create_service_order",
        "Создать заказ только по просьбе пользователя после подтверждения.",
        {
            "part_id": _PART,
            "warehouse": _WAREHOUSE,
            "qty": {"type": "integer", "minimum": 1, "description": "Количество, названное пользователем"},
            "model": {"type": "string", "pattern": r"^(AR|BZ|GS)-\d{3}$"},
        },
    ),
]
_PARAMETERS = {schema["function"]["name"]: schema["function"]["parameters"] for schema in TOOL_SCHEMAS}


def validate_arguments(name: str, arguments: dict) -> dict:
    """Одинаковая проверка для инструментов и ещё не выполненного pending_action."""
    if name not in _PARAMETERS:
        raise ValueError("Неизвестный инструмент")
    if not isinstance(arguments, dict) or set(arguments) != set(_PARAMETERS[name]["required"]):
        raise ValueError("Нужен JSON-объект со всеми обязательными аргументами без дополнительных полей")
    args = dict(arguments)
    for key in ("part_id", "model"):
        if key in args:
            value = args[key]
            normalized = normalize_code(value) if isinstance(value, str) else None
            expected = "Part" if key == "part_id" else "Model"
            if normalized is None or code_type(normalized) != expected:
                raise ValueError(f"Некорректный {key}")
            args[key] = normalized
    if "warehouse" in args and args["warehouse"] not in WAREHOUSES:
        raise ValueError("Неизвестный склад: выберите " + ", ".join(WAREHOUSES))
    if "qty" in args and (type(args["qty"]) is not int or args["qty"] <= 0):
        raise ValueError("Количество должно быть целым числом больше нуля")
    if "question" in args and (not isinstance(args["question"], str) or not args["question"].strip()):
        raise ValueError("Вопрос должен быть непустой строкой")
    return args


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """

    def search(question: str) -> dict:
        answer = local_search(question, kb, llm)
        return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources, "refused": answer.refused}

    def summarize(question: str) -> dict:
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    implementations = {
        "search_knowledge": search,
        "summarize_topic": summarize,
        "check_stock": erp.get_stock,
        "get_price": erp.get_price,
        "create_service_order": erp.create_order,
    }

    def wrap(name: str, implementation: Callable[..., dict]) -> Callable[..., dict]:
        def invoke(**arguments) -> dict:
            try:
                args = validate_arguments(name, arguments)
            except ValueError as exc:
                return {"error": str(exc), "error_type": "invalid_arguments"}
            try:
                return implementation(**args)
            except ErpNotFound as exc:
                return {"error": f"Данных в ERP нет: {exc}", "error_type": "not_found"}
            except ErpRejected as exc:
                return {"error": f"ERP отклонила запрос: {exc}", "error_type": "rejected"}

        return invoke

    return {name: wrap(name, implementation) for name, implementation in implementations.items()}
