"""Видимый текст Markdown для проверки фактов; адреса ссылок не являются фактами."""

from markdown_it import MarkdownIt
from markdown_it.token import Token

# Сохраняем строки, маркеры списков и разделители таблиц для grounding. При этом
# ссылки, включая reference links, разбираются штатными правилами CommonMark.
_MARKDOWN = MarkdownIt("zero").enable([
    "code", "fence", "reference", "newline", "escape", "backticks", "emphasis", "link", "image", "autolink", "entity",
])


def _inline_text(tokens: list[Token]) -> str:
    parts = []
    for token in tokens:
        if token.type in {"softbreak", "hardbreak"}:
            parts.append("\n")
        elif token.type == "image":
            parts.append(_inline_text(token.children or []))
        elif token.type in {"text", "code_inline"}:
            parts.append(token.content)
    return "".join(parts)


def visible_markdown_text(text: str) -> str:
    """Читает подписи ссылок, не их URL/title; ничего не загружает по сети."""
    blocks = []
    for token in _MARKDOWN.parse(text):
        if token.type == "inline":
            blocks.append(_inline_text(token.children or []))
        elif token.type in {"code_block", "fence"}:
            # Внутри блока строки вида «[Цена]: 9999» — видимый текст,
            # а не скрытые определения ссылок. Не разбираем их как Markdown снова.
            blocks.append(token.content.rstrip("\n"))
    return "\n\n".join(blocks)
