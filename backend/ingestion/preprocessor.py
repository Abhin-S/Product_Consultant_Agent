from __future__ import annotations

import re
import unicodedata

from ingestion.loader import Document


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_BLANK_LINES_RE = re.compile(r"\n\s*\n+")


def _looks_informative_line(line: str) -> bool:
    if not line:
        return False

    if line.startswith("[Page "):
        return True

    # Preserve markdown table structure.
    if line.count("|") >= 2:
        return True

    alnum_chars = sum(1 for char in line if char.isalnum())
    if alnum_chars == 0:
        return False

    if len(line) <= 2:
        return False

    ratio = alnum_chars / max(len(line), 1)
    return ratio >= 0.25


def preprocess_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    cleaned = _CONTROL_CHARS_RE.sub("", normalized)

    filtered_lines: list[str] = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            filtered_lines.append("")
            continue

        if _looks_informative_line(stripped):
            filtered_lines.append(stripped)

    collapsed = _BLANK_LINES_RE.sub("\n\n", "\n".join(filtered_lines))
    return collapsed.strip()


def preprocess_documents(documents: list[Document]) -> list[Document]:
    processed: list[Document] = []
    for doc in documents:
        text = preprocess_text(doc.text)
        if text:
            processed.append(
                Document(
                    text=text,
                    source=doc.source,
                    metadata=dict(doc.metadata),
                )
            )
    return processed