from __future__ import annotations

import re
import unicodedata

from ingestion.loader import Document


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_BLANK_LINES_RE = re.compile(r"\n\s*\n+")


def preprocess_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    cleaned = _CONTROL_CHARS_RE.sub("", normalized)

    filtered_lines: list[str] = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            filtered_lines.append("")
            continue
        if len(stripped) >= 20:
            filtered_lines.append(stripped)

    collapsed = _BLANK_LINES_RE.sub("\n\n", "\n".join(filtered_lines))
    return collapsed.strip()


def preprocess_documents(documents: list[Document]) -> list[Document]:
    processed: list[Document] = []
    for doc in documents:
        text = preprocess_text(doc.text)
        if text:
            processed.append(Document(text=text, source=doc.source))
    return processed