from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from config import settings


logger = logging.getLogger(__name__)


@dataclass
class Document:
    text: str
    source: str


def _load_pdf(path: Path) -> str:
    def table_to_markdown(table: list[list[str | None]]) -> str:
        rows = [[(cell or "").strip() for cell in row] for row in table if row]
        if not rows:
            return ""

        width = max(len(row) for row in rows)
        normalized = [row + [""] * (width - len(row)) for row in rows]

        header = normalized[0]
        sep = ["---"] * width
        body = normalized[1:] if len(normalized) > 1 else []

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in body)
        return "\n".join(lines)

    chunks: list[str] = []
    with pdfplumber.open(path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            page_tables = page.extract_tables() or []

            table_blocks = [table_to_markdown(table) for table in page_tables]
            table_blocks = [block for block in table_blocks if block]

            section_parts = [f"[Page {idx}]", page_text]
            if table_blocks:
                section_parts.append("\n\n".join(table_blocks))

            chunks.append("\n\n".join(part for part in section_parts if part.strip()))
    return "\n".join(chunks)


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(docs_dir: str | None = None) -> list[Document]:
    root = Path(docs_dir or settings.DOCS_DIR)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Docs directory not found: {root}")

    supported = {".pdf", ".txt", ".md"}
    documents: list[Document] = []

    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in supported:
            continue

        if file_path.suffix.lower() == ".pdf":
            text = _load_pdf(file_path)
        else:
            text = _load_text(file_path)

        text = text.strip()
        if not text:
            continue

        logger.info("Loaded file %s (%d chars)", file_path.name, len(text))
        documents.append(Document(text=text, source=str(file_path)))

    return documents