from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import settings
from ingestion.chunker import chunk_documents_hierarchical
from ingestion.embedder import upsert_local_chunks
from ingestion.loader import load_documents
from ingestion.preprocessor import preprocess_documents
from retrieval.parent_store import save_parent_chunks
from retrieval.vector_store import get_client


COLLECTION_NAME = "case_studies"


def reset_local_index() -> None:
    client = get_client()

    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted existing Chroma collection '{COLLECTION_NAME}'.")
    except Exception:
        print(f"No existing Chroma collection '{COLLECTION_NAME}' to delete.")

    parent_store_path = Path(settings.PARENT_STORE_PATH)
    if parent_store_path.exists():
        parent_store_path.unlink()
        print(f"Deleted parent store file: {parent_store_path}")


def run_ingestion(docs_dir: str | None) -> dict:
    docs = load_documents(docs_dir)
    preprocessed = preprocess_documents(docs)

    parents, child_chunks = chunk_documents_hierarchical(
        preprocessed,
        parent_chunk_size=settings.PARENT_CHUNK_SIZE,
        parent_overlap=settings.PARENT_CHUNK_OVERLAP,
        child_chunk_size=settings.CHILD_CHUNK_SIZE,
        child_overlap=settings.CHILD_CHUNK_OVERLAP,
    )

    parent_saved = save_parent_chunks(parents, replace_existing_sources=True)
    inserted = upsert_local_chunks(child_chunks, replace_existing_sources=True)
    sources = sorted({doc.source for doc in preprocessed})

    return {
        "docs_loaded": len(docs),
        "docs_preprocessed": len(preprocessed),
        "child_chunks_total": len(child_chunks),
        "chunks_ingested": inserted,
        "parent_chunks_saved": parent_saved,
        "sources": sources,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or update the local RAG index from DOCS_DIR or a provided path."
    )
    parser.add_argument(
        "--docs-dir",
        default=None,
        help="Optional documents directory. If omitted, uses DOCS_DIR from backend/.env.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing Chroma collection and parent store before ingesting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.rebuild:
        reset_local_index()

    result = run_ingestion(args.docs_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()