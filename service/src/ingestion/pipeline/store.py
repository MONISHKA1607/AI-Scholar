import hashlib
import os
from datetime import datetime, timezone
from typing import List, Optional

import chromadb

from ...config import (
    CHROMA_HOST,
    CHROMA_PORT,
    CHROMA_COLLECTION,
    UPLOAD_DIR,
)
from ..types import Chunk, PaperRecord
from .embed import get_embedding_function

_client = None
_collection = None


def get_chroma_client():
    global _client
    if _client is None:
        _client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return _client


def get_collection():
    global _collection
    if _collection is None:
        _collection = get_chroma_client().get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=get_embedding_function(),
        )
    return _collection


def _metadata_for(record: PaperRecord, chunk: Chunk) -> dict:
    """Build Chroma metadata. Chroma rejects list/None values for some fields,
    so we coerce lists to ';' joined strings and drop None entries.
    """
    authors_str = "; ".join(record.authors or [])
    categories_str = ",".join(record.categories or [])
    meta = {
        "canonical_id": record.canonical_id or "",
        "title": record.title or "",
        "authors": authors_str,
        "source": record.source,
        "chunk_idx": chunk.chunk_idx,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    if record.doi:
        meta["doi"] = record.doi
    if record.arxiv_id:
        meta["arxiv_id"] = record.arxiv_id
    if record.s2_paper_id:
        meta["s2_paper_id"] = record.s2_paper_id
    if record.venue:
        meta["venue"] = record.venue
    if record.year is not None:
        meta["year"] = int(record.year)
    if categories_str:
        meta["categories"] = categories_str
    if record.url:
        meta["url"] = record.url
    elif record.pdf_url:
        meta["url"] = record.pdf_url
    if chunk.section_heading:
        meta["section_heading"] = chunk.section_heading
    if record.citation_count is not None:
        meta["citation_count"] = int(record.citation_count)
    if record.uploader_id:
        meta["uploader_id"] = record.uploader_id
    return meta


def store_chunks(record: PaperRecord, chunks: List[Chunk]) -> int:
    """Upsert chunks for a paper into Chroma. Returns number of chunks stored."""
    if not chunks:
        return 0
    if not record.canonical_id:
        raise ValueError("record.canonical_id must be set before storing")

    ids = [f"{record.canonical_id}#chunk_{c.chunk_idx}" for c in chunks]
    docs = [c.text for c in chunks]
    metas = [_metadata_for(record, c) for c in chunks]

    coll = get_collection()
    coll.upsert(ids=ids, documents=docs, metadatas=metas)
    return len(chunks)


def already_indexed(canonical_id: str) -> bool:
    """Cheap check: does Chroma already have any chunk for this canonical_id?"""
    try:
        res = get_collection().get(where={"canonical_id": canonical_id}, limit=1)
        return bool(res and res.get("ids"))
    except Exception:
        return False


def save_blob(user_id: Optional[str], filename: str, data: bytes) -> str:
    """Persist a user-uploaded PDF blob. Returns absolute path."""
    sub = user_id or "anon"
    digest = hashlib.sha256(data).hexdigest()
    ext = os.path.splitext(filename or "")[1].lower() or ".pdf"
    target_dir = os.path.join(UPLOAD_DIR, sub)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, f"{digest}{ext}")
    if not os.path.exists(target_path):
        with open(target_path, "wb") as f:
            f.write(data)
    return target_path
