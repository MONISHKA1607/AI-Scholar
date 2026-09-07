"""Orchestrates a single paper through extract → chunk → dedupe → store,
updating the ledger as it progresses. Idempotent: re-running with the same
canonical_id will short-circuit if already 'done'.
"""
import os
from typing import Optional, Dict, Any

from ..state import ledger
from ..types import PaperRecord
from . import extract, chunk as chunker
from .dedupe import compute_canonical_id
from .store import store_chunks, already_indexed, save_blob

DEFAULT_MAX_PAGES = 12  # abstract + intro + first ~3 sections of a typical paper


def process_paper(
    record: PaperRecord,
    *,
    pdf_bytes: Optional[bytes] = None,
    max_pages: int = DEFAULT_MAX_PAGES,
    job_id: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Run the full ingestion pipeline for one paper.

    Sources may either set record.pdf_path or record.pdf_url (or pass pdf_bytes).
    Returns a dict with status, canonical_id, chunks, error.
    """
    if not record.canonical_id:
        record.canonical_id = compute_canonical_id(record)

    cid = record.canonical_id
    if not force:
        if ledger.is_done(cid) or already_indexed(cid):
            ledger.upsert(
                cid,
                status="done",
                source=record.source,
                title=record.title,
                job_id=job_id,
                uploader_id=record.uploader_id,
            )
            return {"status": "skipped_dup", "canonical_id": cid, "chunks": 0}

    ledger.upsert(
        cid,
        status="processing",
        source=record.source,
        title=record.title,
        job_id=job_id,
        uploader_id=record.uploader_id,
    )

    try:
        if pdf_bytes is not None:
            doc = extract.extract_from_bytes(pdf_bytes, max_pages=max_pages)
        elif record.pdf_path and os.path.exists(record.pdf_path):
            doc = extract.extract_from_path(record.pdf_path, max_pages=max_pages)
        elif record.pdf_url:
            doc = extract.extract_from_url(record.pdf_url, max_pages=max_pages)
        else:
            # No PDF available — index abstract + title metadata only.
            abstract = (record.abstract or "").strip()
            if not abstract:
                ledger.upsert(cid, status="failed", error="no pdf, no abstract")
                return {"status": "failed", "canonical_id": cid, "error": "no pdf, no abstract"}
            chunks = chunker.chunk_text(abstract)
            n = store_chunks(record, chunks)
            ledger.upsert(cid, status="done", chunks=n)
            return {"status": "done", "canonical_id": cid, "chunks": n}

        if doc.is_scanned:
            ledger.upsert(cid, status="skipped_ocr", error="scanned PDF (OCR not supported in v1)")
            return {"status": "skipped_ocr", "canonical_id": cid, "chunks": 0}

        # Prepend abstract+title so retrieval can match high-level queries even if
        # body text doesn't contain the keywords.
        full_text = doc.full_text
        if record.abstract and record.abstract.strip() not in full_text:
            full_text = f"{record.title}\n\n{record.abstract}\n\n{full_text}"

        chunks = chunker.chunk_text(full_text)
        if not chunks:
            ledger.upsert(cid, status="failed", error="no chunks produced")
            return {"status": "failed", "canonical_id": cid, "error": "no chunks produced"}

        n = store_chunks(record, chunks)
        ledger.upsert(cid, status="done", chunks=n)
        return {"status": "done", "canonical_id": cid, "chunks": n}

    except Exception as e:
        ledger.upsert(cid, status="failed", error=str(e)[:500])
        return {"status": "failed", "canonical_id": cid, "error": str(e)}


__all__ = ["process_paper", "save_blob"]
