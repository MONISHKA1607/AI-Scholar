"""FastAPI router for user paper ingestion.

POST /fastapi/ingest/upload — accepts a multipart PDF, saves the blob,
    enqueues an ingestion job, returns { job_id, canonical_id }.

GET  /fastapi/ingest/status/{job_id} — returns job status from RQ +
    ledger context.

GET  /fastapi/ingest/ledger — debug endpoint: most recent ledger rows.
"""
import os
import hashlib
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..pipeline.dedupe import compute_canonical_id
from ..pipeline.store import save_blob
from ..state import ledger
from ..types import PaperRecord


router = APIRouter(prefix="/fastapi/ingest", tags=["ingest"])


@router.post("/upload", status_code=202)
async def upload_paper(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    uploader_id: Optional[str] = Form(None),
):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=415, detail="Only application/pdf is accepted")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds 50MB limit")

    path = save_blob(uploader_id, file.filename or "upload.pdf", data)
    digest = hashlib.sha256(data).hexdigest()
    inferred_title = (
        title
        or os.path.splitext(os.path.basename(file.filename or ""))[0]
        or f"upload-{digest[:8]}"
    )

    record = PaperRecord(
        title=inferred_title,
        authors=[],
        source="upload",
        pdf_path=path,
        uploader_id=uploader_id,
    )
    record.canonical_id = compute_canonical_id(record)

    try:
        from ..workers.queue import enqueue_ingest

        job_id = enqueue_ingest(record.to_dict())
        ledger.upsert(
            record.canonical_id,
            status="queued",
            source="upload",
            title=record.title,
            job_id=job_id,
            uploader_id=uploader_id,
        )
        return {
            "job_id": job_id,
            "canonical_id": record.canonical_id,
            "status": "queued",
        }
    except Exception as e:
        # RQ unavailable — run synchronously as a fallback so the upload doesn't fail.
        from ..pipeline.runner import process_paper

        result = process_paper(record, pdf_bytes=data)
        return {
            "job_id": None,
            "canonical_id": record.canonical_id,
            "status": result.get("status", "failed"),
            "sync_fallback": True,
            "error": result.get("error"),
            "queue_error": str(e),
        }


@router.get("/status/{job_id}")
def status(job_id: str):
    try:
        from ..workers.queue import get_job_status

        rq_status = get_job_status(job_id)
    except Exception as e:
        rq_status = {"status": "unknown", "error": str(e)}

    led = ledger.by_job_id(job_id)
    return {"job_id": job_id, "rq": rq_status, "ledger": led}


@router.get("/ledger")
def recent_ledger(limit: int = 20):
    return {"counts": ledger.counts(), "recent": ledger.list_recent(limit=limit)}
