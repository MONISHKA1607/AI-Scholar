"""Eval runner: scores kb_retrieval against a labeled benchmark.

Metrics:
    hit@k       — was the expected paper in the top-k retrieved chunks?
    MRR@10      — mean reciprocal rank of the first matching chunk (within top 10)
    p50_ms, p95_ms — latency percentiles

A retrieved chunk matches the expected paper when any of these are equal:
    metadata.canonical_id ends with expected_arxiv_id
    metadata.arxiv_id == expected_arxiv_id
    metadata.doi == expected_doi
"""
import json
import os
import time
from typing import List, Dict, Any, Optional

from ..rag.tools import retrieve_chunks
from ..ingestion.pipeline.dedupe import normalize_arxiv_id, normalize_doi

DEFAULT_BENCHMARK = os.path.join(os.path.dirname(__file__), "benchmark.jsonl")


def _load_benchmark(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _chunk_matches(chunk: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    md = chunk.get("metadata") or {}
    exp_arxiv = normalize_arxiv_id(expected.get("expected_arxiv_id"))
    exp_doi = normalize_doi(expected.get("expected_doi"))
    if exp_arxiv:
        if normalize_arxiv_id(md.get("arxiv_id")) == exp_arxiv:
            return True
        cid = (md.get("canonical_id") or "").lower()
        if cid.endswith(f"arxiv:{exp_arxiv}"):
            return True
    if exp_doi:
        if normalize_doi(md.get("doi")) == exp_doi:
            return True
        cid = (md.get("canonical_id") or "").lower()
        if cid.endswith(f"doi:{exp_doi}"):
            return True
    return False


def _percentile(sorted_vals: List[float], pct: float) -> Optional[float]:
    if not sorted_vals:
        return None
    k = int(round((pct / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[k]


def run_benchmark(
    benchmark_path: Optional[str] = None, *, top_k: int = 5
) -> Dict[str, Any]:
    """Run the benchmark; return per-query results and aggregate metrics."""
    path = benchmark_path or DEFAULT_BENCHMARK
    rows = _load_benchmark(path)
    if not rows:
        return {"error": "empty benchmark", "path": path}

    per_query: List[Dict[str, Any]] = []
    hit5_count = 0
    rr_sum = 0.0
    latencies: List[float] = []

    for row in rows:
        q = row.get("question")
        t0 = time.monotonic()
        chunks = retrieve_chunks(q) or []
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        latencies.append(elapsed_ms)

        rank = None
        for idx, c in enumerate(chunks[:10], start=1):
            if _chunk_matches(c, row):
                rank = idx
                break

        hit5 = bool(rank is not None and rank <= top_k)
        rr = 1.0 / rank if rank else 0.0
        if hit5:
            hit5_count += 1
        rr_sum += rr

        per_query.append(
            {
                "id": row.get("id"),
                "question": q,
                "rank": rank,
                "hit@5": hit5,
                "reciprocal_rank": rr,
                "latency_ms": round(elapsed_ms, 1),
                "retrieved": [
                    {
                        "title": (c.get("metadata") or {}).get("title"),
                        "arxiv_id": (c.get("metadata") or {}).get("arxiv_id"),
                        "doi": (c.get("metadata") or {}).get("doi"),
                    }
                    for c in chunks[:top_k]
                ],
            }
        )

    sorted_lat = sorted(latencies)
    summary = {
        "n": len(rows),
        f"hit@{top_k}": round(hit5_count / len(rows), 4),
        "MRR@10": round(rr_sum / len(rows), 4),
        "p50_ms": round(_percentile(sorted_lat, 50) or 0.0, 1),
        "p95_ms": round(_percentile(sorted_lat, 95) or 0.0, 1),
        "benchmark_path": path,
    }
    return {"summary": summary, "per_query": per_query}
