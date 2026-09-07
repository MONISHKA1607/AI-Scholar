"""Ingestion CLI.

Usage:
    python -m src.ingestion.cli arxiv --bootstrap --kaggle PATH [--limit N] [--sync]
    python -m src.ingestion.cli arxiv --oai --since YYYY-MM-DD [--set SET] [--limit N] [--sync]
    python -m src.ingestion.cli status
    python -m src.ingestion.cli worker
    python -m src.ingestion.cli ingest-file PATH [--title T] [--source upload]

Run from the `service/` directory so `src.*` is on the path.
"""
import argparse
import json
import os
import sys
from typing import Iterable

from .pipeline.runner import process_paper
from .sources import arxiv as arxiv_source
from .sources import semantic_scholar as s2_source
from .state import ledger
from .types import PaperRecord


def _drain(records: Iterable[PaperRecord], *, sync: bool, limit: int) -> dict:
    counts = {"done": 0, "skipped_dup": 0, "skipped_ocr": 0, "failed": 0, "queued": 0}
    processed = 0
    for rec in records:
        if processed >= limit:
            break
        if sync:
            res = process_paper(rec)
            status = res.get("status", "failed")
            counts[status] = counts.get(status, 0) + 1
            print(
                f"[{processed + 1}/{limit}] {status:12s} {rec.canonical_id or rec.arxiv_id} :: {rec.title[:80]}"
            )
        else:
            from .workers.queue import enqueue_ingest

            jid = enqueue_ingest(rec.to_dict())
            counts["queued"] += 1
            print(f"[{processed + 1}/{limit}] queued      {jid} :: {rec.title[:80]}")
        processed += 1
    return counts


def _cmd_arxiv(args: argparse.Namespace) -> int:
    if args.bootstrap:
        if not args.kaggle or not os.path.exists(args.kaggle):
            print(f"error: --kaggle path does not exist: {args.kaggle}", file=sys.stderr)
            return 2
        records = arxiv_source.iter_kaggle_dump(
            args.kaggle,
            per_category_cap=args.per_category_cap,
            overall_cap=args.limit,
        )
        counts = _drain(records, sync=args.sync, limit=args.limit)
    elif args.oai:
        if not args.since:
            print("error: --since YYYY-MM-DD is required for --oai", file=sys.stderr)
            return 2
        records = arxiv_source.iter_oai_pmh(
            since=args.since,
            set_spec=args.oai_set,
            until=args.until,
            max_records=args.limit,
        )
        counts = _drain(records, sync=args.sync, limit=args.limit)
    else:
        print("error: pick one of --bootstrap or --oai", file=sys.stderr)
        return 2

    print("\nSummary:")
    print(json.dumps(counts, indent=2))
    return 0


def _cmd_s2(args: argparse.Namespace) -> int:
    if not args.query:
        print("error: --query is required", file=sys.stderr)
        return 2
    records = s2_source.search_bulk(
        args.query,
        max_records=args.limit,
        min_citations=args.min_citations,
    )
    counts = _drain(records, sync=args.sync, limit=args.limit)
    print("\nSummary:")
    print(json.dumps(counts, indent=2))
    return 0


def _cmd_status(_: argparse.Namespace) -> int:
    print(json.dumps(ledger.counts(), indent=2))
    return 0


def _cmd_worker(_: argparse.Namespace) -> int:
    from rq import Worker
    from .workers.queue import get_queue, get_redis

    q = get_queue()
    Worker([q], connection=get_redis()).work(with_scheduler=False)
    return 0


def _cmd_ingest_file(args: argparse.Namespace) -> int:
    if not os.path.exists(args.path):
        print(f"error: file not found: {args.path}", file=sys.stderr)
        return 2
    with open(args.path, "rb") as f:
        data = f.read()
    rec = PaperRecord(
        title=args.title or os.path.basename(args.path),
        authors=[],
        source="upload",
    )
    res = process_paper(rec, pdf_bytes=data)
    print(json.dumps(res, indent=2))
    return 0 if res.get("status") in ("done", "skipped_dup") else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ingestion")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("arxiv", help="ingest from arXiv (Kaggle dump or OAI-PMH)")
    a.add_argument("--bootstrap", action="store_true", help="bootstrap from Kaggle JSONL dump")
    a.add_argument("--oai", action="store_true", help="delta harvest via OAI-PMH")
    a.add_argument("--kaggle", help="path to Kaggle arXiv-metadata-oai-snapshot.json[.gz]")
    a.add_argument("--since", help="YYYY-MM-DD (OAI-PMH `from`)")
    a.add_argument("--until", help="YYYY-MM-DD (OAI-PMH `until`)")
    a.add_argument("--oai-set", default="physics:physics", help="OAI-PMH setSpec")
    a.add_argument("--limit", type=int, default=25000, help="max papers to enqueue/process")
    a.add_argument("--per-category-cap", type=int, default=5000)
    a.add_argument("--sync", action="store_true", help="run inline, do not enqueue to RQ")
    a.set_defaults(func=_cmd_arxiv)

    s2 = sub.add_parser("s2", help="ingest from Semantic Scholar bulk search")
    s2.add_argument("--query", required=False, help="search query (broad topic)")
    s2.add_argument("--limit", type=int, default=5000)
    s2.add_argument("--min-citations", type=int, default=5)
    s2.add_argument("--sync", action="store_true")
    s2.set_defaults(func=_cmd_s2)

    s = sub.add_parser("status", help="print ledger status counts")
    s.set_defaults(func=_cmd_status)

    w = sub.add_parser("worker", help="run an RQ worker against the ingestion queue")
    w.set_defaults(func=_cmd_worker)

    f = sub.add_parser("ingest-file", help="ingest a single local PDF (smoke test)")
    f.add_argument("path")
    f.add_argument("--title")
    f.set_defaults(func=_cmd_ingest_file)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
