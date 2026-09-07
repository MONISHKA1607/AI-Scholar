import hashlib
import re
from typing import Optional

from ..types import PaperRecord

_DOI_PREFIX = re.compile(r"^https?://(dx\.)?doi\.org/", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_ARXIV_VERSION = re.compile(r"v\d+$", re.IGNORECASE)


def normalize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    s = _DOI_PREFIX.sub("", doi.strip()).lower()
    return s or None


def normalize_arxiv_id(arxiv_id: Optional[str]) -> Optional[str]:
    if not arxiv_id:
        return None
    s = arxiv_id.strip().lower()
    s = s.replace("arxiv:", "")
    s = _ARXIV_VERSION.sub("", s)
    return s or None


def _normalize_title(title: str) -> str:
    s = title.lower()
    s = _NON_ALNUM.sub(" ", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s


def _first_author_surname(authors) -> str:
    if not authors:
        return ""
    first = authors[0] if isinstance(authors, list) else str(authors)
    parts = re.split(r"[,\s]+", first.strip())
    parts = [p for p in parts if p]
    if not parts:
        return ""
    # Heuristic: surname is the longest token, or last token for "First Last" order
    if len(parts) == 1:
        return parts[0].lower()
    return parts[-1].lower()


def title_author_hash(title: str, authors) -> str:
    base = f"{_normalize_title(title)}|{_first_author_surname(authors)}"
    return "ta_" + hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]


def compute_canonical_id(record: PaperRecord) -> str:
    """Resolve canonical_id from a PaperRecord via DOI -> arxiv_id -> s2_paper_id -> title hash."""
    doi = normalize_doi(record.doi)
    if doi:
        return f"doi:{doi}"
    arxiv = normalize_arxiv_id(record.arxiv_id)
    if arxiv:
        return f"arxiv:{arxiv}"
    if record.s2_paper_id:
        return f"s2:{record.s2_paper_id.strip()}"
    return title_author_hash(record.title or "", record.authors or [])
