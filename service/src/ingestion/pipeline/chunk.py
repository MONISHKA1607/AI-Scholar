import re
from typing import List, Optional, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..types import Chunk
from .extract import ExtractedDocument

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

NUMBERED_HEADING = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+([A-Z][A-Za-z0-9 \-:()/&,]{2,80})\s*$", re.MULTILINE)
NAMED_HEADING_WORDS = (
    "abstract",
    "introduction",
    "background",
    "related work",
    "methods",
    "method",
    "methodology",
    "approach",
    "experiments",
    "experimental setup",
    "results",
    "evaluation",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
    "appendix",
    "acknowledgments",
    "acknowledgements",
)
NAMED_HEADING_PATTERN = re.compile(
    r"^\s*(" + "|".join(re.escape(w) for w in NAMED_HEADING_WORDS) + r")\s*:?\s*$",
    re.MULTILINE | re.IGNORECASE,
)

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _find_section_boundaries(text: str) -> List[Tuple[int, str]]:
    """Return [(char_offset, heading), ...] sorted by offset.

    A pseudo-boundary at offset 0 with heading None is prepended by callers.
    """
    boundaries: List[Tuple[int, str]] = []
    for m in NUMBERED_HEADING.finditer(text):
        boundaries.append((m.start(), m.group(0).strip()))
    for m in NAMED_HEADING_PATTERN.finditer(text):
        boundaries.append((m.start(), m.group(0).strip().rstrip(":")))
    boundaries.sort(key=lambda x: x[0])
    return boundaries


def section_aware_split(text: str) -> List[Tuple[Optional[str], str]]:
    """Split text by detected section headings.

    Returns a list of (heading_or_None, body) tuples in document order.
    Falls back to a single (None, text) if no headings detected.
    """
    text = text or ""
    if not text.strip():
        return []
    boundaries = _find_section_boundaries(text)
    if not boundaries:
        return [(None, text)]

    out: List[Tuple[Optional[str], str]] = []
    if boundaries[0][0] > 0:
        out.append((None, text[: boundaries[0][0]]))
    for idx, (start, heading) in enumerate(boundaries):
        end = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(text)
        body = text[start:end]
        out.append((heading, body))
    return out


def chunk_document(doc: ExtractedDocument) -> List[Chunk]:
    """Section-aware → recursive-character chunking. Returns Chunks with metadata."""
    full = doc.full_text
    sections = section_aware_split(full)
    chunks: List[Chunk] = []
    idx = 0
    for heading, body in sections:
        if not body.strip():
            continue
        for piece in _splitter.split_text(body):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(Chunk(text=piece, chunk_idx=idx, section_heading=heading))
            idx += 1
    return chunks


def chunk_text(text: str) -> List[Chunk]:
    """Chunk plain text (no extracted-doc metadata)."""
    if not text or not text.strip():
        return []
    sections = section_aware_split(text)
    chunks: List[Chunk] = []
    idx = 0
    for heading, body in sections:
        for piece in _splitter.split_text(body):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(Chunk(text=piece, chunk_idx=idx, section_heading=heading))
            idx += 1
    return chunks
