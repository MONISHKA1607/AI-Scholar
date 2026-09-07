import io
import os
from dataclasses import dataclass
from typing import List, Optional

import requests

try:
    import pypdf
    PdfReader = pypdf.PdfReader
except ImportError:
    import PyPDF2
    PdfReader = PyPDF2.PdfReader


DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AI-Scholar/1.0; +https://github.com/)",
    "Accept": "application/pdf,*/*",
}


@dataclass
class ExtractedPage:
    page: int
    text: str


@dataclass
class ExtractedDocument:
    pages: List[ExtractedPage]
    total_pages: int
    is_scanned: bool

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


def fetch_pdf_bytes(pdf_url: str, timeout: int = 30) -> bytes:
    resp = requests.get(pdf_url, headers=DEFAULT_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def extract_from_bytes(
    pdf_bytes: bytes, max_pages: Optional[int] = None
) -> ExtractedDocument:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)
    n = min(total_pages, max_pages) if max_pages else total_pages

    pages: List[ExtractedPage] = []
    chars_total = 0
    for i in range(n):
        try:
            text = reader.pages[i].extract_text() or ""
        except Exception:
            text = ""
        chars_total += len(text)
        pages.append(ExtractedPage(page=i, text=text))

    is_scanned = chars_total < 200 * max(n, 1) // 10  # < ~20 chars per page avg
    return ExtractedDocument(pages=pages, total_pages=total_pages, is_scanned=is_scanned)


def extract_from_path(path: str, max_pages: Optional[int] = None) -> ExtractedDocument:
    with open(path, "rb") as f:
        return extract_from_bytes(f.read(), max_pages=max_pages)


def extract_from_url(pdf_url: str, max_pages: Optional[int] = None) -> ExtractedDocument:
    data = fetch_pdf_bytes(pdf_url)
    return extract_from_bytes(data, max_pages=max_pages)
