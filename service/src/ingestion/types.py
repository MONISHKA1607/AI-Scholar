from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, List, Dict, Any

PaperSource = Literal["arxiv", "s2", "upload"]


@dataclass
class PaperRecord:
    title: str
    authors: List[str]
    source: PaperSource
    abstract: Optional[str] = None
    year: Optional[int] = None
    categories: List[str] = field(default_factory=list)
    venue: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    s2_paper_id: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    pdf_path: Optional[str] = None
    citation_count: Optional[int] = None
    uploader_id: Optional[str] = None
    canonical_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PaperRecord":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})


@dataclass
class Chunk:
    text: str
    chunk_idx: int
    section_heading: Optional[str] = None
    page: Optional[int] = None
