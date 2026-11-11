"""Load a lease (PDF, TXT, MD) and split it into clauses."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from leaseguard.models import Clause

SUPPORTED = {".pdf", ".txt", ".md"}

# "1." "1)" "12.3" "Section 4" "Clause 7:" at the start of a line
CLAUSE_START = re.compile(r"^\s*(?:(?:section|clause|article)\s+)?(\d{1,3}(?:\.\d{1,2})?)[.):]?\s+(?=\S)", re.I)


@dataclass
class Page:
    number: int
    text: str


def load_pages(path: str | Path) -> list[Page]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type '{suffix}'. Use PDF, TXT or MD.")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [Page(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
        if not any(p.text.strip() for p in pages):
            raise ValueError("No text found in this PDF. It may be a scanned image; try a text PDF or a .txt file.")
        return pages
    return [Page(1, path.read_text(encoding="utf-8", errors="replace"))]


def pages_from_text(text: str) -> list[Page]:
    return [Page(1, text)]


def split_clauses(pages: list[Page], min_chars: int = 15) -> list[Clause]:
    """Split pages into clauses.

    1. Numbered clauses ("7. The tenant ...") start a new clause; following lines
       without a number are joined to it (PDF lines often wrap).
    2. If a page has no numbered lines, each paragraph (blank-line separated) is a clause.
    Header lines before the first numbered clause (names, address) are skipped:
    key terms are read from the whole text separately (extract.py).
    """
    clauses: list[Clause] = []
    for page in pages:
        lines = page.text.splitlines()
        numbered = [ln for ln in lines if CLAUSE_START.match(ln)]
        if numbered:
            current_num, buf = None, []
            for ln in lines:
                m = CLAUSE_START.match(ln)
                if m:
                    if current_num is not None:
                        clauses.append(Clause(number=current_num, text=_clean(" ".join(buf)), page=page.number))
                    current_num, buf = m.group(1), [ln[m.end():]]
                elif current_num is not None:
                    if ln.strip():
                        buf.append(ln.strip())
            if current_num is not None:
                clauses.append(Clause(number=current_num, text=_clean(" ".join(buf)), page=page.number))
        else:
            paras = [p for p in re.split(r"\n\s*\n", page.text) if p.strip()]
            for i, p in enumerate(paras, 1):
                clauses.append(Clause(number=f"p{page.number}-{i}", text=_clean(p), page=page.number))
    return [c for c in clauses if len(c.text) >= min_chars and not _is_boilerplate(c.text)]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_boilerplate(text: str) -> bool:
    return text.upper().startswith("SAMPLE LEASE FOR TESTING")
