"""
Document text extraction utilities for TreeRAG.
Supports PDF and Markdown inputs.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Optional


def extract_pdf_pages(pdf_path: str) -> list[str]:
    """
    Extract text from each page of a PDF.
    Returns a list where index i contains the text of page i+1 (1-based pages).
    """
    try:
        import pypdf
    except ImportError:
        raise ImportError("pypdf is required for PDF support. Install with: pip install pypdf")

    pages = []
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text.strip())
    return pages


def extract_pdf_toc(pdf_path: str) -> list[dict]:
    """
    Attempt to extract embedded table of contents from a PDF.
    Returns list of {title, page} dicts or empty list if none found.
    """
    try:
        import pypdf
    except ImportError:
        return []

    toc = []
    try:
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            outlines = reader.outline
            if outlines:
                _flatten_outlines(outlines, reader, toc)
    except Exception:
        pass
    return toc


def _flatten_outlines(outlines, reader, result: list, depth: int = 0):
    for item in outlines:
        if isinstance(item, list):
            _flatten_outlines(item, reader, result, depth + 1)
        else:
            try:
                page_num = reader.get_destination_page_number(item) + 1
                result.append({"title": item.title, "page": page_num, "depth": depth})
            except Exception:
                pass


def extract_markdown_sections(md_path: str) -> list[dict]:
    """
    Parse a Markdown file into sections based on heading hierarchy.
    Returns list of {title, level, start_line, content} dicts.
    """
    with open(md_path, encoding="utf-8") as f:
        lines = f.readlines()

    sections = []
    current_section = None

    for i, line in enumerate(lines):
        heading_match = re.match(r'^(#{1,6})\s+(.+)', line.rstrip())
        if heading_match:
            if current_section is not None:
                current_section["content"] = "".join(
                    lines[current_section["start_line"]:i]
                )
                sections.append(current_section)
            level = len(heading_match.group(1))
            current_section = {
                "title": heading_match.group(2).strip(),
                "level": level,
                "start_line": i,
                "content": "",
            }

    if current_section is not None:
        current_section["content"] = "".join(lines[current_section["start_line"]:])
        sections.append(current_section)

    return sections


def chunk_pages(pages: list[str], max_pages: int = 10) -> list[tuple[int, int, str]]:
    """
    Split page list into chunks of at most max_pages pages.
    Returns list of (start_page, end_page, combined_text) tuples (1-based pages).
    """
    chunks = []
    total = len(pages)
    i = 0
    while i < total:
        end = min(i + max_pages, total)
        chunk_text = "\n\n--- Page Break ---\n\n".join(
            f"[Page {i + j + 1}]\n{pages[i + j]}" for j in range(end - i)
        )
        chunks.append((i + 1, end, chunk_text))
        i = end
    return chunks


def pages_to_text(pages: list[str], start: int, end: int) -> str:
    """
    Combine pages[start-1 .. end-1] into a single text block.
    start and end are 1-based inclusive.
    """
    parts = []
    for p in range(start, end + 1):
        if 1 <= p <= len(pages):
            parts.append(f"[Page {p}]\n{pages[p - 1]}")
    return "\n\n--- Page Break ---\n\n".join(parts)
