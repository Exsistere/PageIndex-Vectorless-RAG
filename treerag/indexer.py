"""
TreeIndexer: Builds a hierarchical tree index from documents using LLM reasoning.

This is the core of TreeRAG - equivalent to PageIndex's indexing step.
The indexer:
1. Extracts text from each page
2. Optionally detects existing TOC from PDF metadata or first pages
3. Uses LLM to create a semantic tree structure with summaries
4. Splits large sections recursively until sections are small enough
"""

from __future__ import annotations
import json
import logging
from typing import Optional

from .models import TreeNode, TreeIndex
from .client import OpenAIClient
from .extractor import (
    extract_pdf_pages,
    extract_pdf_toc,
    extract_markdown_sections,
    pages_to_text,
)

logger = logging.getLogger("treerag")


# ── Prompts ────────────────────────────────────────────────────────────────────

DETECT_TOC_PROMPT = """You are a document analysis expert. The following text is from the first pages of a document.

Your task: Determine if there is a Table of Contents (TOC) present in this text.
If yes, extract the TOC entries. If no, return an empty list.

Return ONLY valid JSON in this exact format:
{{
  "has_toc": true,
  "entries": [
    {{"title": "Section Title", "page": 5}},
    ...
  ]
}}

If no TOC found:
{{"has_toc": false, "entries": []}}

Document text:
{text}
"""

BUILD_TREE_FROM_TOC_PROMPT = """You are a document analysis expert. Build a hierarchical tree index from this Table of Contents.

TOC entries (title → page number):
{toc_text}

Total pages: {total_pages}

Rules:
- Infer parent-child relationships from numbering (1, 1.1, 1.2, 2, ...) or indentation
- Each node must have a start_page and end_page (end_page = next sibling's start_page - 1, or total_pages for last node)
- Assign node_ids as 4-digit zero-padded numbers (0001, 0002, ...) in depth-first order
- Include a brief summary field (leave empty string "" for now, it will be filled later)

Return ONLY valid JSON:
{{
  "title": "Document Title",
  "description": "Brief one-sentence description of what this document covers",
  "roots": [
    {{
      "node_id": "0001",
      "title": "Chapter 1: Introduction",
      "start_page": 1,
      "end_page": 10,
      "summary": "",
      "children": [
        {{
          "node_id": "0002",
          "title": "1.1 Background",
          "start_page": 1,
          "end_page": 5,
          "summary": "",
          "children": []
        }}
      ]
    }}
  ]
}}
"""

SEGMENT_PAGES_PROMPT = """You are a document analysis expert. Analyze the following document pages and identify the major sections.

Pages {start_page} to {end_page} of a {total_pages}-page document.

Your task:
1. Identify all distinct sections/chapters within these pages
2. For each section, determine its exact start and end pages
3. Give each section a descriptive title

Return ONLY valid JSON:
{{
  "doc_title": "Overall document title (if visible)",
  "doc_description": "Brief description of the document",
  "sections": [
    {{
      "title": "Section Title",
      "start_page": {start_page},
      "end_page": 5
    }},
    ...
  ]
}}

If the pages form a single coherent section with no sub-divisions, return one section spanning all pages.

Document text:
{text}
"""

SUMMARIZE_NODE_PROMPT = """You are a document analysis expert. Write a concise summary of the following document section.

Section: "{title}" (Pages {start_page}–{end_page})

The summary should:
- Be 1-3 sentences
- Capture the key topics, findings, or information in this section
- Be useful for deciding if this section is relevant to a query

Return ONLY the summary text, no JSON, no preamble.

Section text:
{text}
"""

SPLIT_NODE_PROMPT = """You are a document analysis expert. The following section is too long and needs to be split into subsections.

Section: "{title}" (Pages {start_page}–{end_page}, {page_count} pages)

Identify 2-6 logical subsections within this section. Each subsection should be a coherent unit.

Return ONLY valid JSON:
{{
  "subsections": [
    {{
      "title": "Subsection Title",
      "start_page": {start_page},
      "end_page": 12
    }},
    ...
  ]
}}

The subsections must:
- Cover all pages from {start_page} to {end_page} without gaps or overlaps
- Have meaningful titles describing their content

Document text:
{text}
"""


class TreeIndexer:
    """
    Builds a hierarchical tree index from a PDF or Markdown document.
    
    Usage:
        indexer = TreeIndexer(api_key="sk-...", model="gpt-4o")
        tree = indexer.index_pdf("report.pdf")
        tree.save("report_index.json")
        print(tree.to_outline())
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_pages_per_node: int = 10,
        toc_check_pages: int = 20,
        add_summaries: bool = True,
        verbose: bool = False,
    ):
        self.client = OpenAIClient(api_key=api_key, model=model)
        self.max_pages_per_node = max_pages_per_node
        self.toc_check_pages = toc_check_pages
        self.add_summaries = add_summaries

        if verbose:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # ── Public API ─────────────────────────────────────────────────────────────

    def index_pdf(self, pdf_path: str) -> TreeIndex:
        """Build a tree index from a PDF file."""
        logger.info(f"Extracting text from {pdf_path}...")
        pages = extract_pdf_pages(pdf_path)
        total_pages = len(pages)
        logger.info(f"Extracted {total_pages} pages.")

        # Try embedded PDF TOC first
        pdf_toc = extract_pdf_toc(pdf_path)
        if pdf_toc and len(pdf_toc) >= 3:
            logger.info(f"Found embedded PDF TOC with {len(pdf_toc)} entries.")
            tree = self._build_from_toc(pdf_toc, total_pages, pdf_path)
        else:
            # Try to detect TOC from first pages using LLM
            logger.info("Checking first pages for table of contents...")
            check_pages = min(self.toc_check_pages, total_pages)
            toc_text = pages_to_text(pages, 1, check_pages)
            detected = self._detect_toc(toc_text)

            if detected and len(detected) >= 3:
                logger.info(f"Detected TOC with {len(detected)} entries via LLM.")
                tree = self._build_from_toc(detected, total_pages, pdf_path)
            else:
                logger.info("No TOC found. Segmenting document via LLM...")
                tree = self._build_by_segmentation(pages, pdf_path)

        # Fill summaries
        if self.add_summaries:
            logger.info("Generating node summaries...")
            self._fill_summaries(tree, pages)

        logger.info("Tree index complete.")
        return tree

    def index_markdown(self, md_path: str) -> TreeIndex:
        """Build a tree index from a Markdown file."""
        logger.info(f"Parsing markdown sections from {md_path}...")
        sections = extract_markdown_sections(md_path)

        # Convert markdown sections to a pseudo-page tree
        # Each section = 1 "page"
        roots, node_counter = self._sections_to_tree(sections)

        # Build description from first section or filename
        import os
        title = os.path.splitext(os.path.basename(md_path))[0]

        tree = TreeIndex(
            title=title,
            description=f"Markdown document with {len(sections)} sections",
            total_pages=len(sections),
            source=md_path,
            roots=roots,
        )

        if self.add_summaries:
            # For markdown, use section content directly
            self._fill_md_summaries(tree, sections)

        return tree

    # ── TOC-based building ─────────────────────────────────────────────────────

    def _detect_toc(self, text: str) -> list[dict]:
        prompt = DETECT_TOC_PROMPT.format(text=text[:8000])
        try:
            result = self.client.chat_json([{"role": "user", "content": prompt}])
            if result.get("has_toc") and result.get("entries"):
                return result["entries"]
        except Exception as e:
            logger.warning(f"TOC detection failed: {e}")
        return []

    def _build_from_toc(self, toc: list[dict], total_pages: int, source: str) -> TreeIndex:
        toc_text = "\n".join(
            f"- {e['title']} → page {e['page']}" for e in toc
        )
        prompt = BUILD_TREE_FROM_TOC_PROMPT.format(
            toc_text=toc_text, total_pages=total_pages
        )
        try:
            result = self.client.chat_json(
                [{"role": "user", "content": prompt}], max_tokens=8192
            )
            roots = [TreeNode.from_dict(r) for r in result.get("roots", [])]
            tree = TreeIndex(
                title=result.get("title", "Document"),
                description=result.get("description", ""),
                total_pages=total_pages,
                source=source,
                roots=roots,
            )
            # Renumber node IDs
            self._renumber_nodes(tree)
            return tree
        except Exception as e:
            logger.warning(f"TOC tree build failed: {e}. Falling back to segmentation.")
            pages = extract_pdf_pages(source)
            return self._build_by_segmentation(pages, source)

    # ── Segmentation-based building ────────────────────────────────────────────

    def _build_by_segmentation(self, pages: list[str], source: str) -> TreeIndex:
        total_pages = len(pages)
        # Segment the whole document into top-level sections
        text = pages_to_text(pages, 1, min(total_pages, 60))  # Use first 60 pages for overview
        prompt = SEGMENT_PAGES_PROMPT.format(
            start_page=1,
            end_page=total_pages,
            total_pages=total_pages,
            text=text[:12000],
        )
        try:
            result = self.client.chat_json(
                [{"role": "user", "content": prompt}], max_tokens=4096
            )
            sections = result.get("sections", [])
            doc_title = result.get("doc_title", "Document")
            doc_desc = result.get("doc_description", "")
        except Exception as e:
            logger.warning(f"Segmentation failed: {e}. Creating flat structure.")
            # Fall back: one section per max_pages_per_node pages
            sections = []
            for start in range(1, total_pages + 1, self.max_pages_per_node):
                end = min(start + self.max_pages_per_node - 1, total_pages)
                sections.append({"title": f"Pages {start}–{end}", "start_page": start, "end_page": end})
            doc_title = "Document"
            doc_desc = ""

        node_counter = [1]
        roots = []
        for sec in sections:
            node = self._make_node(sec, pages, node_counter)
            roots.append(node)

        return TreeIndex(
            title=doc_title,
            description=doc_desc,
            total_pages=total_pages,
            source=source,
            roots=roots,
        )

    def _make_node(self, sec: dict, pages: list[str], counter: list[int]) -> TreeNode:
        node_id = f"{counter[0]:04d}"
        counter[0] += 1
        start, end = sec["start_page"], sec["end_page"]
        page_count = end - start + 1

        node = TreeNode(
            node_id=node_id,
            title=sec["title"],
            start_page=start,
            end_page=end,
        )

        # If section is too large, split it recursively
        if page_count > self.max_pages_per_node:
            children = self._split_node(node, pages, counter)
            node.children = children

        return node

    def _split_node(
        self, node: TreeNode, pages: list[str], counter: list[int]
    ) -> list[TreeNode]:
        text = pages_to_text(pages, node.start_page, node.end_page)
        prompt = SPLIT_NODE_PROMPT.format(
            title=node.title,
            start_page=node.start_page,
            end_page=node.end_page,
            page_count=node.page_count(),
            text=text[:12000],
        )
        try:
            result = self.client.chat_json(
                [{"role": "user", "content": prompt}], max_tokens=2048
            )
            subsections = result.get("subsections", [])
            if len(subsections) < 2:
                return []
        except Exception as e:
            logger.warning(f"Node splitting failed for '{node.title}': {e}")
            # Fall back: split evenly
            mid = (node.start_page + node.end_page) // 2
            subsections = [
                {"title": f"{node.title} (Part 1)", "start_page": node.start_page, "end_page": mid},
                {"title": f"{node.title} (Part 2)", "start_page": mid + 1, "end_page": node.end_page},
            ]

        children = []
        for sub in subsections:
            child = self._make_node(sub, pages, counter)
            children.append(child)
        return children

    # ── Summaries ──────────────────────────────────────────────────────────────

    def _fill_summaries(self, tree: TreeIndex, pages: list[str]) -> None:
        for node in tree.all_nodes():
            if not node.summary:
                node.summary = self._summarize_node(node, pages)
                logger.info(f"  Summarized [{node.node_id}] {node.title}")

    def _summarize_node(self, node: TreeNode, pages: list[str]) -> str:
        # Only summarize leaf nodes or small nodes to save tokens
        text = pages_to_text(pages, node.start_page, node.end_page)
        prompt = SUMMARIZE_NODE_PROMPT.format(
            title=node.title,
            start_page=node.start_page,
            end_page=node.end_page,
            text=text[:6000],
        )
        try:
            return self.client.chat(
                [{"role": "user", "content": prompt}], max_tokens=256
            ).strip()
        except Exception as e:
            logger.warning(f"Summary failed for '{node.title}': {e}")
            return ""

    # ── Markdown helpers ───────────────────────────────────────────────────────

    def _sections_to_tree(self, sections: list[dict]) -> tuple[list[TreeNode], list[int]]:
        """Convert flat markdown section list into a nested tree."""
        roots = []
        stack: list[TreeNode] = []  # stack of (level, node) pairs
        level_stack: list[int] = []
        counter = [1]

        for i, sec in enumerate(sections):
            node = TreeNode(
                node_id=f"{counter[0]:04d}",
                title=sec["title"],
                start_page=i + 1,
                end_page=i + 1,
            )
            counter[0] += 1
            level = sec["level"]

            # Pop stack until we find parent
            while level_stack and level_stack[-1] >= level:
                stack.pop()
                level_stack.pop()

            if stack:
                stack[-1].children.append(node)
            else:
                roots.append(node)

            stack.append(node)
            level_stack.append(level)

        return roots, counter

    def _fill_md_summaries(self, tree: TreeIndex, sections: list[dict]) -> None:
        for node in tree.all_nodes():
            idx = node.start_page - 1
            if 0 <= idx < len(sections):
                content = sections[idx].get("content", "")[:3000]
                try:
                    prompt = SUMMARIZE_NODE_PROMPT.format(
                        title=node.title,
                        start_page=node.start_page,
                        end_page=node.end_page,
                        text=content,
                    )
                    node.summary = self.client.chat(
                        [{"role": "user", "content": prompt}], max_tokens=256
                    ).strip()
                except Exception:
                    node.summary = content[:200]

    # ── Utility ───────────────────────────────────────────────────────────────

    def _renumber_nodes(self, tree: TreeIndex) -> None:
        counter = [1]
        def renumber(node: TreeNode):
            node.node_id = f"{counter[0]:04d}"
            counter[0] += 1
            for child in node.children:
                renumber(child)
        for root in tree.roots:
            renumber(root)
