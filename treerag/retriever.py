"""
TreeRetriever: Reasoning-based retrieval over the tree index.

Implements agentic tree search where an LLM:
1. Examines the top-level index nodes
2. Reasons which branches are relevant to the query
3. Drills down into chosen branches
4. Returns the most relevant page ranges

No vectors, no embeddings — pure LLM reasoning.
"""

from __future__ import annotations
import logging
from typing import Optional

from .models import TreeNode, TreeIndex
from .client import OpenAIClient
from .extractor import extract_pdf_pages, pages_to_text
import time
import hashlib
logger = logging.getLogger("treerag")


# ── Prompts ──────────────────────────────────────────────────────────────────

TREE_SEARCH_PROMPT = """You are a document retrieval expert. Your task is to identify which sections of a document are most relevant to answer a query.

Document: "{doc_title}"
{doc_description}

Available sections (examine their titles and summaries):
{sections_outline}

Instructions:
- Select the sections most likely to contain information relevant to the query
- You may select 1-5 sections
- Prefer more specific/leaf sections over broad parent sections when possible
- Think step by step: what does the query ask for? Which sections cover that topic?

Query: {query}

Return ONLY valid JSON:
{{
  "selected_node_ids": ["0003", "0007"]
}}
"""

DRILL_DOWN_PROMPT = """You are a document retrieval expert. You're navigating a document tree to find sections relevant to a query.

Query: {query}

You are currently examining the node: "{parent_title}" (Pages {start_page}–{end_page})
This node has the following subsections:

{children_outline}

Instructions:
- Determine which subsections are most relevant to the query
- You may select 1-3 subsections
- If none seem relevant, select the one most likely to contain anything useful

Return ONLY valid JSON:
{{
  "reasoning": "Why these subsections were selected",
  "selected_node_ids": ["0005", "0006"]
}}
"""

EXTRACT_ANSWER_PROMPT = """You are a precise document analyst. Based on the following document excerpts, answer the query.

Query: {query}

Retrieved document sections:
{sections_text}

Instructions:
- Answer the query using ONLY the information in the provided sections
- Be specific and cite page numbers where relevant (e.g., "According to page 5...")
- If the sections don't contain enough information, say so clearly
- Be concise but complete

Answer:
"""


class RetrievalResult:
    """Result of a tree search retrieval."""
    def __init__(
        self,
        nodes: list[TreeNode],
        reasoning: str,
        page_ranges: list[tuple[int, int]],
    ):
        self.nodes = nodes
        self.reasoning = reasoning
        self.page_ranges = page_ranges

    def __repr__(self):
        ranges = ", ".join(f"pp.{s}-{e}" for s, e in self.page_ranges)
        return f"RetrievalResult(nodes={len(self.nodes)}, pages=[{ranges}])"


class TreeRetriever:
    """
    Performs reasoning-based retrieval over a TreeIndex.
    
    Usage:
        retriever = TreeRetriever(api_key="sk-...", model="gpt-4o")
        result = retriever.search(tree, query="What is the revenue growth?")
        print(result.reasoning)
        for node in result.nodes:
            print(node.title, node.start_page, node.end_page)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_depth: int = 3,
        top_k_nodes: int = 3,
    ):
        self.client = OpenAIClient(api_key=api_key, model=model)
        self.max_depth = max_depth
        self.top_k_nodes = top_k_nodes

    def search(self, tree: TreeIndex, query: str) -> RetrievalResult:
        """
        Perform agentic tree search to find relevant sections.
        Returns a RetrievalResult with the most relevant TreeNodes.
        """
        logger.info(f"Searching tree for: {query!r}")
        selected_nodes = self._tree_search(tree, query)
        page_ranges = list({(n.start_page, n.end_page) for n in selected_nodes})
        page_ranges.sort()

        return RetrievalResult(
            nodes=selected_nodes,
            reasoning="Reasoning-based tree search completed.",
            page_ranges=page_ranges,
        )

    def search_and_extract(
        self, tree: TreeIndex, query: str, pdf_path: str
    ) -> tuple[RetrievalResult, str]:
        """
        Search for relevant sections AND extract their text from the PDF.
        Returns (RetrievalResult, extracted_text).
        """
        result = self.search(tree, query)
        pages = extract_pdf_pages(pdf_path)
        extracted_parts = []
        for node in result.nodes:
            text = pages_to_text(pages, node.start_page, node.end_page)
            extracted_parts.append(f"### {node.title} (Pages {node.start_page}–{node.end_page})\n\n{text}")
        extracted_text = "\n\n---\n\n".join(extracted_parts)
        return result, extracted_text

    def answer(
        self, tree: TreeIndex, query: str, pdf_path: str, max_context_chars: int = 150000
    ) -> dict:
        """
        Full RAG pipeline: search → extract → generate answer.
        Returns dict with keys: answer, nodes_used, reasoning, page_ranges.
        """
        result, sections_text = self.search_and_extract(tree, query, pdf_path)
        # Truncate context if needed
        sections_text = sections_text[:max_context_chars]

        prompt = EXTRACT_ANSWER_PROMPT.format(
            query=query,
            sections_text=sections_text,
        )
        answer_text = self.client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
        )

        return {
            "answer": answer_text.strip(),
            "nodes_used": [n.title for n in result.nodes],
            "page_ranges": result.page_ranges,
            "reasoning": result.reasoning,
        }

    # ── Internal tree search ───────────────────────────────────────────────────

    def _tree_search(self, tree: TreeIndex, query: str) -> list[TreeNode]:
        """Multi-level tree search using LLM reasoning at each level."""
        # First: search at root level
        selected = self._select_from_level(
            nodes=tree.roots,
            query=query,
            doc_title=tree.title,
            doc_description=tree.description,
        )

        # Expand: for each selected node, optionally drill deeper
        final_nodes = []
        for node in selected:
            if node.children and self.max_depth > 1:
                deeper = self._drill_down(node, query, depth=1)
                final_nodes.extend(deeper)
            else:
                final_nodes.append(node)

        # Deduplicate and limit
        seen = set()
        unique = []
        for n in final_nodes:
            if n.node_id not in seen:
                seen.add(n.node_id)
                unique.append(n)

        return unique[:self.top_k_nodes * 2]  # Allow some extra before final answer

    def _select_from_level(
        self,
        nodes: list[TreeNode],
        query: str,
        doc_title: str,
        doc_description: str,
    ) -> list[TreeNode]:
        """LLM selects from a flat list of sibling nodes."""
        outline = self._nodes_to_outline(nodes)
        prompt = TREE_SEARCH_PROMPT.format(
            query=query,
            doc_title=doc_title,
            doc_description=doc_description,
            sections_outline=outline,
        )
        static_content = f"{doc_title}{doc_description}{outline}"
        cache_key = hashlib.sha256(static_content.encode()).hexdigest()
        try:
            start_time = time.perf_counter()
            result = self.client.chat_json(
                [{"role": "user", "content": prompt}], max_tokens=512, cache_key= cache_key
            )
            logger.info(f"Node Selection time: {time.perf_counter() - start_time}")
            selected_ids = set(result.get("selected_node_ids", []))
            logger.info(f"Selected nodes: {selected_ids}")
            # logger.info(f"  Reasoning: {result.get('reasoning', '')}")

            selected = [n for n in nodes if n.node_id in selected_ids]
            if not selected:
                # Fallback: select first node
                selected = nodes[:1]
            return selected[:self.top_k_nodes]
        except Exception as e:
            logger.warning(f"Node selection failed: {e}. Using first nodes.")
            return nodes[:self.top_k_nodes]

    def _drill_down(self, node: TreeNode, query: str, depth: int) -> list[TreeNode]:
        """Recursively drill into children nodes if they exist."""
        if not node.children or depth >= self.max_depth:
            return [node]

        outline = self._nodes_to_outline(node.children)
        prompt = DRILL_DOWN_PROMPT.format(
            query=query,
            parent_title=node.title,
            start_page=node.start_page,
            end_page=node.end_page,
            children_outline=outline,
        )
        try:
            result = self.client.chat_json(
                [{"role": "user", "content": prompt}], max_tokens=512
            )
            selected_ids = set(result.get("selected_node_ids", []))
            selected_children = [c for c in node.children if c.node_id in selected_ids]
            if not selected_children:
                selected_children = node.children[:1]
        except Exception:
            selected_children = node.children[:2]

        # Recurse
        result_nodes = []
        for child in selected_children:
            result_nodes.extend(self._drill_down(child, query, depth + 1))
        return result_nodes

    def _nodes_to_outline(self, nodes: list[TreeNode]) -> str:
        lines = []
        for n in nodes: # TODO increase the summary text length, must end with punctuation
            summary_preview = f" — {n.summary[:100]}..." if n.summary else ""
            lines.append(f"[{n.node_id}] {n.title} (pp. {n.start_page}–{n.end_page}){summary_preview}")
        return "\n".join(lines)
