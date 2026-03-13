"""
TreeRAG: High-level interface combining TreeIndexer and TreeRetriever.

Provides a simple one-stop API for vectorless, reasoning-based RAG.
"""

from __future__ import annotations
import os
import logging
import time
from typing import Optional

from .models import TreeIndex
from .indexer import TreeIndexer
from .retriever import TreeRetriever

logger = logging.getLogger("treerag")


class TreeRAG:
    """
    High-level interface for vectorless, reasoning-based RAG over documents.

    Example:
        rag = TreeRAG(api_key="sk-...", model="gpt-4o")

        # One-time indexing (save for reuse)
        rag.index("annual_report.pdf", index_path="annual_report.json")

        # Query
        result = rag.query("What was the total revenue in 2023?")
        print(result["answer"])
        print("Found in:", result["page_ranges"])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_pages_per_node: int = 10,
        toc_check_pages: int = 20,
        add_summaries: bool = True,
        max_search_depth: int = 3,
        top_k_nodes: int = 3,
        verbose: bool = False,
    ):
        if verbose:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

        self.indexer = TreeIndexer(
            api_key=api_key,
            model=model,
            max_pages_per_node=max_pages_per_node,
            toc_check_pages=toc_check_pages,
            add_summaries=add_summaries,
            verbose=verbose,
        )
        self.retriever = TreeRetriever(
            api_key=api_key,
            model=model,
            max_depth=max_search_depth,
            top_k_nodes=top_k_nodes,
        )

        self._tree: Optional[TreeIndex] = None
        self._doc_path: Optional[str] = None

    # ── Index management ───────────────────────────────────────────────────────

    def index(
        self,
        doc_path: str,
        index_path: Optional[str] = None,
        force_rebuild: bool = False,
    ) -> TreeIndex:
        """
        Build (or load) a tree index for a document.

        Args:
            doc_path: Path to PDF or Markdown file.
            index_path: Where to save/load the JSON index. Auto-derived if None.
            force_rebuild: Rebuild even if index file already exists.

        Returns:
            The TreeIndex for the document.
        """
        # Derive default index path
        if index_path is None:
            base = os.path.splitext(doc_path)[0]
            index_path = base + "_treeindex.json"

        # Load existing index if available
        if os.path.exists(index_path) and not force_rebuild:
            logger.info(f"Loading existing index from {index_path}")
            self._tree = TreeIndex.load(index_path)
            self._doc_path = doc_path
            return self._tree

        # Build new index
        ext = os.path.splitext(doc_path)[1].lower()
        if ext == ".pdf":
            tree = self.indexer.index_pdf(doc_path)
        elif ext in (".md", ".markdown"):
            tree = self.indexer.index_markdown(doc_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Use .pdf or .md")

        # Save index
        tree.save(index_path)
        logger.info(f"Index saved to {index_path}")

        self._tree = tree
        self._doc_path = doc_path
        return tree

    def load_index(self, index_path: str, doc_path: str) -> TreeIndex:
        """Load a previously saved tree index."""
        self._tree = TreeIndex.load(index_path)
        self._doc_path = doc_path
        return self._tree

    # ── Querying ───────────────────────────────────────────────────────────────

    def query(self, question: str, return_context: bool = False) -> dict:
        """
        Answer a question using reasoning-based tree search + generation.

        Args:
            question: Natural language question about the document.
            return_context: If True, include the retrieved text in the result.

        Returns:
            Dict with keys:
            - answer: The generated answer
            - nodes_used: List of section titles used
            - page_ranges: List of (start, end) page tuples
            - reasoning: Brief retrieval reasoning
            - context (optional): The raw extracted text used
        """
        self._check_ready()
        start_time = time.perf_counter()
        result, sections_text = self.retriever.search_and_extract(
            self._tree, question, self._doc_path
        )
        logger.info(f"First LLM Call time: {time.perf_counter() - start_time}")
        # Generate answer
        from .retriever import EXTRACT_ANSWER_PROMPT
        prompt = EXTRACT_ANSWER_PROMPT.format(
            query=question,
            sections_text=sections_text[:15000],
        )
        logger.info(f"llm call 2 PROMPT: {prompt}")
        start_time = time.perf_counter()
        answer = self.retriever.client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
        ).strip()
        logger.info(f"Second LLM Call time: {time.perf_counter() - start_time}")
        output = {
            "answer": answer,
            "nodes_used": [n.title for n in result.nodes],
            "page_ranges": result.page_ranges,
            # "reasoning": result.reasoning,
        }
        if return_context:
            output["context"] = sections_text
        return output

    def search(self, query: str):
        """Search for relevant sections without generating an answer."""
        self._check_ready()
        return self.retriever.search(self._tree, query)

    def get_outline(self) -> str:
        """Return the tree index outline as a human-readable string."""
        self._check_ready()
        return self._tree.to_outline()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _check_ready(self):
        if self._tree is None or self._doc_path is None:
            raise RuntimeError(
                "No document indexed. Call .index(doc_path) first."
            )
