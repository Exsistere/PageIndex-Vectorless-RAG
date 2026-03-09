"""
TreeRAG: Vectorless, Reasoning-based RAG with Hierarchical Tree Index
An open-source alternative to PageIndex using OpenAI API.
"""

from .indexer import TreeIndexer
from .retriever import TreeRetriever
from .models import TreeNode, TreeIndex
from .rag import TreeRAG

__version__ = "0.1.0"
__all__ = ["TreeIndexer", "TreeRetriever", "TreeNode", "TreeIndex", "TreeRAG"]
