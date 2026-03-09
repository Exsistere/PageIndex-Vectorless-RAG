"""
Data models for TreeRAG tree index structures.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class TreeNode:
    """
    Represents a node in the hierarchical document tree index.
    Corresponds to a section/chapter of the document.
    """
    node_id: str
    title: str
    start_page: int          # 1-based, inclusive
    end_page: int            # 1-based, inclusive
    summary: str = ""
    children: list[TreeNode] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "title": self.title,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "summary": self.summary,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: dict) -> TreeNode:
        node = cls(
            node_id=d["node_id"],
            title=d["title"],
            start_page=d["start_page"],
            end_page=d["end_page"],
            summary=d.get("summary", ""),
        )
        node.children = [cls.from_dict(c) for c in d.get("children", [])]
        return node

    def page_count(self) -> int:
        return self.end_page - self.start_page + 1

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def flatten(self) -> list[TreeNode]:
        """Return this node and all descendants in depth-first order."""
        result = [self]
        for child in self.children:
            result.extend(child.flatten())
        return result

    def to_outline(self, indent: int = 0) -> str:
        prefix = "  " * indent
        line = f"{prefix}[{self.node_id}] {self.title} (pp. {self.start_page}–{self.end_page})"
        if self.summary:
            line += f"\n{prefix}    → {self.summary[:120]}{'...' if len(self.summary) > 120 else ''}"
        child_lines = [c.to_outline(indent + 1) for c in self.children]
        return "\n".join([line] + child_lines)


@dataclass
class TreeIndex:
    """
    The full hierarchical tree index for a document.
    """
    title: str
    description: str
    total_pages: int
    source: str                        # original file path
    roots: list[TreeNode] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "total_pages": self.total_pages,
            "source": self.source,
            "roots": [r.to_dict() for r in self.roots],
        }

    @classmethod
    def from_dict(cls, d: dict) -> TreeIndex:
        idx = cls(
            title=d["title"],
            description=d.get("description", ""),
            total_pages=d["total_pages"],
            source=d.get("source", ""),
        )
        idx.roots = [TreeNode.from_dict(r) for r in d.get("roots", [])]
        return idx

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> TreeIndex:
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def all_nodes(self) -> list[TreeNode]:
        result = []
        for root in self.roots:
            result.extend(root.flatten())
        return result

    def to_outline(self) -> str:
        lines = [f"📄 {self.title}", f"   {self.description}", ""]
        for root in self.roots:
            lines.append(root.to_outline())
        return "\n".join(lines)

    def get_node(self, node_id: str) -> Optional[TreeNode]:
        for node in self.all_nodes():
            if node.node_id == node_id:
                return node
        return None
