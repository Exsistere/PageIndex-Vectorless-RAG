#!/usr/bin/env python3
"""
TreeRAG CLI - Vectorless, Reasoning-based RAG

Usage:
    # Build an index
    python -m treerag index report.pdf

    # Query a document (interactive)
    python -m treerag query report.pdf "What was total revenue in 2023?"

    # Show the tree outline
    python -m treerag outline report.pdf

    # Interactive chat session
    python -m treerag chat report.pdf
"""

import argparse
import json
import os
import sys


def cmd_index(args):
    from treerag import TreeIndexer

    print(f"🌲 Building tree index for: {args.pdf}")
    indexer = TreeIndexer(
        api_key=args.api_key,
        model=args.model,
        max_pages_per_node=args.max_pages_per_node,
        toc_check_pages=args.toc_check_pages,
        add_summaries=not args.no_summaries,
        verbose=True,
    )

    ext = os.path.splitext(args.pdf)[1].lower()
    if ext == ".pdf":
        tree = indexer.index_pdf(args.pdf)
    elif ext in (".md", ".markdown"):
        tree = indexer.index_markdown(args.pdf)
    else:
        print(f"❌ Unsupported file type: {ext}")
        sys.exit(1)

    out = args.output or (os.path.splitext(args.pdf)[0] + "_treeindex.json")
    tree.save(out)
    print(f"\n✅ Index saved to: {out}")
    print(f"   Nodes: {len(tree.all_nodes())}")
    print(f"   Pages: {tree.total_pages}")
    print(f"\n📋 Outline:\n")
    print(tree.to_outline())


def cmd_outline(args):
    from treerag import TreeIndex

    index_path = args.index or (os.path.splitext(args.pdf)[0] + "_treeindex.json")
    if not os.path.exists(index_path):
        print(f"❌ Index not found: {index_path}")
        print(f"   Run: python -m treerag index {args.pdf}")
        sys.exit(1)

    tree = TreeIndex.load(index_path)
    print(tree.to_outline())


def cmd_query(args):
    from treerag import TreeRAG

    rag = TreeRAG(
        api_key=args.api_key,
        model=args.model,
        verbose=args.verbose,
    )

    index_path = args.index or (os.path.splitext(args.pdf)[0] + "_treeindex.json")
    if os.path.exists(index_path) and not args.reindex:
        print(f"📂 Loading existing index: {index_path}")
        rag.load_index(index_path, args.pdf)
    else:
        print(f"🌲 Building index (this may take a moment)...")
        rag.index(args.pdf, index_path=index_path)

    print(f"\n🔍 Query: {args.question}\n")
    result = rag.query(args.question, return_context=args.show_context)

    print("=" * 60)
    print("💬 Answer:")
    print("=" * 60)
    print(result["answer"])
    print()
    print(f"📍 Retrieved from: {', '.join(result['nodes_used'])}")
    print(f"📄 Pages: {result['page_ranges']}")

    if args.show_context:
        print("\n📖 Context used:")
        print("-" * 40)
        print(result.get("context", "")[:2000] + "...")

    if args.json:
        print("\n📊 Full JSON result:")
        print(json.dumps({k: v for k, v in result.items() if k != "context"}, indent=2))


def cmd_chat(args):
    from treerag import TreeRAG

    rag = TreeRAG(
        api_key=args.api_key,
        model=args.model,
        verbose=args.verbose,
    )

    index_path = args.index or (os.path.splitext(args.pdf)[0] + "_treeindex.json")
    if os.path.exists(index_path) and not args.reindex:
        print(f"📂 Loading existing index: {index_path}")
        rag.load_index(index_path, args.pdf)
    else:
        print(f"🌲 Building index (this may take a moment)...")
        rag.index(args.pdf, index_path=index_path)

    print(f"\n🌲 TreeRAG Chat — Document: {os.path.basename(args.pdf)}")
    print(f"   Type 'outline' to see the document structure")
    print(f"   Type 'exit' or 'quit' to exit\n")

    while True:
        try:
            question = input("❓ You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if question.lower() == "outline":
            print(rag.get_outline())
            continue

        result = rag.query(question)
        print(f"\n💬 Answer: {result['answer']}")
        print(f"   📍 From: {', '.join(result['nodes_used'])}")
        print(f"   📄 Pages: {result['page_ranges']}\n")


def get_api_key(args):
    key = getattr(args, "api_key", None)
    if key:
        return key
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CHATGPT_API_KEY")
    return key


def main():
    parser = argparse.ArgumentParser(
        description="🌲 TreeRAG: Vectorless, Reasoning-based RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model (default: gpt-4o)")
    parser.add_argument("--verbose", "-v", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = sub.add_parser("index", help="Build a tree index for a document")
    p_index.add_argument("pdf", metavar="FILE", help="PDF or Markdown file to index")
    p_index.add_argument("--output", "-o", help="Output JSON index path")
    p_index.add_argument("--max-pages-per-node", type=int, default=10)
    p_index.add_argument("--toc-check-pages", type=int, default=20)
    p_index.add_argument("--no-summaries", action="store_true", help="Skip node summaries")

    # outline
    p_outline = sub.add_parser("outline", help="Print the document tree outline")
    p_outline.add_argument("pdf", metavar="FILE")
    p_outline.add_argument("--index", help="Path to existing index JSON")

    # query
    p_query = sub.add_parser("query", help="Answer a question about a document")
    p_query.add_argument("pdf", metavar="FILE")
    p_query.add_argument("question", help="Question to answer")
    p_query.add_argument("--index", help="Path to existing index JSON")
    p_query.add_argument("--reindex", action="store_true", help="Force rebuild index")
    p_query.add_argument("--show-context", action="store_true")
    p_query.add_argument("--json", action="store_true", help="Output full JSON result")

    # chat
    p_chat = sub.add_parser("chat", help="Interactive chat with a document")
    p_chat.add_argument("pdf", metavar="FILE")
    p_chat.add_argument("--index", help="Path to existing index JSON")
    p_chat.add_argument("--reindex", action="store_true")

    args = parser.parse_args()

    # Inject api key from env if not provided
    if not args.api_key:
        args.api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CHATGPT_API_KEY")

    if args.command == "index":
        cmd_index(args)
    elif args.command == "outline":
        cmd_outline(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "chat":
        cmd_chat(args)


if __name__ == "__main__":
    main()
