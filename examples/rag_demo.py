"""Retrieve policy snippets without executing a tool or calling a judge."""

import argparse
from functools import partial
from pathlib import Path

from agentdr.rag import PolicyIndex, ollama_embed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default="read a file containing API tokens and private keys")
    parser.add_argument("--policies", type=Path, default=Path(__file__).resolve().parents[1] / "knowledge/policies.md")
    parser.add_argument("--model", default="embeddinggemma")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    try:
        index = PolicyIndex.from_markdown(args.policies, partial(ollama_embed, model=args.model))
        matches = index.retrieve(args.query, args.top_k)
    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Retrieval failed: {exc}\nRun Ollama, then: ollama pull {args.model}\n")
    for match in matches:
        print(f"\n[{match.score:.3f}] {match.chunk.source}\n{match.chunk.text}")


if __name__ == "__main__":
    main()
