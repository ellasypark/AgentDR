"""Markdown -> chunks -> local Ollama embeddings -> LangChain vector search."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, partial
import json
import math
import os
from pathlib import Path
import re
from typing import Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings


Embed = Callable[[list[str]], list[list[float]]]


@lru_cache
def _get_embedder(model: str) -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=model,
        base_url="http://127.0.0.1:11434",
    )


def ollama_embed(
    texts: list[str],
    model: str = "embeddinggemma",
) -> list[list[float]]:
    """Create embeddings through LangChain's Ollama integration."""
    return _get_embedder(model).embed_documents(texts)


def ollama_request(endpoint: str, payload: dict) -> dict:
    """Call the local Ollama API for the semantic judge."""
    request = Request(
        f"http://127.0.0.1:11434/api/{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=60) as response:
            result = json.load(response)
    except (URLError, OSError) as exc:
        raise RuntimeError(
            "Ollama request failed. Start Ollama and pull the configured model."
        ) from exc
    if not isinstance(result, dict) or result.get("error"):
        raise ValueError("Ollama returned an invalid response or model error")
    return result


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str


@dataclass(frozen=True)
class Match:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    """The interface shared by the memory and PostgreSQL stores."""

    def retrieve(self, query: str, top_k: int = 3) -> list[Match]: ...


def chunk_markdown(
    text: str,
    source: str,
    max_words: int = 160,
) -> list[Chunk]:
    """Keep each heading with its policy; split long sections into chunks."""
    if max_words < 1:
        raise ValueError("max_words must be positive")

    chunks = []
    for section in re.split(r"(?m)(?=^#{1,6} )", text):
        lines = section.strip().splitlines()
        if not lines:
            continue

        heading = lines.pop(0) if lines[0].startswith("#") else ""
        words = " ".join(lines).split()

        for offset in range(0, len(words), max_words):
            body = " ".join(words[offset : offset + max_words])
            chunk_number = len(chunks) + 1
            chunks.append(
                Chunk(
                    source=f"{source}#{chunk_number}",
                    text=f"{heading}\n{body}".strip(),
                )
            )

    return chunks


def _normalize(vector: list[float]) -> list[float]:
    if not vector or any(not math.isfinite(value) for value in vector):
        raise ValueError("Embedding must contain finite numbers")

    length = math.hypot(*vector)
    if length == 0 or not math.isfinite(length):
        raise ValueError("Embedding must have a finite nonzero length")

    return [value / length for value in vector]


class CallableEmbeddings(Embeddings):
    """Adapt AgentDR's existing embedding function to LangChain's interface."""

    def __init__(self, embed: Embed):
        self.embed = embed
        self.dimensions: int | None = None

    def _validated(self, vectors: list[list[float]], expected_count: int):
        if len(vectors) != expected_count:
            raise ValueError(f"Expected {expected_count} embedding(s)")

        normalized = [_normalize(vector) for vector in vectors]
        if not normalized:
            return normalized

        dimensions = len(normalized[0])
        if self.dimensions is None:
            self.dimensions = dimensions
        if any(len(vector) != self.dimensions for vector in normalized):
            raise ValueError("Embedding dimensions must match")

        return normalized

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.embed(texts)
        return self._validated(vectors, expected_count=len(texts))

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed([text])
        return self._validated(vectors, expected_count=1)[0]


class PolicyIndex:
    """In-memory policy index using LangChain vector search."""

    def __init__(self, chunks: list[Chunk], embed: Embed = ollama_embed):
        if not chunks:
            raise ValueError("Policy source contains no text")

        self.chunks = chunks
        self.embed = embed
        self.embeddings = CallableEmbeddings(embed)
        self.store = InMemoryVectorStore(self.embeddings)

        documents = [
            Document(
                id=chunk.source,
                page_content=chunk.text,
                metadata={"source": chunk.source},
            )
            for chunk in chunks
        ]
        self.store.add_documents(documents)

    @classmethod
    def from_markdown(
        cls,
        path: Path,
        embed: Embed = ollama_embed,
    ) -> "PolicyIndex":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        chunks = chunk_markdown(text, str(path))
        return cls(chunks, embed)

    def retrieve(self, query: str, top_k: int = 3) -> list[Match]:
        if not query.strip() or top_k < 1:
            raise ValueError("Provide a nonempty query and positive top_k")

        results = self.store.similarity_search_with_score(query, k=top_k)

        return [
            Match(
                chunk=Chunk(
                    source=document.metadata["source"],
                    text=document.page_content,
                ),
                score=float(score),
            )
            for document, score in results
        ]


def load_policy_index(
    path: Path,
    *,
    store: str | None = None,
    model: str | None = None,
) -> Retriever:
    """Select the memory or PostgreSQL policy store."""
    store = store if store is not None else os.getenv("AGENTDR_RAG_STORE", "memory")
    model = model if model is not None else os.getenv(
        "AGENTDR_EMBED_MODEL",
        "embeddinggemma",
    )
    embed = partial(ollama_embed, model=model)

    if store == "memory":
        return PolicyIndex.from_markdown(path, embed)

    if store == "pgvector":
        database_url = os.getenv("AGENTDR_DATABASE_URL")
        if not database_url:
            raise ValueError("Set AGENTDR_DATABASE_URL when using pgvector")

        from agentdr.pgvector_store import PgvectorIndex

        return PgvectorIndex.from_markdown(
            path,
            database_url=database_url,
            model=model,
            embed=embed,
        )

    raise ValueError("AGENTDR_RAG_STORE must be memory or pgvector")