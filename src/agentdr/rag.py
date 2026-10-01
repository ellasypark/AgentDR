"""Markdown -> chunks -> local Ollama embeddings -> in-memory cosine search."""

from collections.abc import Callable
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from urllib.error import URLError
from urllib.request import Request, urlopen

Embed = Callable[[list[str]], list[list[float]]]


def ollama_request(endpoint: str, payload: dict) -> dict:
    """Use the local service only; no hosted API key or Ollama SDK is required."""
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


def ollama_embed(texts: list[str], model: str = "embeddinggemma") -> list[list[float]]:
    return ollama_request(
        "embed", {"model": model, "input": texts, "truncate": False}
    )["embeddings"]


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str


@dataclass(frozen=True)
class Match:
    chunk: Chunk
    score: float


def chunk_markdown(text: str, source: str, max_words: int = 160) -> list[Chunk]:
    """Keep each heading with its policy; split longer sections into small chunks."""
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
            body = " ".join(words[offset:offset + max_words])
            chunks.append(Chunk(f"{source}#{len(chunks) + 1}", f"{heading}\n{body}".strip()))
    return chunks


def _normalize(vector: list[float]) -> list[float]:
    if not vector or any(not math.isfinite(value) for value in vector):
        raise ValueError("Embedding must contain finite numbers")
    length = math.hypot(*vector)
    if length == 0 or not math.isfinite(length):
        raise ValueError("Embedding must have a finite nonzero length")
    return [value / length for value in vector]


class PolicyIndex:
    """A tiny vector store. Reuse one instance so policies are embedded only once."""

    def __init__(self, chunks: list[Chunk], embed: Embed = ollama_embed):
        if not chunks:
            raise ValueError("Policy source contains no text")
        self.chunks = chunks
        self.embed = embed
        vectors = embed([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("Expected one embedding per policy chunk")
        self.vectors = [_normalize(vector) for vector in vectors]
        self.dimensions = len(self.vectors[0])
        if any(len(vector) != self.dimensions for vector in self.vectors):
            raise ValueError("Embedding dimensions must match")

    @classmethod
    def from_markdown(cls, path: Path, embed: Embed = ollama_embed) -> "PolicyIndex":
        path = Path(path)
        return cls(chunk_markdown(path.read_text(encoding="utf-8"), str(path)), embed)

    def retrieve(self, query: str, top_k: int = 3) -> list[Match]:
        if not query.strip() or top_k < 1:
            raise ValueError("Provide a nonempty query and positive top_k")
        vectors = self.embed([query])
        if len(vectors) != 1:
            raise ValueError("Expected one query embedding")
        query_vector = _normalize(vectors[0])
        if len(query_vector) != self.dimensions:
            raise ValueError("Query and policy embedding dimensions must match")
        matches = [
            Match(chunk, sum(a * b for a, b in zip(query_vector, vector, strict=True)))
            for chunk, vector in zip(self.chunks, self.vectors, strict=True)
        ]
        return sorted(matches, key=lambda match: match.score, reverse=True)[:top_k]
