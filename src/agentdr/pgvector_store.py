"""Persistent policy chunks with PostgreSQL + pgvector (an optional dependency)."""

from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path

from agentdr.rag import Chunk, Embed, Match, PolicyIndex, _normalize, chunk_markdown


@contextmanager
def _connect(database_url: str):
    try:
        import psycopg
    except ImportError:
        raise RuntimeError("Install the postgres extra: pip install -e '.[postgres]'") from None
    try:
        with psycopg.connect(
            database_url, connect_timeout=5, options="-c statement_timeout=10000"
        ) as connection:
            yield connection
    except psycopg.Error:
        # A database error can contain connection details. Do not put them in events.
        raise RuntimeError(
            "PostgreSQL operation failed. Check AGENTDR_DATABASE_URL, server availability, "
            "and permission to use the vector extension and agentdr_policy_chunks table."
        ) from None


class PgvectorIndex:
    """One immutable index per collection, source text, and embedding model.

    Queries use pgvector's exact cosine search. Small policy collections do not
    need approximate indexes, an ORM, connection pools, or a migration framework.
    """

    def __init__(self, database_url: str, index_key: str, dimensions: int, embed: Embed):
        self.database_url = database_url
        self.index_key = index_key
        self.dimensions = dimensions
        self.embed = embed

    @classmethod
    def from_markdown(
        cls, path: Path, *, database_url: str, model: str, embed: Embed,
        collection: str = "policies",
    ) -> "PgvectorIndex":
        if not database_url or not model or not collection:
            raise ValueError("Provide a database URL, embedding model, and collection")
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        # A logical filename makes the same source reusable across local checkouts.
        chunks = chunk_markdown(text, path.name)
        if not chunks:
            raise ValueError("Policy source contains no text")
        identity = json.dumps(["chunks-v1", collection, path.name, text, model])
        index_key = sha256(identity.encode("utf-8")).hexdigest()

        with _connect(database_url) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS agentdr_policy_chunks (
                    index_key text NOT NULL,
                    chunk_number integer NOT NULL,
                    collection text NOT NULL,
                    model text NOT NULL,
                    source text NOT NULL,
                    content text NOT NULL,
                    embedding vector NOT NULL,
                    PRIMARY KEY (index_key, chunk_number)
                )
            """)
            count, dimensions, max_dimensions = connection.execute("""
                SELECT count(*), min(vector_dims(embedding)), max(vector_dims(embedding))
                FROM agentdr_policy_chunks WHERE index_key = %s
            """, (index_key,)).fetchone()
            if count:
                if count != len(chunks) or dimensions != max_dimensions:
                    raise ValueError("Stored policy index is incomplete or inconsistent")
                return cls(database_url, index_key, dimensions, embed)

        # Model work happens outside the database transaction. All rows are then
        # committed together; concurrent identical ingestions can safely reuse them.
        memory_index = PolicyIndex(chunks, embed)
        with _connect(database_url) as connection:
            with connection.cursor() as cursor:
                cursor.executemany("""
                    INSERT INTO agentdr_policy_chunks
                        (index_key, chunk_number, collection, model, source, content, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (index_key, chunk_number) DO NOTHING
                """, [
                    (index_key, number, collection, model, chunk.source, chunk.text, json.dumps(vector))
                    for number, (chunk, vector) in enumerate(
                        zip(chunks, memory_index.vectors, strict=True)
                    )
                ])
        return cls(database_url, index_key, memory_index.dimensions, embed)

    def retrieve(self, query: str, top_k: int = 3) -> list[Match]:
        if not query.strip() or top_k < 1:
            raise ValueError("Provide a nonempty query and positive top_k")
        vectors = self.embed([query])
        if len(vectors) != 1:
            raise ValueError("Expected one query embedding")
        vector = _normalize(vectors[0])
        if len(vector) != self.dimensions:
            raise ValueError("Query and policy embedding dimensions must match")
        with _connect(self.database_url) as connection:
            rows = connection.execute("""
                SELECT source, content, 1 - (embedding <=> %s::vector) AS score
                FROM agentdr_policy_chunks
                WHERE index_key = %s
                ORDER BY score DESC, chunk_number
                LIMIT %s
            """, (json.dumps(vector), self.index_key, top_k)).fetchall()
        return [Match(Chunk(source, content), score) for source, content, score in rows]
