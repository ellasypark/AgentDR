from contextlib import contextmanager
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from agentdr.pgvector_store import PgvectorIndex
from agentdr.rag import load_policy_index


class StoreSelectionTests(unittest.TestCase):
    def test_memory_is_default_without_database_driver(self):
        with patch.dict(os.environ, {}, clear=True), patch("agentdr.rag.PolicyIndex.from_markdown") as build:
            self.assertIs(load_policy_index(Path("policies.md")), build.return_value)
            self.assertEqual(build.call_args.args[1].keywords["model"], "embeddinggemma")

    def test_pgvector_uses_same_factory_for_cli_and_mcp(self):
        settings = {"AGENTDR_RAG_STORE": "pgvector", "AGENTDR_DATABASE_URL": "test-url", "AGENTDR_EMBED_MODEL": "test-model"}
        with patch.dict(os.environ, settings), patch.object(PgvectorIndex, "from_markdown") as build:
            self.assertIs(load_policy_index(Path("policies.md")), build.return_value)
            self.assertEqual(build.call_args.kwargs["database_url"], "test-url")
            self.assertEqual(build.call_args.kwargs["model"], "test-model")

    def test_missing_database_url_and_unknown_store_fail_clearly(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "AGENTDR_DATABASE_URL"):
                load_policy_index(Path("policies.md"), store="pgvector")
            with self.assertRaisesRegex(ValueError, "memory or pgvector"):
                load_policy_index(Path("policies.md"), store="typo")

    def test_query_validation_precedes_database_access(self):
        for vectors in ([[0, 0]], [[1]], [[float("nan"), 1]], []):
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                PgvectorIndex("unused", "index", 2, lambda _: vectors).retrieve("query")
        index = PgvectorIndex("unused", "index", 2, Mock())
        for query, top_k in (("", 1), ("read", 0)):
            with self.assertRaises(ValueError):
                index.retrieve(query, top_k)
        index.embed.assert_not_called()

    def test_cache_hit_does_not_embed_documents(self):
        @contextmanager
        def connect(_):
            connection = Mock()
            connection.execute.return_value.fetchone.return_value = (1, 2, 2)
            yield connection

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "policies.md"
            source.write_text("## Docs\nRead project docs.")
            embed = Mock()
            with patch("agentdr.pgvector_store._connect", connect):
                index = PgvectorIndex.from_markdown(source, database_url="unused", model="test", embed=embed)
            self.assertEqual(index.dimensions, 2)
            embed.assert_not_called()


DATABASE_URL = os.getenv("AGENTDR_TEST_DATABASE_URL")


@unittest.skipUnless(DATABASE_URL, "Set AGENTDR_TEST_DATABASE_URL to run real pgvector checks")
class PgvectorIntegrationTests(unittest.TestCase):
    def setUp(self):
        import psycopg

        self.psycopg = psycopg
        self.collection = "test-" + uuid4().hex
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "policies.md"
        self.path.write_text("## Docs\nRead README.\n\n## Secrets\nNever read tokens.")
        self.addCleanup(self.cleanup_rows)

    def cleanup_rows(self):
        with self.psycopg.connect(DATABASE_URL) as connection:
            if connection.execute("SELECT to_regclass('agentdr_policy_chunks')").fetchone()[0]:
                connection.execute("DELETE FROM agentdr_policy_chunks WHERE collection = %s", (self.collection,))

    def build(self, embed, model="fixed-v1"):
        return PgvectorIndex.from_markdown(
            self.path, database_url=DATABASE_URL, model=model,
            embed=embed, collection=self.collection,
        )

    def test_persistent_cache_cosine_ranking_and_same_match_contract(self):
        first_embed = Mock(return_value=[[10, 0], [0, 5]])
        first = self.build(first_embed)
        # A new object/connection must reuse stored documents and embed only queries.
        query_embed = Mock(return_value=[[0, 2]])
        reopened = self.build(query_embed)
        query_embed.assert_not_called()
        hits = reopened.retrieve("private keys", top_k=10)
        self.assertEqual(reopened.index_key, first.index_key)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].chunk.text, "## Secrets\nNever read tokens.")
        self.assertEqual(hits[0].chunk.source, "policies.md#2")
        self.assertAlmostEqual(hits[0].score, 1.0, places=5)
        query_embed.assert_called_once_with(["private keys"])
        first_embed.assert_called_once()

    def test_document_and_model_changes_create_isolated_indexes(self):
        original = self.build(lambda texts: [[1, 0] for _ in texts])
        self.path.write_text("## Updated\nNew approved policy.")
        updated = self.build(lambda texts: [[0, 1] for _ in texts])
        other_model = self.build(lambda texts: [[0, 0, 1] for _ in texts], model="fixed-v2")
        self.assertEqual(len({original.index_key, updated.index_key, other_model.index_key}), 3)
        self.assertEqual(len(original.retrieve("read", 10)), 2)
        self.assertIn("New approved policy", updated.retrieve("read")[0].chunk.text)
        self.assertEqual(other_model.dimensions, 3)
        self.assertEqual(len(other_model.retrieve("read")), 1)

    def test_failed_ingestion_keeps_previous_index_usable(self):
        original = self.build(lambda texts: [[1, 0] for _ in texts])
        self.path.write_text("## Broken\nNew policy.")
        with self.assertRaises(ValueError):
            self.build(lambda _: [[0, 0]])
        self.assertEqual(len(original.retrieve("read", 10)), 2)
        with self.psycopg.connect(DATABASE_URL) as connection:
            count = connection.execute("SELECT count(*) FROM agentdr_policy_chunks WHERE collection = %s", (self.collection,)).fetchone()[0]
        self.assertEqual(count, 2)

    def test_database_failure_becomes_alert_without_calling_judge(self):
        from agentdr.semantic_policy import SemanticPolicy

        index = self.build(lambda texts: [[1, 0] for _ in texts])
        with patch("agentdr.pgvector_store._connect", side_effect=RuntimeError("database offline")), patch("agentdr.semantic_policy.ollama_request") as judge:
            decision = SemanticPolicy(index).check("read_file", {"path": "README.md"})
        self.assertEqual(decision.decision, "alert")
        judge.assert_not_called()


if __name__ == "__main__":
    unittest.main()
