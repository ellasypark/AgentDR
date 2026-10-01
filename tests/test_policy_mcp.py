import asyncio
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agentdr.events import Event
from agentdr.policy import check_file_access
from agentdr.recorder import record_event
from agentdr.semantic_policy import SemanticDecision

HAS_MCP = importlib.util.find_spec("mcp") is not None


class PolicyTests(unittest.TestCase):
    def test_credentials_and_system_paths_are_denied(self):
        for path in ("/etc/shadow", "/etc/passwd", "./.env", "./.env.production", "~/.ssh/id_rsa", "~/.aws/credentials"):
            with self.subTest(path=path):
                self.assertFalse(check_file_access(path)[0])
        self.assertTrue(check_file_access("README.md")[0])

    def test_symlink_to_secret_is_denied(self):
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "innocent.txt"
            link.symlink_to(Path(directory) / ".env")
            self.assertFalse(check_file_access(str(link))[0])

    def test_datetime_and_existing_string_timestamps_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            for stamp in (datetime.now(timezone.utc), "2026-09-26T21:00:00Z"):
                record_event(Event("one", "run", stamp, "tool_call", {"tool": "read_file"}), path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertIn("+00:00", rows[0]["timestamp"])
        self.assertEqual(rows[1]["timestamp"], "2026-09-26T21:00:00Z")


@unittest.skipUnless(HAS_MCP, "Install .[mcp] to run MCP integration checks")
class MCPTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("example_server", Path(__file__).resolve().parents[1] / "examples/mcp_server.py")
        self.server = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.server)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.server.TRACE_PATH = self.root / "trace.jsonl"
        self.document = self.root / "README.md"
        self.document.write_text("Hello, project!")

    def events(self):
        return [json.loads(line) for line in self.server.TRACE_PATH.read_text().splitlines()]

    def test_deterministic_block_never_reads_or_calls_models(self):
        with patch.dict("os.environ", {"AGENTDR_SEMANTIC": "1"}), patch.object(self.server, "get_semantic_policy") as semantic, patch.object(Path, "read_text") as read:
            result = self.server.read_file("/etc/shadow")
        self.assertTrue(result.startswith("BLOCKED:"))
        semantic.assert_not_called()
        read.assert_not_called()
        self.assertEqual(self.events()[-1]["data"]["decision"], "block")

    def test_optional_layer_off_keeps_read_and_telemetry(self):
        with patch.dict("os.environ", {"AGENTDR_SEMANTIC": "0"}), patch.object(self.server, "get_semantic_policy") as semantic:
            self.assertEqual(self.server.read_file(str(self.document)), "Hello, project!")
        semantic.assert_not_called()
        rows = self.events()
        self.assertEqual([row["event_type"] for row in rows], ["tool_call", "policy_decision", "tool_finished"])
        self.assertNotIn("Hello, project!", json.dumps(rows))
        self.assertEqual(len({row["run_id"] for row in rows}), 1)

    def test_semantic_alert_or_initialization_failure_withholds_read(self):
        for fail in (False, True):
            with self.subTest(fail=fail), patch.dict("os.environ", {"AGENTDR_SEMANTIC": "1"}), patch.object(self.server, "get_semantic_policy") as semantic, patch.object(Path, "read_text") as read:
                if fail:
                    semantic.side_effect = RuntimeError("missing model")
                else:
                    semantic.return_value.check.return_value = SemanticDecision("alert", "Review needed", ["policies.md#1"])
                self.assertTrue(self.server.read_file(str(self.document)).startswith("ALERT:"))
                read.assert_not_called()
            self.assertEqual(self.events()[-1]["data"]["decision"], "alert")

    def test_semantic_allow_records_sources_before_tool_completion(self):
        with patch.dict("os.environ", {"AGENTDR_SEMANTIC": "1"}), patch.object(self.server, "get_semantic_policy") as semantic:
            semantic.return_value.check.return_value = SemanticDecision("allow", "Project docs", ["policies.md#2"])
            self.assertEqual(self.server.read_file(str(self.document)), "Hello, project!")
        rows = self.events()
        self.assertEqual(rows[-2]["data"]["sources"], ["policies.md#2"])
        self.assertEqual(rows[-1]["event_type"], "tool_finished")

    def test_read_error_is_recorded(self):
        with patch.dict("os.environ", {"AGENTDR_SEMANTIC": "0"}), self.assertRaises(FileNotFoundError):
            self.server.read_file(str(self.root / "missing.txt"))
        self.assertEqual(self.events()[-1]["event_type"], "tool_error")

    def test_registered_mcp_tool_returns_content(self):
        with patch.dict("os.environ", {"AGENTDR_SEMANTIC": "0"}):
            result = asyncio.run(self.server.mcp.call_tool("read_file", {"path": str(self.document)}))
        self.assertFalse(result.is_error)
        self.assertIn("Hello, project!", str(result))


if __name__ == "__main__":
    unittest.main()
