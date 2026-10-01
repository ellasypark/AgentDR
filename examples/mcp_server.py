from datetime import datetime, timezone
from functools import lru_cache, partial
import os
from pathlib import Path
from uuid import uuid4

from mcp.server import MCPServer

from agentdr.events import Event
from agentdr.policy import check_file_access
from agentdr.rag import PolicyIndex, ollama_embed
from agentdr.recorder import record_event
from agentdr.semantic_policy import SemanticDecision, SemanticPolicy

mcp = MCPServer("AgentDR Test")
RUN_ID = str(uuid4())
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = PROJECT_ROOT / "traces/manual.json"


def record(event_type: str, data: dict):
    record_event(Event(
        event_id=str(uuid4()),
        run_id=RUN_ID,
        timestamp=datetime.now(timezone.utc),
        event_type=event_type,
        data=data,
    ), TRACE_PATH)


@lru_cache(maxsize=1)
def get_semantic_policy() -> SemanticPolicy:
    # Lazy: deterministic blocks never require a model or policy ingestion.
    embed = partial(ollama_embed, model=os.getenv("AGENTDR_EMBED_MODEL", "embeddinggemma"))
    index = PolicyIndex.from_markdown(PROJECT_ROOT / "knowledge/policies.md", embed)
    return SemanticPolicy(index, model=os.getenv("AGENTDR_JUDGE_MODEL", "llama3.2:3b"))


@mcp.tool()
def read_file(path: str) -> str:
    """Read a file after deterministic and optional semantic policy checks."""
    action = {"tool": "read_file", "arguments": {"path": path}}
    record("tool_call", action)
    allowed, reason = check_file_access(path)
    record("policy_decision", {
        **action, "layer": "deterministic",
        "decision": "allow" if allowed else "block", "reason": reason,
    })
    if not allowed:
        return f"BLOCKED: {reason}"

    if os.getenv("AGENTDR_SEMANTIC") == "1":
        try:
            decision = get_semantic_policy().check("read_file", {"path": path})
        except (RuntimeError, OSError, ValueError, KeyError, TypeError) as exc:
            decision = SemanticDecision("alert", f"Policy loading failed ({type(exc).__name__})", [])
        record("policy_decision", {
            **action, "layer": "semantic", "decision": decision.decision,
            "reason": decision.reason, "sources": decision.sources,
        })
        if decision.decision == "alert":
            return f"ALERT: {decision.reason}"

    try:
        result = Path(path).expanduser().read_text(encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as exc:
        record("tool_error", {**action, "error": type(exc).__name__})
        raise
    record("tool_finished", {**action, "characters": len(result)})
    return result


if __name__ == "__main__":
    mcp.run()
