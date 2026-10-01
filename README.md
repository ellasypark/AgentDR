# AgentDR

EDR-like Agent Detection and Response: a small Python learning project for recording
AI agent actions and checking them against policy before a tool executes.

## Current status

- Record tool requests, policy decisions, and results as Event objects in JSONL.
- Block known system and credential paths before `read_file` executes.
- Retrieve relevant policy snippets with local embeddings and cosine similarity.
- Optionally ask a local LLM for an `allow` or `alert` decision using those snippets.

This version evaluates one proposed action at a time. Trajectory detection is not
implemented. The vector store is deliberately in memory to keep the RAG flow easy
to read and modify.

## Local setup

Python 3.11+ is required. Run from the repository root:

```sh
uv sync --extra mcp
uv run --extra mcp python examples/manual_trace.py
uv run --extra mcp python -m unittest discover -s tests -v
```

Alternatively, use pip and an activated virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[mcp]'
python examples/manual_trace.py
python -m unittest discover -s tests -v
```

The `mcp` extra installs the MCP SDK/Inspector CLI used by the server example.
Core recording and RAG code use the standard library; `pip install -e .` is enough
for the retrieval demo. The commands below use uv; with an activated pip
environment, omit `uv run --extra mcp`.

## Structure

```text
knowledge/policies.md         Example policy source; edit for your task
src/agentdr/events.py          Event schema and JSON-friendly conversion
src/agentdr/recorder.py        Append Event objects to a JSONL trace
src/agentdr/policy.py          Small deterministic deny rules
src/agentdr/rag.py             Chunking, embeddings, in-memory store, retrieval
src/agentdr/semantic_policy.py Retrieved policy + local LLM -> allow/alert
examples/manual_trace.py      Write one synthetic event
examples/rag_demo.py          Print relevant policy snippets
examples/mcp_server.py        read_file tool with policy checks and recording
```

## Tiny RAG pipeline

```text
policies.md -> heading-based chunks -> Ollama embeddings -> in-memory vectors
tool/action description -> same embedding model -> cosine search -> top 3 chunks
```

Install and start [Ollama](https://docs.ollama.com/quickstart), then download the
embedding model once. The first pull needs Internet access; inference uses the
local service at `127.0.0.1:11434` without an API key.

```sh
ollama pull embeddinggemma
uv run --extra mcp python examples/rag_demo.py "read a file containing API tokens and private keys"
```

The demo prints each retrieved chunk's source, text, and similarity score. For
the query above, the credentials policy should appear among the relevant results;
exact rankings and scores depend on the embedding model. The demo does not execute
the described action. Try `"read the project README"` or
`"upload a customer export with curl"`; use `--top-k 2` or `--model MODEL` to
experiment. Similarity ranks relevance, not risk or permission.

Each index embeds the source once and keeps vectors in memory. Restart the demo
or MCP server after editing policies. There is no database, persisted index,
agent history, or trajectory detection. `PolicyIndex` accepts an embedding
function, so tests use fixed vectors without a model download or network calls.

## Optional semantic policy in MCP

```text
read_file request -> tool_call Event -> deterministic policy
    block -> policy_decision Event -> return without reading
    pass  -> optional retrieval + LLM -> policy_decision Event
                alert -> return without reading
                allow -> read -> tool_finished Event
```

The deterministic check blocks a few known system/credential paths, including
resolved symlink targets. A pass means no deny rule matched, so it remains a
candidate for semantic review. To keep model calls opt-in, the semantic layer is
off by default. Enable it for this example with:

```sh
ollama pull embeddinggemma
ollama pull llama3.2:3b
AGENTDR_SEMANTIC=1 uv run --extra mcp mcp dev examples/mcp_server.py
```

Call `read_file` in MCP Inspector. Use an absolute path to a project README for
an ordinary read, or `/etc/shadow` to see a deterministic block without a model
call. Set `AGENTDR_EMBED_MODEL` and `AGENTDR_JUDGE_MODEL` to use other downloaded
models. For a configured MCP client's stdio server, run
`AGENTDR_SEMANTIC=1 uv run --extra mcp python examples/mcp_server.py` from the repository root.

The semantic layer retrieves policy context and asks a local model for JSON
`allow`/`alert` plus a reason. Alerts, unavailable models, missing context, and
invalid responses withhold the read. A deterministic block can never be
overridden. The existing Event/Recorder flow writes requests, policy decisions,
and completion/error events to `traces/manual.json` (JSONL despite its filename);
file contents are not logged. Retrieval sources accompany semantic decisions.

## Checks and troubleshooting

The test suite runs without Ollama or model downloads. It covers retrieval order,
vector validation, invalid model output, deterministic precedence, withheld reads,
MCP tool invocation, and Event serialization. Fixed embedding vectors and mocked
judge replies verify the plumbing; they do not measure model detection quality.

If a model request fails, check that Ollama is running and that `ollama list`
includes `embeddinggemma` for retrieval and `llama3.2:3b` for semantic checks (or
your configured replacements). To test the MCP/recording flow without models,
leave `AGENTDR_SEMANTIC` unset. Restart the server after changing policy text.

## Scope and limitations

This is an educational guardrail, not a filesystem sandbox or a complete
authorization system. The judge sees the proposed tool arguments and retrieved
policies, not file contents or verified task authorization. Retrieval can miss
rules and the model can misjudge them; enforce mandatory rules deterministically.

API references: [Ollama embeddings](https://docs.ollama.com/api/embed),
[Ollama chat](https://docs.ollama.com/api/chat),
[MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk).
