# AgentDR

EDR-like Agent Detection and Response: a small Python learning project for recording
AI agent actions and checking them against policy before a tool executes.

## Current status

- Record tool requests, policy decisions, and results as Event objects in JSONL.
- Block known system and credential paths before `read_file` executes.
- Retrieve relevant policy snippets with local embeddings and cosine similarity.
- Optionally persist policy vectors in PostgreSQL with pgvector and reuse them across runs.
- Optionally ask a local LLM for an `allow` or `alert` decision using those snippets.

This version evaluates one proposed action at a time. Trajectory detection is not
implemented. The default store is in memory; PostgreSQL + pgvector is an opt-in
alternative with the same retrieval interface.

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
Core recording and in-memory RAG use the standard library; `pip install -e .` is
enough for the memory demo. The optional `postgres` extra adds the Psycopg driver.
The commands below use uv; with an activated pip
environment, omit `uv run --extra mcp`.

## Structure

```text
knowledge/policies.md         Example policy source; edit for your task
src/agentdr/events.py          Event schema and JSON-friendly conversion
src/agentdr/recorder.py        Append Event objects to a JSONL trace
src/agentdr/policy.py          Small deterministic deny rules
src/agentdr/rag.py             Chunking, embeddings, in-memory store, retrieval
src/agentdr/pgvector_store.py  Persistent PostgreSQL vector store
src/agentdr/semantic_policy.py Retrieved policy + local LLM -> allow/alert
examples/manual_trace.py      Write one synthetic event
examples/rag_demo.py          Print relevant policy snippets
examples/mcp_server.py        read_file tool with policy checks and recording
compose.yaml                  Local PostgreSQL with the pgvector extension
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

The default index embeds the source once per instance and keeps vectors in memory.
Restart the demo or MCP server after editing policies. Both stores accept an
embedding function, so tests use fixed vectors without a model download.

## PostgreSQL + pgvector

Use this store to keep policy chunks and their embeddings between program runs.
Ollama still generates embeddings; pgvector stores them and performs cosine
search in SQL. The semantic judge and Event/Recorder flow stay the same.

With Docker running, start the included local database and run the demo:

```sh
docker compose up -d --wait
uv sync --extra mcp --extra postgres
export AGENTDR_DATABASE_URL='postgresql://agentdr:agentdr_local_only@127.0.0.1:5433/agentdr'
ollama pull embeddinggemma
uv run --extra mcp --extra postgres python examples/rag_demo.py --store pgvector "read a file containing API tokens"
```

The first run creates the `vector` extension and `agentdr_policy_chunks` table,
then embeds and inserts the policy chunks in one transaction. Later runs reuse
the stored document vectors; only the search query needs a new embedding. Changed
policy text or a different embedding model gets its own index, so its vectors
cannot mix with the old version. The source file must still be available to
identify the current version. Old versions remain stored; cleanup is manual.
Use stable model names/tags: replacing weights under an unchanged name does not
invalidate this small cache.

For MCP, select the same store in the environment:

```sh
export AGENTDR_RAG_STORE=pgvector
AGENTDR_SEMANTIC=1 uv run --extra mcp --extra postgres mcp dev examples/mcp_server.py
```

Keep `AGENTDR_DATABASE_URL` exported and install the judge model as described below.
Set `AGENTDR_RAG_STORE=memory` (or pass `--store memory` to the demo) to use memory
again. Missing database configuration or connection errors produce an error in
the demo and an alert that withholds the read in the semantic MCP path.

The Compose credentials are for local development only; the port binds to
localhost. Set `AGENTDR_PG_PORT` before starting Compose if 5433 is occupied and
adjust the connection URL. `docker compose down` stops the database while keeping
its named data volume. An existing database needs pgvector installed and a role
allowed to create/use the extension and table. The Python driver extra does not
install the PostgreSQL extension.

This small version uses exact search, with no approximate vector index or ORM.
See [pgvector's documentation](https://github.com/pgvector/pgvector) for the SQL
operators and indexing options.

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

The default test suite runs without a database, Ollama, or model downloads. It covers retrieval order,
vector validation, invalid model output, deterministic precedence, withheld reads,
MCP tool invocation, and Event serialization. Fixed embedding vectors and mocked
judge replies verify the plumbing; they do not measure model detection quality.

Run the additional persistence and SQL-search tests against the local database:

```sh
AGENTDR_TEST_DATABASE_URL="$AGENTDR_DATABASE_URL" uv run --extra mcp --extra postgres python -m unittest discover -s tests -v
```

These tests use unique test collections and remove only their own rows. They
check reuse without re-embedding, cosine ranking, document/model isolation, and
failed ingestion. They are skipped when `AGENTDR_TEST_DATABASE_URL` is unset.

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
