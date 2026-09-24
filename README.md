# AgentDR

Agent Detection and Response: a Python learning project exploring how to record
AI agent activity and identify actions outside a task's allowed scope.

**Status:** initial scaffold. Event recording, detection, and response are not
implemented. The Python files contain short implementation prompts only.

## First milestone

Record one simulated tool call in a local file, then read it back. This gives
later detection rules a concrete input format.

Planned flow:

```text
simulated tool call -> event -> recorder -> traces/manual.jsonl
```

JSONL means one JSON object per line. The first version will use Python's
standard library and have no model API costs.

## Structure

```text
AgentDR/
├── README.md                 # Purpose, scope, and implementation order
├── pyproject.toml            # Package metadata and installation settings
├── .gitignore                # Local environments and generated traces
├── src/
│   └── agentdr/
│       ├── __init__.py       # Makes agentdr an importable package
│       ├── events.py         # What information an event contains
│       └── recorder.py       # How an event is saved
└── examples/
    └── manual_trace.py       # A small demo to write after the core modules
```

The package lives under `src/`; demonstrations live under `examples/`.
Add a `tests/` directory when the first behavior exists to check.

## Local setup

Requires Python 3.11 or newer. On macOS/Linux, clone the repository first
(skip this if you already have a local copy):

```sh
git clone https://github.com/ellasypark/AgentDR.git
cd AgentDR
```

Then, from the project directory:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -c "import agentdr; print(agentdr.__file__)"
```

The editable installation lets imports use your source files as you change
them. The import command only checks package setup; there is no working demo yet.
Setuptools is a build dependency, not an agent framework. This layout follows
the [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/).

## What to implement first

1. **Define an event in `events.py`.** Start with a dataclass. Suggested fields:
   `event_id` identifies an event, `run_id` groups events from one run,
   `timestamp` records UTC time, `event_type` names what happened, and `data`
   holds tool details. Choose a JSON-friendly representation for each field.
2. **Write `record_event(event, path)` in `recorder.py`.** Convert the event to
   JSON and append one line. Create the output directory if it is missing.
3. **Complete `examples/manual_trace.py`.** Construct a synthetic event with a
   made-up tool result and write it to `traces/manual.jsonl`. Then run
   `python examples/manual_trace.py` from this directory.
4. **Check the behavior.** Add tests using `unittest` and a temporary directory.
   Write two events to the same file, read each line as JSON, and check that both
   events survive with their original fields. Run `python -m unittest discover -s tests`.

An example of the intended data, not an implemented schema:

```json
{
  "event_id": "event-001",
  "run_id": "run-001",
  "timestamp": "2026-09-24T00:00:00Z",
  "event_type": "tool_finished",
  "data": {
    "tool_name": "read_file",
    "arguments": {"path": "notes.txt"},
    "result": "Synthetic example text"
  }
}
```

Done means you can explain each field, produce a trace, and read it back without
losing earlier events. Generated traces stay local and are ignored by Git.

## After recording works

Add one rule in `src/agentdr/detection.py`: flag a tool name that is absent from
an explicitly supplied allowlist. Test one allowed and one disallowed event.
Begin the response behavior with a printed finding containing the event ID and
the reason. A finding from a completed event is an alert; preventing a tool call
would require a later check before execution.

Then wrap one small Python tool to emit events automatically. If LangChain or
LangGraph becomes useful, add an adapter that translates its exposed events into
the same format. Keep the event model and recorder independent of that adapter.
Capture observable activity and explicitly emitted summaries; do not assume
access to a model's private internal reasoning.

Choose further features after trying that flow. Databases, dashboards, causal
graphs, benchmark environments, and automatic blocking are future decisions.

## Development approach

The initial commit establishes the structure. Keep the first working recorder
as a separate, small commit you can explain line by line.
