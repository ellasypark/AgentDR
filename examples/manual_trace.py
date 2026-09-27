from agentdr.events import Event
from agentdr.recorder import record_event

event = Event(
    event_id="event-001",
    run_id="run-001",
    timestamp="2026-09-26T21:00:00Z",
    event_type="tool_call",
    data={
        "tool": "read_file",
        "arguments": {"path": "hello.txt"}
    }
)

record_event(event, "traces/manual.json")

print("event recorded")