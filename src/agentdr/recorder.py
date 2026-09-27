import json
from pathlib import Path

from agentdr.events import Event


def record_event(event: Event, path: Path):
    file_path=Path(path)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a") as f:
        json.dump(event.to_dict(), f)
        f.write("\n")