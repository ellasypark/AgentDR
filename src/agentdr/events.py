from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

@dataclass
class Event:
    event_id: str
    run_id: str
    timestamp: datetime
    event_type: str
    data: dict[str, Any]

    def to_dict(self):
        return asdict(self)
