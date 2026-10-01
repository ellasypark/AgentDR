from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

@dataclass
class Event:
    event_id: str
    run_id: str
    timestamp: datetime | str
    event_type: str
    data: dict[str, Any]

    def to_dict(self):
        data = asdict(self)
        if isinstance(self.timestamp, datetime):
            data["timestamp"] = self.timestamp.isoformat()
        return data
