"""Small deterministic deny rules; passing them is not proof of task authorization."""

from pathlib import Path


def check_file_access(path: str) -> tuple[bool, str]:
    try:
        resolved = Path(path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return False, "invalid path"
    blocked = {Path(name).resolve() for name in ("/etc/passwd", "/etc/shadow")}
    if resolved in blocked:
        return False, "blocked system path"
    # Check both names so a symlink named .env cannot evade this example rule.
    for candidate in (Path(path), resolved):
        if any(part in {".ssh", ".aws", ".env"} or part.startswith(".env.")
               for part in candidate.parts):
            return False, "blocked credential path"
    return True, "no deterministic deny rule matched"
