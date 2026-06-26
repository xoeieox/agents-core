"""Thread comments — append-only JSONL per thread id.

Storage: /srv/lapis/targets/comments/<thread_id>.jsonl
One JSON object per line — safe for concurrent agent appends.

Comment shape:
    {id, ts (ISO Pacific), author, author_type, content, tags}

author_type: "user" | "agent" | "system"
tags: list of namespaced strings (e.g. "pm:brief", "human:directive"). Default [].
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from agents_core.room_paths import room_path

PACIFIC = ZoneInfo("America/Los_Angeles")
DEFAULT_DIR = room_path("targets.comments")
VALID_AUTHOR_TYPES = {"user", "agent", "system"}

_KNOWN_FIELDS = {"id", "ts", "author", "author_type", "content", "tags"}


@dataclass
class Comment:
    id: str
    ts: str
    author: str
    author_type: str
    content: str
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class CommentStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else DEFAULT_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, thread_id: str) -> Path:
        # thread_id comes from target.id which is a filename stem — already safe,
        # but strip slashes defensively.
        safe = thread_id.replace("/", "_").replace("..", "_")
        return self.root / f"{safe}.jsonl"

    def append(
        self,
        thread_id: str,
        content: str,
        author: str,
        author_type: str = "agent",
        tags: list[str] | None = None,
    ) -> Comment:
        if author_type not in VALID_AUTHOR_TYPES:
            raise ValueError(f"author_type must be one of {VALID_AUTHOR_TYPES}")
        content = content.strip()
        if not content:
            raise ValueError("content cannot be empty")

        c = Comment(
            id=str(uuid.uuid4()),
            # Microsecond precision — avoids same-second cursor collisions
            # when agents perceive/encode in tight loops.
            ts=datetime.now(PACIFIC).isoformat(timespec="microseconds"),
            author=author,
            author_type=author_type,
            content=content,
            tags=list(tags) if tags else [],
        )
        path = self._path(thread_id)
        # O_APPEND on POSIX guarantees atomic appends for small writes.
        line = json.dumps(c.to_dict(), ensure_ascii=False) + "\n"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
        return c

    def list(self, thread_id: str) -> list[Comment]:
        path = self._path(thread_id)
        if not path.exists():
            return []
        out: list[Comment] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    # Filter to known fields so older records (pre-tags) load cleanly
                    # and any future extras don't break the dataclass.
                    clean = {k: v for k, v in d.items() if k in _KNOWN_FIELDS}
                    out.append(Comment(**clean))
                except (json.JSONDecodeError, TypeError):
                    continue
        return out

    def count(self, thread_id: str) -> int:
        path = self._path(thread_id)
        if not path.exists():
            return 0
        n = 0
        with open(path, "rb") as f:
            for _ in f:
                n += 1
        return n

    def latest(self, thread_id: str) -> Comment | None:
        comments = self.list(thread_id)
        return comments[-1] if comments else None

    def summary(self, thread_id: str) -> dict:
        """Count + latest-comment preview. Single-pass read."""
        comments = self.list(thread_id)
        if not comments:
            return {"count": 0, "latest": None}
        last = comments[-1]
        preview = last.content if len(last.content) <= 140 else last.content[:137] + "..."
        return {
            "count": len(comments),
            "latest": {
                "ts": last.ts,
                "author": last.author,
                "author_type": last.author_type,
                "preview": preview,
            },
        }
