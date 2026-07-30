"""Persistent find/replace rules applied to finished transcripts.

Unlike the glossary, which only *hints* to the model before decoding and is ignored
entirely by engines with no prompt interface, these rules run on the text afterwards, so
they always take effect. Kept free of Streamlit imports so the rules stay unit-testable.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

RULES_VERSION = 1
# Losing more than one entry in a single save is not a normal per-row delete; it usually
# means a stale in-memory set is being flushed over a larger file. Keep a copy first.
SHRINK_BACKUP_THRESHOLD = 1


def default_rules() -> dict[str, Any]:
    return {"version": RULES_VERSION, "rules": []}


def _rule_key(find: str) -> str:
    """Rules are keyed on the search text, which is matched case-sensitively."""
    return " ".join(str(find).split())


def normalize(data: Any) -> dict[str, Any]:
    """Coerce whatever was on disk into a valid rule set, dropping unusable entries.

    Deliberately permissive about the outer shape and the on/off field name. A
    hand-written file is very likely to be a bare `[{...}]` array rather than the wrapped
    object this module writes, and to say `enabled` instead of `active`; silently
    discarding a rule list someone typed out by hand is far worse than accepting both.
    """
    normalized = default_rules()
    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = data.get("rules") or []
    else:
        return normalized

    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        find = _rule_key(entry.get("find", ""))
        if not find or find in seen:
            continue
        seen.add(find)
        active = entry.get("active", entry.get("enabled", True))
        normalized["rules"].append(
            {
                "find": find,
                "replace": str(entry.get("replace", "")),
                "active": bool(active),
            }
        )
    return normalized


def load_rules(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return default_rules()
    try:
        with path.open(encoding="utf-8") as handle:
            return normalize(json.load(handle))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        # A corrupt file must never block editing; fall back to an empty rule set.
        return default_rules()


def backup_if_shrinking(path: str | Path, incoming: dict[str, Any]) -> Path | None:
    """Copy `path` aside when the incoming set drops more than one rule. Returns the copy.

    Guards a real accident: a session that loaded the file before a loader fix held an
    empty rule set in memory, and the first UI edit flushed that emptiness over a file of
    125 hand-written rules. A save that shrinks the set is cheap to snapshot and
    impossible to undo otherwise.
    """
    path = Path(path)
    if not path.exists():
        return None
    lost = len(load_rules(path)["rules"]) - len(normalize(incoming)["rules"])
    if lost <= SHRINK_BACKUP_THRESHOLD:
        return None
    backup = path.with_name(f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}.bak{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def save_rules(path: str | Path, data: dict[str, Any]) -> Path | None:
    """Write atomically so an interrupted save cannot leave a truncated file.

    Returns the path of a safety copy if this save would have discarded several rules.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    backup = backup_if_shrinking(path, data)
    payload = json.dumps(normalize(data), ensure_ascii=False, indent=2)
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            handle.write(payload)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return backup


def find_rule(data: dict[str, Any], find: str) -> dict[str, Any] | None:
    key = _rule_key(find)
    for entry in data.get("rules", []):
        if entry["find"] == key:
            return entry
    return None


def add_rule(data: dict[str, Any], find: str, replace: str, *, active: bool = True) -> bool:
    """Store a rule. Re-adding an existing search text updates its replacement."""
    key = _rule_key(find)
    if not key:
        return False
    existing = find_rule(data, key)
    if existing is not None:
        existing["replace"] = str(replace)
        existing["active"] = bool(active)
        return False
    data.setdefault("rules", []).append(
        {"find": key, "replace": str(replace), "active": bool(active)}
    )
    return True


def remove_rule(data: dict[str, Any], find: str) -> bool:
    key = _rule_key(find)
    rules = data.get("rules", [])
    for index, entry in enumerate(rules):
        if entry["find"] == key:
            del rules[index]
            return True
    return False


def set_rule_active(data: dict[str, Any], find: str, active: bool) -> bool:
    entry = find_rule(data, find)
    if entry is None:
        return False
    entry["active"] = bool(active)
    return True


def set_all_active(data: dict[str, Any], active: bool) -> None:
    for entry in data.get("rules", []):
        entry["active"] = bool(active)


def active_rules(data: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (entry["find"], entry["replace"])
        for entry in data.get("rules", [])
        if entry.get("active")
    ]


def apply_to_text(text: str, rules: list[tuple[str, str]]) -> str:
    for find, replace in rules:
        text = text.replace(find, replace)
    return text
