"""Persistent glossary of proper nouns fed to the ASR engine as an initial prompt.

Kept free of Streamlit imports so the storage rules stay unit-testable.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

GLOSSARY_VERSION = 1
# Losing more than one term in a single save is not a normal per-row delete; it usually
# means a stale in-memory set is being flushed over a larger file. Keep a copy first.
SHRINK_BACKUP_THRESHOLD = 1
DEFAULT_PREFIX = "以下是一段中文為主、中英文夾雜的內容，請以原本說出的語言逐字記錄，不要翻譯。"
# Whisper truncates a long prompt from the left, so terms are appended last on purpose.
PROMPT_SOFT_LIMIT = 200

_TERM_SEPARATORS = re.compile(r"[、,，;；/｜|\n\r\t]+")


def default_glossary() -> dict[str, Any]:
    return {"version": GLOSSARY_VERSION, "prefix": DEFAULT_PREFIX, "terms": []}


def _term_key(text: str) -> str:
    """Dedupe key: case-insensitive so "archicad" cannot shadow "ArchiCAD"."""
    return text.strip().casefold()


def split_terms(raw: str) -> list[str]:
    """Split a pasted string on 、 , ; / | and newlines, keeping spaces inside a term."""
    if not raw:
        return []
    seen: set[str] = set()
    terms: list[str] = []
    for chunk in _TERM_SEPARATORS.split(str(raw)):
        term = " ".join(chunk.split())
        key = _term_key(term)
        if term and key not in seen:
            seen.add(key)
            terms.append(term)
    return terms


def normalize(data: Any) -> dict[str, Any]:
    """Coerce whatever was on disk into a valid glossary, dropping unusable entries.

    Accepts a bare list of terms as well as the wrapped object this module writes, and
    treats `enabled` as a synonym for `active`, so a hand-written file is not silently
    discarded for having the wrong outer shape.
    """
    normalized = default_glossary()
    if isinstance(data, list):
        entries: Any = data
    elif isinstance(data, dict):
        entries = data.get("terms") or []
        prefix = data.get("prefix")
        if isinstance(prefix, str):
            normalized["prefix"] = prefix.strip()
    else:
        return normalized

    seen: set[str] = set()
    for entry in entries:
        if isinstance(entry, str):
            entry = {"text": entry, "active": True}
        if not isinstance(entry, dict):
            continue
        text = " ".join(str(entry.get("text", "")).split())
        key = _term_key(text)
        if not text or key in seen:
            continue
        seen.add(key)
        active = entry.get("active", entry.get("enabled", True))
        normalized["terms"].append({"text": text, "active": bool(active)})
    return normalized


def load_glossary(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return default_glossary()
    try:
        with path.open(encoding="utf-8") as handle:
            return normalize(json.load(handle))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        # A corrupt file must never block transcription; fall back to an empty glossary.
        return default_glossary()


def backup_if_shrinking(path: str | Path, incoming: dict[str, Any]) -> Path | None:
    """Copy `path` aside when the incoming glossary drops more than one term."""
    path = Path(path)
    if not path.exists():
        return None
    lost = len(load_glossary(path)["terms"]) - len(normalize(incoming)["terms"])
    if lost <= SHRINK_BACKUP_THRESHOLD:
        return None
    backup = path.with_name(f"{path.stem}.{time.strftime('%Y%m%d-%H%M%S')}.bak{path.suffix}")
    shutil.copy2(path, backup)
    return backup


def save_glossary(path: str | Path, data: dict[str, Any]) -> Path | None:
    """Write atomically so an interrupted save cannot leave a truncated glossary.

    Returns the path of a safety copy if this save would have discarded several terms.
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


def find_term(data: dict[str, Any], text: str) -> dict[str, Any] | None:
    key = _term_key(text)
    for entry in data.get("terms", []):
        if _term_key(entry["text"]) == key:
            return entry
    return None


def add_terms(data: dict[str, Any], raw: str, *, active: bool = True) -> list[str]:
    """Add every term in `raw`; returns only the ones that were not already stored."""
    added: list[str] = []
    for term in split_terms(raw):
        existing = find_term(data, term)
        if existing is not None:
            existing["active"] = existing["active"] or active
            continue
        data.setdefault("terms", []).append({"text": term, "active": active})
        added.append(term)
    return added


def remove_term(data: dict[str, Any], text: str) -> bool:
    key = _term_key(text)
    terms = data.get("terms", [])
    for index, entry in enumerate(terms):
        if _term_key(entry["text"]) == key:
            del terms[index]
            return True
    return False


def set_term_active(data: dict[str, Any], text: str, active: bool) -> bool:
    entry = find_term(data, text)
    if entry is None:
        return False
    entry["active"] = bool(active)
    return True


def set_all_active(data: dict[str, Any], active: bool) -> None:
    for entry in data.get("terms", []):
        entry["active"] = bool(active)


def active_terms(data: dict[str, Any]) -> list[str]:
    return [entry["text"] for entry in data.get("terms", []) if entry.get("active")]


def inactive_terms(data: dict[str, Any]) -> list[str]:
    return [entry["text"] for entry in data.get("terms", []) if not entry.get("active")]


def build_initial_prompt(data: dict[str, Any]) -> str:
    prefix = str(data.get("prefix", "")).strip()
    terms = active_terms(data)
    parts = [part for part in (prefix, "、".join(terms)) if part]
    return " ".join(parts)
