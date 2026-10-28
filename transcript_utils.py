from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np


# Above this reading speed a Chinese subtitle line is usually too dense to follow.
FAST_SUBTITLE_CPS = 9.0


class MediaProcessingError(RuntimeError):
    """Raised when FFmpeg or FFprobe cannot process the selected media."""


def _startupinfo() -> subprocess.STARTUPINFO | None:
    if os.name != "nt":
        return None
    info = subprocess.STARTUPINFO()
    info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return info


def _tool_path(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise MediaProcessingError(f"找不到 {name}，請先安裝 FFmpeg 並加入 PATH。")
    return path


def probe_duration(source: str | Path) -> float:
    command = [
        _tool_path("ffprobe"),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(source),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        startupinfo=_startupinfo(),
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or "無法讀取媒體長度。"
        raise MediaProcessingError(message)
    try:
        duration = float(json.loads(completed.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MediaProcessingError("媒體沒有可用的時間長度資訊。") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise MediaProcessingError("媒體長度必須大於 0 秒。")
    return duration


def build_waveform(
    source: str | Path,
    duration: float,
    target_points: int = 2400,
    sample_rate: int = 8000,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode the first audio stream and return downsampled peak amplitudes."""
    command = [
        _tool_path("ffmpeg"),
        "-v",
        "error",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        startupinfo=_startupinfo(),
    )
    assert process.stdout is not None
    samples_per_bin = max(1, math.ceil(duration * sample_rate / target_points))
    pending = np.empty(0, dtype=np.int16)
    peaks: list[np.ndarray] = []

    while True:
        chunk = process.stdout.read(256 * 1024)
        if not chunk:
            break
        samples = np.frombuffer(chunk, dtype="<i2")
        if pending.size:
            samples = np.concatenate((pending, samples))
        complete_count = samples.size // samples_per_bin * samples_per_bin
        if complete_count:
            blocks = samples[:complete_count].reshape(-1, samples_per_bin)
            block_peaks = np.max(np.abs(blocks.astype(np.int32)), axis=1)
            peaks.append(block_peaks)
        pending = samples[complete_count:].copy()

    process.stdout.close()
    stderr = process.stderr.read() if process.stderr is not None else b""
    if process.stderr is not None:
        process.stderr.close()
    return_code = process.wait()
    if return_code != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise MediaProcessingError(message or "無法從媒體讀取音軌。")
    if pending.size:
        peaks.append(np.array([np.max(np.abs(pending.astype(np.int32)))], dtype=np.int32))
    if not peaks:
        raise MediaProcessingError("媒體中沒有可顯示的音訊取樣。")

    amplitudes = np.concatenate(peaks).astype(np.float32) / 32768.0
    seconds_per_bin = duration / amplitudes.size
    times = (np.arange(amplitudes.size, dtype=np.float64) + 0.5) * seconds_per_bin
    return times, amplitudes


def extract_audio_range(
    source: str | Path,
    destination: str | Path,
    start_seconds: float,
    end_seconds: float,
) -> Path:
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise ValueError("A–B 範圍無效。")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _tool_path("ffmpeg"),
        "-y",
        "-v",
        "error",
        "-ss",
        f"{start_seconds:.6f}",
        "-i",
        str(source),
        "-t",
        f"{end_seconds - start_seconds:.6f}",
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        startupinfo=_startupinfo(),
        check=False,
    )
    if completed.returncode != 0 or not destination.exists():
        message = completed.stderr.strip() or "無法擷取選定音訊範圍。"
        raise MediaProcessingError(message)
    return destination


_CUE_TIME = re.compile(
    r"(\d{1,3}):([0-5]?\d):([0-5]?\d)[,.](\d{1,3})\s*-->\s*"
    r"(\d{1,3}):([0-5]?\d):([0-5]?\d)[,.](\d{1,3})"
)


def _cue_seconds(hours: str, minutes: str, seconds: str, millis: str) -> float:
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis.ljust(3, "0")) / 1000.0
    )


def parse_srt(text: str) -> dict[str, Any]:
    """Parse SRT cues into a Whisper-style result so existing subtitles can be re-edited.

    Deliberately lenient: tolerates a BOM, CRLF, a WEBVTT header, missing or misnumbered
    cue indices, and VTT's `.` millisecond separator. A cue spanning several lines is
    joined with a space, because the editor and the writers both work one line per cue.
    """
    lines = text.lstrip("﻿").splitlines()
    segments: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        match = _CUE_TIME.search(lines[index])
        if not match:
            index += 1
            continue
        start = _cue_seconds(*match.group(1, 2, 3, 4))
        end = _cue_seconds(*match.group(5, 6, 7, 8))
        index += 1

        cue: list[str] = []
        while index < len(lines):
            line = lines[index].strip()
            if not line or _CUE_TIME.search(line):
                break
            cue.append(line)
            index += 1
        # A cue index sitting directly above the next timestamp, with no blank line
        # between the two cues, would otherwise be swallowed as subtitle text.
        if len(cue) > 1 and cue[-1].isdigit():
            cue.pop()

        content = " ".join(cue).strip()
        if content:
            segments.append({"start": start, "end": max(end, start), "text": content})

    return {"text": " ".join(segment["text"] for segment in segments), "segments": segments}


def read_wav_mono(source: str | Path) -> np.ndarray:
    """Read a 16-bit PCM WAV as float32 samples in [-1, 1], averaging channels to mono.

    Used for the Hugging Face pipelines, which want a bare waveform array. Only the
    format `extract_audio_range` writes (16 kHz mono s16le) needs to be supported.
    """
    with wave.open(str(source), "rb") as handle:
        if handle.getsampwidth() != 2:
            raise MediaProcessingError("只支援 16-bit PCM WAV。")
        channels = handle.getnchannels()
        frames = handle.readframes(handle.getnframes())
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples


def build_result_from_chunks(
    chunks: list[dict[str, Any]],
    total_duration: float | None = None,
) -> dict[str, Any]:
    """Convert Hugging Face ASR pipeline chunks into a Whisper-style result.

    The pipeline yields {"timestamp": (start, end), "text": ...}. Whisper's sequential
    long-form decoding routinely leaves the final chunk's end open (None) and can emit a
    chunk with no timestamp at all, so both ends are repaired from the neighbouring
    chunk, falling back to the clip duration.
    """
    parsed: list[tuple[float | None, float | None, str]] = []
    for chunk in chunks or []:
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        timestamp = chunk.get("timestamp") or ()
        start = timestamp[0] if len(timestamp) > 0 else None
        end = timestamp[1] if len(timestamp) > 1 else None
        parsed.append(
            (
                float(start) if start is not None else None,
                float(end) if end is not None else None,
                text,
            )
        )
    if not parsed:
        return {"text": "", "segments": []}

    # Missing starts resolve forwards from the previous chunk's end.
    starts: list[float] = []
    previous_end = 0.0
    for start, end, _ in parsed:
        resolved = start if start is not None else previous_end
        starts.append(resolved)
        previous_end = end if end is not None else resolved

    # Missing ends resolve backwards from the next chunk's start.
    ends: list[float] = [0.0] * len(parsed)
    next_start = float(total_duration) if total_duration is not None else None
    for index in range(len(parsed) - 1, -1, -1):
        end = parsed[index][1]
        if end is None:
            end = next_start if next_start is not None else starts[index]
        ends[index] = max(float(end), starts[index])
        next_start = starts[index]

    segments = [
        {"start": starts[index], "end": ends[index], "text": text}
        for index, (_, _, text) in enumerate(parsed)
    ]
    return {"text": " ".join(segment["text"] for segment in segments), "segments": segments}


def offset_result_timestamps(result: dict[str, Any], offset_seconds: float) -> dict[str, Any]:
    shifted = copy.deepcopy(result)
    if not offset_seconds:
        return shifted
    for segment in shifted.get("segments", []):
        for key in ("start", "end"):
            if segment.get(key) is not None:
                segment[key] = float(segment[key]) + offset_seconds
        for word in segment.get("words") or []:
            for key in ("start", "end"):
                if word.get(key) is not None:
                    word[key] = float(word[key]) + offset_seconds
    return shifted


def convert_result_text(result: dict[str, Any], converter: Any) -> dict[str, Any]:
    converted = copy.deepcopy(result)
    if isinstance(converted.get("text"), str):
        converted["text"] = converter.convert(converted["text"])
    for segment in converted.get("segments", []):
        if isinstance(segment.get("text"), str):
            segment["text"] = converter.convert(segment["text"])
        for word in segment.get("words") or []:
            if isinstance(word.get("word"), str):
                word["word"] = converter.convert(word["word"])
    return converted


def set_segment_text(segment: dict[str, Any], text: str) -> bool:
    """Replace a segment's text, keeping its original leading whitespace. True if changed."""
    old_text = str(segment.get("text", ""))
    prefix = old_text[: len(old_text) - len(old_text.lstrip())]
    new_text = prefix + str(text).strip()
    if new_text == old_text:
        return False
    segment["text"] = new_text
    return True


def rebuild_result_text(result: dict[str, Any]) -> None:
    result["text"] = "".join(str(segment.get("text", "")) for segment in result.get("segments", []))


def update_result_from_editor(record: dict[str, Any], edited_rows: Any) -> int:
    """Apply edited table rows back onto segments, matching on the 段落 number.

    Row position is deliberately not used: the table may be filtered by a search box,
    in which case row 0 is not segment 0.
    """
    rows = edited_rows.to_dict("records") if hasattr(edited_rows, "to_dict") else list(edited_rows)
    segments = record["result"].get("segments", [])
    changed = 0
    for row in rows:
        try:
            index = int(row.get("段落", 0)) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(segments) and set_segment_text(segments[index], row.get("字幕", "")):
            changed += 1
    if changed:
        rebuild_result_text(record["result"])
        record["edited"] = True
    return changed


def replace_across_segments(record: dict[str, Any], search: str, replacement: str) -> int:
    """Replace every occurrence of `search` in every segment; returns segments touched."""
    if not search:
        return 0
    changed = 0
    for segment in record["result"].get("segments", []):
        text = str(segment.get("text", ""))
        if search in text and set_segment_text(segment, text.replace(search, replacement)):
            changed += 1
    if changed:
        rebuild_result_text(record["result"])
        record["edited"] = True
    return changed


def apply_replacement_rules(
    record: dict[str, Any],
    rules: list[tuple[str, str]],
) -> tuple[int, dict[str, int]]:
    """Run every (find, replace) pair over all segments in one pass.

    Returns the number of segments touched plus a per-rule hit count, so the UI can show
    which saved rules actually did something on this transcript.
    """
    if not rules:
        return 0, {}
    hits = {find: 0 for find, _ in rules}
    changed = 0
    for segment in record["result"].get("segments", []):
        text = str(segment.get("text", ""))
        updated = text
        for find, replace in rules:
            if find and find in updated:
                hits[find] += updated.count(find)
                updated = updated.replace(find, replace)
        if updated != text and set_segment_text(segment, updated):
            changed += 1
    if changed:
        rebuild_result_text(record["result"])
        record["edited"] = True
    return changed, hits


def restore_original_text(record: dict[str, Any]) -> int:
    """Undo every edit using the snapshot taken right after transcription."""
    originals = record.get("original_texts") or []
    segments = record["result"].get("segments", [])
    changed = sum(
        set_segment_text(segment, original)
        for segment, original in zip(segments, originals)
    )
    if changed:
        rebuild_result_text(record["result"])
    record["edited"] = False
    return changed


def segment_rows(
    segments: list[dict[str, Any]],
    search: str = "",
    only_dense: bool = False,
) -> list[dict[str, Any]]:
    """Build editor table rows, optionally filtered by text or by reading speed."""
    rows: list[dict[str, Any]] = []
    needle = search.strip().lower()
    for index, segment in enumerate(segments):
        text = str(segment.get("text", "")).strip()
        if needle and needle not in text.lower():
            continue
        start = float(segment.get("start", 0.0))
        span = max(float(segment.get("end", start)) - start, 0.0)
        speed = round(len(text) / span, 1) if span >= 0.05 else 0.0
        if only_dense and speed <= FAST_SUBTITLE_CPS:
            continue
        rows.append(
            {
                "段落": index + 1,
                "開始": format_hhmmss(start, include_milliseconds=True),
                "長度": round(span, 2),
                "字/秒": speed,
                "字幕": text,
            }
        )
    return rows


def paginate_rows(
    rows: list[dict[str, Any]],
    page: int,
    page_size: int = 10,
) -> tuple[list[dict[str, Any]], int, int]:
    """Return one editor page plus a clamped page number and the total page count."""
    if page_size < 1:
        raise ValueError("page_size must be at least 1")
    total_pages = max(1, (len(rows) + page_size - 1) // page_size)
    try:
        requested_page = int(page)
    except (TypeError, ValueError):
        requested_page = 1
    current_page = max(1, min(requested_page, total_pages))
    start = (current_page - 1) * page_size
    return rows[start : start + page_size], current_page, total_pages


def editor_rows_from_delta(
    visible_rows: list[dict[str, Any]],
    widget_state: Any,
) -> list[dict[str, Any]]:
    """Map Streamlit data-editor cell changes back to stable subtitle numbers."""
    if not isinstance(widget_state, dict):
        return []
    edited = widget_state.get("edited_rows")
    if not isinstance(edited, dict):
        return []

    changed_rows: list[dict[str, Any]] = []
    for raw_index, changes in edited.items():
        try:
            row_index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if (
            not 0 <= row_index < len(visible_rows)
            or not isinstance(changes, dict)
            or "字幕" not in changes
        ):
            continue
        changed_rows.append(
            {
                "段落": visible_rows[row_index].get("段落"),
                "字幕": changes.get("字幕", ""),
            }
        )
    return changed_rows


def result_for_output(record: dict[str, Any]) -> dict[str, Any]:
    """Return the result to hand to whisper's writers, dropping stale word timings.

    whisper.utils' SRT/VTT writers rebuild each line by joining segment["words"]
    whenever that key is present, so manually edited text would be silently discarded.
    The key has to go from every segment, not just the edited ones: those writers test
    only segments[0] before indexing ["words"] on all of them.
    """
    result = record["result"]
    if not record.get("edited") or not any("words" in seg for seg in result.get("segments", [])):
        return result
    stripped = copy.deepcopy(result)
    for segment in stripped.get("segments", []):
        segment.pop("words", None)
    return stripped


def range_suffix(start_seconds: float, end_seconds: float, full_duration: float) -> str:
    if start_seconds <= 0.05 and end_seconds >= full_duration - 0.05:
        return ""

    def compact(seconds: float) -> str:
        milliseconds = int(round(seconds * 1000))
        hours, remainder = divmod(milliseconds, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, millis = divmod(remainder, 1000)
        return f"{hours:02d}-{minutes:02d}-{secs:02d}.{millis:03d}"

    return f"__{compact(start_seconds)}_to_{compact(end_seconds)}"


def format_hhmmss(seconds: float, *, include_milliseconds: bool = False) -> str:
    """Format a media position as HH:MM:SS, optionally including milliseconds."""
    if not include_milliseconds:
        whole_seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(whole_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def parse_hhmmss(value: str) -> float:
    """Parse HH:MM:SS[.mmm]; hours may be greater than 23."""
    parts = value.strip().split(":")
    if len(parts) != 3 or not parts[0].isdigit() or not parts[1].isdigit():
        raise ValueError("時間必須使用 HH:MM:SS 或 HH:MM:SS.mmm 格式。")

    second_parts = parts[2].split(".")
    if (
        len(second_parts) > 2
        or not second_parts[0].isdigit()
        or (len(second_parts) == 2 and not second_parts[1].isdigit())
    ):
        raise ValueError("時間必須使用 HH:MM:SS 或 HH:MM:SS.mmm 格式。")

    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(second_parts[0])
    fraction = float(f"0.{second_parts[1]}") if len(second_parts) == 2 else 0.0
    if minutes >= 60 or seconds >= 60:
        raise ValueError("分鐘與秒數必須介於 00 到 59。")
    return float(hours * 3600 + minutes * 60 + seconds) + fraction


_SENTENCE_END_CHARS = set(".!?。！？")
_NO_LEADING_SPACE_CHARS = set(",.!?;:，。！？；：、）】」』")


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7A3
        or 0xF900 <= code <= 0xFAFF
    )


def _prefix_word(prev_char: str | None, token: str) -> str:
    """Prepend a space to `token` the way Whisper's own tokenizer would (CJK-aware).

    No space between two CJK characters or before attaching punctuation; a space is
    kept at CJK<->Latin boundaries so mixed Chinese/English text stays readable.
    """
    if prev_char is None:
        return token
    next_char = token[0]
    if next_char in _NO_LEADING_SPACE_CHARS:
        needs_space = False
    elif _is_cjk(prev_char) and _is_cjk(next_char):
        needs_space = False
    else:
        needs_space = True
    return (" " if needs_space else "") + token


def build_result_from_words(
    words: list[str],
    timestamps_ms: list[tuple[float, float]],
    max_segment_seconds: float = 12.0,
    max_segment_words: int = 40,
) -> dict[str, Any]:
    """Group word-level ASR timestamps (ms) into Whisper-style segments/result.

    Each stored word carries its leading space baked in (matching OpenAI Whisper's
    own tokenizer convention), because whisper.utils's subtitle writers reconstruct
    text via a plain "".join() over segment["words"] whenever that key is present.
    """
    segments: list[dict[str, Any]] = []
    current_words: list[dict[str, Any]] = []

    def flush() -> None:
        if not current_words:
            return
        segment_text = "".join(w["word"] for w in current_words).strip()
        segments.append(
            {
                "start": current_words[0]["start"],
                "end": current_words[-1]["end"],
                "text": segment_text,
                "words": [dict(w) for w in current_words],
            }
        )
        current_words.clear()

    for word, (start_ms, end_ms) in zip(words, timestamps_ms):
        prev_char = current_words[-1]["word"][-1] if current_words else None
        current_words.append(
            {
                "word": _prefix_word(prev_char, word),
                "start": start_ms / 1000.0,
                "end": end_ms / 1000.0,
            }
        )
        duration = current_words[-1]["end"] - current_words[0]["start"]
        if word.strip() in _SENTENCE_END_CHARS:
            flush()
        elif duration >= max_segment_seconds or len(current_words) >= max_segment_words:
            flush()
    flush()

    full_text = " ".join(segment["text"] for segment in segments)
    return {"text": full_text, "segments": segments}
