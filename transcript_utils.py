from __future__ import annotations

import copy
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


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
