from __future__ import annotations

import gc
import hashlib
import importlib
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
import torch
import whisper
from opencc import OpenCC

from transcript_utils import (
    MediaProcessingError,
    build_waveform,
    convert_result_text,
    extract_audio_range,
    format_hhmmss,
    offset_result_timestamps,
    parse_hhmmss,
    probe_duration,
    range_suffix,
)

st.set_page_config(page_title="Whisper AI 語音轉文字工具", page_icon="🎙️", layout="wide")

TEMP_DIR = Path("temp_uploads")
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")
UPLOAD_MODE = "拖曳上傳檔案（限制由系統設定）"
PATH_MODE = "直接輸入電腦中的檔案路徑（無大小限制，最快）"
OUTPUT_FORMATS = ["txt", "vtt", "srt", "tsv", "json"]
MODEL_OPTIONS = {
    "快速｜Turbo（建議先試）": "turbo",
    "平衡｜Medium": "medium",
    "品質｜Large v3": "large-v3",
    "省資源｜Small": "small",
    "極速預覽｜Base": "base",
    "最低資源｜Tiny": "tiny",
}
LANGUAGE_OPTIONS = {
    "中文": "Chinese",
    "自動偵測": None,
    "英文": "English",
    "日文": "Japanese",
    "韓文": "Korean",
}
MIME_TYPES = {
    "txt": "text/plain",
    "vtt": "text/vtt",
    "srt": "application/x-subrip",
    "tsv": "text/tab-separated-values",
    "json": "application/json",
}
CHINESE_OUTPUT_OPTIONS = {
    "台灣繁體（含台灣常用詞，建議）": "s2twp",
    "台灣繁體（只轉字形）": "s2tw",
    "標準繁體": "s2t",
    "保留 Whisper 原文": None,
}


@st.cache_resource(show_spinner=False, max_entries=1)
def load_whisper_model(model_size: str, device: str):
    MODEL_DIR.mkdir(exist_ok=True)
    return whisper.load_model(model_size, device=device, download_root=str(MODEL_DIR))


@st.cache_resource(show_spinner=False)
def load_chinese_converter(config: str):
    return OpenCC(config)


@st.cache_data(show_spinner=False, max_entries=8)
def analyze_media(path: str, file_size: int, modified_ns: int):
    del file_size, modified_ns
    duration = probe_duration(path)
    times, amplitudes = build_waveform(path, duration, target_points=1600)
    return duration, times, amplitudes


def save_uploaded_file(uploaded_file) -> Path:
    TEMP_DIR.mkdir(exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    upload_token = (
        getattr(uploaded_file, "file_id", None),
        safe_name,
        getattr(uploaded_file, "size", None),
    )
    upload_id = hashlib.sha1(repr(upload_token).encode("utf-8")).hexdigest()[:10]
    destination = TEMP_DIR / f"{upload_id}_{safe_name}"
    if st.session_state.get("saved_upload_token") != upload_token or not destination.exists():
        uploaded_file.seek(0)
        with destination.open("wb") as output_file:
            shutil.copyfileobj(uploaded_file, output_file)
        saved_modified_ns = st.session_state.get("saved_upload_modified_ns")
        if st.session_state.get("saved_upload_token") == upload_token and saved_modified_ns:
            os.utime(destination, ns=(saved_modified_ns, saved_modified_ns))
        else:
            st.session_state.saved_upload_modified_ns = destination.stat().st_mtime_ns
        st.session_state.saved_upload_token = upload_token
    return destination


def cleanup_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        st.warning(f"暫存檔清理失敗：{exc}")


def media_state_id(path: Path) -> str:
    stat = path.stat()
    value = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def selection_bounds(event, duration: float) -> tuple[float, float] | None:
    if not event:
        return None
    selection = event.get("selection", {})
    boxes = selection.get("box") or []
    if boxes:
        box = boxes[-1]
        x_values = box.get("x") or box.get("xrange")
        if isinstance(x_values, (list, tuple)) and len(x_values) >= 2:
            start, end = float(x_values[0]), float(x_values[1])
        elif box.get("x0") is not None and box.get("x1") is not None:
            start, end = float(box["x0"]), float(box["x1"])
        else:
            start = end = 0.0
        if end > start:
            return max(0.0, start), min(duration, end)

    points = selection.get("points") or []
    x_values = [float(point["x"]) for point in points if point.get("x") is not None]
    if len(x_values) >= 2:
        return max(0.0, min(x_values)), min(duration, max(x_values))
    return None


def timeline_ticks(duration: float, target_count: int = 8) -> tuple[list[float], list[str]]:
    desired_step = max(1.0, duration / target_count)
    candidates = [
        1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1200,
        1800, 3600, 7200, 10800, 21600, 43200,
    ]
    step = next((value for value in candidates if value >= desired_step), candidates[-1])
    tick_values = [float(value) for value in range(0, int(duration) + 1, step)]
    if not tick_values:
        tick_values = [0.0]
    if duration - tick_values[-1] > step * 0.35:
        tick_values.append(float(duration))
    return tick_values, [format_hhmmss(value) for value in tick_values]


def waveform_figure(times, amplitudes, selected_range: tuple[float, float], duration: float):
    hover_times = [format_hhmmss(float(value)) for value in times]
    tick_values, tick_labels = timeline_ticks(duration)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=times,
            y=amplitudes,
            mode="lines",
            fill="tozeroy",
            line={"color": "#4f8bf9", "width": 1},
            customdata=hover_times,
            hovertemplate="時間 %{customdata}<extra></extra>",
        )
    )
    figure.add_vrect(
        x0=selected_range[0],
        x1=selected_range[1],
        fillcolor="#ffb347",
        opacity=0.18,
        line_width=0,
        line_color="#ff9800",
    )
    if selected_range[0] > 0:
        figure.add_vrect(x0=0, x1=selected_range[0], fillcolor="#111827", opacity=0.42, line_width=0)
    if selected_range[1] < duration:
        figure.add_vrect(x0=selected_range[1], x1=duration, fillcolor="#111827", opacity=0.42, line_width=0)
    figure.add_vline(
        x=selected_range[0],
        line_color="#22c55e",
        line_width=3,
        annotation_text=f"A  {format_hhmmss(selected_range[0])}",
        annotation_position="top right",
    )
    figure.add_vline(
        x=selected_range[1],
        line_color="#ef4444",
        line_width=3,
        annotation_text=f"B  {format_hhmmss(selected_range[1])}",
        annotation_position="top left",
    )
    figure.update_layout(
        height=340,
        margin={"l": 10, "r": 10, "t": 45, "b": 55},
        dragmode="select",
        selectdirection="h",
        showlegend=False,
        xaxis={
            "title": "原影片時間（HH:MM:SS）",
            "fixedrange": True,
            "tickmode": "array",
            "tickvals": tick_values,
            "ticktext": tick_labels,
        },
        yaxis={"title": "音量", "range": [0, 1.02], "fixedrange": True},
    )
    return figure


def reset_selected_range(
    range_key: str,
    start_time_key: str,
    end_time_key: str,
    selection_token_key: str,
    plot_version_key: str,
    duration: float,
) -> None:
    st.session_state[range_key] = (0.0, float(duration))
    st.session_state[start_time_key] = format_hhmmss(0.0, include_milliseconds=True)
    st.session_state[end_time_key] = format_hhmmss(duration, include_milliseconds=True)
    st.session_state[selection_token_key] = None
    st.session_state[plot_version_key] = st.session_state.get(plot_version_key, 0) + 1


def release_cached_model() -> None:
    load_whisper_model.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    st.session_state["model_release_notice"] = True


def result_state_key(source_id: str) -> str:
    return f"transcription_result_{source_id}"


def preview_state_key(source_id: str) -> str:
    return f"audio_preview_{source_id}"


def update_result_from_editor(record: dict, edited_rows) -> None:
    if hasattr(edited_rows, "to_dict"):
        rows = edited_rows.to_dict("records")
    else:
        rows = list(edited_rows)

    segments = record["result"].get("segments", [])
    for index, row in enumerate(rows):
        if index >= len(segments):
            break
        old_text = str(segments[index].get("text", ""))
        prefix_length = len(old_text) - len(old_text.lstrip())
        prefix = old_text[:prefix_length]
        segments[index]["text"] = prefix + str(row.get("字幕", "")).strip()
    record["result"]["text"] = "".join(str(segment.get("text", "")) for segment in segments)
    record["edited"] = True


def write_result_files(
    record: dict,
    input_path: Path,
    source_id: str,
    output_formats: list[str],
) -> list[tuple[str, Path]]:
    from whisper.utils import get_writer

    output_directory = OUTPUT_DIR / f"{input_path.stem}_{source_id}"
    output_directory.mkdir(parents=True, exist_ok=True)
    suffix = range_suffix(record["start"], record["end"], record["source_duration"])
    output_stem = f"{input_path.stem}{suffix}"
    output_anchor = output_directory / f"{output_stem}{input_path.suffix}"
    files: list[tuple[str, Path]] = []
    for extension in output_formats:
        writer = get_writer(extension, str(output_directory))
        writer(record["result"], str(output_anchor))
        output_file = output_directory / f"{output_stem}.{extension}"
        if output_file.exists():
            files.append((extension, output_file))
    record["output_directory"] = str(output_directory.absolute())
    return files


@contextmanager
def show_whisper_progress(progress_bar, status_text):
    transcribe_module = importlib.import_module("whisper.transcribe")
    tqdm_target = transcribe_module.tqdm

    if hasattr(tqdm_target, "tqdm"):
        original_tqdm = tqdm_target.tqdm

        def set_tqdm(value):
            tqdm_target.tqdm = value
    else:
        original_tqdm = tqdm_target

        def set_tqdm(value):
            transcribe_module.tqdm = value

    class StreamlitTqdm:
        def __init__(self, *args, **kwargs):
            self.disable = kwargs.get("disable", False)
            self.total = int(kwargs.get("total") or 0)
            self.n = int(kwargs.get("initial") or 0)
            self.last_percent = -1
            self.last_render_time = 0.0
            if not self.disable:
                progress_bar.progress(0)
                status_text.text("正在進行語音識別... 0%")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            if not self.disable and exc_type is None:
                progress_bar.progress(100)
                status_text.text("語音識別完成，正在產生檔案...")
            return False

        def update(self, amount=1):
            if self.disable:
                return
            self.n += int(amount or 0)
            if self.total <= 0:
                return
            percent = max(0, min(100, int(self.n / self.total * 100)))
            now = time.monotonic()
            should_render = percent != self.last_percent and (
                now - self.last_render_time >= 0.2 or percent in (0, 100)
            )
            if should_render:
                progress_bar.progress(percent)
                status_text.text(f"正在進行語音識別... {percent}%")
                self.last_percent = percent
                self.last_render_time = now

        def close(self):
            pass

    set_tqdm(StreamlitTqdm)
    try:
        yield
    finally:
        set_tqdm(original_tqdm)


def main():
    st.title("🎙️ Whisper AI 語音轉文字工具")
    st.caption("選擇媒體、圈選片段，再輸出保留原影片時間碼的字幕。所有處理都在本機完成。")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device_label = f"GPU｜{torch.cuda.get_device_name(0)}"
    else:
        device_label = "CPU｜未偵測到 CUDA"

    st.sidebar.header("轉譯設定")
    st.sidebar.caption(device_label)
    model_label = st.sidebar.selectbox(
        "速度與品質",
        list(MODEL_OPTIONS),
        index=0,
        help="Turbo 適合先快速產生字幕；重要成品可再用 Large v3 比對。",
    )
    model_size = MODEL_OPTIONS[model_label]
    language_label = st.sidebar.selectbox("語音語言", list(LANGUAGE_OPTIONS), index=0)
    language = LANGUAGE_OPTIONS[language_label]
    chinese_output_label = st.sidebar.selectbox(
        "中文字體輸出",
        list(CHINESE_OUTPUT_OPTIONS),
        index=0,
        help="OpenCC 會在辨識後轉換字形與台灣常用詞，不會改動字幕時間碼。",
    )
    output_format = st.sidebar.multiselect(
        "輸出格式", OUTPUT_FORMATS, default=["srt", "txt"]
    )
    with st.sidebar.expander("進階設定"):
        initial_prompt = st.text_area(
            "專有名詞提示",
            placeholder="可輸入人名、公司名或術語，以頓號或換行分隔。",
            height=90,
        )
        cleanup_temp_file = st.checkbox("轉換後清除上傳的原始暫存檔", value=False)
        st.caption(f"模型位置：{MODEL_DIR.absolute()}")
        st.button("釋放 GPU 模型", on_click=release_cached_model, use_container_width=True)
    if st.session_state.pop("model_release_notice", False):
        st.sidebar.success("已清除模型快取。")

    st.header("1. 選擇媒體")
    input_mode = st.radio(
        "輸入方式",
        [UPLOAD_MODE, PATH_MODE],
        horizontal=True,
        label_visibility="collapsed",
    )
    input_path: Path | None = None
    uploaded_temp_file = False

    if input_mode == UPLOAD_MODE:
        uploaded_file = st.file_uploader(
            "選擇影片或音檔", type=["mp4", "mkv", "mov", "avi", "mp3", "wav", "m4a"]
        )
        if uploaded_file is not None:
            input_path = save_uploaded_file(uploaded_file)
            uploaded_temp_file = True
            st.info(f"已選擇檔案：{uploaded_file.name}")
    else:
        path_input = st.text_input(
            "請貼上檔案完整路徑（例如 D:\\Videos\\1222.mp4）", ""
        ).strip().strip('"')
        if path_input:
            candidate = Path(path_input)
            if candidate.exists() and candidate.is_file():
                input_path = candidate
                st.success(f"✅ 偵測到檔案：{candidate.name}")
            else:
                st.error("❌ 找不到該檔案，請檢查完整路徑與副檔名。")

    if input_path is None:
        return

    try:
        source_stat = input_path.stat()
        with st.spinner("正在分析音軌並建立波形..."):
            duration, times, amplitudes = analyze_media(
                str(input_path.resolve()), source_stat.st_size, source_stat.st_mtime_ns
            )
    except (OSError, MediaProcessingError) as exc:
        st.error(f"無法建立音訊波形：{exc}")
        return

    source_id = media_state_id(input_path)
    range_key = f"transcript_range_{source_id}"
    start_time_key = f"transcript_start_time_{source_id}"
    end_time_key = f"transcript_end_time_{source_id}"
    selection_key = f"waveform_selection_{source_id}"
    selection_token_key = f"waveform_selection_token_{source_id}"
    plot_version_key = f"waveform_version_{source_id}"
    if range_key not in st.session_state:
        st.session_state[range_key] = (0.0, float(duration))
    if start_time_key not in st.session_state:
        st.session_state[start_time_key] = format_hhmmss(0.0, include_milliseconds=True)
    if end_time_key not in st.session_state:
        st.session_state[end_time_key] = format_hhmmss(duration, include_milliseconds=True)
    if plot_version_key not in st.session_state:
        st.session_state[plot_version_key] = 0
    selected_range = tuple(st.session_state[range_key])

    st.header("2. 選擇片段")
    st.caption(
        "在波形上由左往右拖曳即可套用範圍；也可以直接輸入精確到毫秒的時間碼。"
    )
    event = st.plotly_chart(
        waveform_figure(times, amplitudes, selected_range, duration),
        use_container_width=True,
        key=f"{selection_key}_{st.session_state[plot_version_key]}",
        on_select="rerun",
        selection_mode="box",
        config={"displaylogo": False, "displayModeBar": False, "scrollZoom": False},
    )
    bounds = selection_bounds(event, duration)
    if bounds and bounds[1] - bounds[0] >= 0.1:
        normalized = (
            float(max(0, round(bounds[0], 3))),
            float(min(duration, round(bounds[1], 3))),
        )
        if st.session_state.get(selection_token_key) != normalized:
            st.session_state[selection_token_key] = normalized
            st.session_state[range_key] = normalized
            st.session_state[start_time_key] = format_hhmmss(
                normalized[0], include_milliseconds=True
            )
            st.session_state[end_time_key] = format_hhmmss(
                normalized[1], include_milliseconds=True
            )
            st.session_state[plot_version_key] += 1
            st.rerun()

    with st.form(f"time_range_form_{source_id}", border=True):
        start_col, end_col, update_col = st.columns([2, 2, 1])
        with start_col:
            start_text = st.text_input(
                "開始 A（HH:MM:SS.mmm）",
                key=start_time_key,
                placeholder="00:00:00.000",
            )
        with end_col:
            end_text = st.text_input(
                "結束 B（HH:MM:SS.mmm）",
                key=end_time_key,
                placeholder="00:00:00.000",
            )
        with update_col:
            st.write("")
            update_range = st.form_submit_button("套用時間", use_container_width=True)

    if update_range:
        try:
            entered_start = parse_hhmmss(start_text)
            entered_end = min(parse_hhmmss(end_text), float(duration))
            if entered_start >= entered_end:
                raise ValueError("結束 B 必須晚於開始 A。")
            st.session_state[range_key] = (entered_start, entered_end)
            st.session_state[selection_token_key] = None
            st.session_state[plot_version_key] += 1
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    selected_start, selected_end = tuple(st.session_state[range_key])
    duration_selected = selected_end - selected_start
    reset_col, preview_col = st.columns(2)
    with reset_col:
        st.button(
            "重設為全片",
            key=f"reset_{source_id}",
            use_container_width=True,
            on_click=reset_selected_range,
            args=(
                range_key,
                start_time_key,
                end_time_key,
                selection_token_key,
                plot_version_key,
                duration,
            ),
        )
    with preview_col:
        make_preview = st.button(
            "試聽選取起點 30 秒",
            key=f"preview_{source_id}",
            disabled=duration_selected < 0.5,
            use_container_width=True,
        )

    if make_preview:
        preview_file = TEMP_DIR / f"preview_{uuid.uuid4().hex}.wav"
        try:
            preview_end = min(selected_end, selected_start + 30.0)
            with st.spinner("正在建立試聽片段..."):
                extract_audio_range(input_path, preview_file, selected_start, preview_end)
            st.session_state[preview_state_key(source_id)] = {
                "range": (selected_start, selected_end),
                "data": preview_file.read_bytes(),
                "seconds": preview_end - selected_start,
            }
        except (OSError, MediaProcessingError, ValueError) as exc:
            st.error(f"無法建立試聽片段：{exc}")
        finally:
            cleanup_file(preview_file)

    preview_record = st.session_state.get(preview_state_key(source_id))
    if preview_record and tuple(preview_record["range"]) == (selected_start, selected_end):
        st.audio(preview_record["data"], format="audio/wav")
        st.caption(f"試聽長度：{preview_record['seconds']:.1f} 秒")

    range_columns = st.columns(3)
    range_columns[0].metric(
        "開始",
        format_hhmmss(selected_start, include_milliseconds=True),
    )
    range_columns[1].metric(
        "結束",
        format_hhmmss(selected_end, include_milliseconds=True),
    )
    range_columns[2].metric(
        "選取長度",
        format_hhmmss(duration_selected, include_milliseconds=True),
    )
    if duration_selected < 0.5:
        st.warning("選取範圍至少需要 0.5 秒。")

    st.header("3. 轉譯與輸出")
    result_key = result_state_key(source_id)
    run_transcription = st.button(
        f"開始轉譯｜{model_size}",
        type="primary",
        disabled=duration_selected < 0.5 or not output_format,
        use_container_width=True,
    )
    if not output_format:
        st.warning("請至少選擇一種輸出格式。")

    if run_transcription:
        extracted_audio: Path | None = None
        total_started = time.perf_counter()
        try:
            TEMP_DIR.mkdir(exist_ok=True)
            extracted_audio = TEMP_DIR / f"segment_{uuid.uuid4().hex}.wav"

            phase_started = time.perf_counter()
            with st.spinner("1/3 正在準備音訊..."):
                extract_audio_range(input_path, extracted_audio, selected_start, selected_end)
            extraction_seconds = time.perf_counter() - phase_started

            phase_started = time.perf_counter()
            with st.spinner(f"2/3 正在載入 {model_size} 模型..."):
                model = load_whisper_model(model_size, device)
            model_load_seconds = time.perf_counter() - phase_started

            phase_started = time.perf_counter()
            status_text = st.empty()
            progress_bar = st.progress(0)
            status_text.text("3/3 正在進行語音識別...")
            transcribe_options = {
                "language": language,
                "verbose": False,
                "fp16": device == "cuda",
            }
            if initial_prompt.strip():
                transcribe_options["initial_prompt"] = initial_prompt.strip()
            if model_size == "turbo":
                transcribe_options["temperature"] = 0.0
            with show_whisper_progress(progress_bar, status_text):
                result = model.transcribe(str(extracted_audio), **transcribe_options)
            inference_seconds = time.perf_counter() - phase_started

            phase_started = time.perf_counter()
            result = offset_result_timestamps(result, selected_start)
            conversion_config = CHINESE_OUTPUT_OPTIONS[chinese_output_label]
            if conversion_config:
                result = convert_result_text(
                    result,
                    load_chinese_converter(conversion_config),
                )
            postprocess_seconds = time.perf_counter() - phase_started
            total_seconds = time.perf_counter() - total_started
            st.session_state[result_key] = {
                "id": uuid.uuid4().hex,
                "result": result,
                "start": selected_start,
                "end": selected_end,
                "source_duration": duration,
                "model": model_size,
                "language": language_label,
                "conversion": chinese_output_label,
                "initial_prompt": initial_prompt.strip(),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "edited": False,
                "timings": {
                    "extraction": extraction_seconds,
                    "model_load": model_load_seconds,
                    "inference": inference_seconds,
                    "postprocess": postprocess_seconds,
                    "total": total_seconds,
                },
            }
            st.success(f"轉譯完成，共耗時 {total_seconds:.1f} 秒。")
        except Exception as exc:
            st.error(f"轉譯失敗：{exc}")
        finally:
            if extracted_audio is not None:
                cleanup_file(extracted_audio)
            if uploaded_temp_file and cleanup_temp_file:
                cleanup_file(input_path)

    record = st.session_state.get(result_key)
    if record is None:
        return

    st.divider()
    st.subheader("轉譯結果")
    if (
        record["model"] != model_size
        or record["language"] != language_label
        or record["conversion"] != chinese_output_label
        or record["initial_prompt"] != initial_prompt.strip()
        or record["start"] != selected_start
        or record["end"] != selected_end
    ):
        st.info("目前顯示的是上一次結果；設定或片段已變更，按上方按鈕才會重新轉譯。")

    timings = record["timings"]
    record_duration = record["end"] - record["start"]
    realtime = record_duration / max(timings["inference"], 0.001)
    result_metrics = st.columns(4)
    result_metrics[0].metric("總耗時", f"{timings['total']:.1f} 秒")
    result_metrics[1].metric("模型推論", f"{timings['inference']:.1f} 秒")
    result_metrics[2].metric("處理速度", f"{realtime:.1f}× realtime")
    result_metrics[3].metric("字幕段落", len(record["result"].get("segments", [])))
    st.caption(
        f"{record['created_at']}｜模型 {record['model']}｜語言 {record['language']}｜"
        f"音訊 {timings['extraction']:.2f}s｜載入 {timings['model_load']:.2f}s｜"
        f"後處理 {timings['postprocess']:.2f}s"
    )

    edit_tab, text_tab, download_tab = st.tabs(["字幕編輯", "純文字", "下載"])
    with edit_tab:
        segments = record["result"].get("segments", [])
        if not segments:
            st.info("這次結果沒有可編輯的字幕段落。")
        else:
            editor_rows = [
                {
                    "段落": index + 1,
                    "開始": format_hhmmss(
                        float(segment.get("start", 0.0)),
                        include_milliseconds=True,
                    ),
                    "結束": format_hhmmss(
                        float(segment.get("end", 0.0)),
                        include_milliseconds=True,
                    ),
                    "字幕": str(segment.get("text", "")).strip(),
                }
                for index, segment in enumerate(segments)
            ]
            with st.form(f"subtitle_editor_{record['id']}"):
                edited_rows = st.data_editor(
                    editor_rows,
                    hide_index=True,
                    use_container_width=True,
                    height=420,
                    disabled=["段落", "開始", "結束"],
                    column_config={
                        "段落": st.column_config.NumberColumn(width="small"),
                        "開始": st.column_config.TextColumn(width="medium"),
                        "結束": st.column_config.TextColumn(width="medium"),
                        "字幕": st.column_config.TextColumn(width="large"),
                    },
                )
                save_edits = st.form_submit_button("儲存字幕修改", type="primary")
            if save_edits:
                update_result_from_editor(record, edited_rows)
                st.session_state[result_key] = record
                st.success("字幕修改已保存；下載檔會使用最新內容。")

    with text_tab:
        st.text_area(
            "完整文字",
            value=str(record["result"].get("text", "")).strip(),
            height=360,
            disabled=True,
        )

    with download_tab:
        if not output_format:
            st.warning("請先在左側選擇至少一種輸出格式。")
        else:
            try:
                output_files = write_result_files(
                    record,
                    input_path,
                    source_id,
                    output_format,
                )
                download_columns = st.columns(len(output_files))
                for index, (extension, output_file) in enumerate(output_files):
                    download_columns[index].download_button(
                        label=f"下載 .{extension}",
                        data=output_file.read_bytes(),
                        file_name=output_file.name,
                        mime=MIME_TYPES.get(extension, "text/plain"),
                        use_container_width=True,
                    )
                st.caption(
                    f"檔案位置：{record['output_directory']}｜字幕時間碼對應原始媒體。"
                )
            except Exception as exc:
                st.error(f"無法產生輸出檔：{exc}")


if __name__ == "__main__":
    main()
