from __future__ import annotations

import hashlib
import importlib
import math
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


@st.cache_resource(show_spinner=False)
def load_whisper_model(model_size: str, device: str):
    MODEL_DIR.mkdir(exist_ok=True)
    return whisper.load_model(model_size, device=device, download_root=str(MODEL_DIR))


@st.cache_resource(show_spinner=False)
def load_chinese_converter(config: str):
    return OpenCC(config)


@st.cache_resource(show_spinner=False)
def analyze_media(path: str, file_size: int, modified_ns: int):
    del file_size, modified_ns
    duration = probe_duration(path)
    times, amplitudes = build_waveform(path, duration, target_points=1600)
    return duration, times, amplitudes


def save_uploaded_file(uploaded_file) -> Path:
    TEMP_DIR.mkdir(exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    destination = TEMP_DIR / safe_name
    upload_token = (
        getattr(uploaded_file, "file_id", None),
        safe_name,
        getattr(uploaded_file, "size", None),
    )
    if st.session_state.get("saved_upload_token") != upload_token or not destination.exists():
        uploaded_file.seek(0)
        with destination.open("wb") as output_file:
            shutil.copyfileobj(uploaded_file, output_file)
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


def reset_pending_range(
    pending_range_key: str,
    start_time_key: str,
    end_time_key: str,
    selection_token_key: str,
    plot_version_key: str,
    duration: float,
) -> None:
    st.session_state[pending_range_key] = (0.0, float(duration))
    st.session_state[start_time_key] = format_hhmmss(0.0)
    st.session_state[end_time_key] = format_hhmmss(duration)
    st.session_state[selection_token_key] = None
    st.session_state[plot_version_key] = st.session_state.get(plot_version_key, 0) + 1


def apply_pending_range(range_key: str, pending_range_key: str) -> None:
    st.session_state[range_key] = tuple(st.session_state[pending_range_key])


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
    st.markdown("先在音訊波形選取 A–B 範圍，再轉錄為保留原影片時間碼的字幕。")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        st.success(f"✅ 偵測到 GPU：{torch.cuda.get_device_name(0)}，將使用 CUDA 加速。")
    else:
        st.warning("⚠️ 未偵測到 CUDA，將使用 CPU 運行（速度較慢）。")

    st.sidebar.header("⚙️ 設定參數")
    st.sidebar.caption(f"模型儲存位置：{MODEL_DIR.absolute()}")
    model_size = st.sidebar.selectbox(
        "選擇模型大小（建議 large-v3）",
        ["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        index=5,
    )
    language = st.sidebar.text_input("語言設定（例如 Chinese）", value="Chinese")
    chinese_output_label = st.sidebar.selectbox(
        "中文字體輸出",
        list(CHINESE_OUTPUT_OPTIONS),
        index=0,
        help="OpenCC 會在辨識後轉換字形與台灣常用詞，不會改動字幕時間碼。",
    )
    output_format = st.sidebar.multiselect(
        "輸出格式", OUTPUT_FORMATS, default=["srt", "txt"]
    )
    cleanup_temp_file = st.sidebar.checkbox("轉換後清除上傳的原始暫存檔", value=False)

    input_mode = st.radio("選擇輸入方式：", [UPLOAD_MODE, PATH_MODE])
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
    pending_range_key = f"pending_transcript_range_{source_id}"
    start_time_key = f"transcript_start_time_{source_id}"
    end_time_key = f"transcript_end_time_{source_id}"
    selection_key = f"waveform_selection_{source_id}"
    selection_token_key = f"waveform_selection_token_{source_id}"
    plot_version_key = f"waveform_version_{source_id}"
    if range_key not in st.session_state:
        st.session_state[range_key] = (0.0, float(duration))
    if pending_range_key not in st.session_state:
        st.session_state[pending_range_key] = tuple(st.session_state[range_key])
    if start_time_key not in st.session_state:
        st.session_state[start_time_key] = format_hhmmss(0.0)
    if end_time_key not in st.session_state:
        st.session_state[end_time_key] = format_hhmmss(duration)
    if plot_version_key not in st.session_state:
        st.session_state[plot_version_key] = 0
    pending_range = tuple(st.session_state[pending_range_key])

    st.subheader("🔊 音訊波形與轉錄範圍")
    st.info(
        "在波形上由左往右拖曳：綠線是開始 A，紅線是結束 B。"
        "拖曳只會更新暫存範圍，不會擷取音訊或執行 Whisper；確認時間後再按「確認選取」。"
    )
    event = st.plotly_chart(
        waveform_figure(times, amplitudes, pending_range, duration),
        use_container_width=True,
        key=f"{selection_key}_{st.session_state[plot_version_key]}",
        on_select="rerun",
        selection_mode="box",
        config={"displaylogo": False, "displayModeBar": False, "scrollZoom": False},
    )
    bounds = selection_bounds(event, duration)
    if bounds and bounds[1] - bounds[0] >= 0.1:
        normalized = (
            float(max(0, math.floor(bounds[0]))),
            float(min(duration, math.ceil(bounds[1]))),
        )
        if st.session_state.get(selection_token_key) != normalized:
            st.session_state[selection_token_key] = normalized
            st.session_state[pending_range_key] = normalized
            st.session_state[start_time_key] = format_hhmmss(normalized[0])
            st.session_state[end_time_key] = format_hhmmss(normalized[1])
            pending_range = normalized

    with st.form(f"time_range_form_{source_id}", border=True):
        start_col, end_col, update_col = st.columns([2, 2, 1])
        with start_col:
            start_text = st.text_input(
                "暫存開始 A（HH:MM:SS）",
                key=start_time_key,
                placeholder="00:00:00",
            )
        with end_col:
            end_text = st.text_input(
                "暫存結束 B（HH:MM:SS）",
                key=end_time_key,
                placeholder="00:00:00",
            )
        with update_col:
            st.write("")
            update_pending = st.form_submit_button("更新時間", use_container_width=True)

    if update_pending:
        try:
            entered_start = parse_hhmmss(start_text)
            entered_end = min(parse_hhmmss(end_text), float(duration))
            if entered_start >= entered_end:
                raise ValueError("結束 B 必須晚於開始 A。")
            pending_range = (entered_start, entered_end)
            st.session_state[pending_range_key] = pending_range
            st.session_state[selection_token_key] = None
            st.session_state[plot_version_key] += 1
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    reset_col, apply_col = st.columns(2)
    with reset_col:
        st.button(
            "重設為全片",
            key=f"reset_{source_id}",
            use_container_width=True,
            on_click=reset_pending_range,
            args=(
                pending_range_key,
                start_time_key,
                end_time_key,
                selection_token_key,
                plot_version_key,
                duration,
            ),
        )
    with apply_col:
        st.button(
            "✅ 確認選取",
            key=f"apply_{source_id}",
            type="primary",
            use_container_width=True,
            on_click=apply_pending_range,
            args=(range_key, pending_range_key),
        )

    selected_start, selected_end = tuple(st.session_state[range_key])
    duration_selected = selected_end - selected_start
    if pending_range != (selected_start, selected_end):
        st.warning(
            f"暫存選取（尚未套用）：A {format_hhmmss(pending_range[0])} → "
            f"B {format_hhmmss(pending_range[1])}"
        )
    st.info(
        f"準備轉錄：A {format_hhmmss(selected_start)} → B {format_hhmmss(selected_end)}　"
        f"｜選取長度 {format_hhmmss(duration_selected)}　｜影片長度 {format_hhmmss(duration)}"
    )
    if duration_selected < 0.5:
        st.warning("選取範圍至少需要 0.5 秒。")

    if not st.button(
        "🚀 轉錄選取範圍",
        type="primary",
        disabled=duration_selected < 0.5,
        use_container_width=True,
    ):
        return
    if not output_format:
        st.warning("請至少選擇一種輸出格式。")
        return

    extracted_audio: Path | None = None
    try:
        TEMP_DIR.mkdir(exist_ok=True)
        extracted_audio = TEMP_DIR / f"segment_{uuid.uuid4().hex}.wav"
        with st.spinner("正在擷取 A–B 音訊範圍..."):
            extract_audio_range(input_path, extracted_audio, selected_start, selected_end)

        with st.spinner(f"正在載入 {model_size} 模型中..."):
            model = load_whisper_model(model_size, device)

        start_time = time.time()
        status_text = st.empty()
        progress_bar = st.progress(0)
        status_text.text("正在準備語音識別...")
        with show_whisper_progress(progress_bar, status_text):
            result = model.transcribe(
                str(extracted_audio),
                language=language.strip() or None,
                verbose=False,
                fp16=(device == "cuda"),
            )

        result = offset_result_timestamps(result, selected_start)
        conversion_config = CHINESE_OUTPUT_OPTIONS[chinese_output_label]
        if conversion_config:
            result = convert_result_text(result, load_chinese_converter(conversion_config))

        st.success(f"✨ 轉換完成！耗時：{time.time() - start_time:.2f} 秒")
        st.subheader("📝 字幕預覽（前 1000 字）")
        st.text_area("內容回傳", value=result.get("text", "")[:1000], height=200)

        st.subheader("💾 下載檔案")
        from whisper.utils import get_writer

        OUTPUT_DIR.mkdir(exist_ok=True)
        suffix = range_suffix(selected_start, selected_end, duration)
        output_stem = f"{input_path.stem}{suffix}"
        output_anchor = input_path.with_name(f"{output_stem}{input_path.suffix}")
        for ext in output_format:
            writer = get_writer(ext, str(OUTPUT_DIR))
            writer(result, str(output_anchor))

        columns = st.columns(len(output_format))
        for index, ext in enumerate(output_format):
            output_file = OUTPUT_DIR / f"{output_stem}.{ext}"
            if output_file.exists():
                columns[index].download_button(
                    label=f"下載 .{ext}",
                    data=output_file.read_bytes(),
                    file_name=output_file.name,
                    mime=MIME_TYPES.get(ext, "text/plain"),
                )
        st.info(
            f"檔案已儲存至：{OUTPUT_DIR.absolute()}\n\n"
            "選段檔名會附上 A–B 範圍，但 SRT 內部時間碼仍對應原影片。"
        )
    except Exception as exc:
        st.error(f"發生錯誤：{exc}")
    finally:
        if extracted_audio is not None:
            cleanup_file(extracted_audio)
        if uploaded_temp_file and cleanup_temp_file:
            cleanup_file(input_path)


if __name__ == "__main__":
    main()
