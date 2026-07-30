from __future__ import annotations

import gc
import hashlib
import importlib
import re
import shutil
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
import torch
import whisper
from opencc import OpenCC

from glossary import (
    PROMPT_SOFT_LIMIT,
    active_terms,
    add_terms,
    build_initial_prompt,
    inactive_terms,
    load_glossary,
    remove_term,
    save_glossary,
    set_all_active,
    set_term_active,
)
from replacements import (
    active_rules,
    add_rule,
    load_rules,
    remove_rule,
    save_rules,
)
from replacements import set_all_active as set_all_rules_active
from replacements import set_rule_active
from transcript_utils import (
    FAST_SUBTITLE_CPS,
    MediaProcessingError,
    apply_replacement_rules,
    build_result_from_chunks,
    build_result_from_words,
    build_waveform,
    convert_result_text,
    extract_audio_range,
    format_hhmmss,
    offset_result_timestamps,
    parse_hhmmss,
    parse_srt,
    probe_duration,
    range_suffix,
    read_wav_mono,
    replace_across_segments,
    restore_original_text,
    result_for_output,
    segment_rows,
    update_result_from_editor,
)

st.set_page_config(page_title="Whisper AI 語音轉文字工具", page_icon="🎙️", layout="wide")

TEMP_DIR = Path("temp_uploads")
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")
GLOSSARY_PATH = Path("glossary.json")
GLOSSARY_OPEN_KEY = "glossary_panel_open"
RULES_PATH = Path("replacements.json")
RULES_OPEN_KEY = "rules_panel_open"
ADVANCED_OPEN_KEY = "advanced_panel_open"
STATUS_LABELS = {"pending": "待轉譯", "running": "轉譯中", "done": "完成", "error": "失敗"}
UPLOAD_MODE = "拖曳上傳檔案（限制由系統設定）"
PATH_MODE = "直接輸入電腦中的檔案路徑（無大小限制，最快）"
SUBTITLE_MODE = "匯入現有字幕檔（SRT／VTT，直接編輯）"
SUBTITLE_SUFFIXES = (".srt", ".vtt")
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
ENGINE_WHISPER = "Whisper（OpenAI）"
ENGINE_BREEZE = "Breeze ASR 25（台灣華語＋中英夾雜推薦）"
ENGINE_SENSEVOICE = "SenseVoice-small（中英混合）"
SENSEVOICE_MODEL_LABEL = "SenseVoice-small"
BREEZE_MODEL_ID = "MediaTek-Research/Breeze-ASR-25"
BREEZE_MODEL_LABEL = "Breeze-ASR-25"
SENSEVOICE_LANGUAGE_OPTIONS = {
    "自動偵測（中英混合建議）": "auto",
    "中文": "zh",
    "英文": "en",
    "粵語": "yue",
    "日文": "ja",
    "韓文": "ko",
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


@st.cache_resource(show_spinner=False, max_entries=1)
def load_breeze_model(device: str):
    from transformers import (
        AutomaticSpeechRecognitionPipeline,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    processor = WhisperProcessor.from_pretrained(BREEZE_MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(
        BREEZE_MODEL_ID,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model = model.to(device).eval()
    # chunk_length_s is deliberately not set. The upstream model card passes 0, which was
    # how transformers v4 selected sequential long-form decoding; under v5 that same value
    # takes the experimental chunked path and silently stops after the first 30s window
    # (a 45s clip came back as one segment ending at 15s). Leaving it unset uses Whisper's
    # native long-form algorithm, which covers the whole clip and also handles inputs
    # under 30s — calling model.generate directly is finer-grained on long audio but
    # truncates short clips.
    return AutomaticSpeechRecognitionPipeline(
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=model.device,
    )


def transcribe_with_breeze(
    pipeline,
    audio_path: str,
    language: str | None,
    prevent_drift: bool,
) -> dict:
    """Transcribe via the Hugging Face pipeline. The glossary prompt is deliberately unused.

    Passing `prompt_ids` aborts transformers' long-form decoding after the first window:
    a 45s clip came back as a single segment ending at 15s, with the rest silently
    discarded. That held for both prompt_condition_type values and with
    condition_on_prev_tokens either way, and whether it triggers depends on the prompt's
    length, so there is no safe threshold to cap it at. Losing 2/3 of a transcript is a
    far worse failure than losing term biasing, so the prompt is only applied on the
    Whisper engine, whose initial_prompt handles long audio correctly.
    """
    waveform = read_wav_mono(audio_path)
    generate_kwargs: dict = {
        "task": "transcribe",
        "condition_on_prev_tokens": not prevent_drift,
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    }
    if language:
        generate_kwargs["language"] = language
    clip_seconds = len(waveform) / 16_000
    output = pipeline(waveform, return_timestamps=True, generate_kwargs=generate_kwargs)
    result = build_result_from_chunks(output.get("chunks") or [], total_duration=clip_seconds)

    # Truncation here is silent, so check rather than trust: transformers has already
    # changed which decoding path a given argument selects once.
    segments = result["segments"]
    covered = segments[-1]["end"] if segments else 0.0
    if clip_seconds - covered > max(3.0, clip_seconds * 0.05):
        st.warning(
            f"Breeze 只轉出前 {format_hhmmss(covered)}，選取範圍長度是 "
            f"{format_hhmmss(clip_seconds)}，後面的內容可能遺漏。"
            "請改用 Whisper 引擎，並回報這個狀況。"
        )
    return result


@st.cache_resource(show_spinner=False, max_entries=1)
def load_sensevoice_model(device: str):
    from funasr import AutoModel

    return AutoModel(
        model="FunAudioLLM/SenseVoiceSmall",
        hub="hf",
        trust_remote_code=True,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="cuda:0" if device == "cuda" else "cpu",
        disable_update=True,
    )


# SenseVoice can emit special tokens for language / emotion / audio events
# (e.g. <|zh|>, <|HAPPY|>, <|BGM|>, <|Laughter|>). These must not leak into subtitles.
_SENSEVOICE_TAG = re.compile(r"<\|[^|]*\|>")


def transcribe_with_sensevoice(model, audio_path: str, language: str) -> dict:
    results = model.generate(
        input=audio_path,
        cache={},
        language=language or "auto",
        use_itn=True,
        batch_size_s=60,
        merge_vad=False,
        merge_length_s=15,
        output_timestamp=True,
    )
    if not results:
        return {"text": "", "segments": []}
    item = results[0]
    raw_words = item.get("words") or []
    raw_timestamps = item.get("timestamp") or []
    words: list[str] = []
    timestamps: list = []
    for word, timestamp in zip(raw_words, raw_timestamps):
        cleaned = _SENSEVOICE_TAG.sub("", word).strip()
        if cleaned:
            words.append(cleaned)
            timestamps.append(timestamp)
    return build_result_from_words(words, timestamps)


def get_glossary() -> dict:
    if "glossary" not in st.session_state:
        st.session_state.glossary = load_glossary(GLOSSARY_PATH)
    return st.session_state.glossary


def persist_glossary() -> None:
    """Save to disk from a widget callback; errors are surfaced on the next render."""
    # Every mutation runs through here, and any mutation means the user is working inside
    # the panel, so it must not close underneath them on the resulting rerun.
    st.session_state[GLOSSARY_OPEN_KEY] = True
    try:
        backup = save_glossary(GLOSSARY_PATH, st.session_state.glossary)
        st.session_state.pop("glossary_error", None)
        if backup is not None:
            st.session_state.glossary_error = (
                f"這次儲存讓詞庫變少了，已先把原本的內容備份到 {backup.name}。"
                "如果不是你刻意刪除的，請關掉程式重新啟動後再檢查。"
            )
    except OSError as exc:
        st.session_state.glossary_error = f"詞庫儲存失敗：{exc}"


def glossary_activate(term: str) -> None:
    set_term_active(get_glossary(), term, True)
    persist_glossary()


def glossary_deactivate(term: str) -> None:
    set_term_active(get_glossary(), term, False)
    persist_glossary()


def glossary_delete(term: str) -> None:
    remove_term(get_glossary(), term)
    persist_glossary()


def glossary_set_all(active: bool) -> None:
    set_all_active(get_glossary(), active)
    persist_glossary()


def glossary_add_from_input(input_key: str) -> None:
    raw = st.session_state.get(input_key, "")
    added = add_terms(get_glossary(), raw)
    persist_glossary()
    st.session_state[input_key] = ""
    st.session_state.glossary_notice = (
        f"已新增 {len(added)} 個詞：{'、'.join(added)}" if added else "沒有新的詞可以新增（可能已在詞庫中）。"
    )


def glossary_update_prefix(input_key: str) -> None:
    get_glossary()["prefix"] = str(st.session_state.get(input_key, "")).strip()
    persist_glossary()


def term_key(prefix: str, term: str) -> str:
    return f"{prefix}_{hashlib.sha1(term.encode('utf-8')).hexdigest()[:10]}"


def keep_advanced_open() -> None:
    st.session_state[ADVANCED_OPEN_KEY] = True


def get_rules() -> dict:
    if "replacement_rules" not in st.session_state:
        st.session_state.replacement_rules = load_rules(RULES_PATH)
    return st.session_state.replacement_rules


def persist_rules() -> None:
    st.session_state[RULES_OPEN_KEY] = True
    try:
        backup = save_rules(RULES_PATH, st.session_state.replacement_rules)
        st.session_state.pop("rules_error", None)
        if backup is not None:
            st.session_state.rules_error = (
                f"這次儲存讓規則變少了，已先把原本的內容備份到 {backup.name}。"
                "如果不是你刻意刪除的，請關掉程式重新啟動後再檢查。"
            )
    except OSError as exc:
        st.session_state.rules_error = f"取代規則儲存失敗：{exc}"


def rule_toggle(find: str, checkbox_key: str) -> None:
    # Read the checkbox's own state rather than a value captured at render time.
    set_rule_active(get_rules(), find, bool(st.session_state.get(checkbox_key, True)))
    persist_rules()


def rule_delete(find: str) -> None:
    st.session_state.pop(term_key("rule_on", find), None)
    remove_rule(get_rules(), find)
    persist_rules()


def rules_set_all(active: bool) -> None:
    data = get_rules()
    set_all_rules_active(data, active)
    # Widget state outranks the `value=` argument, so the per-row checkboxes would keep
    # rendering their old state after a bulk change. Dropping the keys re-seeds them.
    for entry in data.get("rules", []):
        st.session_state.pop(term_key("rule_on", entry["find"]), None)
    persist_rules()


def rule_add_from_inputs(find_key: str, replace_key: str) -> None:
    find = str(st.session_state.get(find_key, "")).strip()
    replace = str(st.session_state.get(replace_key, ""))
    if not find:
        st.session_state.rules_notice = "請先填寫要尋找的文字。"
        return
    created = add_rule(get_rules(), find, replace)
    persist_rules()
    st.session_state[find_key] = ""
    st.session_state[replace_key] = ""
    st.session_state.rules_notice = (
        f"已新增規則：「{find}」→「{replace}」" if created else f"已更新「{find}」的取代內容。"
    )


def apply_replacement(record: dict, search_key: str, replace_key: str, notice_key: str) -> None:
    """Replace across all segments, then clear the find/replace pair.

    Runs as a widget callback so the two text inputs can be reset: after a replacement the
    search text no longer matches anything by definition, which would filter the table
    down to nothing and strand the user. Clearing them leaves the editor ready for the
    next correction.
    """
    search = str(st.session_state.get(search_key, "")).strip()
    replacement = str(st.session_state.get(replace_key, ""))
    if not search:
        st.session_state[notice_key] = (0, "", replacement, False, 0)
        return
    changed = replace_across_segments(record, search, replacement)
    saved = False
    if changed and st.session_state.get(f"{search_key}_save_rule"):
        add_rule(get_rules(), search, replacement)
        persist_rules()
        saved = True
    # Counted here rather than shown live: a caption that appeared and vanished as the
    # user typed was one of the things making the panel jump around.
    near_misses = 0
    if not changed:
        needle = search.lower()
        near_misses = sum(
            1
            for segment in record["result"].get("segments", [])
            if needle in str(segment.get("text", "")).lower()
        )
    st.session_state[search_key] = ""
    st.session_state[replace_key] = ""
    st.session_state[notice_key] = (changed, search, replacement, saved, near_misses)


def clear_subtitle_filters(search_key: str, dense_key: str) -> None:
    st.session_state[search_key] = ""
    st.session_state[dense_key] = False


def apply_saved_rules(record: dict, notice_key: str) -> None:
    changed, hits = apply_replacement_rules(record, active_rules(get_rules()))
    st.session_state[notice_key] = (changed, {f: n for f, n in hits.items() if n})


@st.fragment
def glossary_panel(prompt_supported: bool = True) -> None:
    """Two-column click-to-toggle glossary. A fragment so clicks don't redraw the waveform."""
    data = get_glossary()
    enabled = active_terms(data)
    available = inactive_terms(data)
    prompt_preview = build_initial_prompt(data)
    total = len(enabled) + len(available)

    # Open by default only for a first-time empty glossary; after that the user's own
    # toggle decides. This must never be recomputed from the term counts: doing so closed
    # the panel the moment the first term was added.
    if GLOSSARY_OPEN_KEY not in st.session_state:
        st.session_state[GLOSSARY_OPEN_KEY] = total == 0

    # The counts live out here because Streamlit resets an expander's open state whenever
    # its label changes, and the label would change on every single add or removal.
    st.caption(f"詞庫啟用 {len(enabled)} 個 / 共 {total} 個")
    with st.expander("📚 管理專有名詞詞庫", expanded=st.session_state[GLOSSARY_OPEN_KEY]):
        st.caption(
            "詞庫會存進 glossary.json，關掉程式也不會消失。啟用的詞會在轉譯時當作提示，"
            "幫助模型正確寫出人名、產品名與術語。"
        )
        if not prompt_supported:
            st.info(
                "目前選擇的引擎不會使用詞庫提示，詞庫仍會保留。"
                "Breeze 與 SenseVoice 沒有可靠的提示詞介面："
                "在 Breeze 上送出提示詞會讓長音訊只轉出開頭一小段，其餘內容被丟棄，"
                "因此寧可不送。想確實修正固定用詞，請改用下方的「取代規則」，它一定會生效。"
            )
        if error := st.session_state.pop("glossary_error", None):
            st.error(error)
        if notice := st.session_state.pop("glossary_notice", None):
            st.success(notice)

        prefix_key = "glossary_prefix_input"
        st.text_input(
            "開頭提示語句",
            value=data.get("prefix", ""),
            key=prefix_key,
            on_change=glossary_update_prefix,
            args=(prefix_key,),
            help="用來把模型的語境定在中文，可降低中英夾雜時被誤翻成英文的機率。",
        )

        available_column, enabled_column = st.columns(2, gap="medium")

        with available_column:
            st.markdown(f"**詞庫（未啟用）｜{len(available)}**")
            st.caption("點擊詞可加入本次轉譯，🗑 會從詞庫永久刪除。")
            add_key = "glossary_add_input"
            input_column, button_column = st.columns([3, 1])
            input_column.text_input(
                "新增詞",
                key=add_key,
                placeholder="ArchiCAD、subagent、容積率",
                label_visibility="collapsed",
                on_change=glossary_add_from_input,
                args=(add_key,),
            )
            button_column.button(
                "新增",
                key="glossary_add_button",
                use_container_width=True,
                on_click=glossary_add_from_input,
                args=(add_key,),
            )
            search = st.text_input(
                "搜尋詞庫",
                key="glossary_search",
                placeholder="🔍 搜尋詞庫",
                label_visibility="collapsed",
            ).strip()
            matches = [term for term in available if search.lower() in term.lower()]
            with st.container(height=260, border=True):
                if not available:
                    st.caption("詞庫裡的詞都已啟用。")
                elif not matches:
                    st.caption("沒有符合搜尋的詞。")
                for term in matches:
                    add_column, delete_column = st.columns([6, 1], gap="small")
                    add_column.button(
                        f"＋ {term}",
                        key=term_key("gloss_on", term),
                        use_container_width=True,
                        on_click=glossary_activate,
                        args=(term,),
                    )
                    delete_column.button(
                        "🗑",
                        key=term_key("gloss_del", term),
                        help=f"從詞庫永久刪除「{term}」",
                        on_click=glossary_delete,
                        args=(term,),
                    )
            st.button(
                "全部啟用",
                key="glossary_enable_all",
                use_container_width=True,
                disabled=not available,
                on_click=glossary_set_all,
                args=(True,),
            )

        with enabled_column:
            st.markdown(f"**本次啟用｜{len(enabled)}**")
            st.caption("點擊詞即可移出本次轉譯，詞仍會留在詞庫裡。")
            with st.container(height=316, border=True):
                if not enabled:
                    st.caption("尚未啟用任何詞，轉譯時只會使用上方的開頭提示語句。")
                for term in enabled:
                    st.button(
                        f"✓ {term}",
                        key=term_key("gloss_off", term),
                        use_container_width=True,
                        on_click=glossary_deactivate,
                        args=(term,),
                    )
            st.button(
                "全部停用",
                key="glossary_disable_all",
                use_container_width=True,
                disabled=not enabled,
                on_click=glossary_set_all,
                args=(False,),
            )

        if prompt_preview:
            st.text_area("送給模型的提示語", value=prompt_preview, height=80, disabled=True)
            if len(prompt_preview) > PROMPT_SOFT_LIMIT:
                st.warning(
                    f"提示語有 {len(prompt_preview)} 字，超過建議的 {PROMPT_SOFT_LIMIT} 字。"
                    "Whisper 只會讀取結尾的部分，建議停用一些不會出現在這段影片裡的詞。"
                )
        st.caption(f"詞庫檔案：{GLOSSARY_PATH.absolute()}")


@st.fragment
def rules_panel() -> None:
    """Saved find/replace pairs. A fragment so edits don't redraw the whole page."""
    data = get_rules()
    rules = data.get("rules", [])
    enabled = active_rules(data)

    if RULES_OPEN_KEY not in st.session_state:
        st.session_state[RULES_OPEN_KEY] = not rules

    st.caption(f"取代規則啟用 {len(enabled)} 條 / 共 {len(rules)} 條")
    with st.expander("🔁 管理常用取代規則", expanded=st.session_state[RULES_OPEN_KEY]):
        st.caption(
            "規則存進 replacements.json，關掉程式也不會消失。這些規則是在轉譯完成後套用到文字上，"
            "所以不論用哪個引擎都一定會生效——這點和只能「提示」模型的專有名詞詞庫不同。"
        )
        if error := st.session_state.pop("rules_error", None):
            st.error(error)
        if notice := st.session_state.pop("rules_notice", None):
            st.success(notice)

        find_key, replace_key = "rule_find_input", "rule_replace_input"
        find_column, replace_column, button_column = st.columns([3, 3, 1])
        find_column.text_input("尋找", key=find_key, placeholder="agent teams")
        replace_column.text_input(
            "取代為", key=replace_key, placeholder="Agent Teams（留空表示刪除）"
        )
        with button_column:
            st.write("")
            st.button(
                "新增",
                key="rule_add_button",
                use_container_width=True,
                on_click=rule_add_from_inputs,
                args=(find_key, replace_key),
            )

        with st.container(height=240, border=True):
            if not rules:
                st.caption("尚未建立任何取代規則。")
            for entry in rules:
                find = entry["find"]
                toggle_column, text_column, delete_column = st.columns([1, 8, 1])
                checkbox_key = term_key("rule_on", find)
                toggle_column.checkbox(
                    "啟用",
                    value=entry["active"],
                    key=checkbox_key,
                    label_visibility="collapsed",
                    on_change=rule_toggle,
                    args=(find, checkbox_key),
                )
                shown = entry["replace"] or "（刪除）"
                text_column.markdown(f"`{find}` → `{shown}`")
                delete_column.button(
                    "🗑",
                    key=term_key("rule_del", find),
                    help=f"刪除規則「{find}」",
                    on_click=rule_delete,
                    args=(find,),
                )

        enable_column, disable_column = st.columns(2)
        enable_column.button(
            "全部啟用",
            key="rules_enable_all",
            use_container_width=True,
            disabled=len(enabled) == len(rules),
            on_click=rules_set_all,
            args=(True,),
        )
        disable_column.button(
            "全部停用",
            key="rules_disable_all",
            use_container_width=True,
            disabled=not enabled,
            on_click=rules_set_all,
            args=(False,),
        )
        st.caption(f"規則檔案：{RULES_PATH.absolute()}")


@st.cache_data(show_spinner=False, max_entries=8)
def analyze_media(path: str, file_size: int, modified_ns: int):
    del file_size, modified_ns
    duration = probe_duration(path)
    times, amplitudes = build_waveform(path, duration, target_points=1600)
    return duration, times, amplitudes


def save_uploaded_file(uploaded_file) -> Path:
    """Stage an upload on disk, keeping its original filename.

    The dedupe hash names the *folder*, not the file. It used to be a filename prefix,
    which leaked into every output because the subtitle files are named after the input
    stem, so a download of 0107.mp4 arrived as `a3f9c2e14b_0107.srt`. Writing only when
    the file is absent also keeps the mtime — and therefore the media cache key — stable
    across reruns without having to restore it by hand.
    """
    safe_name = Path(uploaded_file.name).name
    upload_token = (
        getattr(uploaded_file, "file_id", None),
        safe_name,
        getattr(uploaded_file, "size", None),
    )
    upload_id = hashlib.sha1(repr(upload_token).encode("utf-8")).hexdigest()[:10]
    folder = TEMP_DIR / upload_id
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / safe_name
    if not destination.exists():
        uploaded_file.seek(0)
        with destination.open("wb") as output_file:
            shutil.copyfileobj(uploaded_file, output_file)
    return destination


def cleanup_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        st.warning(f"暫存檔清理失敗：{exc}")


def preview_state_key(source_id: str) -> str:
    return f"audio_preview_{source_id}"


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
    load_breeze_model.clear()
    load_sensevoice_model.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    st.session_state["model_release_notice"] = True


def output_stem(record: dict) -> str:
    """Base filename for this result: the source's own name plus any A-B range."""
    suffix = range_suffix(record["start"], record["end"], record["source_duration"])
    return f"{Path(record['source_path']).stem}{suffix}"


def resolve_output_directory(record: dict) -> Path:
    """Save next to the source file, so subtitles land beside the video they belong to.

    An imported subtitle therefore writes back over itself, which is the point. Uploads
    are the exception: their "source" is a temp folder, so those still go to outputs/.
    """
    source = Path(record["source_path"])
    if not record.get("is_temp") and source.parent.is_dir():
        return source.parent
    return OUTPUT_DIR / source.stem


def render_subtitle_files(
    record: dict,
    output_formats: list[str],
    directory: Path,
    stem: str,
) -> list[tuple[str, Path]]:
    from whisper.utils import get_writer

    directory.mkdir(parents=True, exist_ok=True)
    # The writers derive each output name from this anchor's stem; its suffix is ignored.
    anchor = directory / f"{stem}.media"
    result = result_for_output(record)
    files: list[tuple[str, Path]] = []
    for extension in output_formats:
        writer = get_writer(extension, str(directory))
        writer(result, str(anchor))
        output_file = directory / f"{stem}.{extension}"
        if output_file.exists():
            files.append((extension, output_file))
    return files


def subtitle_payloads(record: dict, output_formats: list[str]) -> list[tuple[str, str, bytes]]:
    """Build the download bytes in a temp folder.

    Rendering used to write straight to the destination on every rerun, which was
    harmless while everything landed in outputs/ but would silently overwrite the user's
    own file now that results save next to their source. Nothing touches the destination
    until the save button is pressed.
    """
    stem = output_stem(record)
    with tempfile.TemporaryDirectory() as scratch:
        return [
            (extension, path.name, path.read_bytes())
            for extension, path in render_subtitle_files(
                record, output_formats, Path(scratch), stem
            )
        ]


def save_result_files(record: dict, output_formats: list[str]) -> tuple[Path, list[Path]]:
    directory = resolve_output_directory(record)
    files = render_subtitle_files(record, output_formats, directory, output_stem(record))
    record["output_directory"] = str(directory.absolute())
    return directory, [path for _, path in files]


def get_queue() -> list[dict]:
    if "queue" not in st.session_state:
        st.session_state.queue = []
    return st.session_state.queue


def enqueue_media(path: Path, *, is_temp: bool) -> str | None:
    """Add one file to the queue. Returns an error string, or None on success."""
    queue = get_queue()
    resolved = str(path.resolve())
    if any(job["path"] == resolved for job in queue):
        return f"{path.name}：已在佇列中"
    try:
        duration = probe_duration(path)
    except (OSError, MediaProcessingError) as exc:
        return f"{path.name}：{exc}"
    queue.append(
        {
            "id": uuid.uuid4().hex[:8],
            "path": resolved,
            "name": path.name,
            "duration": float(duration),
            "start": 0.0,
            "end": float(duration),
            "status": "pending",
            "error": None,
            "record": None,
            "is_temp": is_temp,
        }
    )
    return None


def read_text_guess(path: Path) -> str:
    """Read a subtitle file whose encoding is unknown.

    Subtitles written on a Traditional Chinese Windows box are frequently cp950 rather
    than UTF-8, and guessing wrong turns the whole file into mojibake.
    """
    for encoding in ("utf-8-sig", "cp950", "gb18030", "utf-16"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError, LookupError):
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def enqueue_subtitle(path: Path, *, is_temp: bool) -> str | None:
    """Import an existing subtitle straight into the results area for re-editing."""
    queue = get_queue()
    resolved = str(path.resolve())
    if any(job["path"] == resolved for job in queue):
        return f"{path.name}：已在佇列中"
    try:
        result = parse_srt(read_text_guess(path))
    except OSError as exc:
        return f"{path.name}：{exc}"
    segments = result["segments"]
    if not segments:
        return f"{path.name}：找不到任何字幕時間碼，請確認是 SRT 或 VTT 檔"

    duration = max(float(segment["end"]) for segment in segments)
    queue.append(
        {
            "id": uuid.uuid4().hex[:8],
            "path": resolved,
            "name": path.name,
            "duration": duration,
            "start": 0.0,
            "end": duration,
            "status": "done",
            "error": None,
            "is_temp": is_temp,
            "record": {
                "id": uuid.uuid4().hex,
                "result": result,
                "source_path": resolved,
                "source_name": path.name,
                "start": 0.0,
                "end": duration,
                "source_duration": duration,
                "engine": "匯入字幕",
                "model": "—",
                "language": "—",
                "conversion": "—",
                "initial_prompt": "",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "edited": False,
                "rule_hits": {},
                "is_subtitle": True,
                "is_temp": is_temp,
                "original_texts": [str(segment.get("text", "")) for segment in segments],
                "timings": dict.fromkeys(
                    ("extraction", "model_load", "inference", "postprocess", "total"), 0.0
                ),
            },
        }
    )
    return None


def remove_job(job_id: str) -> None:
    st.session_state.queue = [job for job in get_queue() if job["id"] != job_id]
    if st.session_state.get("focused_job") == job_id:
        st.session_state.pop("focused_job", None)


def clear_finished_jobs() -> None:
    st.session_state.queue = [job for job in get_queue() if job["status"] != "done"]


def requeue_job(job_id: str) -> None:
    for job in get_queue():
        if job["id"] == job_id:
            job["status"] = "pending"
            job["error"] = None


def focus_job(job_id: str) -> None:
    st.session_state.focused_job = job_id


def transcribe_job(job: dict, settings: dict, device: str) -> dict:
    """Run one queue entry end to end and return its result record."""
    source_path = Path(job["path"])
    start, end = float(job["start"]), float(job["end"])
    engine = settings["engine"]
    extracted_audio = TEMP_DIR / f"segment_{uuid.uuid4().hex}.wav"
    total_started = time.perf_counter()
    try:
        TEMP_DIR.mkdir(exist_ok=True)
        phase_started = time.perf_counter()
        extract_audio_range(source_path, extracted_audio, start, end)
        extraction_seconds = time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        if engine == ENGINE_WHISPER:
            model = load_whisper_model(settings["model_size"], device)
        elif engine == ENGINE_BREEZE:
            model = load_breeze_model(device)
        else:
            model = load_sensevoice_model(device)
        model_load_seconds = time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        if engine == ENGINE_WHISPER:
            status_text = st.empty()
            progress_bar = st.progress(0)
            transcribe_options = {
                "language": settings["language"],
                # Pinned so a mixed-language clip can never be decoded as "translate".
                "task": "transcribe",
                "verbose": False,
                "fp16": device == "cuda",
                "condition_on_previous_text": not settings["prevent_drift"],
                # Whisper's own fallback ladder: a segment that decodes badly is retried
                # at a higher temperature instead of being kept as-is.
                "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            }
            if settings["initial_prompt"]:
                transcribe_options["initial_prompt"] = settings["initial_prompt"]
            with show_whisper_progress(progress_bar, status_text):
                result = model.transcribe(str(extracted_audio), **transcribe_options)
            progress_bar.empty()
            status_text.empty()
        elif engine == ENGINE_BREEZE:
            result = transcribe_with_breeze(
                model, str(extracted_audio), settings["language"], settings["prevent_drift"]
            )
        else:
            result = transcribe_with_sensevoice(
                model, str(extracted_audio), settings["language"]
            )
        inference_seconds = time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        result = offset_result_timestamps(result, start)
        conversion_config = CHINESE_OUTPUT_OPTIONS[settings["chinese_output_label"]]
        if conversion_config:
            result = convert_result_text(result, load_chinese_converter(conversion_config))
        rules = active_rules(get_rules())
        rule_hits: dict[str, int] = {}
        if rules:
            # Applied before the snapshot below, so the saved rules count as part of the
            # pipeline and "restore original" does not undo them.
            _, rule_hits = apply_replacement_rules({"result": result, "edited": False}, rules)
        postprocess_seconds = time.perf_counter() - phase_started

        return {
            "id": uuid.uuid4().hex,
            "result": result,
            "source_path": str(source_path),
            "source_name": job["name"],
            "is_temp": job["is_temp"],
            "start": start,
            "end": end,
            "source_duration": float(job["duration"]),
            "engine": engine,
            "model": settings["model_display"],
            "language": settings["language_label"],
            "conversion": settings["chinese_output_label"],
            "initial_prompt": settings["initial_prompt"],
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "edited": False,
            "rule_hits": {find: n for find, n in rule_hits.items() if n},
            "original_texts": [
                str(segment.get("text", "")) for segment in result.get("segments", [])
            ],
            "timings": {
                "extraction": extraction_seconds,
                "model_load": model_load_seconds,
                "inference": inference_seconds,
                "postprocess": postprocess_seconds,
                "total": time.perf_counter() - total_started,
            },
        }
    finally:
        cleanup_file(extracted_audio)


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


@st.fragment
def subtitle_editor(record: dict, input_path: Path) -> None:
    """Searchable subtitle table with replace-all, per-segment audition and undo.

    A fragment so typing in the search box reruns only this panel, not the waveform.
    """
    segments = record["result"].get("segments", [])
    if not segments:
        st.info("這次結果沒有可編輯的字幕段落。")
        return

    total_characters = sum(len(str(seg.get("text", "")).strip()) for seg in segments)
    dense_count = len(segment_rows(segments, only_dense=True))

    metrics = st.columns(4)
    metrics[0].metric("段落", len(segments))
    metrics[1].metric("總字數", f"{total_characters:,}")
    metrics[2].metric(
        "偏快段落",
        dense_count,
        help=f"每秒超過 {FAST_SUBTITLE_CPS:.0f} 個字，通常是斷句太長或辨識黏在一起。",
    )
    metrics[3].metric("狀態", "已修改" if record.get("edited") else "原始")

    search_key = f"subtitle_search_{record['id']}"
    replace_key = f"subtitle_replace_{record['id']}"
    search_filter_key = f"subtitle_filter_{record['id']}"
    dense_key = f"subtitle_dense_only_{record['id']}"
    notice_key = f"subtitle_replace_notice_{record['id']}"
    rules_notice_key = f"subtitle_rules_notice_{record['id']}"

    # Find/replace and the table filter each live in their own form. Widgets inside a form
    # do not trigger a rerun when they change, so tabbing between 尋找 and 取代為 or ticking
    # a checkbox no longer re-renders the panel mid-edit — which is what made the layout
    # jump and briefly drop controls. Nothing reruns until a submit button is pressed.
    with st.form(f"subtitle_replace_form_{record['id']}", border=True):
        st.markdown("**尋找與取代**（會套用到整份字幕，不受下方篩選影響）")
        find_column, to_column = st.columns(2)
        find_column.text_input(
            "尋找文字",
            key=search_key,
            placeholder="要修正的字，需與原文大小寫一致",
        )
        to_column.text_input(
            "取代為",
            key=replace_key,
            placeholder="留空表示刪除這段文字",
        )
        save_column, submit_column = st.columns([3, 2])
        save_column.checkbox(
            "取代後存成常用規則",
            key=f"{search_key}_save_rule",
            help="勾選後按「全部取代」，這組尋找／取代會存進 replacements.json 重複使用。",
        )
        with submit_column:
            st.write("")
            st.form_submit_button(
                "全部取代",
                type="primary",
                use_container_width=True,
                on_click=apply_replacement,
                args=(record, search_key, replace_key, notice_key),
            )

    if notice := st.session_state.pop(notice_key, None):
        changed, replaced, replacement, saved, near_misses = notice
        if not replaced:
            st.warning("請先在「尋找文字」填入要修正的字。")
        elif changed:
            action = f"換成「{replacement}」" if replacement else "刪除"
            extra = "，並已存成常用規則" if saved else ""
            st.success(f"已在 {changed} 個段落把「{replaced}」{action}{extra}，欄位已清空。")
        elif near_misses:
            st.warning(
                f"有 {near_misses} 段包含「{replaced}」，但大小寫不完全相同，因此沒有取代。"
                "取代要求大小寫完全一致，請照字幕裡的寫法輸入。"
            )
        else:
            st.info(f"沒有段落包含「{replaced}」，未做任何變更。")

    filter_row = st.container(border=True)
    with filter_row:
        rules_column, apply_column = st.columns([3, 2])
        saved_rules = active_rules(get_rules())
        rules_column.caption(f"已儲存 {len(saved_rules)} 條啟用中的取代規則")
        apply_column.button(
            f"套用儲存的規則（{len(saved_rules)}）",
            key=f"subtitle_apply_rules_{record['id']}",
            disabled=not saved_rules,
            use_container_width=True,
            on_click=apply_saved_rules,
            args=(record, rules_notice_key),
        )
        with st.form(f"subtitle_filter_form_{record['id']}", border=False):
            keyword_column, dense_column, filter_button_column = st.columns([4, 3, 2])
            keyword_column.text_input(
                "只顯示包含這串文字的段落",
                key=search_filter_key,
                placeholder="🔍 留空顯示全部",
            )
            dense_column.checkbox(
                "只顯示偏快的段落",
                key=dense_key,
                disabled=not dense_count,
            )
            with filter_button_column:
                st.write("")
                st.form_submit_button("套用篩選", use_container_width=True)

    search = str(st.session_state.get(search_filter_key, "")).strip()
    only_dense = bool(st.session_state.get(dense_key, False))

    if notice := st.session_state.pop(rules_notice_key, None):
        changed, hits_by_rule = notice
        if changed:
            detail = "、".join(f"{find}×{n}" for find, n in hits_by_rule.items())
            st.success(f"套用規則後修改了 {changed} 個段落：{detail}")
        else:
            st.info("目前啟用的規則在這份字幕裡沒有找到任何符合的文字。")

    rows = segment_rows(segments, search=search, only_dense=only_dense)
    if not rows:
        # Deliberately not an early return: bailing out here used to remove the table, the
        # audition control and the undo button all at once, which read as the panel
        # breaking rather than as an empty filter result.
        st.info("沒有符合篩選條件的段落。")
        st.button(
            "清除篩選",
            key=f"subtitle_clear_filters_{record['id']}",
            on_click=clear_subtitle_filters,
            args=(search_filter_key, dense_key),
        )
    else:
        if search or only_dense:
            st.caption(
                f"顯示 {len(rows)} / {len(segments)} 段。"
                "切換篩選前請先儲存修改，否則未儲存的編輯會遺失。"
            )
        # New widget identity per filter so pending cell edits are never mapped onto a
        # different set of rows. Safe to key on the filter now that the filter only
        # changes on an explicit submit rather than on every keystroke.
        filter_token = hashlib.sha1(f"{search}|{only_dense}".encode("utf-8")).hexdigest()[:8]
        with st.form(f"subtitle_editor_{record['id']}_{filter_token}"):
            edited_rows = st.data_editor(
                rows,
                hide_index=True,
                use_container_width=True,
                height=min(620, 90 + 35 * len(rows)),
                disabled=["段落", "開始", "長度", "字/秒"],
                column_config={
                    "段落": st.column_config.NumberColumn(width="small", help="對應原始字幕編號"),
                    "開始": st.column_config.TextColumn(width="small"),
                    "長度": st.column_config.NumberColumn(width="small", format="%.2f 秒"),
                    "字/秒": st.column_config.NumberColumn(
                        width="small",
                        format="%.1f",
                        help=f"超過 {FAST_SUBTITLE_CPS:.0f} 代表這段字幕偏快",
                    ),
                    "字幕": st.column_config.TextColumn(width="large"),
                },
            )
            save_edits = st.form_submit_button("儲存字幕修改", type="primary")
        if save_edits:
            changed = update_result_from_editor(record, edited_rows)
            if changed:
                st.success(f"已更新 {changed} 個段落；下載檔會使用最新內容。")
            else:
                st.info("沒有偵測到變更。")

    audition_column, revert_column = st.columns([3, 1])
    with audition_column:
        number_column, play_column = st.columns([2, 1])
        target = number_column.number_input(
            "試聽段落",
            min_value=1,
            max_value=len(segments),
            value=int(rows[0]["段落"]) if rows else 1,
            step=1,
            key=f"subtitle_audition_{record['id']}",
            label_visibility="collapsed",
        )
        with play_column:
            play = st.button(
                "▶ 試聽此段",
                key=f"subtitle_play_{record['id']}",
                disabled=not input_path.exists(),
                use_container_width=True,
            )
    with revert_column:
        if st.button(
            "↩ 還原全部",
            key=f"subtitle_revert_{record['id']}",
            disabled=not record.get("edited"),
            use_container_width=True,
            help="把所有字幕還原成剛轉譯完的原始文字。",
        ):
            st.success(f"已還原 {restore_original_text(record)} 個段落。")
            st.rerun(scope="fragment")

    audition_key = f"subtitle_audio_{record['id']}"
    if play:
        segment = segments[int(target) - 1]
        start = max(0.0, float(segment.get("start", 0.0)) - 0.15)
        end = min(record["source_duration"], float(segment.get("end", start)) + 0.15)
        clip_path = TEMP_DIR / f"segment_preview_{uuid.uuid4().hex}.wav"
        try:
            with st.spinner("正在擷取這一段的音訊..."):
                extract_audio_range(input_path, clip_path, start, max(end, start + 0.2))
            st.session_state[audition_key] = {
                "number": int(target),
                "text": str(segment.get("text", "")).strip(),
                "data": clip_path.read_bytes(),
            }
        except (OSError, MediaProcessingError, ValueError) as exc:
            st.error(f"無法試聽這一段：{exc}")
        finally:
            cleanup_file(clip_path)

    if clip := st.session_state.get(audition_key):
        st.audio(clip["data"], format="audio/wav")
        st.caption(f"第 {clip['number']} 段：{clip['text']}")


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
    engine = st.sidebar.selectbox(
        "辨識引擎",
        [ENGINE_BREEZE, ENGINE_WHISPER, ENGINE_SENSEVOICE],
        index=0,
        help="Breeze ASR 25 是聯發科研究院從 Whisper-large-v2 微調的模型，針對台灣華語與"
        "中英夾雜優化，混語錯誤率約為原版的一半，且直接輸出繁體；首次使用會下載約 3GB 模型。"
        "SenseVoice-small 較輕量、推論最快，首次使用下載約 1GB。",
    )
    if engine == ENGINE_WHISPER:
        model_label = st.sidebar.selectbox(
            "速度與品質",
            list(MODEL_OPTIONS),
            index=0,
            help="Turbo 適合先快速產生字幕；重要成品可再用 Large v3 比對。",
        )
        model_size = MODEL_OPTIONS[model_label]
        model_display = model_size
        language_label = st.sidebar.selectbox("語音語言", list(LANGUAGE_OPTIONS), index=0)
        language = LANGUAGE_OPTIONS[language_label]
    elif engine == ENGINE_BREEZE:
        model_size = None
        model_display = BREEZE_MODEL_LABEL
        # Same language space as Whisper: Breeze is a Whisper-large-v2 fine-tune.
        language_label = st.sidebar.selectbox("語音語言", list(LANGUAGE_OPTIONS), index=0)
        language = LANGUAGE_OPTIONS[language_label]
    else:
        model_size = None
        model_display = SENSEVOICE_MODEL_LABEL
        language_label = st.sidebar.selectbox(
            "語音語言", list(SENSEVOICE_LANGUAGE_OPTIONS), index=0
        )
        language = SENSEVOICE_LANGUAGE_OPTIONS[language_label]
    chinese_output_label = st.sidebar.selectbox(
        "中文字體輸出",
        list(CHINESE_OUTPUT_OPTIONS),
        index=0,
        help="OpenCC 會在辨識後轉換字形與台灣常用詞，不會改動字幕時間碼。",
    )
    output_format = st.sidebar.multiselect(
        "輸出格式", OUTPUT_FORMATS, default=["srt", "txt"]
    )
    # Without a remembered state this expander collapses on every rerun, so changing a
    # setting inside it made it snap shut immediately afterwards.
    with st.sidebar.expander("進階設定", expanded=st.session_state.get(ADVANCED_OPEN_KEY, False)):
        prevent_drift = st.checkbox(
            "防止英文翻譯漂移",
            value=True,
            key="prevent_drift",
            on_change=keep_advanced_open,
            help="不把上一段的結果當作下一段的提示。中英夾雜時 Whisper 偶爾會把某段改成翻譯"
            "成英文，並沿著上下文一路傳染下去；關閉這個關聯可以擋住連鎖效應，"
            "代價是跨段落的用詞一致性略降。",
        )
        cleanup_temp_file = st.checkbox(
            "轉換後清除上傳的原始暫存檔",
            value=False,
            key="cleanup_temp_file",
            on_change=keep_advanced_open,
            help="勾選會在轉譯完成後刪掉上傳的暫存影片，之後就無法在字幕編輯器裡逐段試聽。",
        )
        st.caption(f"模型位置：{MODEL_DIR.absolute()}")
        st.button("釋放 GPU 模型", on_click=release_cached_model, use_container_width=True)
    if st.session_state.pop("model_release_notice", False):
        st.sidebar.success("已清除模型快取。")

    st.header("1. 加入待轉譯檔案")
    input_mode = st.radio(
        "輸入方式",
        [UPLOAD_MODE, PATH_MODE, SUBTITLE_MODE],
        horizontal=True,
        label_visibility="collapsed",
    )

    if input_mode == SUBTITLE_MODE:
        st.caption(
            "匯入以前轉好的字幕，直接套用取代規則、修改文字後存回原檔。"
            "不需要影片，但也因此無法逐段試聽。"
        )
        subtitle_upload, subtitle_path = st.tabs(["上傳字幕檔", "輸入字幕路徑"])
        with subtitle_upload:
            uploaded_subtitles = st.file_uploader(
                "選擇字幕檔（可一次選多個）",
                type=["srt", "vtt"],
                accept_multiple_files=True,
                key="subtitle_uploader",
            )
            if uploaded_subtitles and st.button(
                f"匯入 {len(uploaded_subtitles)} 個字幕檔",
                type="primary",
                use_container_width=True,
                key="import_uploaded_subtitles",
            ):
                problems = []
                with st.spinner("正在讀取字幕..."):
                    for uploaded_file in uploaded_subtitles:
                        try:
                            problems.append(
                                enqueue_subtitle(save_uploaded_file(uploaded_file), is_temp=True)
                            )
                        except OSError as exc:
                            problems.append(f"{uploaded_file.name}：{exc}")
                st.session_state.enqueue_report = [p for p in problems if p]
                st.rerun()
            if uploaded_subtitles:
                st.caption("上傳的字幕會存到 outputs/，無法覆蓋原始檔案；要覆蓋請改用路徑分頁。")
        with subtitle_path:
            subtitle_input = st.text_area(
                "貼上字幕檔完整路徑，一行一個",
                height=110,
                placeholder="D:\\Videos\\1222.srt\nD:\\Videos\\1223.srt",
                key="subtitle_path_input",
            )
            subtitle_candidates = [
                line.strip().strip('"') for line in subtitle_input.splitlines()
            ]
            subtitle_candidates = [line for line in subtitle_candidates if line]
            if subtitle_candidates and st.button(
                f"匯入 {len(subtitle_candidates)} 個字幕檔",
                type="primary",
                use_container_width=True,
                key="import_subtitle_paths",
            ):
                problems = []
                with st.spinner("正在讀取字幕..."):
                    for line in subtitle_candidates:
                        candidate = Path(line)
                        if not candidate.exists() or not candidate.is_file():
                            problems.append(f"{line}：找不到檔案")
                        elif candidate.suffix.lower() not in SUBTITLE_SUFFIXES:
                            problems.append(f"{candidate.name}：不是 .srt 或 .vtt")
                        else:
                            problems.append(enqueue_subtitle(candidate, is_temp=False))
                st.session_state.enqueue_report = [p for p in problems if p]
                st.rerun()
            st.caption("以路徑匯入時，儲存會直接覆蓋原始字幕檔。")
    elif input_mode == UPLOAD_MODE:
        uploaded_files = st.file_uploader(
            "選擇影片或音檔（可一次選多個）",
            type=["mp4", "mkv", "mov", "avi", "mp3", "wav", "m4a"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button(
            f"加入佇列（{len(uploaded_files)} 個檔案）", type="primary", use_container_width=True
        ):
            problems = []
            with st.spinner("正在讀取檔案..."):
                for uploaded_file in uploaded_files:
                    try:
                        problems.append(
                            enqueue_media(save_uploaded_file(uploaded_file), is_temp=True)
                        )
                    except OSError as exc:
                        problems.append(f"{uploaded_file.name}：{exc}")
            st.session_state.enqueue_report = [p for p in problems if p]
            st.rerun()
    else:
        path_input = st.text_area(
            "貼上檔案完整路徑，一行一個（例如 D:\\Videos\\1222.mp4）",
            height=110,
            placeholder="D:\\Videos\\1222.mp4\nD:\\Videos\\1223.mp4",
        )
        candidates = [line.strip().strip('"') for line in path_input.splitlines()]
        candidates = [line for line in candidates if line]
        if candidates and st.button(
            f"加入佇列（{len(candidates)} 個路徑）", type="primary", use_container_width=True
        ):
            problems = []
            with st.spinner("正在讀取檔案..."):
                for line in candidates:
                    candidate = Path(line)
                    if candidate.exists() and candidate.is_file():
                        problems.append(enqueue_media(candidate, is_temp=False))
                    else:
                        problems.append(f"{line}：找不到檔案")
            st.session_state.enqueue_report = [p for p in problems if p]
            st.rerun()

    for problem in st.session_state.pop("enqueue_report", []):
        st.warning(problem)

    queue = get_queue()
    pending_jobs = [job for job in queue if job["status"] == "pending"]
    done_jobs = [job for job in queue if job["status"] == "done"]
    if queue:
        st.header("2. 佇列與片段")
        summary = st.columns(4)
        summary[0].metric("佇列檔案", len(queue))
        summary[1].metric("待轉譯", len(pending_jobs))
        summary[2].metric("完成", len(done_jobs))
        summary[3].metric(
            "待轉譯總長",
            format_hhmmss(sum(job["end"] - job["start"] for job in pending_jobs)),
        )
        render_queue_table(queue, done_jobs)

    # Rendered even with an empty queue: the glossary and the replacement rules are
    # global, persistent settings, and hiding them behind "add a file first" made a saved
    # rule set look like it had been lost.
    st.header("3. 專有名詞與取代規則")
    prompt_supported = engine == ENGINE_WHISPER
    glossary_panel(prompt_supported=prompt_supported)
    # Only the prompt that actually reaches a model counts as a setting worth re-running for.
    initial_prompt = build_initial_prompt(get_glossary()) if prompt_supported else ""
    rules_panel()

    if not queue:
        st.info("佇列是空的，請在上方加入至少一個檔案才能開始轉譯。")
        return

    settings = {
        "engine": engine,
        "model_size": model_size,
        "model_display": model_display,
        "language": language,
        "language_label": language_label,
        "chinese_output_label": chinese_output_label,
        "initial_prompt": initial_prompt.strip(),
        "prevent_drift": prevent_drift,
        "output_format": output_format,
    }
    render_transcription_step(queue, pending_jobs, settings, device, cleanup_temp_file)
    render_results(output_format)


def render_queue_table(queue: list[dict], done_jobs: list[dict]) -> None:
    with st.container(border=True):
        for job in queue:
            name_col, range_col, status_col, action_col, remove_col = st.columns(
                [5, 3, 2, 2, 1]
            )
            name_col.markdown(f"**{job['name']}**")
            name_col.caption(job["path"])
            trimmed = job["start"] > 0.05 or job["end"] < job["duration"] - 0.05
            range_col.caption(
                f"{format_hhmmss(job['start'])} – {format_hhmmss(job['end'])}"
                + ("（已裁切）" if trimmed else "（全片）")
            )
            status_col.caption(STATUS_LABELS.get(job["status"], job["status"]))
            if job["status"] == "error":
                status_col.caption(f"⚠️ {job['error']}")
            if job["status"] == "done":
                action_col.button(
                    "重新轉譯",
                    key=f"requeue_{job['id']}",
                    use_container_width=True,
                    on_click=requeue_job,
                    args=(job["id"],),
                )
            else:
                action_col.button(
                    "設定片段",
                    key=f"focus_{job['id']}",
                    use_container_width=True,
                    on_click=focus_job,
                    args=(job["id"],),
                )
            remove_col.button(
                "🗑",
                key=f"remove_{job['id']}",
                help=f"從佇列移除 {job['name']}",
                on_click=remove_job,
                args=(job["id"],),
            )

    if done_jobs:
        st.button(
            "清除已完成的佇列項目（結果會保留在下方）",
            key="clear_done",
            on_click=clear_finished_jobs,
        )

    focused_id = st.session_state.get("focused_job")
    focused = next((job for job in queue if job["id"] == focused_id), None)
    if focused is not None and focused["status"] != "done":
        render_range_editor(focused)


def render_transcription_step(
    queue: list[dict],
    pending_jobs: list[dict],
    settings: dict,
    device: str,
    cleanup_temp_file: bool,
) -> None:
    output_format = settings["output_format"]
    model_display = settings["model_display"]
    st.header("4. 開始轉譯")
    if not output_format:
        st.warning("請至少選擇一種輸出格式。")
    run_queue = st.button(
        f"轉譯佇列中的 {len(pending_jobs)} 個檔案｜{model_display}",
        type="primary",
        disabled=not pending_jobs or not output_format,
        use_container_width=True,
    )

    if run_queue:
        overall = st.progress(0)
        status_line = st.empty()
        for index, job in enumerate(pending_jobs):
            status_line.info(f"（{index + 1}/{len(pending_jobs)}）正在轉譯 {job['name']}...")
            job["status"] = "running"
            try:
                job["record"] = transcribe_job(job, settings, device)
                job["status"] = "done"
                job["error"] = None
                if job["is_temp"] and cleanup_temp_file:
                    cleanup_file(Path(job["path"]))
            except Exception as exc:
                job["status"] = "error"
                job["error"] = str(exc)
            overall.progress((index + 1) / len(pending_jobs))
        status_line.empty()
        overall.empty()
        failed = sum(1 for job in pending_jobs if job["status"] == "error")
        st.session_state.queue_report = (len(pending_jobs) - failed, failed)
        st.rerun()

    if report := st.session_state.pop("queue_report", None):
        succeeded, failed = report
        if succeeded:
            st.success(f"完成 {succeeded} 個檔案的轉譯。")
        if failed:
            st.error(f"{failed} 個檔案轉譯失敗，詳情見上方佇列。")


def render_range_editor(job: dict) -> None:
    """Waveform and A-B range editor for one queue entry."""
    source_path = Path(job["path"])
    st.subheader(f"設定片段：{job['name']}")
    try:
        source_stat = source_path.stat()
        with st.spinner("正在分析音軌並建立波形..."):
            duration, times, amplitudes = analyze_media(
                str(source_path.resolve()), source_stat.st_size, source_stat.st_mtime_ns
            )
    except (OSError, MediaProcessingError) as exc:
        st.error(f"無法建立音訊波形：{exc}")
        return

    source_id = job["id"]
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
                extract_audio_range(source_path, preview_file, selected_start, preview_end)
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
        return

    # The queue entry owns the range; session_state only backs the widgets above.
    job["start"], job["end"] = float(selected_start), float(selected_end)
    st.button(
        "收起片段設定",
        key=f"unfocus_{source_id}",
        on_click=lambda: st.session_state.pop("focused_job", None),
    )


def render_results(output_formats: list[str]) -> None:
    """Staging area: every finished transcript stays here for editing and saving."""
    finished = [job for job in get_queue() if job["status"] == "done" and job["record"]]
    st.header("5. 結果暫存區")
    if not finished:
        st.info("還沒有完成的轉譯結果。轉譯完成後會留在這裡，可以隨時回來繼續編輯與下載。")
        return

    st.caption(
        f"共 {len(finished)} 份結果暫存在這個工作階段中。編輯後的內容會保留，"
        "重新整理瀏覽器才會清空。"
    )
    labels = {
        f"{job['name']}｜{job['record']['created_at']}": job for job in finished
    }
    selected_label = st.selectbox("選擇要編輯的結果", list(labels), index=len(labels) - 1)
    job = labels[selected_label]
    record = job["record"]
    source_path = Path(record["source_path"])

    if output_formats and st.button(
        f"把全部 {len(finished)} 份結果存到各自的來源資料夾",
        key="write_all_results",
        use_container_width=True,
        help="每份結果都會存在它自己的原始檔案旁邊，並覆蓋同名檔案。",
    ):
        written, failures = 0, []
        for entry in finished:
            try:
                save_result_files(entry["record"], output_formats)
                written += 1
            except Exception as exc:
                failures.append(f"{entry['name']}：{exc}")
        st.success(f"已儲存 {written} 份結果。")
        for failure in failures:
            st.error(failure)

    timings = record["timings"]
    segment_count = len(record["result"].get("segments", []))
    if record.get("is_subtitle"):
        # An imported subtitle has no inference to report; realtime would divide by zero.
        result_metrics = st.columns(3)
        result_metrics[0].metric("字幕段落", segment_count)
        result_metrics[1].metric("字幕長度", format_hhmmss(record["source_duration"]))
        result_metrics[2].metric("狀態", "已修改" if record.get("edited") else "原始")
        st.caption(f"{record['created_at']}｜匯入自 {record['source_path']}")
    else:
        record_duration = record["end"] - record["start"]
        realtime = record_duration / max(timings["inference"], 0.001)
        result_metrics = st.columns(4)
        result_metrics[0].metric("總耗時", f"{timings['total']:.1f} 秒")
        result_metrics[1].metric("模型推論", f"{timings['inference']:.1f} 秒")
        result_metrics[2].metric("處理速度", f"{realtime:.1f}× realtime")
        result_metrics[3].metric("字幕段落", segment_count)
        st.caption(
            f"{record['created_at']}｜模型 {record['model']}｜語言 {record['language']}｜"
            f"音訊 {timings['extraction']:.2f}s｜載入 {timings['model_load']:.2f}s｜"
            f"後處理 {timings['postprocess']:.2f}s"
        )
    if record.get("rule_hits"):
        detail = "、".join(f"{find}×{count}" for find, count in record["rule_hits"].items())
        st.caption(f"轉譯後已自動套用取代規則：{detail}")

    edit_tab, text_tab, download_tab = st.tabs(["字幕編輯", "純文字", "下載"])
    with edit_tab:
        subtitle_editor(record, source_path)

    with text_tab:
        segments = record["result"].get("segments", [])
        layout_column, timecode_column = st.columns(2)
        per_line = layout_column.toggle(
            "逐段換行", value=True, key=f"text_per_line_{record['id']}"
        )
        with_timecode = timecode_column.toggle(
            "顯示時間碼", value=False, key=f"text_timecode_{record['id']}", disabled=not per_line
        )
        if per_line:
            lines = []
            for segment in segments:
                text = str(segment.get("text", "")).strip()
                if with_timecode:
                    text = f"[{format_hhmmss(float(segment.get('start', 0.0)))}] {text}"
                lines.append(text)
            full_text = "\n".join(lines)
        else:
            full_text = str(record["result"].get("text", "")).strip()
        st.text_area("完整文字", value=full_text, height=420, disabled=True)
        st.caption(f"共 {len(full_text):,} 個字元；點進文字框可全選複製。")

    with download_tab:
        if not output_formats:
            st.warning("請先在左側選擇至少一種輸出格式。")
        else:
            try:
                destination = resolve_output_directory(record)
                stem = output_stem(record)
                st.markdown(f"**儲存位置**　`{destination}`")
                if record.get("is_temp"):
                    st.caption("上傳的檔案沒有原始路徑可寫回，因此存到 outputs/。")
                else:
                    st.caption("與原始檔案同一個資料夾。")

                existing = [
                    f"{stem}.{extension}"
                    for extension in output_formats
                    if (destination / f"{stem}.{extension}").exists()
                ]
                if existing:
                    st.warning(f"按下儲存會覆蓋既有檔案：{'、'.join(existing)}")

                if st.button(
                    "儲存到上述位置",
                    key=f"save_files_{record['id']}",
                    type="primary",
                    use_container_width=True,
                ):
                    saved_directory, saved_files = save_result_files(record, output_formats)
                    st.success(
                        f"已儲存 {len(saved_files)} 個檔案到 {saved_directory}："
                        f"{'、'.join(path.name for path in saved_files)}"
                    )

                st.divider()
                payloads = subtitle_payloads(record, output_formats)
                download_columns = st.columns(len(payloads))
                for index, (extension, filename, data) in enumerate(payloads):
                    download_columns[index].download_button(
                        label=f"下載 .{extension}",
                        data=data,
                        file_name=filename,
                        mime=MIME_TYPES.get(extension, "text/plain"),
                        use_container_width=True,
                    )
                st.caption("下載不會寫入上面的位置；字幕時間碼對應原始媒體。")
            except Exception as exc:
                st.error(f"無法產生輸出檔：{exc}")


if __name__ == "__main__":
    main()
