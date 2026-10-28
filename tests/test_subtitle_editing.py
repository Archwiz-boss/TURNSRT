from __future__ import annotations

import unittest

from transcript_utils import (
    FAST_SUBTITLE_CPS,
    rebuild_result_text,
    editor_rows_from_delta,
    paginate_rows,
    replace_across_segments,
    restore_original_text,
    result_for_output,
    segment_rows,
    set_segment_text,
    update_result_from_editor,
)


def make_record(texts, *, spans=None, words=False):
    spans = spans or [(index * 2.0, index * 2.0 + 2.0) for index in range(len(texts))]
    segments = []
    for text, (start, end) in zip(texts, spans):
        segment = {"start": start, "end": end, "text": text}
        if words:
            segment["words"] = [{"word": text, "start": start, "end": end}]
        segments.append(segment)
    return {
        "result": {"text": "".join(texts), "segments": segments},
        "original_texts": list(texts),
        "edited": False,
    }


class SetSegmentTextTests(unittest.TestCase):
    def test_preserves_whispers_leading_space(self):
        segment = {"text": " 原本的字幕"}
        self.assertTrue(set_segment_text(segment, "新的字幕"))
        self.assertEqual(segment["text"], " 新的字幕")

    def test_reports_no_change_when_text_is_identical(self):
        segment = {"text": " 原本的字幕"}
        self.assertFalse(set_segment_text(segment, "原本的字幕"))

    def test_strips_whitespace_the_user_typed_at_the_edges(self):
        segment = {"text": "字幕"}
        set_segment_text(segment, "  新字幕  ")
        self.assertEqual(segment["text"], "新字幕")


class UpdateFromEditorTests(unittest.TestCase):
    def test_maps_rows_by_segment_number_not_position(self):
        record = make_record(["第一段", "第二段", "第三段"])
        # What a search-filtered table hands back: only segment 3.
        changed = update_result_from_editor(record, [{"段落": 3, "字幕": "改過的第三段"}])
        self.assertEqual(changed, 1)
        texts = [seg["text"] for seg in record["result"]["segments"]]
        self.assertEqual(texts, ["第一段", "第二段", "改過的第三段"])

    def test_rebuilds_the_full_text_after_an_edit(self):
        record = make_record(["第一段", "第二段"])
        update_result_from_editor(record, [{"段落": 1, "字幕": "改了"}])
        self.assertEqual(record["result"]["text"], "改了第二段")

    def test_marks_the_record_as_edited_only_when_something_changed(self):
        record = make_record(["第一段"])
        self.assertEqual(update_result_from_editor(record, [{"段落": 1, "字幕": "第一段"}]), 0)
        self.assertFalse(record["edited"])
        update_result_from_editor(record, [{"段落": 1, "字幕": "換掉"}])
        self.assertTrue(record["edited"])

    def test_ignores_rows_pointing_outside_the_segment_list(self):
        record = make_record(["第一段"])
        rows = [{"段落": 0, "字幕": "x"}, {"段落": 99, "字幕": "y"}, {"段落": "abc", "字幕": "z"}]
        self.assertEqual(update_result_from_editor(record, rows), 0)
        self.assertEqual(record["result"]["segments"][0]["text"], "第一段")

    def test_accepts_a_dataframe_like_object(self):
        class FakeFrame:
            def to_dict(self, orient):
                assert orient == "records"
                return [{"段落": 1, "字幕": "來自 DataFrame"}]

        record = make_record(["第一段"])
        self.assertEqual(update_result_from_editor(record, FakeFrame()), 1)
        self.assertEqual(record["result"]["segments"][0]["text"], "來自 DataFrame")


class ReplaceAcrossSegmentsTests(unittest.TestCase):
    def test_replaces_every_occurrence_in_every_segment(self):
        record = make_record(["用 agent teams", "另一個 agent teams", "無關"])
        self.assertEqual(replace_across_segments(record, "agent teams", "Agent Teams"), 2)
        self.assertEqual(record["result"]["segments"][0]["text"], "用 Agent Teams")
        self.assertEqual(record["result"]["segments"][1]["text"], "另一個 Agent Teams")

    def test_empty_replacement_deletes_the_text(self):
        record = make_record(["嗯嗯這樣子"])
        replace_across_segments(record, "嗯嗯", "")
        self.assertEqual(record["result"]["segments"][0]["text"], "這樣子")

    def test_empty_search_is_a_no_op(self):
        record = make_record(["原文"])
        self.assertEqual(replace_across_segments(record, "", "x"), 0)
        self.assertFalse(record["edited"])

    def test_search_with_no_hits_leaves_the_record_untouched(self):
        record = make_record(["原文"])
        self.assertEqual(replace_across_segments(record, "不存在", "x"), 0)
        self.assertFalse(record["edited"])


class RestoreOriginalTextTests(unittest.TestCase):
    def test_undoes_every_edit(self):
        record = make_record(["第一段", "第二段"])
        replace_across_segments(record, "第一段", "亂改")
        self.assertEqual(restore_original_text(record), 1)
        self.assertEqual(
            [seg["text"] for seg in record["result"]["segments"]], ["第一段", "第二段"]
        )
        self.assertFalse(record["edited"])

    def test_clears_the_edited_flag_even_when_text_already_matches(self):
        record = make_record(["第一段"])
        record["edited"] = True
        self.assertEqual(restore_original_text(record), 0)
        self.assertFalse(record["edited"])

    def test_restores_the_joined_text(self):
        record = make_record(["甲", "乙"])
        update_result_from_editor(record, [{"段落": 1, "字幕": "改"}])
        restore_original_text(record)
        self.assertEqual(record["result"]["text"], "甲乙")


class SegmentRowsTests(unittest.TestCase):
    def test_numbers_rows_from_one_and_formats_the_start_time(self):
        record = make_record(["甲", "乙"])
        rows = segment_rows(record["result"]["segments"])
        self.assertEqual([row["段落"] for row in rows], [1, 2])
        self.assertEqual(rows[1]["開始"], "00:00:02.000")

    def test_search_keeps_the_original_segment_number(self):
        record = make_record(["甲", "乙", "丙"])
        rows = segment_rows(record["result"]["segments"], search="丙")
        self.assertEqual([row["段落"] for row in rows], [3])

    def test_search_is_case_insensitive(self):
        record = make_record(["用 SubAgent 跑"])
        self.assertEqual(len(segment_rows(record["result"]["segments"], search="subagent")), 1)

    def test_reading_speed_is_characters_per_second(self):
        record = make_record(["一二三四"], spans=[(0.0, 2.0)])
        self.assertEqual(segment_rows(record["result"]["segments"])[0]["字/秒"], 2.0)

    def test_only_dense_keeps_lines_above_the_threshold(self):
        record = make_record(["短", "字" * 40], spans=[(0.0, 2.0), (2.0, 4.0)])
        rows = segment_rows(record["result"]["segments"], only_dense=True)
        self.assertEqual([row["段落"] for row in rows], [2])
        self.assertGreater(rows[0]["字/秒"], FAST_SUBTITLE_CPS)

    def test_zero_length_segment_does_not_divide_by_zero(self):
        record = make_record(["卡住"], spans=[(1.0, 1.0)])
        self.assertEqual(segment_rows(record["result"]["segments"])[0]["字/秒"], 0.0)


class PaginationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"段落": number, "字幕": str(number)} for number in range(1, 26)]

    def test_returns_one_page_without_overlapping_rows(self):
        visible, page, total_pages = paginate_rows(self.rows, 2, 10)
        self.assertEqual([row["段落"] for row in visible], list(range(11, 21)))
        self.assertEqual((page, total_pages), (2, 3))

    def test_clamps_a_stale_page_after_filtering(self):
        visible, page, total_pages = paginate_rows(self.rows[:3], 99, 10)
        self.assertEqual([row["段落"] for row in visible], [1, 2, 3])
        self.assertEqual((page, total_pages), (1, 1))

    def test_empty_results_still_report_one_page(self):
        self.assertEqual(paginate_rows([], 4, 10), ([], 1, 1))

    def test_rejects_invalid_page_size(self):
        with self.assertRaises(ValueError):
            paginate_rows(self.rows, 1, 0)


class EditorDeltaTests(unittest.TestCase):
    def test_maps_visible_row_index_to_stable_segment_number(self):
        visible = [{"段落": 41, "字幕": "甲"}, {"段落": 42, "字幕": "乙"}]
        state = {"edited_rows": {"1": {"字幕": "修改後"}}}
        self.assertEqual(
            editor_rows_from_delta(visible, state),
            [{"段落": 42, "字幕": "修改後"}],
        )

    def test_ignores_non_text_changes_and_invalid_indexes(self):
        visible = [{"段落": 8, "字幕": "原文"}]
        state = {"edited_rows": {"0": {"開始": "00:00:01"}, "9": {"字幕": "錯列"}}}
        self.assertEqual(editor_rows_from_delta(visible, state), [])


class ResultForOutputTests(unittest.TestCase):
    def test_unedited_result_is_passed_through_unchanged(self):
        record = make_record(["甲"], words=True)
        self.assertIs(result_for_output(record), record["result"])

    def test_edited_result_drops_word_timings_so_srt_uses_the_new_text(self):
        record = make_record(["甲", "乙"], words=True)
        update_result_from_editor(record, [{"段落": 1, "字幕": "改過"}])
        prepared = result_for_output(record)
        self.assertTrue(all("words" not in seg for seg in prepared["segments"]))
        self.assertEqual(prepared["segments"][0]["text"], "改過")

    def test_stripping_does_not_mutate_the_in_memory_result(self):
        record = make_record(["甲"], words=True)
        update_result_from_editor(record, [{"段落": 1, "字幕": "改過"}])
        result_for_output(record)
        self.assertIn("words", record["result"]["segments"][0])

    def test_edited_result_without_word_timings_is_passed_through(self):
        record = make_record(["甲"])
        update_result_from_editor(record, [{"段落": 1, "字幕": "改過"}])
        self.assertIs(result_for_output(record), record["result"])


class RebuildResultTextTests(unittest.TestCase):
    def test_joins_segments_without_adding_separators(self):
        result = {"segments": [{"text": " 甲"}, {"text": " 乙"}], "text": ""}
        rebuild_result_text(result)
        self.assertEqual(result["text"], " 甲 乙")


if __name__ == "__main__":
    unittest.main()
