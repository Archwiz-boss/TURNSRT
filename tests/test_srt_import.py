from __future__ import annotations

import unittest

from transcript_utils import parse_srt

BASIC = """1
00:00:00,000 --> 00:00:02,500
第一段字幕

2
00:00:02,500 --> 00:00:05,000
第二段字幕
"""


class ParseSrtTests(unittest.TestCase):
    def test_parses_cues_into_whisper_style_segments(self):
        result = parse_srt(BASIC)
        self.assertEqual(
            result["segments"],
            [
                {"start": 0.0, "end": 2.5, "text": "第一段字幕"},
                {"start": 2.5, "end": 5.0, "text": "第二段字幕"},
            ],
        )
        self.assertEqual(result["text"], "第一段字幕 第二段字幕")

    def test_parses_hours_and_milliseconds(self):
        result = parse_srt("1\n01:17:47,700 --> 02:26:45,317\n內容\n")
        segment = result["segments"][0]
        self.assertAlmostEqual(segment["start"], 4667.7, places=3)
        self.assertAlmostEqual(segment["end"], 8805.317, places=3)

    def test_joins_a_multi_line_cue_with_a_space(self):
        result = parse_srt("1\n00:00:00,000 --> 00:00:01,000\n上一行\n下一行\n")
        self.assertEqual(result["segments"][0]["text"], "上一行 下一行")

    def test_tolerates_crlf_and_a_byte_order_mark(self):
        result = parse_srt("﻿1\r\n00:00:00,000 --> 00:00:01,000\r\n內容\r\n")
        self.assertEqual(result["segments"][0]["text"], "內容")

    def test_accepts_vtt_style_decimal_separator_and_header(self):
        result = parse_srt("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\n內容\n")
        self.assertEqual(result["segments"][0]["end"], 1.5)

    def test_cue_index_is_not_swallowed_without_a_blank_line(self):
        text = "1\n00:00:00,000 --> 00:00:01,000\n甲\n2\n00:00:01,000 --> 00:00:02,000\n乙\n"
        self.assertEqual([s["text"] for s in parse_srt(text)["segments"]], ["甲", "乙"])

    def test_a_numeric_subtitle_line_is_preserved(self):
        result = parse_srt("1\n00:00:00,000 --> 00:00:01,000\n2024\n")
        self.assertEqual(result["segments"][0]["text"], "2024")

    def test_misnumbered_and_missing_indices_still_parse(self):
        text = "00:00:00,000 --> 00:00:01,000\n甲\n\n99\n00:00:01,000 --> 00:00:02,000\n乙\n"
        self.assertEqual([s["text"] for s in parse_srt(text)["segments"]], ["甲", "乙"])

    def test_empty_cues_are_dropped(self):
        text = "1\n00:00:00,000 --> 00:00:01,000\n\n2\n00:00:01,000 --> 00:00:02,000\n有內容\n"
        self.assertEqual([s["text"] for s in parse_srt(text)["segments"]], ["有內容"])

    def test_reversed_timestamps_never_produce_a_negative_span(self):
        result = parse_srt("1\n00:00:05,000 --> 00:00:01,000\n內容\n")
        segment = result["segments"][0]
        self.assertGreaterEqual(segment["end"], segment["start"])

    def test_empty_input_yields_an_empty_result(self):
        self.assertEqual(parse_srt(""), {"text": "", "segments": []})
        self.assertEqual(parse_srt("沒有任何時間碼的檔案"), {"text": "", "segments": []})

    def test_round_trips_through_the_editing_helpers(self):
        from transcript_utils import replace_across_segments, segment_rows

        record = {"result": parse_srt(BASIC), "edited": False}
        rows = segment_rows(record["result"]["segments"])
        self.assertEqual([row["段落"] for row in rows], [1, 2])
        self.assertEqual(replace_across_segments(record, "第一段", "第 1 段"), 1)
        self.assertEqual(record["result"]["segments"][0]["text"], "第 1 段字幕")

    def test_result_can_be_written_back_out_as_srt(self):
        import tempfile
        from pathlib import Path

        from whisper.utils import get_writer

        with tempfile.TemporaryDirectory() as directory:
            writer = get_writer("srt", directory)
            writer(parse_srt(BASIC), str(Path(directory) / "out.mp4"))
            written = (Path(directory) / "out.srt").read_text(encoding="utf-8")
        self.assertIn("第一段字幕", written)
        self.assertIn("00:00:02,500 --> 00:00:05,000", written)


if __name__ == "__main__":
    unittest.main()
