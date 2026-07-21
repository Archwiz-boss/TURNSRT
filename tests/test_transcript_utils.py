from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from transcript_utils import (
    build_waveform,
    convert_result_text,
    extract_audio_range,
    format_hhmmss,
    offset_result_timestamps,
    parse_hhmmss,
    probe_duration,
    range_suffix,
)


class TranscriptUtilsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "sample.wav"

        sample_rate = 16_000
        tone_time = np.arange(sample_rate, dtype=np.float64) / sample_rate
        tone = (np.sin(2 * np.pi * 440 * tone_time) * 12_000).astype("<i2")
        silence = np.zeros(sample_rate, dtype="<i2")
        samples = np.concatenate((tone, silence, tone))
        with wave.open(str(self.source), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(samples.tobytes())

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_media_probe_waveform_and_range_extraction(self):
        duration = probe_duration(self.source)
        self.assertAlmostEqual(duration, 3.0, places=2)

        times, amplitudes = build_waveform(self.source, duration, target_points=300)
        self.assertEqual(len(times), len(amplitudes))
        self.assertGreater(len(times), 200)
        middle = amplitudes[(times >= 1.1) & (times <= 1.9)]
        outer = amplitudes[(times <= 0.9) | (times >= 2.1)]
        self.assertLess(float(middle.mean()), float(outer.mean()) * 0.1)

        extracted = extract_audio_range(self.source, self.root / "clip.wav", 0.75, 1.75)
        self.assertAlmostEqual(probe_duration(extracted), 1.0, places=2)

    def test_timestamp_offset_preserves_original_result(self):
        original = {
            "text": " test",
            "segments": [
                {
                    "start": 0.25,
                    "end": 1.0,
                    "text": " test",
                    "words": [{"start": 0.3, "end": 0.6, "word": " test"}],
                }
            ],
        }
        shifted = offset_result_timestamps(original, 10.0)
        self.assertEqual(original["segments"][0]["start"], 0.25)
        self.assertEqual(shifted["segments"][0]["start"], 10.25)
        self.assertEqual(shifted["segments"][0]["end"], 11.0)
        self.assertEqual(shifted["segments"][0]["words"][0]["start"], 10.3)

    def test_srt_writer_uses_original_media_timeline(self):
        from whisper.utils import get_writer

        shifted = offset_result_timestamps(
            {
                "text": " 測試",
                "segments": [{"start": 0.25, "end": 1.0, "text": " 測試"}],
            },
            10.0,
        )
        get_writer("srt", str(self.root))(shifted, str(self.root / "clip.wav"))
        srt_text = (self.root / "clip.srt").read_text(encoding="utf-8")
        self.assertIn("00:00:10,250 --> 00:00:11,000", srt_text)

    def test_opencc_conversion_covers_result_segments_and_words(self):
        try:
            from opencc import OpenCC
        except ImportError as exc:
            self.skipTest(str(exc))
        original = {
            "text": "软件里面",
            "segments": [
                {"text": "软件里面", "words": [{"word": "软件"}, {"word": "里面"}]}
            ],
        }
        converted = convert_result_text(original, OpenCC("s2twp"))
        self.assertEqual(converted["text"], "軟體裡面")
        self.assertEqual(converted["segments"][0]["text"], "軟體裡面")
        self.assertEqual(converted["segments"][0]["words"][0]["word"], "軟體")
        self.assertEqual(original["text"], "软件里面")

    def test_range_suffix(self):
        self.assertEqual(range_suffix(0.0, 20.0, 20.0), "")
        self.assertEqual(
            range_suffix(65.25, 70.5, 100.0),
            "__00-01-05.250_to_00-01-10.500",
        )

    def test_hhmmss_format_and_parse(self):
        self.assertEqual(format_hhmmss(0), "00:00:00")
        self.assertEqual(format_hhmmss(7981.567), "02:13:02")
        self.assertEqual(format_hhmmss(27 * 3600 + 65), "27:01:05")
        self.assertEqual(parse_hhmmss("02:13:02"), 7982.0)
        self.assertEqual(parse_hhmmss("27:01:05"), 27 * 3600 + 65)
        with self.assertRaises(ValueError):
            parse_hhmmss("02:99:00")
        with self.assertRaises(ValueError):
            parse_hhmmss("7982")


if __name__ == "__main__":
    unittest.main()
