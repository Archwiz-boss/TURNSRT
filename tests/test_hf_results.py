from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from transcript_utils import MediaProcessingError, build_result_from_chunks, read_wav_mono


def chunk(start, end, text):
    return {"timestamp": (start, end), "text": text}


class BuildResultFromChunksTests(unittest.TestCase):
    def test_maps_chunks_onto_whisper_style_segments(self):
        result = build_result_from_chunks([chunk(0.0, 1.5, " 我去呼叫他的"), chunk(1.5, 3.0, " 好的")])
        self.assertEqual(
            result["segments"],
            [
                {"start": 0.0, "end": 1.5, "text": "我去呼叫他的"},
                {"start": 1.5, "end": 3.0, "text": "好的"},
            ],
        )
        self.assertEqual(result["text"], "我去呼叫他的 好的")

    def test_open_ended_final_chunk_falls_back_to_clip_duration(self):
        result = build_result_from_chunks([chunk(0.0, 1.0, "甲"), chunk(1.0, None, "乙")], 4.5)
        self.assertEqual(result["segments"][1]["end"], 4.5)

    def test_open_ended_final_chunk_without_duration_collapses_to_its_start(self):
        result = build_result_from_chunks([chunk(2.0, None, "甲")])
        self.assertEqual(result["segments"][0], {"start": 2.0, "end": 2.0, "text": "甲"})

    def test_missing_end_resolves_from_the_next_chunk_start(self):
        result = build_result_from_chunks([chunk(0.0, None, "甲"), chunk(2.5, 3.0, "乙")], 9.0)
        self.assertEqual(result["segments"][0]["end"], 2.5)

    def test_missing_start_resolves_from_the_previous_chunk_end(self):
        result = build_result_from_chunks([chunk(0.0, 1.0, "甲"), chunk(None, 2.0, "乙")])
        self.assertEqual(result["segments"][1]["start"], 1.0)

    def test_chunk_without_any_timestamp_is_still_kept(self):
        result = build_result_from_chunks([{"text": "沒有時間碼"}], 5.0)
        self.assertEqual(result["segments"], [{"start": 0.0, "end": 5.0, "text": "沒有時間碼"}])

    def test_end_is_never_earlier_than_start(self):
        result = build_result_from_chunks([chunk(3.0, 1.0, "時間碼顛倒")])
        segment = result["segments"][0]
        self.assertGreaterEqual(segment["end"], segment["start"])

    def test_blank_chunks_are_dropped(self):
        result = build_result_from_chunks([chunk(0.0, 1.0, "  "), chunk(1.0, 2.0, "有內容")])
        self.assertEqual([seg["text"] for seg in result["segments"]], ["有內容"])

    def test_empty_input_yields_an_empty_result(self):
        self.assertEqual(build_result_from_chunks([]), {"text": "", "segments": []})
        self.assertEqual(build_result_from_chunks(None), {"text": "", "segments": []})

    def test_result_survives_the_downstream_offset_helper(self):
        from transcript_utils import offset_result_timestamps

        shifted = offset_result_timestamps(build_result_from_chunks([chunk(0.0, 1.0, "甲")]), 60.0)
        self.assertEqual(shifted["segments"][0]["start"], 60.0)
        self.assertEqual(shifted["segments"][0]["end"], 61.0)


class ReadWavMonoTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_wav(self, name, samples, channels=1, sampwidth=2):
        path = self.root / name
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(sampwidth)
            handle.setframerate(16_000)
            handle.writeframes(samples.tobytes())
        return path

    def test_scales_pcm16_into_the_minus_one_to_one_range(self):
        path = self.write_wav("mono.wav", np.array([0, 16384, -16384], dtype="<i2"))
        samples = read_wav_mono(path)
        self.assertEqual(samples.dtype, np.float32)
        np.testing.assert_allclose(samples, [0.0, 0.5, -0.5])

    def test_averages_stereo_channels_to_mono(self):
        # Interleaved L/R: frame 1 is (0.0, 0.5) -> 0.25, frame 2 is (0.25, 0.5) -> 0.375.
        path = self.write_wav("stereo.wav", np.array([0, 16384, 8192, 16384], dtype="<i2"), channels=2)
        np.testing.assert_allclose(read_wav_mono(path), [0.25, 0.375])

    def test_rejects_unsupported_sample_widths(self):
        path = self.write_wav("wide.wav", np.zeros(4, dtype="<i4"), sampwidth=4)
        with self.assertRaises(MediaProcessingError):
            read_wav_mono(path)

    def test_reads_a_file_produced_by_extract_audio_range(self):
        from transcript_utils import extract_audio_range

        source = self.write_wav(
            "source.wav",
            (np.sin(np.arange(32_000) / 16_000 * 2 * np.pi * 440) * 12_000).astype("<i2"),
        )
        clip = self.root / "clip.wav"
        extract_audio_range(source, clip, 0.5, 1.5)
        samples = read_wav_mono(clip)
        self.assertGreater(samples.size, 0)
        self.assertLessEqual(float(np.abs(samples).max()), 1.0)


if __name__ == "__main__":
    unittest.main()
