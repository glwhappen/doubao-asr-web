"""说话人标签对齐的单元测试（不联网、不需要 sherpa-onnx）。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import diarization  # noqa: E402


class AssignSpeakersCase(unittest.TestCase):
    def test_assigns_by_max_overlap(self):
        lines = [
            {"text": "第一句", "start": 0.0, "end": 3.0},
            {"text": "第二句", "start": 3.5, "end": 6.0},
        ]
        segments = [
            diarization.Segment(0.0, 3.2, 1),
            diarization.Segment(3.4, 6.2, 2),
        ]
        speakers = diarization.assign_speakers(lines, segments)
        self.assertEqual(speakers, 2)
        self.assertEqual([l["spk"] for l in lines], [1, 2])

    def test_partial_overlap_wins_by_area(self):
        """一句横跨两个人的片段时，取重叠更多的那个人。"""
        lines = [{"text": "重叠句", "start": 2.0, "end": 5.0}]
        segments = [
            diarization.Segment(0.0, 2.4, 1),      # 重叠 0.4s
            diarization.Segment(2.4, 5.2, 2),      # 重叠 2.6s
        ]
        diarization.assign_speakers(lines, segments)
        self.assertEqual(lines[0]["spk"], 2)

    def test_no_overlap_leaves_line_untouched(self):
        lines = [{"text": "孤句", "start": 20.0, "end": 21.0}]
        diarization.assign_speakers(lines, [diarization.Segment(0.0, 5.0, 1)])
        self.assertNotIn("spk", lines[0])

    def test_empty_segments_is_noop(self):
        lines = [{"text": "x", "start": 0.0, "end": 1.0}]
        self.assertEqual(diarization.assign_speakers(lines, []), 0)
        self.assertNotIn("spk", lines[0])

    def test_tiny_speaker_merges_into_neighbour(self):
        """总时长过短的簇（模型常把同一人的一小截单独分出来）应并到相邻说话人。"""
        segments = [
            diarization.Segment(0.0, 5.0, 0),
            diarization.Segment(5.0, 12.0, 1),
            diarization.Segment(12.0, 12.4, 2),     # 只有 0.4s
            diarization.Segment(12.4, 18.0, 1),
        ]
        merged = diarization._merge_tiny_speakers(segments)
        self.assertEqual({s.speaker for s in merged}, {0, 1})
        self.assertEqual(merged[2].speaker, 1)

    def test_all_tiny_keeps_longest(self):
        segments = [
            diarization.Segment(0.0, 0.4, 0),
            diarization.Segment(0.4, 1.0, 1),
        ]
        merged = diarization._merge_tiny_speakers(segments)
        self.assertTrue(merged)


if __name__ == "__main__":
    unittest.main(verbosity=2)
