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

    def test_nearby_gap_gets_nearest_speaker(self):
        """和分离片段没有重叠、但时间挨着 → 用最近的说话人补上（避免一段里有的有标签有的没有）。"""
        lines = [{"text": "漏检的短句", "start": 5.5, "end": 6.0}]
        diarization.assign_speakers(lines, [diarization.Segment(0.0, 5.0, 1)])
        self.assertEqual(lines[0]["spk"], 1)

    def test_far_gap_leaves_line_untouched(self):
        """离所有片段都太远（>10s）就不猜，保持没有标签。"""
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


class VoiceprintCase(unittest.TestCase):
    """声纹库匹配（不需要模型：直接喂造出来的向量）。"""

    def setUp(self):
        import numpy as np
        self.np = np
        self.tracker = diarization.SpeakerTracker()

    @staticmethod
    def vec(*values):
        import numpy as np
        v = np.array(values, dtype="float32")
        return v / (float((v ** 2).sum()) ** 0.5)

    def test_enrolled_voice_is_recognized(self):
        self.tracker.add_voiceprint("甲", self.vec(1, 0, 0, 0, 0, 0, 0, 0))
        spk, name = self.tracker.match_vector(self.vec(0.99, 0.02, 0, 0, 0, 0, 0, 0))
        self.assertEqual(name, "甲")
        self.assertEqual(spk, 1)

    def test_stranger_gets_new_speaker_not_a_name(self):
        """陌生人不能被硬塞成已注册的名字；他自己占一个会话内编号。"""
        self.tracker.add_voiceprint("甲", self.vec(1, 0, 0, 0, 0, 0, 0, 0))
        spk, name = self.tracker.match_vector(self.vec(0, 1, 0, 0, 0, 0, 0, 0))
        self.assertIsNone(name)
        self.assertEqual(self.tracker.count, 1)       # 会话内建立 1 个说话人
        # 之后「甲」本人出现 → 拿另一个编号，并且带上名字
        spk2, name2 = self.tracker.match_vector(self.vec(1, 0, 0, 0, 0, 0, 0, 0))
        self.assertEqual(name2, "甲")
        self.assertNotEqual(spk, spk2)
        self.assertEqual(self.tracker.count, 2)

    def test_same_person_twice_keeps_one_speaker(self):
        spk1, _ = self.tracker.match_vector(self.vec(1, 0, 0, 0, 0, 0, 0, 0))
        spk2, _ = self.tracker.match_vector(self.vec(0.97, 0.05, 0, 0, 0, 0, 0, 0))
        self.assertEqual(spk1, spk2)
        self.assertEqual(self.tracker.count, 1)

    def test_bad_voiceprint_is_ignored(self):
        self.assertFalse(self.tracker.add_voiceprint("空", []))
        self.assertFalse(self.tracker.add_voiceprint("短", [1, 2]))
        self.assertFalse(self.tracker.add_voiceprint("", self.vec(1, 0, 0, 0, 0, 0, 0, 0)))
        self.assertEqual(self.tracker.count, 0)

    def test_vectors_exports_centroids(self):
        self.tracker.match_vector(self.vec(1, 0, 0, 0, 0, 0, 0, 0))
        vectors = self.tracker.vectors()
        self.assertEqual(list(vectors), ["1"])
        self.assertEqual(len(vectors["1"]), 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
