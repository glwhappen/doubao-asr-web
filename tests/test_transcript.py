"""上游结果 → 行/段 的回归测试（不需要联网）。

重点覆盖 2026-10-16 线上反馈的坑：说一个「哎」，上游对同一个短语气词反复改字
（唉→哎→唉，三条消息 start_time 完全相同、只有 end_time 在涨），
以前会被切成三行。
"""
import sys
import time
import unittest

sys.path.insert(0, "/root/doubao-asr-web")
import server  # noqa: E402


def item(text, start, end, interim, vad=False, index=0):
    return {"text": text, "start_time": start, "end_time": end,
            "is_interim": interim, "is_vad_finished": vad, "index": index}


class TranscriptCase(unittest.TestCase):
    def setUp(self):
        # 词库现在按会话传入，测试里用空的（不做替换），需要测替换再显式传
        self.lexicon = server.Lexicon()

    def feed(self, items):
        t = server.Transcript(time.time(), self.lexicon)
        for it in items:
            t.apply(it["text"], it["is_interim"], None,
                    start_hint=it["start_time"], end_hint=it["end_time"],
                    vad_finished=it["is_vad_finished"], cap=it["end_time"])
        t.close()
        return t.lines

    def test_short_interjection_rewrites_stay_one_line(self):
        """「哎」被上游改成 唉/哎/唉，必须还是一行。"""
        lines = self.feed([
            item("唉", 1.33, 2.679, True),
            item("哎", 1.33, 3.319, True),
            item("唉", 1.33, 3.400, False, vad=True),
        ])
        self.assertEqual(len(lines), 1, f"应当只有一行，实际 {[l['text'] for l in lines]}")
        self.assertEqual(lines[0]["start"], 1.33)
        self.assertEqual(lines[0]["end"], 3.4)
        self.assertTrue(lines[0]["final"])

    def test_short_words_contiguous_merge(self):
        """两个短语气词首尾相接、无停顿 → 合并成一行。"""
        lines = self.feed([
            item("哎", 0.0, 1.0, False, vad=True),
            item("唉", 1.0, 2.0, False, vad=True),
        ])
        self.assertEqual(len(lines), 1)

    def test_pause_then_speak_keeps_two_lines(self):
        """停顿后重新开口（起点跳到上一段结束之后）→ 必须保留两行。"""
        lines = self.feed([
            item("你好，这是微信语音接口测试，今天阳光很好。", 0.0, 7.4, False, vad=True),
            item("你", 14.11, 15.46, True),
            item("你好，这是微信语音接口测试，今天阳光很好。", 14.11, 20.56, False, vad=True),
        ])
        self.assertEqual(len(lines), 2, f"实际 {[l['text'] for l in lines]}")

    def test_two_sentences_split_by_punctuation(self):
        """连续说两句（无停顿）→ 按标点切成两行。"""
        lines = self.feed([
            item("你好，这是微信语音接口测试。今天阳光很好。", 0.0, 6.28, False, vad=True),
        ])
        self.assertEqual(len(lines), 2)

    def test_same_message_repeated_sentences_not_merged(self):
        """同一条消息里按标点切出的三个相同句子（连说三遍）→ 保持三行。"""
        lines = self.feed([
            item("你好，这是微信语音接口测试。你好，这是微信语音接口测试。你好，这是微信语音接口测试。",
                 0.0, 20.8, False, vad=True),
        ])
        self.assertEqual(len(lines), 3, f"实际 {[l['text'] for l in lines]}")

    def test_session_lexicon_replaces_text(self):
        """词库随会话传入 → 只对本次会话生效，并留下 orig/fixes 供前端显示。"""
        lex = server.Lexicon.from_payload({"corrections": [{"wrong": "接口测试", "right": "接口验证"}]})
        t = server.Transcript(time.time(), lex)
        t.apply("你好，这是微信语音接口测试。", False, None, start_hint=0, end_hint=2, cap=2)
        t.close()
        self.assertIn("接口验证", t.lines[0]["text"])
        self.assertEqual(t.lines[0]["orig"], "你好，这是微信语音接口测试。")
        self.assertEqual(t.lines[0]["fixes"], [["接口测试", "接口验证"]])

    def test_hotwords_come_from_corrections_and_manual(self):
        """热词 = 手加的 + 纠正对里抽出来的那个词。"""
        lex = server.Lexicon.from_payload({
            "corrections": [{"wrong": "这个模型叫豆包。", "right": "这个模型叫豆包输入法。"}],
            "hotwords": ["飞棚听写"],
        })
        words = lex.hotwords()
        self.assertIn("飞棚听写", words)          # 手加的
        self.assertIn("输入法", words)            # 从纠正对里抽出来的

    def test_empty_lexicon_does_not_touch_text(self):
        """没有词库时不能改动识别结果。"""
        t = server.Transcript(time.time(), server.Lexicon())
        t.apply("你好，这是微信语音接口测试。", False, None, start_hint=0, end_hint=2, cap=2)
        t.close()
        self.assertEqual(t.lines[0]["text"], "你好，这是微信语音接口测试。")
        self.assertNotIn("fixes", t.lines[0])

    def test_long_repeat_not_merged(self):
        """同一句真说了两遍、中间有停顿 → 不能合并。"""
        lines = self.feed([
            item("今天天气不错，我们出去走走吧。", 0.0, 3.0, False, vad=True),
            item("今天天气不错，我们出去走走吧。", 5.0, 8.0, False, vad=True),
        ])
        self.assertEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
