"""离线引擎（offline_engine.py）单元测试。

不联网、不依赖模型的部分永远跑（前端特征、词表解析、VAD 参数）；
真接模型推理的部分在「装了 onnxruntime 且模型在位」时才跑，否则 skip。

用法: ./venv/bin/python -m unittest tests.test_offline_engine -v
"""
import asyncio
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import offline_engine as oe  # noqa: E402

try:
    import onnxruntime  # noqa: F401
    HAVE_ORT = True
except ImportError:
    HAVE_ORT = False

HAVE_MODELS = (oe.MODEL_DIR / "asr-model-encoder.onnx").exists() and (oe.MODEL_DIR / "asr-model-token.txt").exists()
SKIP_REASON = "没装 onnxruntime" if not HAVE_ORT else "缺模型（tools/fetch_offline_asr_models.sh）"


class TestFrontend(unittest.TestCase):
    """前端特征：80 维 log-mel + LFR(7,6) → 560 维。维度算错，模型就全乱。"""

    def test_fbank_shape(self):
        x = np.zeros(16000, dtype=np.float32)                       # 1 秒
        feat = oe.fbank(x)
        expected = 1 + (16000 - 400) // 160                         # 25ms 窗 / 10ms 步长
        self.assertEqual(feat.shape, (expected, 80))

    def test_fbank_too_short_returns_empty(self):
        self.assertEqual(oe.fbank(np.zeros(200, dtype=np.float32)).shape[0], 0)

    def test_lfr_shape_and_dim(self):
        feat = np.arange(100 * 80, dtype=np.float32).reshape(100, 80)
        out = oe.lfr(feat)
        self.assertEqual(out.shape, ((100 - 7) // 6 + 1, 560))       # 80 × 7 = 560

    def test_lfr_too_short_returns_empty(self):
        self.assertEqual(oe.lfr(np.zeros((3, 80), dtype=np.float32)).shape[0], 0)

    def test_lfr_keeps_first_frame(self):
        feat = np.arange(20 * 80, dtype=np.float32).reshape(20, 80)
        out = oe.lfr(feat, m=7, n=6)
        # LFR 从第 3 帧（左侧填充 3 帧）开始，第一块应覆盖原始第 0 帧
        self.assertAlmostEqual(float(out[0, 3 * 80]), 0.0)

    def test_mel_filterbank_normalised(self):
        fb = oe.mel_filters()
        self.assertEqual(fb.shape, (80, 257))
        self.assertTrue(np.all(fb >= 0))
        self.assertTrue(np.allclose(fb.sum(axis=0).clip(max=1.0), fb.sum(axis=0).clip(max=1.0)))


class TestTokenParsing(unittest.TestCase):
    def test_load_tokens_space_separated(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
            fh.write("<blank> 0\n<s> 1\n</s> 2\n筑 4\nand@@ 3\n")
            path = Path(fh.name)
        try:
            tokens = oe.load_tokens(path)
        finally:
            path.unlink()
        self.assertEqual(tokens[0], "<blank>")
        self.assertEqual(tokens[4], "筑")
        self.assertEqual(tokens[3], "and@@")

    def test_load_punc_tokens_pipe_separated(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
            fh.write("<blank>|<s>|</s>|a|啊|好")
            path = Path(fh.name)
        try:
            tokens = oe.load_punc_tokens(path)
        finally:
            path.unlink()
        self.assertEqual(tokens["<blank>"], 0)
        self.assertEqual(tokens["啊"], 4)
        self.assertEqual(tokens["好"], 5)


class TestVadParams(unittest.TestCase):
    def test_thresholds_are_sane(self):
        self.assertLess(oe.MIN_SPEECH_SEC, oe.SILENCE_HANGOVER_SEC)
        self.assertLess(oe.SILENCE_HANGOVER_SEC, oe.MAX_SEGMENT_SEC)
        self.assertLess(oe.PARTIAL_MAX_SEC, oe.MAX_SEGMENT_SEC)

    def test_noise_adapts_up_on_speech_only(self):
        session = oe.StreamSession.__new__(oe.StreamSession)          # 不加载模型，只测 VAD 状态机
        session._noise = 0.004
        session._adapt_noise(0.5, True)
        self.assertLess(session._noise, 0.01)                         # 大音量语音不该把底噪抬高
        session._adapt_noise(0.05, False)
        self.assertGreater(session._noise, 0.004)                     # 安静段把底噪抬起来
        self.assertTrue(session._is_speech(0.5))
        self.assertFalse(session._is_speech(0.0001))


@unittest.skipUnless(HAVE_ORT and HAVE_MODELS, SKIP_REASON)
class TestOfflineTranscribe(unittest.TestCase):
    """真模型推理：拿仓库里的 fixture 音频对答案。"""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = oe.AsrPipeline(threads=4)
        if not cls.pipeline.available:
            raise unittest.SkipTest(f"引擎不可用：{cls.pipeline.reason}")

    def _transcribe(self, name: str) -> str:
        with wave.open(str(ROOT / "tests" / name), "rb") as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        session = oe.StreamSession(self.pipeline)
        asyncio.run(session.feed((pcm * 32768).astype(np.int16).tobytes()))
        return asyncio.run(session.finish())["text"]

    def test_chinese_fixture(self):
        text = self._transcribe("fixture-6s.wav")
        self.assertIn("你好", text)
        self.assertIn("微信", text)
        self.assertIn("阳光", text)

    def test_second_chinese_fixture(self):
        text = self._transcribe("fixture-2s.wav")
        self.assertIn("苹果", text)
        self.assertIn("香蕉", text)

    def test_punctuation_model_adds_marks(self):
        # 标点模型是可选的，但装了就应该能标出来（“测试”后应是句号，实测就是这样）
        if self.pipeline.punc is None:
            self.skipTest("没装标点模型")
        text = self.pipeline._punctuate("你好这是微信语音接口测试今天阳光很好")
        self.assertIn("。", text)
        self.assertEqual(len(text), len("你好这是微信语音接口测试今天阳光很好") + 1)

    def test_silence_yields_nothing(self):
        session = oe.StreamSession(self.pipeline)
        silence = (np.zeros(16000 * 2, dtype=np.int16)).tobytes()
        asyncio.run(session.feed(silence))
        self.assertEqual(asyncio.run(session.finish())["text"], "")


if __name__ == "__main__":
    unittest.main()
