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


class _StubPipeline:
    """不加载模型，只回固定结果：用来单测切句状态机（CI 也能跑）。"""

    available = True
    reason = "stub"

    def __init__(self, quality=0.9, text="模拟文本。"):
        self.quality = quality
        self.text = text
        self.calls = 0

    async def transcribe(self, pcm, punctuate=True):
        self.calls += 1
        return self.text, self.quality


def _pcm(pattern, sr=16000, amp=0.08):
    """pattern: [(语音秒, 静音秒), ...] → PCM16 bytes（语音用固定种子噪声，能量稳定）。"""
    rng = np.random.default_rng(1)
    out = []
    for speech, silence in pattern:
        out.append((rng.normal(0, amp, int(speech * sr))).astype(np.float32))
        out.append(np.zeros(int(silence * sr), dtype=np.float32))
    x = np.clip(np.concatenate(out), -1, 1)
    return (x * 32768).astype(np.int16).tobytes()


def _run(pattern, quality=0.9):
    pipeline = _StubPipeline(quality=quality)
    session = oe.StreamSession(pipeline)
    pcm = _pcm(pattern)
    for i in range(0, len(pcm), 3200):
        asyncio.run(session.feed(pcm[i:i + 3200]))
    result = asyncio.run(session.finish())
    return session, pipeline, result


class TestVadMerging(unittest.TestCase):
    """切句回归：模型在 1~2 秒碎片上会输出乱码（用户真实报障），所以短停顿不许切、短片段不许单独定稿。"""

    def test_short_pauses_do_not_split(self):
        # 1.5s 语音 + 0.5s 停顿 ×3：短停顿不该把一句话切成三段，应合成一句
        session, _, result = _run([(1.5, 0.5)] * 3)
        self.assertEqual(len(session.segments), 1, "短停顿把句子切碎了")
        self.assertGreater(result["sentences"][-1]["end"] / 1000, 5.0)

    def test_tiny_fragments_merge(self):
        # 1.0s 语音 + 0.6s 停顿 ×4：每段都短于 MIN_DECODE_SEC，必须合并
        session, pipeline, result = _run([(1.0, 0.6)] * 4)
        self.assertEqual(len(session.segments), 1, "短片段被单独定稿了")

    def test_long_pause_splits(self):
        # 2s 语音 + 2.5s 停顿 ×2：这是真的说完了，应该分成两句
        session, _, result = _run([(2.0, 2.5), (2.0, 2.5)])
        self.assertEqual(len(session.segments), 2)
        self.assertGreater(result["sentences"][1]["start"], result["sentences"][0]["end"] - 0.01)

    def test_low_quality_defers_close(self):
        # 质量分很低时，1.0s 停顿只“试”不解——必须等到强制断句（1.0+1.5s）才定稿
        session, pipeline, _ = _run([(2.0, 1.2), (2.0, 3.0)], quality=0.05)
        self.assertGreaterEqual(pipeline.calls, 2)          # 至少试过一次 + 最终定稿
        self.assertEqual(len(session.segments), 1)          # 低质量不切碎：合成一句

    def test_silence_compression(self):
        # 段内 2s 长静音应被压到 SILENCE_KEEP_SEC 以内（否则模型会重复吐字）
        session, _, _ = _run([(1.0, 2.0), (1.0, 0.2)])
        audio = np.concatenate([
            np.random.default_rng(2).normal(0, 0.08, 16000).astype(np.float32),
            np.zeros(16000 * 2, dtype=np.float32),
            np.random.default_rng(3).normal(0, 0.08, 16000).astype(np.float32),
        ])
        compressed = session._compress_silence(audio)
        self.assertLess(len(compressed), len(audio) - 16000, "长静音没被压缩")

    def test_sentence_split_by_punctuation(self):
        session, _, _ = _run([(2.0, 2.5)])
        parts = session._split_sentences("第一句话。第二句话长一些。", 0.0, 10.0)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0][2], "第一句话。")
        self.assertTrue(parts[0][0] == 0.0 and abs(parts[1][1] - 10.0) < 0.01)
        self.assertLess(parts[0][1], parts[1][0] + 0.01)


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
