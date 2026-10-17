#!/usr/bin/env python3
"""离线识别引擎（字节跳动 ASR 模型，纯 ONNX，本机 CPU）。

【这是什么】
    把从字节公开算法分发接口拉到的 ASR 模型（见 tools/fetch_offline_asr_models.sh）包成一个本地服务，
    不需要联网、不需要豆包客户端、不需要闭源引擎。网页 Demo 的「离线」档就是连它。

【模型与架构】
    前端：80 维 log-mel（25ms/10ms，HTK mel）→ LFR(7,6) → 560 维 → 逐句 CMVN 归一化
    识别：encoder(560→512) → decoder(CIF/连续积分激发，输出 logits + us_cif_peak) → 贪心解码
    后处理：classifier 质量分 + punc 标点模型（3 类）
    与豆包输入法 Flute 里的 ntp_cif 同源（CIF 架构），但这里是明文 ONNX，onnxruntime 直接跑。

【接口】刻意做成和本机 FunASR 引擎同一形状，server.py 里两套引擎复用同一段客户端逻辑：
    POST /stream                    {"language": "zh"} → {"session_id": ...}
    POST /stream/{sid}/chunk        裸 PCM16/16k/mono  → 累计文本 + 已定稿句子 + 当前未定稿片段
    POST /stream/{sid}/finish       收尾，返回全部句子
    POST /stream/{sid}/close        释放
    GET  /health                    模型/线程/阈值/延迟统计

【怎么切句】
    能量 VAD（自适应底噪）+ 停顿 0.6s 断句 + 单段上限 12s；
    单段说话期间每约 1.2s 用当前缓冲重解一次，作为「未定稿」实时上屏。

用法：
    ./venv/bin/python offline_engine.py                 # 起服务（默认 127.0.0.1:4272）
    ./venv/bin/python offline_engine.py --selftest A.wav [B.wav ...]   # 直接转写文件，验证模型
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

log = logging.getLogger("offline-asr")
logging.basicConfig(level=os.environ.get("DOUBAO_LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

MODEL_DIR = Path(os.environ.get("DOUBAO_OFFLINE_ASR_DIR", Path(__file__).parent / "models" / "offline-asr"))
HOST = os.environ.get("DOUBAO_OFFLINE_ASR_HOST", "127.0.0.1")
PORT = int(os.environ.get("DOUBAO_OFFLINE_ASR_PORT", "4272"))
TOKEN = os.environ.get("DOUBAO_OFFLINE_ASR_TOKEN", "")

MIN_SPEECH_SEC = 0.25        # 片段内至少这么多语音，否则当噪声丢
MIN_DECODE_SEC = 1.20        # 短于这个的片段不急着定稿（等下一句并进来），因为模型在短音频上会输出乱码
SILENCE_HANGOVER_SEC = 1.00  # 停顿超过它才“考虑”断句
MERGE_GAP_SEC = 1.50         # 再宽限这么久：期间又说话就并回同一句；到期无条件定稿
QUALITY_OK = 0.35            # 考虑断句时先解码看一眼：质量分≥它才真的断（防碎片乱码；实测乱码 0.00~0.29，完整句 0.44~0.72）
DECIDE_MIN_INTERVAL_SEC = 1.0  # 这种“试解码”最少隔多久做一次，省 CPU
MAX_SEGMENT_SEC = 15.0       # 单段上限，避免长段解码太慢
PAD_SEC = 0.15               # 解码前后各补一点静音，保全边界音节
SILENCE_KEEP_SEC = 0.18      # 段内长静音压到这么长（模型对长段静音会重复吐字）
SILENCE_CUT_SEC = 0.30       # 超过它才算“长静音”，需要压缩
SENTENCE_END = "。！？!?…"    # 解码后按这些标点拆成一行一句
PARTIAL_INTERVAL_SEC = 1.20  # 未定稿结果的刷新间隔
PARTIAL_MAX_SEC = 8.0        # 超过这个长度不再做未定稿解码（省 CPU）
PARTIAL_MIN_SEC = 2.00       # 短于这个不做未定稿（短音频解码就是乱码，会在页面上闪垃圾字）


# ------------------------------------------------------------------ 前端特征
_MEL_CACHE: dict[tuple, np.ndarray] = {}


def mel_filters(n_mels: int = 80, n_fft: int = 512, sr: int = SAMPLE_RATE,
                fmin: float = 20.0, fmax: float = 8000.0) -> np.ndarray:
    key = (n_mels, n_fft, sr, fmin, fmax)
    if key in _MEL_CACHE:
        return _MEL_CACHE[key]
    hz2mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)          # noqa: E731 (HTK mel，与 Kaldi 一致)
    mel2hz = lambda m: 700.0 * (np.exp(m / 1127.0) - 1.0)        # noqa: E731
    pts = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    bins = np.floor((n_fft // 2) * mel2hz(pts) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(left, center):
            if center > left:
                fb[m - 1, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center:
                fb[m - 1, k] = (right - k) / (right - center)
    _MEL_CACHE[key] = fb
    return fb


def fbank(x: np.ndarray, n_mels: int = 80, win: int = 400, hop: int = 160, n_fft: int = 512) -> np.ndarray:
    """Kaldi 风格 log-mel fbank（25ms 窗 / 10ms 步长，Hann→Hamming 与训练一致用 hamming）。"""
    if len(x) < win:
        return np.zeros((0, n_mels), dtype=np.float32)
    fb = mel_filters(n_mels, n_fft)
    window = np.hamming(win).astype(np.float32)
    n_frames = 1 + (len(x) - win) // hop
    frames = np.lib.stride_tricks.as_strided(
        x, (n_frames, win), (x.strides[0] * hop, x.strides[0])).copy()
    frames *= window
    power = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    return np.log(np.maximum(power @ fb.T, 1e-10)).astype(np.float32)


def lfr(feat: np.ndarray, m: int = 7, n: int = 6) -> np.ndarray:
    """低帧率：把连续 m 帧拼成 1 帧（步长 n），80→560 维。"""
    if feat.shape[0] < m:
        return np.zeros((0, feat.shape[1] * m), dtype=np.float32)
    left = (m - 1) // 2
    padded = np.pad(feat, ((left, m - 1 - left), (0, 0)), mode="edge")
    n_out = (feat.shape[0] - m) // n + 1
    return np.stack([padded[i * n: i * n + m].reshape(-1) for i in range(n_out)]).astype(np.float32)


def load_tokens(path: Path) -> dict[int, str]:
    tokens: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.rstrip("\n").split(" ")
        if len(parts) >= 2 and parts[-1].lstrip("-").isdigit():
            tokens[int(parts[-1])] = parts[0]
    return tokens


def load_punc_tokens(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {tok: i for i, tok in enumerate(text.split("|"))}


# ------------------------------------------------------------------ 识别流水线
@dataclass
class Segment:
    start: float
    end: float
    text: str
    quality: float = 0.0


@dataclass
class AsrPipeline:
    model_dir: Path = MODEL_DIR
    threads: int = 0
    available: bool = False
    reason: str = "未加载"
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self):
        self.enc = self.dec = self.cls = self.punc = None
        self.tokens: dict[int, str] = {}
        self.punc_tokens: dict[str, int] = {}
        self._punc_mark = {0: "，", 1: "", 2: "。"}   # 实测：1=不加标点，2=句号，0=逗号
        self.load()

    def load(self):
        try:
            import onnxruntime as ort
        except ImportError as exc:                   # noqa: BLE001
            self.reason = f"没装 onnxruntime（{exc}）；pip install onnxruntime"
            return
        need = {"asr-model-encoder.onnx", "asr-model-decoder.onnx", "asr-model-token.txt"}
        missing = [n for n in need if not (self.model_dir / n).exists()]
        if missing:
            self.reason = f"缺模型文件 {missing}，先跑 tools/fetch_offline_asr_models.sh"
            return
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.threads:
            opts.intra_op_num_threads = self.threads
        prov = ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(str(self.model_dir / "asr-model-encoder.onnx"), opts, providers=prov)
        self.dec = ort.InferenceSession(str(self.model_dir / "asr-model-decoder.onnx"), opts, providers=prov)
        for attr, name in (("cls", "asr-model-classifier.onnx"), ("punc", "asr-punc.onnx")):
            path = self.model_dir / name
            if path.exists():
                try:
                    setattr(self, attr, ort.InferenceSession(str(path), opts, providers=prov))
                except Exception as exc:             # noqa: BLE001
                    log.warning("加载 %s 失败（可选，跳过）：%r", name, exc)
        self.tokens = load_tokens(self.model_dir / "asr-model-token.txt")
        punc_path = self.model_dir / "asr-punc-token.txt"
        if punc_path.exists() and self.punc is not None:
            self.punc_tokens = load_punc_tokens(punc_path)
        self.available = self.tokens != {}
        self.reason = "已加载" if self.available else "token 表为空"
        if self.available:
            log.info("离线引擎就绪：%s（encoder/decoder%s%s，token %d）", self.model_dir,
                     "+classifier" if self.cls else "", "+punc" if self.punc else "", len(self.tokens))

    # ---- 推理 ----
    def _decode(self, pcm: np.ndarray) -> tuple[str, float]:
        feat = lfr(fbank(pcm))
        if feat.shape[0] == 0:
            return "", 0.0
        feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-10)   # 逐句 CMVN（实测这样才对得上）
        speech = feat[None].astype(np.float32)
        lens = np.array([feat.shape[0]], dtype=np.int32)
        enc, enc_len = self.enc.run(None, {"speech": speech, "speech_lengths": lens})
        logits = self.dec.run(None, {"enc": enc, "enc_len": enc_len})[0]
        idx = logits[0].argmax(-1)
        text = "".join(self.tokens.get(int(i), "") for i in idx if int(i) not in (0, 1, 2)).replace("@@", "")
        quality = 0.0
        if self.cls is not None:
            try:
                quality = float(self.cls.run(None, {"enc": enc, "enc_len": enc_len})[0][0][0])
            except Exception:                        # noqa: BLE001
                quality = 0.0
        return text, quality

    def _punctuate(self, text: str) -> str:
        if not text or self.punc is None or not self.punc_tokens:
            return text
        unk = self.punc_tokens.get("<unk>", 1)
        ids = np.array([[self.punc_tokens.get(ch, unk) for ch in text]], dtype=np.int32)
        try:
            logits = self.punc.run(None, {"inputs": ids, "text_lengths": np.array([ids.shape[1]], dtype=np.int32)})[0][0]
        except Exception as exc:                     # noqa: BLE001
            log.warning("标点模型失败：%r", exc)
            return text
        labels = logits.argmax(-1)
        out = []
        for ch, lab in zip(text, labels):
            out.append(ch + self._punc_mark.get(int(lab), ""))
        return "".join(out)

    async def transcribe(self, pcm: np.ndarray, punctuate: bool = True) -> tuple[str, float]:
        async with self._lock:                       # 推理串行化：CPU 上并发跑只会互相拖慢
            text, quality = await asyncio.to_thread(self._decode, pcm)
        if punctuate and text:
            text = await asyncio.to_thread(self._punctuate, text)
        return text, quality


# ------------------------------------------------------------------ 流式会话
class StreamSession:
    """一块一块收 PCM，做能量 VAD 切句，边收边出「未定稿」，停顿后出「已定稿」。

    切句是这套引擎最容出问题的地方：模型在 **1~2 秒的碎片上会输出乱码**
    （实测：完整句质量分 0.44~0.72、碎片乱码 0.001~0.29），所以策略是“宁可晚一点合，也不要把话切碎”：
      · 停顿 1.0s 只是“考虑断句”：先把当前缓冲解码看一眼，**质量分≥ 0.35 才真的断**；
      · 不够好就继续合，最长再等 1.5s，到期无条件定稿（不会无限拖）；
      · 短于 1.2s 的片段根本不去试断（模型一定会乱码），除非会话结束或段太长。
    """

    def __init__(self, pipeline: AsrPipeline, hotwords: list[str] | None = None):
        self.pipeline = pipeline
        self.hotwords = hotwords or []
        self.frames: list[np.ndarray] = []           # 当前片段（含尾部停顿）
        self.segments: list[Segment] = []            # 已定稿
        self.samples_total = 0
        self.seg_start_sample = 0
        self.silence_frames = 0
        self.speech_frames = 0
        self._pending = np.zeros(0, dtype=np.float32)
        self._recent: list[float] = []               # 最近几帧 RMS，做平滑
        self._noise = 0.004                          # 底噪 RMS（自适应）
        self._last_partial_at = 0.0
        self._last_decide_at = 0.0                   # 上次“试解码决定要不要断句”
        self.partial = ""
        self.created = time.time()

    # ---- VAD ----
    def _is_speech(self, rms: float) -> bool:
        return rms > max(0.0035, 1.8 * self._noise)

    def _adapt_noise(self, rms: float, is_speech: bool):
        if not is_speech:
            self._noise = 0.98 * self._noise + 0.02 * rms
        else:
            self._noise = 0.999 * self._noise + 0.001 * rms

    def _smooth(self, rms: float) -> float:
        """3 帧滑动平均：单帧的能量坑（爆破音/咝音）不断句。"""
        self._recent.append(rms)
        if len(self._recent) > 3:
            self._recent.pop(0)
        return float(sum(self._recent) / len(self._recent))

    async def feed(self, pcm_bytes: bytes) -> dict:
        """吃一块 PCM16，返回当前累计文本/定稿句子/未定稿片段。"""
        data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if self._pending.size:
            data = np.concatenate([self._pending, data])
        n_frames = len(data) // FRAME_SAMPLES
        self._pending = data[n_frames * FRAME_SAMPLES:].copy()
        closed: list[Segment] = []
        for i in range(n_frames):
            frame = data[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES]
            rms = self._smooth(float(np.sqrt(np.mean(frame ** 2)) + 1e-12))
            speech = self._is_speech(rms)
            self._adapt_noise(rms, speech)
            self.samples_total += FRAME_SAMPLES
            if speech:
                if not self.frames:
                    self.seg_start_sample = self.samples_total - FRAME_SAMPLES
                self.frames.append(frame)
                self.speech_frames += 1
                self.silence_frames = 0
            elif self.frames:
                self.frames.append(frame)            # 停顿时先留着
                self.silence_frames += 1
                pause = self.silence_frames * FRAME_MS / 1000
                speech_sec = self.speech_frames * FRAME_MS / 1000
                forced = pause >= SILENCE_HANGOVER_SEC + MERGE_GAP_SEC
                decide = (not forced and pause >= SILENCE_HANGOVER_SEC
                          and speech_sec >= MIN_DECODE_SEC
                          and time.time() - self._last_decide_at >= DECIDE_MIN_INTERVAL_SEC)
                if decide:
                    self._last_decide_at = time.time()
                if forced or decide:
                    seg = await self._maybe_close(force=forced)
                    if seg:
                        closed.append(seg)
            if self.frames and len(self.frames) * FRAME_MS / 1000 >= MAX_SEGMENT_SEC:
                seg = await self._maybe_close(force=True)
                if seg:
                    closed.append(seg)
        # 未定稿：说话中每隔一段时间用当前缓冲重解一次
        if self.frames and not closed:
            dur = len(self.frames) * FRAME_MS / 1000
            now = time.time()
            if (PARTIAL_MIN_SEC <= dur <= PARTIAL_MAX_SEC
                    and now - self._last_partial_at >= PARTIAL_INTERVAL_SEC and not self.partial):
                self._last_partial_at = now
                audio = np.concatenate(self.frames)
                text, _ = await self.pipeline.transcribe(audio, punctuate=False)
                self.partial = text
        return self.snapshot(closed)

    def _peek(self):
        audio = np.concatenate(self.frames) if self.frames else np.zeros(0, dtype=np.float32)
        start = self.seg_start_sample / SAMPLE_RATE
        dur = len(self.frames) * FRAME_MS / 1000
        speech_sec = self.speech_frames * FRAME_MS / 1000
        return audio, start, dur, speech_sec

    def _clear(self):
        self.frames = []
        self.silence_frames = 0
        self.speech_frames = 0
        self.partial = ""

    def _prepare(self, audio: np.ndarray, dur: float) -> np.ndarray:
        """解码前的音频整理：尾巴停顿留一点、段内长静音压短、两头补静音。

        为什么要压段内静音：把一句话中间的停顿合进同一段后，模型看到几秒的纯静音会**重复吐字**
        （实测「微信的。微信语音接口测试」这类重复），压到 0.18s 后就干净了。
        """
        tail = min(self.silence_frames * FRAME_MS / 1000, SILENCE_HANGOVER_SEC)
        keep = int((dur - tail) * SAMPLE_RATE)
        if 0 < keep < audio.size:
            audio = audio[:keep]
        audio = self._compress_silence(audio)
        pad = np.zeros(int(PAD_SEC * SAMPLE_RATE), dtype=np.float32)
        return np.concatenate([pad, audio, pad])

    def _compress_silence(self, audio: np.ndarray) -> np.ndarray:
        """把段内的长静音压到 SILENCE_KEEP_SEC（只动解码用的副本，不动时间戳）。"""
        n = len(audio) // FRAME_SAMPLES
        if n < 2:
            return audio
        frames = audio[:n * FRAME_SAMPLES].reshape(n, FRAME_SAMPLES)
        rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
        quiet = rms < max(0.0035, 1.8 * self._noise)
        keep_frames = int(SILENCE_KEEP_SEC * SAMPLE_RATE / FRAME_SAMPLES)
        cut_frames = int(SILENCE_CUT_SEC * SAMPLE_RATE / FRAME_SAMPLES)
        out, run = [], 0
        for i in range(n):
            if quiet[i]:
                run += 1
                if run <= keep_frames or run < cut_frames:
                    out.append(i)
                elif run == cut_frames:                    # 刚跨过阈值：把这段的开始重新保留
                    out.extend(range(i - run + 1, i - run + 1 + keep_frames))
            else:
                run = 0
                out.append(i)
        if len(out) == n:
            return audio
        keep_idx = np.array(sorted(set(out)), dtype=int)
        return frames[keep_idx].reshape(-1).astype(np.float32)

    def _split_sentences(self, text: str, start: float, end: float) -> list[tuple[float, float, str]]:
        """把一段的文本按句末标点拆成多行，时间按字数比例分（一行一句更可读）。"""
        parts, buf = [], ""
        for ch in text:
            buf += ch
            if ch in SENTENCE_END:
                parts.append(buf)
                buf = ""
        if buf.strip():
            parts.append(buf)
        if len(parts) <= 1:
            return [(start, end, text)]
        total = sum(len(p) for p in parts) or 1
        out, t = [], start
        for p in parts:
            dur = (end - start) * len(p) / total
            out.append((round(t, 2), round(t + dur, 2), p))
            t += dur
        return out

    async def _maybe_close(self, force: bool) -> Segment | None:
        """试断句：force=True 无条件定稿；否则先解码看质量分，不够好就继续合并。"""
        if not self.frames:
            return None
        audio, start, dur, speech_sec = self._peek()
        if speech_sec < MIN_SPEECH_SEC:
            if force:
                self._clear()                        # 纯噪声，丢掉
            return None
        text, quality = await self.pipeline.transcribe(self._prepare(audio, dur))
        if not text:
            if force:
                self._clear()
            return None
        if not force and quality < QUALITY_OK:
            log.info("试着断句但质量不够（%.2fs，质量 %.3f）→ 继续合并：%s", speech_sec, quality, text)
            return None
        end = start + dur
        self._clear()
        for word, repl in ((h.split("=>")[0], h.split("=>")[1]) for h in self.hotwords if "=>" in h):
            text = text.replace(word, repl)          # 热词：模型本身不吃 context，用同义替换兜一层
        emitted = None
        for s, e, part in self._split_sentences(text, start, end):
            seg = Segment(start=s, end=e, text=part, quality=quality)
            self.segments.append(seg)
            emitted = emitted or seg
        log.info("定稿 %.2f-%.2fs（%.1fs 语音，%d 字，质量 %.3f）: %s",
                 start, end, speech_sec, len(text), quality, text)
        return emitted

    def snapshot(self, closed: list[Segment]) -> dict:
        final_text = "".join(s.text for s in self.segments)
        return {
            "text": final_text + self.partial,
            "partial": self.partial,
            "sentences": [{"start": int(s.start * 1000), "end": int(s.end * 1000), "text": s.text,
                           "quality": round(s.quality, 4)} for s in self.segments],
            "closed": [{"start": int(s.start * 1000), "end": int(s.end * 1000), "text": s.text} for s in closed],
            "audio_seconds": round(self.samples_total / SAMPLE_RATE, 2),
        }

    async def finish(self) -> dict:
        closed = []
        if self.frames:
            seg = await self._maybe_close(force=True)
            if seg:
                closed.append(seg)
        return self.snapshot(closed)


# ------------------------------------------------------------------ HTTP 服务
APP = FastAPI(title="doubao-asr-web 离线引擎")
PIPELINE = AsrPipeline(threads=int(os.environ.get("DOUBAO_OFFLINE_ASR_THREADS", "0")))
SESSIONS: dict[str, StreamSession] = {}


def _check_token(request: Request):
    if TOKEN and request.headers.get("x-engine-token") != TOKEN:
        raise HTTPException(status_code=401, detail="bad engine token")


@APP.get("/health")
async def health():
    return {"ok": PIPELINE.available, "engine": "offline", "reason": PIPELINE.reason,
            "model_dir": str(MODEL_DIR), "tokens": len(PIPELINE.tokens),
            "punctuation": PIPELINE.punc is not None, "quality_model": PIPELINE.cls is not None,
            "sessions": len(SESSIONS), "sample_rate": SAMPLE_RATE,
            "vad": {"min_speech_sec": MIN_SPEECH_SEC, "hangover_sec": SILENCE_HANGOVER_SEC,
                    "max_segment_sec": MAX_SEGMENT_SEC}}


@APP.post("/stream")
async def open_session(request: Request):
    _check_token(request)
    if not PIPELINE.available:
        raise HTTPException(status_code=503, detail=f"离线引擎不可用：{PIPELINE.reason}")
    body = {}
    try:
        body = await request.json()
    except Exception:                                # noqa: BLE001
        pass
    hotwords = body.get("hotwords") or []
    if isinstance(hotwords, str):
        hotwords = [h for h in hotwords.split() if h]
    sid = uuid.uuid4().hex
    SESSIONS[sid] = StreamSession(PIPELINE, hotwords)
    log.info("新会话 %s（热词 %d 个）", sid, len(hotwords))
    return {"session_id": sid, "engine": "offline"}


@APP.post("/stream/{sid}/chunk")
async def chunk(sid: str, request: Request):
    _check_token(request)
    session = SESSIONS.get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    pcm = await request.body()
    if not pcm:
        return JSONResponse(session.snapshot([]))
    return JSONResponse(await session.feed(pcm))


@APP.post("/stream/{sid}/finish")
async def finish(sid: str, request: Request):
    _check_token(request)
    session = SESSIONS.pop(sid, None)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return JSONResponse(await session.finish())


@APP.post("/stream/{sid}/close")
async def close(sid: str, request: Request):
    _check_token(request)
    SESSIONS.pop(sid, None)
    return {"ok": True}


@APP.get("/")
async def root():
    return {"service": "doubao-asr-web offline engine", "health": "/health"}


# ------------------------------------------------------------------ 自测
def _read_wav(path: str) -> np.ndarray:
    with wave.open(path) as w:
        assert w.getframerate() == SAMPLE_RATE, f"{path} 需要 16k 采样率"
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


async def _selftest(paths: list[str]):
    if not PIPELINE.available:
        raise SystemExit(f"离线引擎不可用：{PIPELINE.reason}")
    for path in paths:
        audio = _read_wav(path)
        t0 = time.time()
        session = StreamSession(PIPELINE)
        await session.feed((audio * 32768).astype(np.int16).tobytes())
        result = await session.finish()
        audio_sec = len(audio) / SAMPLE_RATE
        log.info("%s：%.1fs 音频 → %.1fs 处理（%.2fx 实时）", Path(path).name, audio_sec,
                 time.time() - t0, (time.time() - t0) / max(audio_sec, 1e-6))
        print(f"  {Path(path).name}: {result['text']}")


def main():
    parser = argparse.ArgumentParser(description="豆包 ASR 离线引擎（纯 ONNX，本机 CPU）")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--selftest", nargs="+", metavar="WAV", help="不进服务，直接转写这些 wav 文件")
    args = parser.parse_args()
    if args.selftest:
        asyncio.run(_selftest(args.selftest))
        return
    if not PIPELINE.available:
        log.warning("引擎不可用：%s（服务仍会启动，/health 里能看到原因）", PIPELINE.reason)
    import uvicorn
    uvicorn.run(APP, host=args.host, port=args.port, log_level=os.environ.get("DOUBAO_LOG_LEVEL", "info").lower())


if __name__ == "__main__":
    main()
