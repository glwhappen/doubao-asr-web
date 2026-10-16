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

MIN_SPEECH_SEC = 0.30        # 短于这个的语音片段直接丢（咳嗽/碰麦）
SILENCE_HANGOVER_SEC = 0.60  # 停顿这么久就断句
MAX_SEGMENT_SEC = 12.0       # 单段上限，避免长段解码太慢
PARTIAL_INTERVAL_SEC = 1.20  # 未定稿结果的刷新间隔
PARTIAL_MAX_SEC = 8.0        # 超过这个长度不再做未定稿解码（省 CPU）


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
    """一块一块收 PCM，做能量 VAD 切句，边收边出「未定稿」，停顿后出「已定稿」。"""

    def __init__(self, pipeline: AsrPipeline, hotwords: list[str] | None = None):
        self.pipeline = pipeline
        self.hotwords = hotwords or []
        self.frames: list[np.ndarray] = []           # 当前语音片段（含尾部停顿）
        self.segments: list[Segment] = []            # 已定稿
        self.samples_total = 0
        self.seg_start_sample = 0
        self.silence_frames = 0
        self._pending = np.zeros(0, dtype=np.float32)
        self._noise = 0.004                          # 底噪 RMS（自适应）
        self._last_partial_at = 0.0
        self.partial = ""
        self.created = time.time()

    # ---- VAD ----
    def _is_speech(self, rms: float) -> bool:
        return rms > max(0.006, 2.2 * self._noise)

    def _adapt_noise(self, rms: float, is_speech: bool):
        if not is_speech:
            self._noise = 0.98 * self._noise + 0.02 * rms
        else:
            self._noise = 0.999 * self._noise + 0.001 * rms

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
            rms = float(np.sqrt(np.mean(frame ** 2)) + 1e-12)
            speech = self._is_speech(rms)
            self._adapt_noise(rms, speech)
            self.samples_total += FRAME_SAMPLES
            if speech:
                if not self.frames:
                    self.seg_start_sample = self.samples_total - FRAME_SAMPLES
                self.frames.append(frame)
                self.silence_frames = 0
            elif self.frames:
                self.frames.append(frame)            # 停顿也先留着，断不断由下面判断
                self.silence_frames += 1
                if self.silence_frames * FRAME_MS / 1000 >= SILENCE_HANGOVER_SEC:
                    seg = await self._close_segment()
                    if seg:
                        closed.append(seg)
            if self.frames and len(self.frames) * FRAME_MS / 1000 >= MAX_SEGMENT_SEC:
                seg = await self._close_segment()
                if seg:
                    closed.append(seg)
        # 未定稿：说话中每隔一段时间用当前缓冲重解一次
        if self.frames and not closed:
            dur = len(self.frames) * FRAME_MS / 1000
            now = time.time()
            if (dur <= PARTIAL_MAX_SEC and dur >= 0.8
                    and now - self._last_partial_at >= PARTIAL_INTERVAL_SEC and not self.partial):
                self._last_partial_at = now
                audio = np.concatenate(self.frames)
                text, _ = await self.pipeline.transcribe(audio, punctuate=False)
                self.partial = text
        return self.snapshot(closed)

    async def _close_segment(self) -> Segment | None:
        audio = np.concatenate(self.frames) if self.frames else np.zeros(0, dtype=np.float32)
        start = self.seg_start_sample / SAMPLE_RATE
        end = self.samples_total / SAMPLE_RATE
        self.frames = []
        self.silence_frames = 0
        self.partial = ""
        speech_sec = (end - start) - SILENCE_HANGOVER_SEC
        if speech_sec < MIN_SPEECH_SEC or audio.size < SAMPLE_RATE * MIN_SPEECH_SEC:
            return None
        text, quality = await self.pipeline.transcribe(audio)
        if not text:
            return None
        for word, repl in ((h.split("=>")[0], h.split("=>")[1]) for h in self.hotwords if "=>" in h):
            text = text.replace(word, repl)          # 热词：模型本身不吃 context，用同义替换兜一层
        seg = Segment(start=round(start, 2), end=round(end, 2), text=text, quality=quality)
        self.segments.append(seg)
        log.info("定稿 %.2f-%.2fs（%d 字，质量 %.3f）: %s", seg.start, seg.end, len(text), quality, text)
        return seg

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
            seg = await self._close_segment()
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
