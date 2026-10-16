"""说话人分离（本地 ONNX 模型，离线可用）。

用途: 会话结束后对整段录音做说话人分离，给每句识别结果打上「谁说的」标签。
      方案与 screenpipe 一致：pyannote 分割模型 + 3D-Speaker 声纹 + 聚类；
      这里直接用 sherpa-onnx 的 OfflineSpeakerDiarization（同一套配方，省得自己接 ONNX）。
用法: from diarization import DIARIZER;  DIARIZER.available();  DIARIZER.diarize(pcm_bytes) -> [Segment]
入参: PCM 为 16k/单声道/16bit 的 bytes；模型目录由 DOUBAO_DIAR_DIR 指定（默认 ./models）
出参: [Segment(start, end, speaker)]，speaker 从 1 开始按首次出现顺序编号
依赖: pip install -r requirements-diarization.txt（sherpa-onnx + numpy）
      模型用 tools/fetch_diarization_models.sh 下载
关键词: 说话人分离, 说话人识别, diarization, speaker, 声纹, pyannote, 3dspeaker, sherpa-onnx,
        campplus, 多人对话, 会议记录, 区分说话的人, speaker embedding, clustering
更新: 2026-10-16
"""
from __future__ import annotations

import logging
import os
import wave
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("doubao-asr.diar")

BASE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("DOUBAO_DIAR_DIR", BASE / "models"))
SEG_MODEL = MODEL_DIR / "segmentation.onnx"
EMB_MODEL = MODEL_DIR / "embedding.onnx"

# 聚类阈值：越小越容易把不同人合成一个，越大越容易把一个人拆成多个。
# sherpa-onnx 示例用 0.5，但实测 0.5 会把同一人的尾巴拆成"第三个人"，0.6~0.8 更稳，0.9 会过度合并。
CLUSTER_THRESHOLD = float(os.environ.get("DOUBAO_DIAR_THRESHOLD", "0.7"))
MIN_DURATION_ON = float(os.environ.get("DOUBAO_DIAR_MIN_ON", "0.3"))
MIN_DURATION_OFF = float(os.environ.get("DOUBAO_DIAR_MIN_OFF", "0.5"))
MAX_SECONDS = float(os.environ.get("DOUBAO_DIAR_MAX_SECONDS", "600"))  # 超过就跳过，防卡死
MIN_SPEAKER_SECONDS = float(os.environ.get("DOUBAO_DIAR_MIN_SPEAKER_SECONDS", "1.0"))  # 短于此的簇会被并掉


@dataclass
class Segment:
    start: float
    end: float
    speaker: int


class Diarizer:
    """惰性加载模型：只有真的要用分离时才占内存（约 60MB）。"""

    def __init__(self):
        self._impl = None
        self._failed = False

    # ---------------------------------------------------------------- 状态

    def available(self) -> bool:
        """sherpa-onnx 装了、模型也在，才算可用。"""
        if self._impl is not None:
            return True
        if self._failed or not (SEG_MODEL.exists() and EMB_MODEL.exists()):
            return False
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError:
            return False
        return True

    def status(self) -> dict:
        return {
            "available": self.available(),
            "model_dir": str(MODEL_DIR),
            "segmentation": SEG_MODEL.exists(),
            "embedding": EMB_MODEL.exists(),
            "threshold": CLUSTER_THRESHOLD,
        }

    # ---------------------------------------------------------------- 加载

    def _load(self):
        if self._impl is not None:
            return self._impl
        import sherpa_onnx

        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(SEG_MODEL)),
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMB_MODEL)),
            clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=CLUSTER_THRESHOLD),
            min_duration_on=MIN_DURATION_ON,
            min_duration_off=MIN_DURATION_OFF,
        )
        if not config.validate():
            raise RuntimeError("说话人分离模型配置不合法（文件可能损坏）")
        log.info("加载说话人分离模型：%s / %s", SEG_MODEL.name, EMB_MODEL.name)
        self._impl = sherpa_onnx.OfflineSpeakerDiarization(config)
        return self._impl

    # ---------------------------------------------------------------- 分离

    def diarize(self, pcm: bytes) -> list[Segment]:
        """对一段 16k/单声道/PCM16 音频做分离，返回按时间排序的片段（说话人从 1 开始）。"""
        import numpy as np

        if len(pcm) < 16000:                        # 不足 0.5 秒不值当
            return []
        seconds = len(pcm) / 32000
        if seconds > MAX_SECONDS:
            log.info("音频 %.0fs 超过上限 %.0fs，跳过说话人分离", seconds, MAX_SECONDS)
            return []

        try:
            impl = self._load()
        except Exception:                            # noqa: BLE001
            self._failed = True
            log.exception("说话人分离模型加载失败，本次跳过")
            return []

        samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
        try:
            raw = impl.process(samples).sort_by_start_time()
        except Exception:                            # noqa: BLE001
            log.exception("说话人分离失败，本次跳过")
            return []

        segments = [Segment(round(float(i.start), 2), round(float(i.end), 2), int(i.speaker)) for i in raw]
        segments = _merge_tiny_speakers(segments)

        # sherpa 给的 speaker id 是聚类编号（可能不从 0 开始、也不连续），
        # 这里按“首次出现顺序”重排成 1,2,3…，前端更好看
        order: dict[int, int] = {}
        out: list[Segment] = []
        for seg in segments:
            if seg.speaker not in order:
                order[seg.speaker] = len(order) + 1
            out.append(Segment(seg.start, seg.end, order[seg.speaker]))
        log.info("说话人分离完成：%.1fs 音频 → %d 段 / %d 人", seconds, len(out), len(order))
        return out


def _merge_tiny_speakers(segments: list[Segment]) -> list[Segment]:
    """把总时长过短的聚类并到相邻说话人上。

    聚类偶尔会把一小截（几百毫秒到一两秒）单独分给“第 3 个人”，
    这既不真实又会把句子标错。短于此阈值的簇按“前一段的说话人”合并。
    """
    if not segments:
        return segments
    total: dict[int, float] = {}
    for seg in segments:
        total[seg.speaker] = total.get(seg.speaker, 0.0) + max(0.0, seg.end - seg.start)
    keep = {spk for spk, dur in total.items() if dur >= MIN_SPEAKER_SECONDS}
    if len(keep) == len(total):                       # 没有需要合并的
        return segments
    if not keep:                                      # 全部都很短 → 只留最长的那个
        keep = {max(total, key=total.get)}
    out, last = [], None
    for seg in segments:
        if seg.speaker in keep:
            last = seg.speaker
            out.append(seg)
        elif last is not None:                        # 并到前一段
            out.append(Segment(seg.start, seg.end, last))
        # 开头就是短簇 → 先跳过（下一段会补上），避免凭空造一个人
    return out


DIARIZER = Diarizer()


# ---------------------------------------------------------------- 结果对齐

def assign_speakers(lines: list[dict], segments: list[Segment]) -> int:
    """把分离结果按“时间重叠最多”贴到每一句上；返回识别出的人数。

    lines 里每项要有 start/end（秒）。没有重叠时保持原样（不硬塞一个说话人）。
    """
    if not segments:
        return 0
    for line in lines:
        start = line.get("start")
        end = line.get("end")
        if start is None:
            continue
        end = start if end is None else end
        best, best_overlap = None, 0.0
        for seg in segments:
            overlap = min(end, seg.end) - max(start, seg.start)
            if overlap > best_overlap:
                best, best_overlap = seg.speaker, overlap
        if best is not None and best_overlap > 0:
            line["spk"] = best
    return max((s.speaker for s in segments), default=0)


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int = 16000) -> bytes:
    """调试用：把 PCM 包成 wav（也方便单独拿去听/复现）。"""
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()
