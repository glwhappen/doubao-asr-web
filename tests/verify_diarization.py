"""验证说话人分离：两段不同人的音频 → 每句带上正确的 spk。

用法: ./venv/bin/python tests/verify_diarization.py [host] [port]
依赖: 需要服务在跑，且已装说话人分离模型（tools/fetch_diarization_models.sh）
      ASR 结果本身由云端给出（会重试），说话人标签由本地模型给出。

测试音频现场合成：中文女声（fixture-6s）+ 1s 静音 + 英文男声（fixture-en-jfk），不依赖 ffmpeg。
"""
import asyncio
import json
import sys
import wave
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
HOST = sys.argv[1] if len(sys.argv) > 1 else "172.17.0.1"
PORT = sys.argv[2] if len(sys.argv) > 2 else "4416"


def build_two_speaker_pcm() -> bytes:
    """把两个不同说话人的 fixture 拼成一段（中间 1 秒静音）。"""
    pcm = b""
    for name in ("fixture-6s.wav", "fixture-en-jfk.wav"):
        with wave.open(str(ROOT / "tests" / name), "rb") as w:
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
            pcm += w.readframes(w.getnframes())
        pcm += b"\x00\x00" * 16000        # 1s 静音
    return pcm


async def main():
    pcm = build_two_speaker_pcm()
    print(f"测试音频 {len(pcm) / 32000:.1f}s（中文女声 + 静音 + 英文男声）")
    final = None
    live_labeled = []          # 录音过程中（还没点停止）就带上标签的句子
    streaming = True
    async with websockets.connect(f"ws://{HOST}:{PORT}/ws?engine=auto", max_size=None, open_timeout=30) as ws:
        await ws.send(json.dumps({"type": "config", "lexicon": {}, "diarize": True}))

        async def drain():
            nonlocal final
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("type") == "status" and "说话人" in (msg.get("message") or ""):
                    print("  服务端:", msg["message"])
                if msg.get("type") == "update" and streaming:
                    for line in msg.get("lines") or []:
                        if line.get("spk") and line.get("final"):
                            live_labeled.append((line.get("spk"), (line.get("text") or "")[:20]))
                if msg.get("type") in ("final", "error"):
                    final = msg
                    return

        task = asyncio.create_task(drain())
        for i in range(0, len(pcm), 640):
            await ws.send(pcm[i:i + 640])
            await asyncio.sleep(0.02)
        streaming = False          # 从这一刻起不再是"录音过程中"
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.wait_for(task, timeout=180)

    assert final and final.get("type") == "final", f"没拿到最终结果: {final}"
    lines = final.get("lines") or []
    print(f"\n录音过程中就拿到标签的句子：{live_labeled[:3]}{' …' if len(live_labeled) > 3 else ''}（共 {len(live_labeled)} 句）")
    print(f"识别 {len(lines)} 句，speakers={final.get('speakers')}")
    for line in lines:
        print(f"   spk={line.get('spk')}  {line.get('start')}-{line.get('end')}s  {line.get('text')[:40]}")
    labels = [l.get("spk") for l in lines if l.get("spk")]
    assert labels, "没有任何句子带上说话人标签"
    assert len(set(labels)) >= 2, f"应该分出至少 2 个人，实际 {sorted(set(labels))}"
    assert live_labeled, "录音过程中没有任何句子拿到标签（实时追踪没生效）"
    print("\n✅ 通过：说话人标签实时生效，且不同人拿到了不同标签")


if __name__ == "__main__":
    asyncio.run(main())
