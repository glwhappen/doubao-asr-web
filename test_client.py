"""本地验证：把 WAV 当浏览器麦克风推给 /ws，检查句子是否累积、时间戳是否存在。

用法: python test_client.py [wav] [host] [port]
"""
import asyncio
import json
import sys
import wave
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent
WAV = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "tests" / "fixture-6s.wav")
HOST = sys.argv[2] if len(sys.argv) > 2 else "172.17.0.1"
PORT = sys.argv[3] if len(sys.argv) > 3 else "4416"
ENGINE = sys.argv[4] if len(sys.argv) > 4 else "auto"
FRAME = 640


async def main():
    with wave.open(WAV, "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000), "需要 16k/mono/PCM16 的 wav"
        pcm = w.readframes(w.getnframes())
    print(f"音频 {len(pcm) / 32000:.2f}s → ws://{HOST}:{PORT}/ws?engine={ENGINE}")

    max_lines = 0
    shrunk = False
    final = None

    async with websockets.connect(f"ws://{HOST}:{PORT}/ws?engine={ENGINE}", max_size=None, open_timeout=60) as ws:
        # 客户端词库（现在由浏览器本地保存，每次会话发过来；测试里发空的一份）
        await ws.send(json.dumps({"type": "config", "lexicon": {"corrections": [], "hotwords": []}}))
        async def drain():
            nonlocal max_lines, shrunk, final
            while True:
                msg = json.loads(await ws.recv())
                kind = msg.get("type")
                if kind == "status":
                    print("  [status]", msg.get("engine"), msg.get("message"))
                if kind == "update":
                    n = len(msg.get("lines") or [])
                    if n < max_lines:
                        shrunk = True
                    max_lines = max(max_lines, n)
                elif kind == "final":
                    final = msg
                    print("最终:", json.dumps(msg.get("text", ""), ensure_ascii=False))
                    for line in msg.get("lines") or []:
                        print(f"   [{line['start']}–{line['end']}s] {line['text']}")
                    return
                elif kind == "error":
                    final = msg
                    return

        task = asyncio.create_task(drain())
        for i in range(0, len(pcm), FRAME):
            await ws.send(pcm[i:i + FRAME])
            await asyncio.sleep(0.02)          # 实时速度，跟浏览器麦克风一致
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.wait_for(task, timeout=60)

    assert final and final.get("type") == "final", f"没拿到最终结果: {final}"
    text = final.get("text") or ""
    lines = final.get("lines") or []
    assert text, "没有拿到文本"
    assert lines and all(l.get("ts") for l in lines), "缺少时间戳"
    assert not shrunk, f"出现了句子数量回退（说明历史被覆盖）：峰值 {max_lines}"
    print(f"✅ 通过：{len(lines)} 句，最长时 {max_lines} 句，无回退")


asyncio.run(main())
