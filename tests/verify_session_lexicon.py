"""验证「词库随会话下发」：客户端发 config 后，服务端只对本次会话生效。

用法: ./venv/bin/python tests/verify_session_lexicon.py [wav] [host] [port]
依赖: 需要服务在跑；不依赖云端（走云端时会自动重试）
"""
import asyncio
import json
import sys
import wave
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
WAV = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "tests" / "fixture-6s.wav")
HOST = sys.argv[2] if len(sys.argv) > 2 else "172.17.0.1"
PORT = sys.argv[3] if len(sys.argv) > 3 else "4416"

# fixture-6s.wav 说的是「你好，这是微信语音接口测试，今天阳光很好。」
CORRECTION = {"wrong": "接口测试", "right": "接口验证"}


async def run(lexicon, expect):
    with wave.open(WAV, "rb") as w:
        pcm = w.readframes(w.getnframes())
    async with websockets.connect(f"ws://{HOST}:{PORT}/ws?engine=auto", max_size=None, open_timeout=30) as ws:
        await ws.send(json.dumps({"type": "config", "lexicon": lexicon}))   # 先发词库
        final = None

        async def drain():
            nonlocal final
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("type") in ("final", "error"):
                    final = msg
                    return

        task = asyncio.create_task(drain())
        for i in range(0, len(pcm), 640):
            await ws.send(pcm[i:i + 640])
            await asyncio.sleep(0.02)
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.wait_for(task, timeout=90)

    text = (final or {}).get("text", "")
    print(f"  词库={lexicon.get('corrections')} → {text!r}")
    assert expect in text, f"期望包含 {expect!r}，实际 {text!r}"
    return text


async def main():
    print("① 带纠正词库：应把「接口测试」改成「接口验证」")
    await run({"corrections": [CORRECTION]}, "接口验证")

    print("② 不带词库：应保持原样「接口测试」")
    await run({}, "接口测试")

    print("③ 只带热词（不应改动文本）")
    await run({"hotwords": ["接口验证"]}, "接口测试")
    print("✅ 全部通过：词库随会话下发，不影响其他会话")


if __name__ == "__main__":
    asyncio.run(main())
