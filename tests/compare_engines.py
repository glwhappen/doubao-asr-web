"""引擎效果对比：同一段音频分别走「在线」和「离线」，比准确率与延迟。

用法: ./venv/bin/python tests/compare_engines.py [host] [port]
依赖: 服务在跑；离线档需要离线引擎（systemctl status doubao-offline-asr）；
      在线档需要云端可用（失败会标记为 --）。

指标：CER = 字级编辑距离 / 标准答案字数（先去标点空格、统一大小写）。
"""
import asyncio
import json
import sys
import time
import wave
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent
HOST = sys.argv[1] if len(sys.argv) > 1 else "172.17.0.1"
PORT = sys.argv[2] if len(sys.argv) > 2 else "4416"

# 标准答案（去标点后比对）
CASES = [
    ("fixture-6s.wav", "你好这是微信语音接口测试今天阳光很好"),
    ("fixture-2s.wav", "现在进行第二次接口验证苹果香蕉橘子"),
    ("fixture-mixed-zh-en.wav", "你好这是微信语音接口测试今天阳光很好andsomytolmagaasknotwhatyourcountrycandoforyouaswhatyoucandoforyourcountry"),
]


def read_pcm(name: str) -> bytes:
    with wave.open(str(ROOT / "tests" / name), "rb") as w:
        return w.readframes(w.getnframes())


def normalize(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def cer(ref: str, hyp: str) -> float:
    """字级编辑距离 / 参考长度。"""
    ref, hyp = normalize(ref), normalize(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


async def run(engine: str, pcm: bytes) -> tuple[str, str, float]:
    """返回 (引擎名, 文本, 秒数)。"""
    t0 = time.time()
    text, used = "", "-"
    try:
        async with websockets.connect(f"ws://{HOST}:{PORT}/ws?engine={engine}", max_size=None, open_timeout=30) as ws:
            await ws.send(json.dumps({"type": "config", "lexicon": {}, "diarize": False}))
            for i in range(0, len(pcm), 3200):
                await ws.send(pcm[i:i + 3200])
                await asyncio.sleep(0.005)
            await ws.send(json.dumps({"type": "stop"}))
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=180))
                if msg.get("type") == "final":
                    text, used = msg.get("text") or "", msg.get("engine") or "-"
                    break
                if msg.get("type") == "error":
                    return f"错误:{msg.get('message')}", "-", time.time() - t0
    except Exception as exc:                          # noqa: BLE001
        return f"失败:{exc}", "-", time.time() - t0
    return text, used, time.time() - t0


async def main():
    print(f"对比目标 ws://{HOST}:{PORT}\n")
    header = f"{'音频':26s} {'引擎':10s} {'CER':>7s} {'耗时':>7s}  文本"
    print(header)
    print("-" * len(header))
    summary: dict[str, list[float]] = {"cloud": [], "offline": []}
    for name, ref in CASES:
        pcm = read_pcm(name)
        for engine in ("cloud", "offline"):
            text, used, secs = await run(engine, pcm)
            score = cer(ref, text) if not text.startswith(("失败", "错误")) else float("nan")
            if used in summary:
                summary[used].append(score)
            print(f"{name:26s} {used:10s} {score:7.3f} {secs:6.1f}s  {text[:70]}")
        print()
    print("平均 CER（越小越好）：")
    for engine, scores in summary.items():
        if scores:
            print(f"  {engine:8s} {sum(scores) / len(scores):.3f}   （{len(scores)} 段）")


if __name__ == "__main__":
    asyncio.run(main())
